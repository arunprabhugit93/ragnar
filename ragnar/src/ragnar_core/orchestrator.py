from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
import shlex
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from .agent_transcripts import AgentTranscriptStore, default_transcripts_path
from .approval_store import ApprovalStore, default_approvals_path
from .approval_broker import ApprovalBroker, Decision
from .config import RagnarConfig, default_config_path, load_config
from .conductor_decision import make_conductor_decision
from .context_broker import ContextBroker, ContextBudget
from .contracts import (
    agent_result_from_reply,
    build_invocation_contract,
    expected_review_output_schema,
    MemoryWriteback,
    provider_error_result,
    provider_free_result,
    review_result_from_reply,
    RoleReviewResult,
    SCHEMA_VERSION,
)
from .context_memory import ContextMemoryProvider, MemoryMode
from .edit_adapter import SafePatchAdapter
from .execution_profiles import profile_by_name
from .execution import LocalExecutionAdapter
from .intent_analyzer import analyze_intent
from .memory_writeback import MemoryWritebackStore, default_writeback_path
from .observability import RunRecorder, default_runs_path
from .pr_adapter import PullRequestDraftStore, build_pr_draft, default_pr_draft_dir
from .project_profiler import build_project_profile, qa_commands_from_profile
from .role_runtime import RoleRuntime, RoleRuntimeMode, RoleRuntimeResult, default_manifest_path, extract_json_object
from .role_registry import RoleContract, RoleRegistry, load_role_registry
from .run_ledger import RunLedgerRecord, RunLedgerStore, default_run_ledger_path, record_from_artifact
from .workspace import RoleWorkspaceManager


BuildRole = Literal["backend_engineer", "frontend_engineer", "workflow_engineer"]
BUILD_SEQUENCE: tuple[BuildRole, ...] = ("backend_engineer", "frontend_engineer", "workflow_engineer")
BUILD_ROLE_SET = set(BUILD_SEQUENCE)

MAX_PLAN_REVISIONS = 2
MAX_QA_REVISIONS = 2


class RagnarState(TypedDict, total=False):
    run_id: str
    objective: str
    mode: str
    phase: str
    selected_build_roles: list[str]
    context_queries: list[dict[str, Any]]
    memory_context: list[dict[str, Any]]
    qa_commands: list[list[str]]
    workspace_enabled: bool
    artifacts: Annotated[list[dict[str, Any]], operator.add]
    audit_events: Annotated[list[dict[str, Any]], operator.add]
    proposed_actions: Annotated[list[dict[str, Any]], operator.add]
    approval_requests: Annotated[list[dict[str, Any]], operator.add]
    blocked: bool
    final_report: str
    plan_review_verdict: str
    plan_review_feedback: str
    plan_revision_count: int
    qa_review_verdict: str
    qa_review_feedback: str
    qa_rework_roles: list[str]
    qa_revision_count: int
    owner_briefing: str
    project_profile: dict[str, Any]
    execution_mode: str
    conductor_decision: dict[str, Any]
    intent_analysis: dict[str, Any]
    clarification_question: str
    agent_call_count: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_roles_path() -> Path:
    return _repo_root() / "roles" / "ragnar_roles.yaml"


def _default_checkpoint_path() -> Path:
    return _repo_root() / ".ragnar" / "orchestrator.sqlite"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(node: str, message: str, **metadata: Any) -> dict[str, Any]:
    return {"at": _now(), "node": node, "message": message, "metadata": metadata}


def _artifact(kind: str, owner_role: str, body: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "owner_role": owner_role, "created_at": _now(), "body": body}


def _tokens(text: str) -> set[str]:
    return {token.strip(".,:;()[]{}<>!?").lower() for token in text.split()}


def _role_summary(role: RoleContract) -> dict[str, Any]:
    return {
        "role_id": role.role_id,
        "display_name": role.display_name,
        "team": role.team,
        "can": role.authority.get("can", []),
        "requires_approval": role.authority.get("requires_approval", []),
        "cannot": role.authority.get("cannot", []),
        "private_memory_namespace": role.private_memory_namespace,
        "shared_memory_namespaces": role.memory.get("shared_namespaces", []),
    }


class RagnarOrchestrator:
    """Strict LangGraph orchestration spine for Ragnar.

    The graph controls phase order and policy gates. Role nodes can become smarter
    internally, but they cannot skip graph gates or perform outward actions.
    """

    def __init__(
        self,
        registry: RoleRegistry,
        memory_mode: MemoryMode = "off",
        qa_commands: list[list[str]] | None = None,
        approvals_path: Path | None = None,
        runs_path: Path | None = None,
        manifest_path: Path | None = None,
        config_path: Path | None = None,
        record_runs: bool = True,
        prepare_worktrees: bool = True,
        role_runtime_mode: RoleRuntimeMode = "offline",
    ) -> None:
        self.registry = registry
        self.approval_broker = ApprovalBroker()
        repo_root = _repo_root()
        self.config: RagnarConfig = load_config(config_path or default_config_path())
        self.memory_provider = ContextMemoryProvider(mode=memory_mode)
        self.execution = LocalExecutionAdapter(repo_root)
        self.workspaces = RoleWorkspaceManager(repo_root, enabled=prepare_worktrees)
        self.patch_adapter = SafePatchAdapter(self.workspaces)
        self.role_runtime = RoleRuntime(
            manifest_path or default_manifest_path(repo_root),
            mode=role_runtime_mode,
            base_url=os.environ.get("LETTA_SERVER_URL", "http://localhost:8283"),
            api_key=os.environ.get("LETTA_API_KEY"),
            enable_agent_messaging=self.config.enable_agent_messaging(),
            agent_max_steps=self.config.agent_max_steps(),
        )
        self.approval_store = ApprovalStore(approvals_path or default_approvals_path(repo_root))
        self.memory_writebacks = MemoryWritebackStore(default_writeback_path(repo_root))
        self.run_ledger = RunLedgerStore(default_run_ledger_path(repo_root))
        self.context_broker = ContextBroker(
            self.registry,
            self.memory_provider,
            self.run_ledger,
            ContextBudget(
                max_total_hits=self.config.memory_max_total_hits(),
                max_hits_per_role=self.config.memory_max_hits_per_role(),
                max_hit_chars=self.config.memory_max_hit_chars(),
                max_handoffs=self.config.memory_max_handoffs(),
                max_handoff_chars=self.config.memory_max_handoff_chars(),
            ),
        )
        self.agent_transcripts = AgentTranscriptStore(default_transcripts_path(repo_root))
        self.pr_drafts = PullRequestDraftStore(default_pr_draft_dir(repo_root))
        self.recorder = RunRecorder(runs_path or default_runs_path(repo_root))
        self.qa_commands = qa_commands or []
        self.record_runs = record_runs

    def build_graph(self, checkpointer: Any | None = None) -> Any:
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError("Missing dependency: langgraph. Run `pip install -e .` from the ragnar directory.") from exc

        graph = StateGraph(RagnarState)
        graph.add_node("intake_objective", self.intake_objective)
        graph.add_node("intent_analyzer", self.intent_analyzer)
        graph.add_node("clarification_gate", self.clarification_gate)
        graph.add_node("project_profiler", self.project_profiler)
        graph.add_node("conductor_triage", self.conductor_triage)
        graph.add_node("researcher", self.researcher)
        graph.add_node("architect_plan", self.architect_plan)
        graph.add_node("conductor_review_plan", self.conductor_review_plan)
        graph.add_node("plan_approval_gate", self.plan_approval_gate)
        graph.add_node("dispatch_build_roles", self.dispatch_build_roles)
        graph.add_node("backend_engineer", self.backend_engineer)
        graph.add_node("frontend_engineer", self.frontend_engineer)
        graph.add_node("workflow_engineer", self.workflow_engineer)
        graph.add_node("qa_gate", self.qa_gate)
        graph.add_node("conductor_review_qa", self.conductor_review_qa)
        graph.add_node("integrator_prepare", self.integrator_prepare)
        graph.add_node("approval_gate", self.approval_gate)
        graph.add_node("final_report", self.final_report)

        graph.add_edge(START, "intake_objective")
        graph.add_edge("intake_objective", "intent_analyzer")
        graph.add_conditional_edges(
            "intent_analyzer",
            self.after_intent_analysis,
            {"clarification_gate": "clarification_gate", "project_profiler": "project_profiler"},
        )
        graph.add_edge("clarification_gate", "final_report")
        graph.add_edge("project_profiler", "conductor_triage")
        graph.add_conditional_edges(
            "conductor_triage",
            self.after_triage,
            {
                "researcher": "researcher",
                "architect_plan": "architect_plan",
                # "dispatch_build_roles" here means "ready for the build phase" --
                # routed through plan_approval_gate first, which itself passes
                # straight through when no plan artifact exists (no architect ran).
                "dispatch_build_roles": "plan_approval_gate",
            },
        )
        graph.add_conditional_edges(
            "researcher",
            self.after_research,
            {"architect_plan": "architect_plan", "dispatch_build_roles": "plan_approval_gate"},
        )
        graph.add_conditional_edges(
            "architect_plan",
            self.after_architect,
            {
                "conductor_review_plan": "conductor_review_plan",
                "dispatch_build_roles": "plan_approval_gate",
                "final_report": "final_report",
            },
        )
        graph.add_conditional_edges(
            "conductor_review_plan",
            self.after_plan_review,
            {
                "architect_plan": "architect_plan",
                "dispatch_build_roles": "plan_approval_gate",
                "final_report": "final_report",
            },
        )
        graph.add_conditional_edges(
            "plan_approval_gate",
            self.after_plan_approval,
            {"dispatch_build_roles": "dispatch_build_roles", "final_report": "final_report"},
        )
        graph.add_conditional_edges(
            "dispatch_build_roles",
            self.next_build_node,
            {
                "backend_engineer": "backend_engineer",
                "frontend_engineer": "frontend_engineer",
                "workflow_engineer": "workflow_engineer",
                "qa_gate": "qa_gate",
            },
        )
        graph.add_conditional_edges(
            "backend_engineer",
            self.after_backend,
            {"frontend_engineer": "frontend_engineer", "workflow_engineer": "workflow_engineer", "qa_gate": "qa_gate"},
        )
        graph.add_conditional_edges(
            "frontend_engineer",
            self.after_frontend,
            {"workflow_engineer": "workflow_engineer", "qa_gate": "qa_gate"},
        )
        graph.add_edge("workflow_engineer", "qa_gate")
        graph.add_conditional_edges(
            "qa_gate",
            self.after_qa_gate,
            {
                "conductor_review_qa": "conductor_review_qa",
                "approval_gate": "approval_gate",
                "final_report": "final_report",
            },
        )
        graph.add_conditional_edges(
            "conductor_review_qa",
            self.after_qa_review,
            {
                "backend_engineer": "backend_engineer",
                "frontend_engineer": "frontend_engineer",
                "workflow_engineer": "workflow_engineer",
                "dispatch_build_roles": "dispatch_build_roles",
                "integrator_prepare": "integrator_prepare",
            },
        )
        graph.add_edge("integrator_prepare", "approval_gate")
        graph.add_edge("approval_gate", "final_report")
        graph.add_edge("final_report", END)

        return graph.compile(checkpointer=checkpointer or InMemorySaver())

    def invoke(
        self,
        objective: str,
        thread_id: str | None = None,
        checkpoint_db: Path | None = None,
        run_id: str | None = None,
    ) -> RagnarState:
        run_id = run_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id or run_id}}
        initial_state = {
            "run_id": run_id,
            "objective": objective,
            "mode": "strict_intelligent",
            "phase": "created",
            "qa_commands": self.qa_commands,
            "workspace_enabled": self.workspaces.enabled,
            "config": {
                "version": self.config.version,
                "memory_mode": self.config.memory_mode(),
                "provider_default": self.config.to_dict().get("providers", {}).get("default"),
            },
        }
        if checkpoint_db is None:
            app = self.build_graph()
            state = app.invoke(initial_state, config=config)
            if self.record_runs:
                state["observability"] = self.recorder.write(state)
                state["memory_promotion"] = self._auto_promote_memory(state)
            return state

        os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: langgraph-checkpoint-sqlite. Run `pip install -e .` from the ragnar directory."
            ) from exc
        checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(checkpoint_db)) as checkpointer:
            app = self.build_graph(checkpointer=checkpointer)
            state = app.invoke(initial_state, config=config)
            if self.record_runs:
                state["observability"] = self.recorder.write(state)
                state["memory_promotion"] = self._auto_promote_memory(state)
            return state

    def _auto_promote_memory(self, state: RagnarState) -> dict[str, Any]:
        """Best-effort: a missing/misconfigured pgvector DB must not crash a run
        that otherwise completed -- promotion improves future runs' retrieval,
        it isn't a correctness requirement for the run that just finished.
        """
        try:
            promoted = self.memory_writebacks.promote_to_pgvector(run_id=str(state["run_id"]))
            return {"promoted": promoted}
        except Exception as exc:
            return {"error": str(exc)}

    def intake_objective(self, state: RagnarState) -> dict[str, Any]:
        objective = state["objective"].strip()
        if not objective:
            raise ValueError("Objective is required")
        return {
            "phase": "intake",
            "audit_events": [_event("intake_objective", "accepted owner objective", objective=objective)],
        }

    def intent_analyzer(self, state: RagnarState) -> dict[str, Any]:
        analysis = analyze_intent(state["objective"])
        analysis_dict = analysis.to_dict()
        return {
            "phase": "intent_analyzed",
            "intent_analysis": analysis_dict,
            "artifacts": [_artifact("intent_analysis", "conductor", analysis_dict)],
            "audit_events": [
                _event(
                    "intent_analyzer",
                    "classified objective intent",
                    intent=analysis.intent,
                    needs_clarification=analysis.needs_clarification,
                    missing_slots=analysis.missing_slots,
                )
            ],
        }

    def clarification_gate(self, state: RagnarState) -> dict[str, Any]:
        analysis = state.get("intent_analysis", {})
        question = str(analysis.get("question") or "Please clarify the target before Ragnar changes files.")
        return {
            "phase": "needs_clarification",
            "blocked": True,
            "clarification_question": question,
            "audit_events": [_event("clarification_gate", "paused for owner clarification", question=question)],
        }

    def project_profiler(self, state: RagnarState) -> dict[str, Any]:
        root = self.workspaces.git_root() or _repo_root()
        profile = build_project_profile(root).to_dict()
        return {
            "phase": "profiled",
            "project_profile": profile,
            "artifacts": [_artifact("project_profile", "conductor", profile)],
            "audit_events": [
                _event(
                    "project_profiler",
                    "detected project profile",
                    languages=profile["languages"],
                    frameworks=profile["frameworks"],
                )
            ],
        }

    def conductor_triage(self, state: RagnarState) -> dict[str, Any]:
        conductor = self.registry.get("conductor")
        objective = state["objective"]
        initial_build_roles = self._select_build_roles(objective)
        decision = make_conductor_decision(objective, initial_build_roles, state.get("project_profile"))
        selected_roles = decision.selected_build_roles
        if not self.config.trivial_fast_path() and decision.execution_profile == "fast_path":
            profile = profile_by_name("standard_path")
            decision = replace(
                decision,
                execution_profile="standard_path",
                architect_required=profile.use_architect,
                review_required=profile.use_conductor_plan_review or profile.use_conductor_qa_review == "always",
                # Matches conductor_decision.py's own formula: every profile allows
                # inter-agent chat now, unconditionally -- a single-role run can still
                # consult a peer, so this must not re-gate on role count.
                inter_agent_comm_required=profile.allow_inter_agent_chat,
                reason="Fast path disabled by configuration; using standard path.",
                budgets={
                    "max_agent_calls": profile.max_agent_calls,
                    "max_review_loops": profile.max_review_loops,
                    "max_memory_hits": profile.max_memory_hits,
                },
            )
        execution_mode = decision.execution_profile
        profile = profile_by_name(execution_mode)
        decision = replace(
            decision,
            budgets={
                **decision.budgets,
                "max_agent_calls": self._max_agent_calls_budget(profile, selected_roles),
            },
        )
        context_queries, memory_context = self.context_broker.retrieve_for_roles(objective, selected_roles)
        return {
            "phase": "triaged",
            "execution_mode": execution_mode,
            "conductor_decision": decision.to_dict(),
            "selected_build_roles": selected_roles,
            "context_queries": context_queries,
            "memory_context": memory_context,
            "artifacts": [
                _artifact(
                    "conductor_triage",
                    conductor.role_id,
                    {
                        "objective": objective,
                        "execution_mode": execution_mode,
                        "conductor_decision": decision.to_dict(),
                        "selected_build_roles": selected_roles,
                        "role_contract": _role_summary(conductor),
                        "routing_rule": "deterministic conductor decision layer with complexity/risk/research profiles",
                        "memory_lookups": {
                            "queries": len(context_queries),
                            "hits": sum(len(item.get("hits", [])) for item in memory_context),
                            "errors": sum(len(item.get("errors", [])) for item in memory_context),
                        },
                    },
                )
            ],
            "audit_events": [
                _event(
                    "conductor_triage",
                    "selected execution profile and build roles",
                    profile=execution_mode,
                    roles=selected_roles,
                    research_required=decision.research_required,
                    review_required=decision.review_required,
                )
            ],
        }

    def architect_plan(self, state: RagnarState) -> dict[str, Any]:
        revision_count = state.get("plan_revision_count", 0)
        feedback = {"conductor_feedback": state.get("plan_review_feedback")} if revision_count > 0 else None
        return self._role_execution_artifact(
            role_id="delivery_architect",
            action="create_delivery_plan",
            state=state,
            output_kind="architecture_plan",
            notes=[
                "Decompose the objective into scoped, ordered tasks with acceptance criteria.",
                "Every role action must be allowed by its role contract.",
                "Role diffs must stay inside each role's file-scope policy.",
                "No outward action may be executed without owner approval.",
                "Plan only -- do not implement.",
            ],
            rework_feedback=feedback,
        )

    def conductor_review_plan(self, state: RagnarState) -> dict[str, Any]:
        plan_artifact = self._latest_artifact(state, "architecture_plan")
        result = self._conductor_review(
            review_kind="plan_review",
            action="review_delivery_plan",
            state=state,
            subject_artifact=plan_artifact,
            revision_count_key="plan_revision_count",
            max_revisions=self.config.max_plan_revisions(default=MAX_PLAN_REVISIONS),
            review_instructions=[
                "Review the architect's delivery plan for completeness and fit to the objective.",
                "verdict='revise' sends it back to the architect with your feedback; "
                "verdict='approve' proceeds to the build roles.",
            ],
        )
        return {
            "phase": "plan_reviewed",
            "plan_review_verdict": result["verdict"],
            "plan_review_feedback": result["feedback"],
            "plan_revision_count": result["revision_count"],
            "artifacts": [result["artifact"]],
            "proposed_actions": result["proposed_actions"],
            "agent_call_count": result["agent_call_count"],
            "audit_events": [result["audit_event"]],
        }

    def plan_approval_gate(self, state: RagnarState) -> dict[str, Any]:
        """Owner checkpoint between plan creation and the build phase.

        Only fires when a delivery plan was actually produced -- fast_path/
        standard_path objectives with no architect step have nothing to
        approve here and pass straight through. Reuses the same
        ApprovalStore/ApprovalBroker pattern as the end-of-run approval_gate,
        so it resolves the same way: rerun with the same run_id after
        `/approve <run_id> conductor start_build_phase`.
        """
        plan_artifact = self._latest_artifact(state, "architecture_plan")
        if plan_artifact is None:
            return {
                "phase": "plan_approval_not_required",
                "audit_events": [
                    _event("plan_approval_gate", "no delivery plan produced; build phase needs no plan approval")
                ],
            }

        conductor = self.registry.get("conductor")
        action = "start_build_phase"
        decision = self.approval_broker.decide(conductor, action)
        if decision.decision is Decision.DENY:
            raise RuntimeError(f"conductor cannot start the build phase: {decision.reason}")
        if decision.decision is Decision.ALLOW:
            return {
                "phase": "plan_approval_not_required",
                "audit_events": [
                    _event("plan_approval_gate", "start_build_phase does not require owner approval by policy")
                ],
            }

        latest = self.approval_store.latest(state["run_id"], conductor.role_id, action)
        if latest and latest.status == "approved":
            return {
                "phase": "plan_approved",
                "audit_events": [_event("plan_approval_gate", "owner approved the delivery plan")],
            }
        if latest and latest.status == "denied":
            raise RuntimeError(f"Owner denied the delivery plan: {latest.reason}")
        if latest is None:
            latest = self.approval_store.record(state["run_id"], conductor.role_id, action, "pending", decision.reason)

        return {
            "phase": "awaiting_plan_approval",
            "blocked": True,
            "approval_requests": [
                {
                    "role_id": conductor.role_id,
                    "action": action,
                    "reason": decision.reason,
                    "requested_at": latest.recorded_at,
                    "status": "pending_owner_approval",
                }
            ],
            "audit_events": [_event("plan_approval_gate", "paused for owner plan approval")],
        }

    def after_plan_approval(self, state: RagnarState) -> str:
        if state.get("phase") == "awaiting_plan_approval":
            return "final_report"
        return "dispatch_build_roles"

    def dispatch_build_roles(self, state: RagnarState) -> dict[str, Any]:
        selected = state.get("selected_build_roles", [])
        return {
            "phase": "dispatched",
            "audit_events": [_event("dispatch_build_roles", "dispatching bounded role nodes", roles=selected)],
        }

    def _rework_feedback_for(self, state: RagnarState, role_id: str) -> dict[str, Any] | None:
        if role_id not in state.get("qa_rework_roles", []):
            return None
        qa_artifact = self._latest_artifact(state, "qa_verdict")
        qa_body = qa_artifact["body"] if qa_artifact else {}
        return {
            "conductor_feedback": state.get("qa_review_feedback"),
            "qa_verdict": qa_body.get("verdict"),
            "qa_commands": qa_body.get("commands", []),
        }

    def researcher(self, state: RagnarState) -> dict[str, Any]:
        return self._role_execution_artifact(
            role_id="researcher",
            action="produce_research_brief",
            state=state,
            output_kind="research_brief",
            notes=[
                "Research only the knowledge gaps named by the conductor decision.",
                "Return concise findings, citations or source notes when available, and implementation implications.",
                "Do not propose code patches or outward actions.",
                "Write durable memory only for reusable project/vendor facts, not one-off run narration.",
            ],
        )

    def backend_engineer(self, state: RagnarState) -> dict[str, Any]:
        return self._role_execution_artifact(
            role_id="backend_engineer",
            action="edit_backend_worktree",
            state=state,
            output_kind="backend_work_packet",
            notes=[
                "Implement the backend portion of the objective in the role worktree.",
                "Return proposed_patches as unified diffs for every file change.",
                "If no safe backend patch is possible, explain the missing repo evidence and do not propose push_branch.",
                "Inspect services, APIs, data models, migrations, and pipelines.",
                "Keep production data mutation out of scope without owner approval.",
            ],
            rework_feedback=self._rework_feedback_for(state, "backend_engineer"),
        )

    def frontend_engineer(self, state: RagnarState) -> dict[str, Any]:
        return self._role_execution_artifact(
            role_id="frontend_engineer",
            action="edit_frontend_worktree",
            state=state,
            output_kind="frontend_work_packet",
            notes=[
                "Implement the frontend portion of the objective in the role worktree.",
                "Return proposed_patches as unified diffs for every file change.",
                "If no safe frontend patch is possible, explain the missing repo evidence and do not propose push_branch.",
                "Inspect UI components, pages, client state, styling, and user-facing flows.",
                "Keep public branding changes behind owner approval.",
            ],
            rework_feedback=self._rework_feedback_for(state, "frontend_engineer"),
        )

    def workflow_engineer(self, state: RagnarState) -> dict[str, Any]:
        return self._role_execution_artifact(
            role_id="workflow_engineer",
            action="edit_workflow_worktree",
            state=state,
            output_kind="workflow_work_packet",
            notes=[
                "Implement the workflow/integration portion of the objective in the role worktree.",
                "Return proposed_patches as unified diffs for every file change.",
                "If no safe workflow patch is possible, explain the missing repo evidence and do not propose external actions.",
                "Inspect workflow configs, automation, webhooks, and system wiring.",
                "Do not enable external webhooks or production workflows without owner approval.",
            ],
            rework_feedback=self._rework_feedback_for(state, "workflow_engineer"),
        )

    def _deterministic_qa_writebacks(
        self,
        run_id: str,
        qa: RoleContract,
        verdict: str,
        failed_packet_facts: list[dict[str, Any]],
        command_results: list[dict[str, Any]],
        denied_commands: list[dict[str, Any]],
    ) -> list[MemoryWriteback]:
        """Durable QA findings captured from ground truth, independent of
        whether the QA agent's own reply chose to write a memory_writeback
        about them. A repeated pattern (e.g. a role's patches keep failing to
        apply) should be discoverable in future runs even if no model ever
        reflects on it.
        """
        if verdict != "fail":
            return []
        patch_failed_roles = sorted(
            {str(fact["owner_role"]) for fact in failed_packet_facts if fact.get("patch_apply_failed")}
        )
        failed_commands = [
            f"{' '.join(result['command'])} (exit {result['exit_code']})"
            for result in command_results
            if result.get("exit_code") != 0
        ]
        reasons = []
        if patch_failed_roles:
            reasons.append(f"patch failed to apply for: {', '.join(patch_failed_roles)}")
        if failed_commands:
            reasons.append(f"failing commands: {'; '.join(failed_commands)}")
        if denied_commands:
            reasons.append(f"{len(denied_commands)} command(s) denied by workspace policy")
        if not reasons:
            return []

        writebacks = [
            MemoryWriteback(
                role_id=qa.role_id,
                namespace="qa_findings",
                scope="shared",
                text=f"QA failed on run {run_id}: {'; '.join(reasons)}",
                tags=["ragnar", "qa_finding", "role:qa_engineer", "ground_truth"],
                source_run_id=run_id,
                source_artifact="qa_verdict",
            )
        ]
        for role_id in patch_failed_roles:
            if role_id not in BUILD_ROLE_SET:
                continue
            role = self.registry.get(role_id)
            writebacks.append(
                MemoryWriteback(
                    role_id=role_id,
                    namespace=role.private_memory_namespace,
                    scope="private",
                    text=(
                        f"A patch this role proposed in run {run_id} failed to apply "
                        "(git apply check failed). Re-verify diff context against the "
                        "actual current file content before proposing patches."
                    ),
                    tags=["ragnar", "qa_finding", "patch_apply_failed", f"role:{role_id}"],
                    source_run_id=run_id,
                    source_artifact="qa_verdict",
                )
            )
        return writebacks

    def qa_gate(self, state: RagnarState) -> dict[str, Any]:
        qa = self.registry.get("qa_engineer")
        decision = self.approval_broker.decide(qa, "produce_qa_verdict")
        if decision.decision is not Decision.ALLOW:
            raise RuntimeError(f"QA gate is not allowed: {decision.reason}")
        build_packets = [artifact for artifact in state.get("artifacts", []) if artifact["kind"].endswith("_work_packet")]
        selected_roles = state.get("selected_build_roles", []) or [
            str(artifact.get("owner_role"))
            for artifact in build_packets
            if artifact.get("owner_role") in BUILD_ROLE_SET
        ]
        reviewed_diffs = [self.workspaces.diff(state["run_id"], role_id).to_dict() for role_id in selected_roles]
        qa_workspace = {
            "role_id": qa.role_id,
            "enabled": self.workspaces.enabled,
            "available": any(report["available"] for report in reviewed_diffs),
            "git_root": str(self.workspaces.git_root()) if self.workspaces.git_root() else None,
            "branch": None,
            "worktree_path": None,
            "status": "reviewing_build_worktrees" if reviewed_diffs else "no_build_worktrees",
            "message": "QA reviews selected build role worktrees; it does not need its own build worktree.",
            # reviewed_worktrees deliberately lives only in observed_facts below, not
            # here -- both used to carry the same diff data into the same invocation
            # contract, doubling that payload's tokens for no reason.
        }
        qa_commands = state.get("qa_commands", [])
        commands_from_profile = False
        if not qa_commands and self.config.enable_qa_profile_discovery():
            qa_commands = qa_commands_from_profile(state.get("project_profile", {}))
            commands_from_profile = bool(qa_commands)
        command_results = []
        denied_commands = []
        for command in qa_commands:
            command_check = self.workspaces.check_command(qa.role_id, command)
            if not command_check.allowed:
                denied_commands.append({"command": command, "policy": command_check.to_dict()})
                continue
            command_results.append(self.execution.run(command).to_dict())
        command_failed = any(result["exit_code"] != 0 for result in command_results)
        failed_packets = [
            artifact
            for artifact in build_packets
            if (artifact.get("body", {}).get("agent_result", {}) or {}).get("status") in {"failed", "blocked"}
            # Ground truth over claimed status: a role can say status="completed" while
            # its unified diff never actually applied (git apply --check failure). QA
            # must not let that silently pass as if the work landed.
            or artifact.get("body", {}).get("patch_apply_failed")
            # A role skipped due to agent-call budget exhaustion gets a "no_provider"
            # agent_result -- the same status benign offline stubs use -- so it would
            # otherwise slip past this filter as if the role had simply run offline
            # rather than been skipped entirely.
            or artifact.get("body", {}).get("budget_exhausted")
        ]
        failed_packet_facts = [
            {
                "kind": artifact.get("kind"),
                "owner_role": artifact.get("owner_role"),
                "status": (artifact.get("body", {}).get("agent_result", {}) or {}).get("status"),
                "summary": (artifact.get("body", {}).get("agent_result", {}) or {}).get("summary"),
                "patch_apply_failed": bool(artifact.get("body", {}).get("patch_apply_failed")),
                "budget_exhausted": bool(artifact.get("body", {}).get("budget_exhausted")),
            }
            for artifact in failed_packets
        ]
        if command_failed or denied_commands or not build_packets or failed_packets:
            verdict = "fail"
        elif command_results:
            verdict = "pass"
        else:
            verdict = "pass_with_warnings"
        warnings = []
        if not command_results:
            warnings.append("No real QA command was configured for this run.")
        elif commands_from_profile:
            warnings.append("QA commands were discovered from the project profile, not explicitly configured.")
        if failed_packets:
            warnings.append("One or more build role packets failed or blocked; stopping without review retries.")

        profile = profile_by_name(str(state.get("execution_mode")))
        if not profile.use_qa_agent or (state.get("execution_mode") == "fast_path" and self.config.trivial_skip_qa_agent()):
            agent_result = provider_free_result(
                state["run_id"],
                qa,
                "qa_verdict",
                f"Deterministic fast-path QA facts; verdict is {verdict}.",
            )
            runtime_result = None
        else:
            runtime_result = "pending"

        # The deterministic verdict above is ground truth from real exit codes and stays
        # authoritative. This adds a narrative reasoning layer on top of it -- QA's agent
        # cannot override "fail" into "pass", it can only comment on/explain the result.
        role_context = self._role_memory_context(state, qa)
        policy = self.workspaces.policies.get("qa_engineer")
        observed_facts = {
            "verdict": verdict,
            "reviewed_artifacts": [artifact["kind"] for artifact in build_packets],
            "reviewed_worktrees": reviewed_diffs,
            "failed_packets": failed_packet_facts,
            "commands": command_results,
            "denied_commands": denied_commands,
            "warnings": warnings,
        }
        invocation = build_invocation_contract(
            run_id=state["run_id"],
            objective=state["objective"],
            action="produce_qa_verdict",
            role=qa,
            model=self.config.role_model("qa_engineer"),
            workspace=qa_workspace,
            policy=policy,
            memory_context=role_context
            + [{"scope": "observed_facts", "namespace": None, "query": {}, "hits": [observed_facts], "errors": []}],
            project_profile=state.get("project_profile"),
            compact=bool(self.config.compact_letta_invocations()),
            agent_messaging_allowed=bool(state.get("conductor_decision", {}).get("inter_agent_comm_required")),
        )
        agent_call_count = state.get("agent_call_count", 0)
        if runtime_result == "pending":
            runtime_result, agent_call_count = self._call_role_runtime(
                qa, state["objective"], "produce_qa_verdict", role_context, state, invocation=invocation
            )
            if runtime_result.raw_reply is not None:
                agent_result = agent_result_from_reply(state["run_id"], qa, "qa_verdict", runtime_result.raw_reply)
            elif runtime_result.status == "provider_error":
                agent_result = provider_error_result(state["run_id"], qa, "qa_verdict", runtime_result.message)
            elif runtime_result.status == "budget_exhausted":
                agent_result = provider_free_result(state["run_id"], qa, "qa_verdict", runtime_result.message)
            else:
                agent_result = provider_free_result(
                    state["run_id"], qa, "qa_verdict", f"Provider-free QA reasoning; deterministic verdict is {verdict}."
                )

        if self.record_runs:
            deterministic_writebacks = self._deterministic_qa_writebacks(
                state["run_id"], qa, verdict, failed_packet_facts, command_results, denied_commands
            )
            self.memory_writebacks.append_many(list(agent_result.memory_writebacks) + deterministic_writebacks)
        if self.record_runs and runtime_result is not None and runtime_result.transcript is not None:
            self.agent_transcripts.append(state["run_id"], qa.role_id, "produce_qa_verdict", runtime_result.transcript)

        proposed_actions = []
        rejected_actions = []
        for item in agent_result.proposed_actions:
            item_decision = self.approval_broker.decide(qa, item["action"])
            if item_decision.decision is Decision.DENY:
                rejected_actions.append({**item, "broker_reason": item_decision.reason})
                continue
            proposed_actions.append({"role_id": item["role_id"], "action": item["action"], "reason": item["reason"]})

        return {
            "phase": "qa_failed" if verdict == "fail" else "qa_complete",
            "artifacts": [
                _artifact(
                    "qa_verdict",
                    qa.role_id,
                    {
                        "verdict": verdict,
                        "reviewed_artifacts": [artifact["kind"] for artifact in build_packets],
                        "reviewed_worktrees": reviewed_diffs,
                        "failed_packets": failed_packet_facts,
                        "commands": command_results,
                        "denied_commands": denied_commands,
                        "warnings": warnings,
                        "agent_invocation": invocation.to_dict(),
                        "agent_result": agent_result.to_dict(),
                        "agent_transcript_summary": self._transcript_summary(runtime_result.transcript)
                        if runtime_result is not None
                        else None,
                        "rejected_actions": rejected_actions,
                    },
                )
            ],
            "proposed_actions": proposed_actions,
            "agent_call_count": agent_call_count,
            "audit_events": [_event("qa_gate", "produced QA verdict", verdict=verdict)],
        }

    def conductor_review_qa(self, state: RagnarState) -> dict[str, Any]:
        qa_artifact = self._latest_artifact(state, "qa_verdict")
        qa_body = qa_artifact.get("body", {}) if qa_artifact else {}
        # Ground truth wins over the conductor's own judgment: a role whose patch never
        # applied cannot be waved through as "approve" just because the model said so.
        patch_failed_roles = [
            str(packet.get("owner_role"))
            for packet in qa_body.get("failed_packets", [])
            if packet.get("patch_apply_failed") and packet.get("owner_role") in BUILD_ROLE_SET
        ]
        result = self._conductor_review(
            review_kind="qa_review",
            action="review_qa_verdict",
            state=state,
            subject_artifact=qa_artifact,
            revision_count_key="qa_revision_count",
            max_revisions=self.config.max_qa_revisions(default=MAX_QA_REVISIONS),
            review_instructions=[
                "Review the QA verdict and the build work packets it reviewed.",
                "verdict='revise' sends work back to the build role(s) named in rework_roles; "
                "verdict='approve' proceeds to integration.",
                "Name specific build role_ids in rework_roles (backend_engineer, frontend_engineer, "
                "workflow_engineer) when you know which one needs rework; leave it empty only if "
                "the whole build phase needs to be redone.",
            ],
            force_revise_roles=patch_failed_roles,
        )
        rework_roles = result["rework_roles"]
        if result["verdict"] == "revise" and not rework_roles:
            # Ambiguous rework target -- fall back to redoing every currently selected build role
            # rather than guessing, by routing through dispatch_build_roles' existing fan-out.
            rework_roles = list(state.get("selected_build_roles", []))
        return {
            "phase": "qa_reviewed",
            "qa_review_verdict": result["verdict"],
            "qa_review_feedback": result["feedback"],
            "qa_rework_roles": rework_roles,
            "qa_revision_count": result["revision_count"],
            "artifacts": [result["artifact"]],
            "proposed_actions": result["proposed_actions"],
            "agent_call_count": result["agent_call_count"],
            "audit_events": [result["audit_event"]],
        }

    def integrator_prepare(self, state: RagnarState) -> dict[str, Any]:
        integrator = self.registry.get("integrator")
        draft_decision = self.approval_broker.decide(integrator, "open_pull_request_draft")
        if draft_decision.decision is not Decision.ALLOW:
            raise RuntimeError(f"Integrator cannot prepare PR draft: {draft_decision.reason}")
        selected_roles = state.get("selected_build_roles", [])
        diff_reports = [self.workspaces.diff(state["run_id"], role_id).to_dict() for role_id in selected_roles]
        pr_draft = build_pr_draft(state["run_id"], state["objective"], diff_reports)
        pr_draft_path = str(self.pr_drafts.save(pr_draft)) if self.record_runs else None
        has_changes = any(report["changed_files"] for report in diff_reports)
        proposed_actions = []
        next_outward_action = None
        integration_summary = "Prepared integration packet from selected role outputs."
        if has_changes:
            next_outward_action = "open_pull_request"
            proposed_actions.append(
                {
                    "role_id": integrator.role_id,
                    "action": "open_pull_request",
                    "reason": "Opening a PR is an outward action and must be approved by the owner.",
                }
            )
        else:
            integration_summary = "No changed files were produced; PR opening is not proposed."
        return {
            "phase": "integration_prepared",
            "artifacts": [
                _artifact(
                    "integration_packet",
                    integrator.role_id,
                    {
                        "summary": integration_summary,
                        "included_artifacts": [artifact["kind"] for artifact in state.get("artifacts", [])],
                        "role_diffs": diff_reports,
                        "workspace_policy_ok": all(report["policy_ok"] for report in diff_reports),
                        "pull_request_draft": pr_draft.to_dict(),
                        "pull_request_draft_path": pr_draft_path,
                        "next_outward_action": next_outward_action,
                    },
                )
            ],
            "proposed_actions": proposed_actions,
            "audit_events": [_event("integrator_prepare", "prepared integration packet and proposed PR action")],
        }

    def approval_gate(self, state: RagnarState) -> dict[str, Any]:
        approval_requests = []
        blocked = False
        for proposed in state.get("proposed_actions", []):
            role = self.registry.get(str(proposed["role_id"]))
            decision = self.approval_broker.decide(role, str(proposed["action"]))
            if decision.decision is Decision.DENY:
                raise RuntimeError(f"Denied proposed action {proposed['action']}: {decision.reason}")
            if decision.decision is Decision.REQUIRE_APPROVAL:
                latest = self.approval_store.latest(state["run_id"], role.role_id, str(proposed["action"]))
                if latest and latest.status == "approved":
                    continue
                if latest and latest.status == "denied":
                    raise RuntimeError(f"Owner denied {role.role_id}/{proposed['action']}: {latest.reason}")
                if latest is None:
                    latest = self.approval_store.record(
                        state["run_id"],
                        role.role_id,
                        str(proposed["action"]),
                        "pending",
                        decision.reason,
                    )
                blocked = True
                approval_requests.append(
                    {
                        "role_id": role.role_id,
                        "action": proposed["action"],
                        "reason": decision.reason,
                        "requested_at": latest.recorded_at,
                        "status": "pending_owner_approval",
                    }
                )
        phase = "awaiting_owner_approval" if blocked else "approved_to_finish"
        return {
            "phase": phase,
            "blocked": blocked,
            "approval_requests": approval_requests,
            "audit_events": [_event("approval_gate", "evaluated proposed outward actions", blocked=blocked)],
        }

    def final_report(self, state: RagnarState) -> dict[str, Any]:
        requests = state.get("approval_requests", [])
        report = {
            "run_id": state["run_id"],
            "objective": state["objective"],
            "phase": state.get("phase"),
            "selected_build_roles": state.get("selected_build_roles", []),
            "artifact_count": len(state.get("artifacts", [])),
            "blocked": bool(state.get("blocked")),
            "approval_requests": requests,
        }
        briefing, agent_call_count = self._conductor_synthesize_briefing(state, report)
        return {
            "final_report": json.dumps(report, indent=2),
            "owner_briefing": briefing,
            "agent_call_count": agent_call_count,
            "audit_events": [_event("final_report", "created final orchestration report")],
        }

    def _conductor_synthesize_briefing(self, state: RagnarState, report: dict[str, Any]) -> tuple[str, int]:
        agent_call_count = state.get("agent_call_count", 0)
        if state.get("phase") == "needs_clarification":
            return str(state.get("clarification_question") or "Ragnar needs clarification before continuing."), agent_call_count
        if state.get("execution_mode") in {"fast_path", "standard_path"}:
            status_line = "Awaiting your approval." if report["blocked"] else "No approvals pending."
            roles_line = ", ".join(report["selected_build_roles"]) or "no build roles"
            return (
                f"Run {report['run_id']}: {report['phase']}. {state.get('execution_mode')} completed across "
                f"{roles_line} with {report['artifact_count']} artifacts. {status_line}"
            ), agent_call_count
        conductor = self.registry.get("conductor")
        decision = self.approval_broker.decide(conductor, "summarize_status")
        if decision.decision is Decision.DENY:
            raise RuntimeError(f"conductor cannot summarize_status: {decision.reason}")

        policy = self.workspaces.policies.get("conductor")
        workspace_report = self.workspaces.prepare(state["run_id"], "conductor")
        memory_context = [
            {
                "scope": "run_summary",
                "namespace": None,
                "query": {},
                "hits": [
                    {
                        "structured_report": report,
                        "plan_review": {
                            "verdict": state.get("plan_review_verdict"),
                            "feedback": state.get("plan_review_feedback"),
                        },
                        "qa_review": {
                            "verdict": state.get("qa_review_verdict"),
                            "feedback": state.get("qa_review_feedback"),
                            "rework_roles": state.get("qa_rework_roles", []),
                        },
                    }
                ],
                "errors": [],
            }
        ]
        invocation = build_invocation_contract(
            run_id=state["run_id"],
            objective=state["objective"],
            action="summarize_status",
            role=conductor,
            model=self.config.role_model("conductor"),
            workspace=workspace_report.to_dict(),
            policy=policy,
            memory_context=memory_context,
            expected_output_schema={
                "schema_version": SCHEMA_VERSION,
                "required_fields": ["briefing"],
                "rules": [
                    "Return JSON only, a single object with one field: briefing.",
                    "briefing is a short, natural-language, decision-first summary for the human owner: "
                    "what happened, what (if anything) needs their decision, and any risks flagged during review.",
                ],
            },
            project_profile=state.get("project_profile"),
            agent_messaging_allowed=bool(state.get("conductor_decision", {}).get("inter_agent_comm_required")),
        )
        runtime_result, agent_call_count = self._call_role_runtime(
            conductor, state["objective"], "summarize_status", memory_context, state, invocation=invocation
        )
        if self.record_runs and runtime_result.transcript is not None:
            self.agent_transcripts.append(state["run_id"], conductor.role_id, "summarize_status", runtime_result.transcript)
        if runtime_result.raw_reply is not None:
            try:
                payload = json.loads(extract_json_object(runtime_result.raw_reply))
                briefing = str(payload.get("briefing", "")).strip()
                if briefing:
                    return briefing, agent_call_count
            except (json.JSONDecodeError, ValueError, AttributeError):
                pass

        status_line = "Awaiting your approval." if report["blocked"] else "No approvals pending."
        roles_line = ", ".join(report["selected_build_roles"]) or "no build roles"
        return (
            f"Run {report['run_id']}: {report['phase']}. "
            f"{report['artifact_count']} artifacts produced across {roles_line}. {status_line}"
        ), agent_call_count

    def after_intent_analysis(self, state: RagnarState) -> str:
        if state.get("intent_analysis", {}).get("needs_clarification"):
            return "clarification_gate"
        return "project_profiler"

    def after_triage(self, state: RagnarState) -> str:
        decision = state.get("conductor_decision", {})
        if decision.get("research_required"):
            return "researcher"
        if decision.get("architect_required"):
            return "architect_plan"
        return "dispatch_build_roles"

    def after_research(self, state: RagnarState) -> str:
        decision = state.get("conductor_decision", {})
        if decision.get("architect_required"):
            return "architect_plan"
        return "dispatch_build_roles"

    def after_architect(self, state: RagnarState) -> str:
        profile = profile_by_name(str(state.get("execution_mode")))
        if profile.use_conductor_plan_review:
            return "conductor_review_plan"
        if state.get("execution_mode") == "planning_only" or not state.get("selected_build_roles", []):
            return "final_report"
        return "dispatch_build_roles"

    def after_qa_gate(self, state: RagnarState) -> str:
        profile = profile_by_name(str(state.get("execution_mode")))
        qa_artifact = self._latest_artifact(state, "qa_verdict")
        body = qa_artifact.get("body", {}) if qa_artifact else {}
        verdict = body.get("verdict")
        patch_apply_failed = any(packet.get("patch_apply_failed") for packet in body.get("failed_packets", []))
        if verdict == "fail":
            # A patch that silently failed to apply is a correctness bug, not a
            # judgment call -- it must always loop back for rework, even on
            # profiles (fast_path) that otherwise skip conductor QA review entirely.
            if patch_apply_failed or profile.use_conductor_qa_review in {"always", "on_failure"}:
                return "conductor_review_qa"
            return "final_report"
        if profile.use_conductor_qa_review == "always":
            return "conductor_review_qa"
        return "approval_gate"

    def next_build_node(self, state: RagnarState) -> str:
        selected = set(state.get("selected_build_roles", []))
        for role_id in BUILD_SEQUENCE:
            if role_id in selected:
                return role_id
        return "qa_gate"

    def after_backend(self, state: RagnarState) -> str:
        selected = set(state.get("selected_build_roles", []))
        if "frontend_engineer" in selected:
            return "frontend_engineer"
        if "workflow_engineer" in selected:
            return "workflow_engineer"
        return "qa_gate"

    def after_frontend(self, state: RagnarState) -> str:
        selected = set(state.get("selected_build_roles", []))
        if "workflow_engineer" in selected:
            return "workflow_engineer"
        return "qa_gate"

    def after_plan_review(self, state: RagnarState) -> str:
        if state.get("plan_review_verdict") == "revise":
            return "architect_plan"
        if state.get("execution_mode") == "planning_only" or not state.get("selected_build_roles", []):
            return "final_report"
        return "dispatch_build_roles"

    def after_qa_review(self, state: RagnarState) -> str:
        verdict = state.get("qa_review_verdict")
        if verdict in ("approve", "escalated_to_owner"):
            return "integrator_prepare"
        rework_roles = state.get("qa_rework_roles", [])
        if len(rework_roles) == 1 and rework_roles[0] in BUILD_SEQUENCE:
            return rework_roles[0]
        return "dispatch_build_roles"

    def _max_agent_calls_budget(self, profile: Any, selected_roles: list[str]) -> int:
        """Recompute the enforced call ceiling from the ACTUAL configured revision
        caps (ragnar.yaml's max_plan_revisions/max_qa_revisions), not a static
        per-profile guess.

        execution_profiles.py's own max_agent_calls constants were pure
        decoration until this budget was actually enforced -- they don't scale
        with max_plan_revisions/max_qa_revisions and would silently truncate a
        legitimately-configured revision loop before it ever reaches its own
        cap (each loop iteration costs 2+ real calls, and the cap defaults to
        5 each). This derives a ceiling generous enough to let the configured
        loops fully play out, still bounding truly pathological runs.
        """
        calls = len(selected_roles) + 2  # initial build pass + final briefing, with slack
        if profile.use_architect:
            plan_loop_iterations = self.config.max_plan_revisions(default=MAX_PLAN_REVISIONS) + 1
            calls += 2 * plan_loop_iterations if profile.use_conductor_plan_review else 1
        if profile.use_qa_agent:
            calls += 1
        # Always -- not just when profile.use_conductor_qa_review != "never" --
        # because a patch that fails to apply forces conductor_review_qa to run
        # regardless of profile (see after_qa_gate's patch_apply_failed override),
        # even on fast_path where this loop otherwise never fires.
        qa_loop_iterations = self.config.max_qa_revisions(default=MAX_QA_REVISIONS) + 1
        calls += 2 * qa_loop_iterations
        return calls

    def _select_build_roles(self, objective: str) -> list[str]:
        normalized = objective.lower().replace("-", "_").replace("/", " ")
        words = _tokens(normalized)
        selected: list[str] = []
        backend_terms = {
            "api",
            "apis",
            "backend",
            "database",
            "db",
            "migration",
            "migrations",
            "schema",
            "service",
            "pipeline",
            "auth",
            "login",
            "endpoint",
            "worker",
            "queue",
        }
        frontend_terms = {
            "html",
            "htm",
            "webpage",
            "webpages",
            "website",
            "websites",
            "static",
            "index",
            "landing",
            "ui",
            "ux",
            "frontend",
            "page",
            "pages",
            "react",
            "screen",
            "component",
            "components",
            "css",
            "style",
            "form",
            "dashboard",
            "button",
        }
        workflow_terms = {
            "workflow",
            "automation",
            "webhook",
            "webhooks",
            "integration",
            "integrations",
            "connector",
            "flow",
            "zapier",
            "n8n",
            "trigger",
        }
        frontend_typos = {"pag", "htm", "frntend", "fronted"}
        if words & backend_terms or any(phrase in normalized for phrase in ("data model", "business logic", "rest api")):
            selected.append("backend_engineer")
        if words & frontend_terms or words & frontend_typos or any(phrase in normalized for phrase in ("user interface", "design system", "landing page")):
            selected.append("frontend_engineer")
        if words & workflow_terms or any(phrase in normalized for phrase in ("third party", "third_party", "external system")):
            selected.append("workflow_engineer")
        if not selected:
            selected.append("backend_engineer")
        return selected

    def _is_trivial_objective(self, objective: str, selected_roles: list[str]) -> bool:
        if len(selected_roles) != 1:
            return False
        normalized = objective.lower().replace("-", " ")
        words = _tokens(normalized)
        trivial_terms = {
            "hello",
            "world",
            "simple",
            "static",
            "html",
            "page",
            "pag",
            "readme",
            "copy",
            "text",
            "css",
        }
        complex_terms = {
            "api",
            "auth",
            "database",
            "migration",
            "deploy",
            "payment",
            "oauth",
            "webhook",
            "integration",
            "production",
            "security",
            "schema",
            "backend",
        }
        if words & complex_terms:
            return False
        return bool(words & trivial_terms or "landing page" in normalized)

    def _merge_build_roles(self, current: list[str], requested: list[str]) -> list[str]:
        selected = set(current)
        selected.update(role_id for role_id in requested if role_id in BUILD_ROLE_SET)
        return [role_id for role_id in BUILD_SEQUENCE if role_id in selected]

    def _context_queries(self, objective: str, selected_roles: list[str]) -> list[dict[str, Any]]:
        return self.context_broker.context_queries(objective, selected_roles)

    def _role_execution_artifact(
        self,
        role_id: str,
        action: str,
        state: RagnarState,
        output_kind: str,
        notes: list[str],
        rework_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        role = self.registry.get(role_id)
        decision = self.approval_broker.decide(role, action)
        if decision.decision is Decision.DENY:
            raise RuntimeError(f"{role_id} cannot run {action}: {decision.reason}")
        if rework_feedback is None:
            existing = self.context_broker.already_done(state, role_id, action)
            if existing:
                return {
                    "phase": f"{role_id}_already_done",
                    "artifacts": [
                        _artifact(
                            output_kind,
                            role_id,
                            {
                                "objective": state["objective"],
                                "allowed_action": action,
                                "agent_result": {
                                    "schema_version": SCHEMA_VERSION,
                                    "run_id": state["run_id"],
                                    "role_id": role_id,
                                    "status": "completed",
                                    "summary": f"Skipped duplicate work; prior artifact already exists: {existing.get('artifact_kind')}.",
                                    "proposed_patches": [],
                                    "handoffs": [],
                                    "memory_writebacks": [],
                                    "proposed_actions": [],
                                    "requested_handoff_roles": [],
                                },
                                "already_done": existing,
                                "handoffs": [],
                                "memory_writebacks": [],
                                "patch_reports": [],
                                "notes": notes,
                            },
                        )
                    ],
                    "proposed_actions": [],
                    "audit_events": [_event(role_id, "skipped duplicate role work", action=action)],
                }
        role_context = self._role_memory_context(state, role)
        workspace_report = self.workspaces.prepare(state["run_id"], role_id)
        diff_report = self.workspaces.diff(state["run_id"], role_id) if workspace_report.available else None
        policy = self.workspaces.policies.get(role_id)
        invocation = build_invocation_contract(
            run_id=state["run_id"],
            objective=state["objective"],
            action=action,
            role=role,
            model=self.config.role_model(role_id),
            workspace=workspace_report.to_dict(),
            policy=policy,
            memory_context=role_context,
            handoff_inputs=self.context_broker.handoff_inputs(state, role_id),
            rework_feedback=rework_feedback,
            project_profile=state.get("project_profile"),
            compact=bool(self.config.compact_letta_invocations()),
            agent_messaging_allowed=bool(state.get("conductor_decision", {}).get("inter_agent_comm_required")),
        )
        runtime_result, agent_call_count = self._call_role_runtime(
            role, state["objective"], action, role_context, state, invocation=invocation
        )
        if runtime_result.raw_reply is not None:
            agent_result = agent_result_from_reply(state["run_id"], role, output_kind, runtime_result.raw_reply)
        elif runtime_result.status == "provider_error":
            agent_result = provider_error_result(state["run_id"], role, output_kind, runtime_result.message)
        elif runtime_result.status == "budget_exhausted":
            agent_result = provider_free_result(state["run_id"], role, output_kind, runtime_result.message)
        else:
            agent_result = provider_free_result(
                state["run_id"],
                role,
                output_kind,
                f"Provider-free bounded packet for {role_id}; no edits proposed.",
            )

        if self.record_runs and runtime_result.transcript is not None:
            self.agent_transcripts.append(state["run_id"], role_id, action, runtime_result.transcript)
        role_runtime_dict = runtime_result.to_dict()
        role_runtime_dict["transcript"] = None  # full detail lives in .ragnar/agent_transcripts.jsonl

        patch_reports: list[dict[str, Any]] = []
        if agent_result.proposed_patches and workspace_report.available and workspace_report.worktree_path:
            if self.config.allow_agent_edits():
                for patch in agent_result.proposed_patches:
                    report = self.patch_adapter.apply(role_id, Path(workspace_report.worktree_path), patch.unified_diff)
                    patch_reports.append({"patch_id": patch.patch_id, **report.to_dict()})
            else:
                patch_reports = [
                    {
                        "patch_id": patch.patch_id,
                        "applied": False,
                        "reason": "execution.allow_agent_edits is false in ragnar.yaml",
                    }
                    for patch in agent_result.proposed_patches
                ]
            diff_report = self.workspaces.diff(state["run_id"], role_id) if workspace_report.available else diff_report

        # Deterministic ground truth, independent of what the role claimed: it proposed
        # a patch, we had a real worktree and edits are configured to apply, but none of
        # the proposed patches actually landed (git apply --check failure, stale diff,
        # etc). qa_gate treats this as a failure regardless of agent_result.status.
        patch_apply_failed = (
            bool(agent_result.proposed_patches)
            and workspace_report.available
            and bool(workspace_report.worktree_path)
            and self.config.allow_agent_edits()
            and not any(report.get("applied") for report in patch_reports)
        )

        if self.record_runs:
            self.memory_writebacks.append_many(agent_result.memory_writebacks)

        # A model can hallucinate an action name the role has no grant for at all --
        # approval_gate() raises on outright DENY, so filter those out here instead
        # of letting untrusted model output crash the run. Anything requiring
        # approval or already allowed still reaches the real approval gate.
        proposed_actions = []
        rejected_actions = []
        for item in agent_result.proposed_actions:
            item_decision = self.approval_broker.decide(role, item["action"])
            if item_decision.decision is Decision.DENY:
                rejected_actions.append({**item, "broker_reason": item_decision.reason})
                continue
            proposed_actions.append({"role_id": item["role_id"], "action": item["action"], "reason": item["reason"]})

        requested_handoffs = (
            agent_result.requested_handoff_roles
            if agent_result.status == "blocked" and not agent_result.proposed_patches
            else []
        )
        next_selected_roles = self._merge_build_roles(state.get("selected_build_roles", []), requested_handoffs)
        artifact_body = {
            "objective": state["objective"],
            "allowed_action": action,
            "policy_decision": decision.decision.value,
            "role_contract": _role_summary(role),
            "context_queries": [
                query
                for query in state.get("context_queries", [])
                if query.get("namespace") in role.memory.get("shared_namespaces", [])
                or query.get("namespace") == role.private_memory_namespace
            ],
            "memory_context": role_context,
            "role_runtime": role_runtime_dict,
            "agent_transcript_summary": self._transcript_summary(runtime_result.transcript),
            "workspace": workspace_report.to_dict(),
            "diff": diff_report.to_dict() if diff_report else None,
            "agent_invocation": invocation.to_dict(),
            "agent_result": agent_result.to_dict(),
            "handoffs": [handoff.to_dict() for handoff in agent_result.handoffs],
            "requested_handoff_roles": requested_handoffs,
            "memory_writebacks": [writeback.to_dict() for writeback in agent_result.memory_writebacks],
            "patch_reports": patch_reports,
            "patch_apply_failed": patch_apply_failed,
            "budget_exhausted": runtime_result.status == "budget_exhausted",
            "rejected_actions": rejected_actions,
            "notes": notes,
        }
        artifact = _artifact(output_kind, role_id, artifact_body)
        ledger_record = record_from_artifact(state["run_id"], artifact)
        if self.record_runs and ledger_record is not None:
            self.run_ledger.append(ledger_record)

        return {
            "phase": f"{role_id}_complete",
            "selected_build_roles": next_selected_roles,
            "artifacts": [artifact],
            "proposed_actions": proposed_actions,
            "agent_call_count": agent_call_count,
            "audit_events": [
                _event(
                    role_id,
                    "completed bounded role node",
                    action=action,
                    requested_handoff_roles=requested_handoffs,
                )
            ],
        }

    def _role_memory_context(self, state: RagnarState, role: RoleContract) -> list[dict[str, Any]]:
        return self.context_broker.role_context(state, role)

    def _call_role_runtime(
        self,
        role: RoleContract,
        objective: str,
        action: str,
        context_hits: list[dict[str, Any]],
        state: RagnarState,
        invocation: Any = None,
    ) -> tuple[RoleRuntimeResult, int]:
        """Enforces conductor_decision's computed max_agent_calls as a real ceiling.

        Only meaningful within a single graph traversal -- a replay after an
        approval gate starts a fresh count, since neither LangGraph state nor
        the orchestrator instance survives across separate invoke() calls
        today. Still closes a real gap: nothing previously stopped a bad
        revision loop from making far more LLM calls than its execution
        profile's budget intended.
        """
        count = state.get("agent_call_count", 0)
        max_calls = (state.get("conductor_decision") or {}).get("budgets", {}).get("max_agent_calls")
        if max_calls and count >= max_calls:
            used_context_hits = sum(len(item.get("hits", [])) for item in context_hits)
            return (
                RoleRuntimeResult(
                    role_id=role.role_id,
                    runtime="budget_exhausted",
                    letta_agent_id=None,
                    memory_namespace=role.private_memory_namespace,
                    message=(
                        f"Skipped: run already used {count} agent calls, at or above the "
                        f"{max_calls}-call budget for {state.get('execution_mode')}."
                    ),
                    used_context_hits=used_context_hits,
                    status="budget_exhausted",
                ),
                count,
            )
        result = self.role_runtime.run(role, objective, action, context_hits, invocation=invocation)
        return result, count + 1

    def _latest_artifact(self, state: RagnarState, kind: str) -> dict[str, Any] | None:
        return next((artifact for artifact in reversed(state.get("artifacts", [])) if artifact["kind"] == kind), None)

    def _transcript_summary(self, transcript: dict[str, Any] | None) -> dict[str, Any] | None:
        if not transcript:
            return None
        messages = transcript.get("messages", [])
        return {
            "message_count": len(messages),
            "tool_calls": [message["tool_name"] for message in messages if message.get("tool_name")],
            "store": "agent_transcripts.jsonl",
        }

    def _conductor_review(
        self,
        review_kind: str,
        action: str,
        state: RagnarState,
        subject_artifact: dict[str, Any] | None,
        revision_count_key: str,
        max_revisions: int,
        review_instructions: list[str],
        force_revise_roles: list[str] | None = None,
    ) -> dict[str, Any]:
        conductor = self.registry.get("conductor")
        decision = self.approval_broker.decide(conductor, action)
        if decision.decision is Decision.DENY:
            raise RuntimeError(f"conductor cannot run {action}: {decision.reason}")

        # Every "rerun after approval" replays the WHOLE graph from START (no true
        # LangGraph interrupt/resume) -- build roles already skip redundant work via
        # already_done(), but conductor_review_plan/qa had no equivalent, so approving
        # a plan and rerunning always re-billed a conductor LLM call reviewing a plan
        # that hadn't changed since the last pass. Reuse the cached verdict when the
        # subject is provably unchanged since the last review of this exact action.
        signature = self._subject_signature(subject_artifact)
        if not force_revise_roles and signature is not None:
            cached = self.run_ledger.latest_for(str(state["run_id"]), conductor.role_id, action)
            if cached and cached.get("artifact_ref") == f"{review_kind}:{signature}":
                return self._reused_conductor_review(review_kind, state, subject_artifact, cached)

        policy = self.workspaces.policies.get("conductor")
        workspace_report = self.workspaces.prepare(state["run_id"], "conductor")
        memory_context = [
            {
                "scope": "review_subject",
                "namespace": None,
                "query": {"kind": review_kind, "instructions": review_instructions},
                "hits": [subject_artifact] if subject_artifact else [],
                "errors": [] if subject_artifact else [f"No {review_kind} subject artifact found."],
            }
        ]
        invocation = build_invocation_contract(
            run_id=state["run_id"],
            objective=state["objective"],
            action=action,
            role=conductor,
            model=self.config.role_model("conductor"),
            workspace=workspace_report.to_dict(),
            policy=policy,
            memory_context=memory_context,
            expected_output_schema=expected_review_output_schema(),
            project_profile=state.get("project_profile"),
            agent_messaging_allowed=bool(state.get("conductor_decision", {}).get("inter_agent_comm_required")),
        )
        runtime_result, agent_call_count = self._call_role_runtime(
            conductor, state["objective"], action, memory_context, state, invocation=invocation
        )
        if runtime_result.raw_reply is not None:
            review = review_result_from_reply(state["run_id"], conductor, runtime_result.raw_reply)
        elif runtime_result.status == "budget_exhausted":
            # Distinct from the true offline/no-provider fallback below: the run is
            # online, this review was simply skipped because it hit the per-run
            # call ceiling. Defaulting to "approve" here would let a real failure
            # silently sail through on a non-answer -- "revise" instead feeds the
            # same revision-count/escalation machinery, so repeated exhaustion
            # eventually escalates to a real gated owner approval instead of a
            # rubber-stamped pass.
            review = RoleReviewResult(
                schema_version=SCHEMA_VERSION,
                run_id=state["run_id"],
                role_id=conductor.role_id,
                verdict="revise",
                feedback=runtime_result.message,
                rework_roles=[],
            )
        else:
            review = RoleReviewResult(
                schema_version=SCHEMA_VERSION,
                run_id=state["run_id"],
                role_id=conductor.role_id,
                verdict="approve",
                feedback="Offline mode: auto-approved.",
                rework_roles=[],
            )

        if force_revise_roles:
            # Deterministic ground truth always wins over the conductor's own verdict --
            # it cannot rubber-stamp "approve" when a role's patch never actually landed.
            review = replace(
                review,
                verdict="revise",
                feedback=review.feedback or "A build role's patch failed to apply; forcing rework.",
                rework_roles=sorted(set(review.rework_roles) | set(force_revise_roles)),
            )

        revision_count = state.get(revision_count_key, 0)
        verdict = review.verdict
        if verdict == "revise":
            if revision_count >= max_revisions:
                verdict = "escalated_to_owner"
            else:
                revision_count += 1

        proposed_actions: list[dict[str, Any]] = []
        if verdict == "escalated_to_owner":
            proposed_actions.append(
                {
                    "role_id": conductor.role_id,
                    "action": "request_approval",
                    "reason": f"{review_kind} revision cap reached; proceeding under owner review flag.",
                }
            )

        if self.record_runs and runtime_result.transcript is not None:
            self.agent_transcripts.append(state["run_id"], conductor.role_id, action, runtime_result.transcript)
        role_runtime_dict = runtime_result.to_dict()
        role_runtime_dict["transcript"] = None  # full detail lives in .ragnar/agent_transcripts.jsonl

        artifact = _artifact(
            f"conductor_{review_kind}",
            conductor.role_id,
            {
                "objective": state["objective"],
                "reviewed_artifact_kind": subject_artifact["kind"] if subject_artifact else None,
                "verdict": verdict,
                "model_verdict": review.verdict,
                "feedback": review.feedback,
                "rework_roles": review.rework_roles,
                "revision_count": revision_count,
                "role_runtime": role_runtime_dict,
                "agent_transcript_summary": self._transcript_summary(runtime_result.transcript),
                "agent_invocation": invocation.to_dict(),
                "review_instructions": review_instructions,
            },
        )
        audit_event = _event(f"conductor_{review_kind}", "conductor reviewed artifact", verdict=verdict)

        if self.record_runs and signature is not None:
            self._cache_conductor_review(
                state, conductor, action, review_kind, signature, verdict, review.feedback, review.rework_roles, revision_count
            )

        return {
            "verdict": verdict,
            "feedback": review.feedback,
            "rework_roles": review.rework_roles,
            "revision_count": revision_count,
            "artifact": artifact,
            "audit_event": audit_event,
            "proposed_actions": proposed_actions,
            "agent_call_count": agent_call_count,
        }

    def _subject_signature(self, artifact: dict[str, Any] | None) -> str | None:
        """Stable content fingerprint for the conductor-review dedup cache.

        Deliberately excludes volatile fields (embedded created_at timestamps
        on nested HandoffArtifact/MemoryWriteback dataclasses) so a replay
        that re-derives identical underlying work -- e.g. a build role's
        already_done skip-artifact standing in for the original real one --
        produces the same signature the original artifact did.
        """
        if not artifact:
            return None
        body = artifact.get("body", {})
        if artifact.get("kind") == "qa_verdict":
            payload = {
                "verdict": body.get("verdict"),
                "failed_packets": body.get("failed_packets"),
                "commands": [
                    {"command": c.get("command"), "exit_code": c.get("exit_code")} for c in body.get("commands", [])
                ],
                "denied_commands": body.get("denied_commands"),
            }
        else:
            already_done = body.get("already_done")
            if already_done:
                payload = {"summary": already_done.get("summary"), "changed_files": already_done.get("changed_files", [])}
            else:
                result = body.get("agent_result", {}) or {}
                diff = body.get("diff") or {}
                payload = {"summary": result.get("summary"), "changed_files": diff.get("changed_files", [])}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _reused_conductor_review(
        self,
        review_kind: str,
        state: RagnarState,
        subject_artifact: dict[str, Any] | None,
        cached: dict[str, Any],
    ) -> dict[str, Any]:
        payload = json.loads(cached.get("summary") or "{}")
        verdict = str(payload.get("verdict", "approve"))
        feedback = str(payload.get("feedback", ""))
        rework_roles = list(payload.get("rework_roles", []))
        revision_count = int(payload.get("revision_count", 0))
        artifact = _artifact(
            f"conductor_{review_kind}",
            "conductor",
            {
                "objective": state["objective"],
                "reviewed_artifact_kind": subject_artifact["kind"] if subject_artifact else None,
                "verdict": verdict,
                "model_verdict": verdict,
                "feedback": feedback,
                "rework_roles": rework_roles,
                "revision_count": revision_count,
                "reused_from_ledger": True,
                "note": (
                    "Subject artifact unchanged since the last review of this action in "
                    "this run; reused the cached verdict instead of re-billing an "
                    "identical conductor LLM call."
                ),
            },
        )
        audit_event = _event(
            f"conductor_{review_kind}", "reused cached conductor review; subject unchanged", verdict=verdict
        )
        proposed_actions: list[dict[str, Any]] = []
        if verdict == "escalated_to_owner":
            proposed_actions.append(
                {
                    "role_id": "conductor",
                    "action": "request_approval",
                    "reason": f"{review_kind} revision cap reached; proceeding under owner review flag.",
                }
            )
        return {
            "verdict": verdict,
            "feedback": feedback,
            "rework_roles": rework_roles,
            "revision_count": revision_count,
            "artifact": artifact,
            "audit_event": audit_event,
            "proposed_actions": proposed_actions,
            "agent_call_count": state.get("agent_call_count", 0),
        }

    def _cache_conductor_review(
        self,
        state: RagnarState,
        conductor: RoleContract,
        action: str,
        review_kind: str,
        signature: str,
        verdict: str,
        feedback: str,
        rework_roles: list[str],
        revision_count: int,
    ) -> None:
        run_id = str(state["run_id"])
        payload = json.dumps(
            {"verdict": verdict, "feedback": feedback, "rework_roles": rework_roles, "revision_count": revision_count},
            sort_keys=True,
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {"run_id": run_id, "role_id": conductor.role_id, "action": action, "signature": signature, "revision_count": revision_count},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self.run_ledger.append(
            RunLedgerRecord(
                run_id=run_id,
                role_id=conductor.role_id,
                action=action,
                status=verdict,
                artifact_kind=f"conductor_{review_kind}",
                artifact_ref=f"{review_kind}:{signature}",
                summary=payload,
                changed_files=[],
                handoff_to=list(rework_roles),
                fingerprint=fingerprint,
            )
        )


def run_objective(
    objective: str,
    roles_path: Path | None = None,
    checkpoint_db: Path | None = None,
    run_id: str | None = None,
    memory_mode: MemoryMode = "off",
    qa_commands: list[list[str]] | None = None,
    record_runs: bool = False,
    prepare_worktrees: bool = False,
    config_path: Path | None = None,
    role_runtime_mode: RoleRuntimeMode = "offline",
    approvals_path: Path | None = None,
) -> RagnarState:
    registry = load_role_registry(roles_path or _default_roles_path())
    orchestrator = RagnarOrchestrator(
        registry,
        memory_mode=memory_mode,
        qa_commands=qa_commands,
        record_runs=record_runs,
        prepare_worktrees=prepare_worktrees,
        config_path=config_path,
        role_runtime_mode=role_runtime_mode,
        approvals_path=approvals_path,
    )
    return orchestrator.invoke(objective, checkpoint_db=checkpoint_db, run_id=run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the strict Ragnar LangGraph orchestrator.")
    parser.add_argument("objective")
    parser.add_argument("--roles", type=Path, default=_default_roles_path())
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--checkpoint-db", type=Path, default=_default_checkpoint_path())
    parser.add_argument("--run-id", help="Reuse a run ID, usually after approving a pending action.")
    parser.add_argument(
        "--memory-mode",
        choices=["off", "auto", "pgvector", "graphiti", "all"],
        default="auto",
        help="Memory retrieval mode. Graphiti requires its local service and provider config.",
    )
    parser.add_argument(
        "--qa-command",
        action="append",
        default=[],
        help="Local QA command to run, split with shell-like spaces. Can be passed multiple times.",
    )
    parser.add_argument("--no-checkpoint", action="store_true", help="Use in-memory checkpoints for this run.")
    parser.add_argument("--no-record-runs", action="store_true", help="Do not write .ragnar/runs observability files.")
    parser.add_argument("--no-worktrees", action="store_true", help="Do not prepare isolated role Git worktrees.")
    parser.add_argument(
        "--role-runtime",
        choices=["offline", "letta"],
        default="letta",
        help="offline: no LLM calls, stub packets only. letta: call each role's provisioned Letta agent.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full graph state as JSON.")
    args = parser.parse_args()

    qa_commands = [shlex.split(command) for command in args.qa_command]
    state = run_objective(
        args.objective,
        args.roles,
        None if args.no_checkpoint else args.checkpoint_db,
        run_id=args.run_id,
        memory_mode=args.memory_mode,
        qa_commands=qa_commands,
        record_runs=not args.no_record_runs,
        prepare_worktrees=not args.no_worktrees,
        config_path=args.config,
        role_runtime_mode=args.role_runtime,
    )
    if args.json:
        print(json.dumps(state, indent=2, default=str))
        return
    print(state["final_report"])


if __name__ == "__main__":
    main()
