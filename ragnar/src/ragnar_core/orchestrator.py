from __future__ import annotations

import argparse
import json
import operator
import os
import shlex
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from .agent_transcripts import AgentTranscriptStore, default_transcripts_path
from .approval_store import ApprovalStore, default_approvals_path
from .approval_broker import ApprovalBroker, Decision
from .config import RagnarConfig, default_config_path, load_config
from .contracts import (
    agent_result_from_reply,
    build_invocation_contract,
    expected_review_output_schema,
    provider_error_result,
    provider_free_result,
    review_result_from_reply,
    RoleReviewResult,
    SCHEMA_VERSION,
)
from .context_memory import ContextMemoryProvider, MemoryMode
from .edit_adapter import SafePatchAdapter
from .execution import LocalExecutionAdapter
from .memory_writeback import MemoryWritebackStore, default_writeback_path
from .observability import RunRecorder, default_runs_path
from .pr_adapter import PullRequestDraftStore, build_pr_draft, default_pr_draft_dir
from .project_profiler import build_project_profile, qa_commands_from_profile
from .role_runtime import RoleRuntime, RoleRuntimeMode, default_manifest_path, extract_json_object
from .role_registry import RoleContract, RoleRegistry, load_role_registry
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
        graph.add_node("project_profiler", self.project_profiler)
        graph.add_node("conductor_triage", self.conductor_triage)
        graph.add_node("architect_plan", self.architect_plan)
        graph.add_node("conductor_review_plan", self.conductor_review_plan)
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
        graph.add_edge("intake_objective", "project_profiler")
        graph.add_edge("project_profiler", "conductor_triage")
        graph.add_edge("conductor_triage", "architect_plan")
        graph.add_edge("architect_plan", "conductor_review_plan")
        graph.add_conditional_edges(
            "conductor_review_plan",
            self.after_plan_review,
            {"architect_plan": "architect_plan", "dispatch_build_roles": "dispatch_build_roles"},
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
        graph.add_edge("qa_gate", "conductor_review_qa")
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
            return state

    def intake_objective(self, state: RagnarState) -> dict[str, Any]:
        objective = state["objective"].strip()
        if not objective:
            raise ValueError("Objective is required")
        return {
            "phase": "intake",
            "audit_events": [_event("intake_objective", "accepted owner objective", objective=objective)],
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
        selected_roles = self._select_build_roles(objective)
        context_queries = self._context_queries(objective, selected_roles)
        memory_context = self.memory_provider.retrieve(context_queries)
        return {
            "phase": "triaged",
            "selected_build_roles": selected_roles,
            "context_queries": context_queries,
            "memory_context": memory_context,
            "artifacts": [
                _artifact(
                    "conductor_triage",
                    conductor.role_id,
                    {
                        "objective": objective,
                        "selected_build_roles": selected_roles,
                        "role_contract": _role_summary(conductor),
                        "routing_rule": "deterministic phrase and keyword routing with conservative backend default",
                        "memory_lookups": {
                            "queries": len(context_queries),
                            "hits": sum(len(item.get("hits", [])) for item in memory_context),
                            "errors": sum(len(item.get("errors", [])) for item in memory_context),
                        },
                    },
                )
            ],
            "audit_events": [_event("conductor_triage", "selected build roles", roles=selected_roles)],
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
            "audit_events": [result["audit_event"]],
        }

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

    def qa_gate(self, state: RagnarState) -> dict[str, Any]:
        qa = self.registry.get("qa_engineer")
        decision = self.approval_broker.decide(qa, "produce_qa_verdict")
        if decision.decision is not Decision.ALLOW:
            raise RuntimeError(f"QA gate is not allowed: {decision.reason}")
        build_packets = [artifact for artifact in state.get("artifacts", []) if artifact["kind"].endswith("_work_packet")]
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
        if command_failed or denied_commands or not build_packets:
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

        # The deterministic verdict above is ground truth from real exit codes and stays
        # authoritative. This adds a narrative reasoning layer on top of it -- QA's agent
        # cannot override "fail" into "pass", it can only comment on/explain the result.
        role_context = self._role_memory_context(state, qa)
        policy = self.workspaces.policies.get("qa_engineer")
        workspace_report = self.workspaces.prepare(state["run_id"], "qa_engineer")
        observed_facts = {
            "verdict": verdict,
            "reviewed_artifacts": [artifact["kind"] for artifact in build_packets],
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
            workspace=workspace_report.to_dict(),
            policy=policy,
            memory_context=role_context
            + [{"scope": "observed_facts", "namespace": None, "query": {}, "hits": [observed_facts], "errors": []}],
            project_profile=state.get("project_profile"),
        )
        runtime_result = self.role_runtime.run(
            qa, state["objective"], "produce_qa_verdict", role_context, invocation=invocation
        )
        if runtime_result.raw_reply is not None:
            agent_result = agent_result_from_reply(state["run_id"], qa, "qa_verdict", runtime_result.raw_reply)
        elif runtime_result.status == "provider_error":
            agent_result = provider_error_result(state["run_id"], qa, "qa_verdict", runtime_result.message)
        else:
            agent_result = provider_free_result(
                state["run_id"], qa, "qa_verdict", f"Provider-free QA reasoning; deterministic verdict is {verdict}."
            )

        if self.record_runs:
            self.memory_writebacks.append_many(agent_result.memory_writebacks)
        if self.record_runs and runtime_result.transcript is not None:
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
            "phase": "qa_complete",
            "artifacts": [
                _artifact(
                    "qa_verdict",
                    qa.role_id,
                    {
                        "verdict": verdict,
                        "reviewed_artifacts": [artifact["kind"] for artifact in build_packets],
                        "commands": command_results,
                        "denied_commands": denied_commands,
                        "warnings": warnings,
                        "agent_invocation": invocation.to_dict(),
                        "agent_result": agent_result.to_dict(),
                        "agent_transcript_summary": self._transcript_summary(runtime_result.transcript),
                        "rejected_actions": rejected_actions,
                    },
                )
            ],
            "proposed_actions": proposed_actions,
            "audit_events": [_event("qa_gate", "produced QA verdict", verdict=verdict)],
        }

    def conductor_review_qa(self, state: RagnarState) -> dict[str, Any]:
        qa_artifact = self._latest_artifact(state, "qa_verdict")
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
        return {
            "final_report": json.dumps(report, indent=2),
            "owner_briefing": self._conductor_synthesize_briefing(state, report),
            "audit_events": [_event("final_report", "created final orchestration report")],
        }

    def _conductor_synthesize_briefing(self, state: RagnarState, report: dict[str, Any]) -> str:
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
        )
        runtime_result = self.role_runtime.run(
            conductor, state["objective"], "summarize_status", memory_context, invocation=invocation
        )
        if self.record_runs and runtime_result.transcript is not None:
            self.agent_transcripts.append(state["run_id"], conductor.role_id, "summarize_status", runtime_result.transcript)
        if runtime_result.raw_reply is not None:
            try:
                payload = json.loads(extract_json_object(runtime_result.raw_reply))
                briefing = str(payload.get("briefing", "")).strip()
                if briefing:
                    return briefing
            except (json.JSONDecodeError, ValueError, AttributeError):
                pass

        status_line = "Awaiting your approval." if report["blocked"] else "No approvals pending."
        roles_line = ", ".join(report["selected_build_roles"]) or "no build roles"
        return (
            f"Run {report['run_id']}: {report['phase']}. "
            f"{report['artifact_count']} artifacts produced across {roles_line}. {status_line}"
        )

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
        return "dispatch_build_roles"

    def after_qa_review(self, state: RagnarState) -> str:
        verdict = state.get("qa_review_verdict")
        if verdict in ("approve", "escalated_to_owner"):
            return "integrator_prepare"
        rework_roles = state.get("qa_rework_roles", [])
        if len(rework_roles) == 1 and rework_roles[0] in BUILD_SEQUENCE:
            return rework_roles[0]
        return "dispatch_build_roles"

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

    def _merge_build_roles(self, current: list[str], requested: list[str]) -> list[str]:
        selected = set(current)
        selected.update(role_id for role_id in requested if role_id in BUILD_ROLE_SET)
        return [role_id for role_id in BUILD_SEQUENCE if role_id in selected]

    def _context_queries(self, objective: str, selected_roles: list[str]) -> list[dict[str, Any]]:
        queries = [{"scope": "shared", "namespace": "project_context", "query": objective}]
        for role_id in selected_roles:
            role = self.registry.get(role_id)
            queries.append({"scope": "private", "namespace": role.private_memory_namespace, "query": objective})
            for namespace in role.memory.get("shared_namespaces", []):
                queries.append({"scope": "shared", "namespace": namespace, "query": objective})
        seen = set()
        deduped = []
        for query in queries:
            key = (query["scope"], query["namespace"], query["query"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(query)
        return deduped

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
            rework_feedback=rework_feedback,
            project_profile=state.get("project_profile"),
        )
        runtime_result = self.role_runtime.run(role, state["objective"], action, role_context, invocation=invocation)
        if runtime_result.raw_reply is not None:
            agent_result = agent_result_from_reply(state["run_id"], role, output_kind, runtime_result.raw_reply)
        elif runtime_result.status == "provider_error":
            agent_result = provider_error_result(state["run_id"], role, output_kind, runtime_result.message)
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

        return {
            "phase": f"{role_id}_complete",
            "selected_build_roles": next_selected_roles,
            "artifacts": [
                _artifact(
                    output_kind,
                    role_id,
                    {
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
                        "rejected_actions": rejected_actions,
                        "notes": notes,
                    },
                )
            ],
            "proposed_actions": proposed_actions,
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
        namespaces = set(role.memory.get("shared_namespaces", []))
        namespaces.add(role.private_memory_namespace)
        role_context = []
        for item in state.get("memory_context", []):
            query = item.get("query", {})
            if query.get("namespace") in namespaces:
                role_context.append(item)
        return role_context

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
    ) -> dict[str, Any]:
        conductor = self.registry.get("conductor")
        decision = self.approval_broker.decide(conductor, action)
        if decision.decision is Decision.DENY:
            raise RuntimeError(f"conductor cannot run {action}: {decision.reason}")

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
        )
        runtime_result = self.role_runtime.run(
            conductor, state["objective"], action, memory_context, invocation=invocation
        )
        if runtime_result.raw_reply is not None:
            review = review_result_from_reply(state["run_id"], conductor, runtime_result.raw_reply)
        else:
            review = RoleReviewResult(
                schema_version=SCHEMA_VERSION,
                run_id=state["run_id"],
                role_id=conductor.role_id,
                verdict="approve",
                feedback="Offline mode: auto-approved.",
                rework_roles=[],
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

        return {
            "verdict": verdict,
            "feedback": review.feedback,
            "rework_roles": review.rework_roles,
            "revision_count": revision_count,
            "artifact": artifact,
            "audit_event": audit_event,
            "proposed_actions": proposed_actions,
        }


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
