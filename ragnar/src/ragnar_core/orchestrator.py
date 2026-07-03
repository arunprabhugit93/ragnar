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

from .approval_store import ApprovalStore, default_approvals_path
from .approval_broker import ApprovalBroker, Decision
from .context_memory import ContextMemoryProvider, MemoryMode
from .execution import LocalExecutionAdapter
from .observability import RunRecorder, default_runs_path
from .role_runtime import RoleRuntime, default_manifest_path
from .role_registry import RoleContract, RoleRegistry, load_role_registry
from .workspace import RoleWorkspaceManager, policies_as_dict


BuildRole = Literal["backend_engineer", "frontend_engineer", "workflow_engineer"]
BUILD_SEQUENCE: tuple[BuildRole, ...] = ("backend_engineer", "frontend_engineer", "workflow_engineer")


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
        record_runs: bool = True,
        prepare_worktrees: bool = True,
    ) -> None:
        self.registry = registry
        self.approval_broker = ApprovalBroker()
        repo_root = _repo_root()
        self.memory_provider = ContextMemoryProvider(mode=memory_mode)
        self.execution = LocalExecutionAdapter(repo_root)
        self.workspaces = RoleWorkspaceManager(repo_root, enabled=prepare_worktrees)
        self.role_runtime = RoleRuntime(manifest_path or default_manifest_path(repo_root))
        self.approval_store = ApprovalStore(approvals_path or default_approvals_path(repo_root))
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
        graph.add_node("conductor_triage", self.conductor_triage)
        graph.add_node("architect_plan", self.architect_plan)
        graph.add_node("dispatch_build_roles", self.dispatch_build_roles)
        graph.add_node("backend_engineer", self.backend_engineer)
        graph.add_node("frontend_engineer", self.frontend_engineer)
        graph.add_node("workflow_engineer", self.workflow_engineer)
        graph.add_node("qa_gate", self.qa_gate)
        graph.add_node("integrator_prepare", self.integrator_prepare)
        graph.add_node("approval_gate", self.approval_gate)
        graph.add_node("final_report", self.final_report)

        graph.add_edge(START, "intake_objective")
        graph.add_edge("intake_objective", "conductor_triage")
        graph.add_edge("conductor_triage", "architect_plan")
        graph.add_edge("architect_plan", "dispatch_build_roles")
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
        graph.add_edge("qa_gate", "integrator_prepare")
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
        architect = self.registry.get("delivery_architect")
        plan_steps = [
            "Confirm objective and acceptance criteria.",
            "Retrieve role-private and shared context before execution.",
            "Prepare isolated role worktrees when Git is available.",
            "Perform role-scoped implementation work only in selected build roles.",
            "Enforce file-scope and command allowlist policy before execution.",
            "Run QA gate before integration.",
            "Request owner approval for outward actions.",
        ]
        return {
            "phase": "planned",
            "artifacts": [
                _artifact(
                    "architecture_plan",
                    architect.role_id,
                    {
                        "objective": state["objective"],
                        "selected_build_roles": state.get("selected_build_roles", []),
                        "plan_steps": plan_steps,
                        "acceptance_criteria": [
                            "Every role action must be allowed by its role contract.",
                            "Every selected build role must receive an isolated workspace report.",
                            "Role diffs must stay inside the role file-scope policy.",
                            "Local commands must match the role command allowlist.",
                            "No outward action may be executed without approval.",
                            "QA must produce a verdict before integration.",
                            "Integrator may prepare artifacts but cannot merge or deploy.",
                        ],
                        "workspace_policy": policies_as_dict(),
                    },
                )
            ],
            "audit_events": [_event("architect_plan", "created strict execution plan")],
        }

    def dispatch_build_roles(self, state: RagnarState) -> dict[str, Any]:
        selected = state.get("selected_build_roles", [])
        return {
            "phase": "dispatched",
            "audit_events": [_event("dispatch_build_roles", "dispatching bounded role nodes", roles=selected)],
        }

    def backend_engineer(self, state: RagnarState) -> dict[str, Any]:
        return self._role_execution_artifact(
            role_id="backend_engineer",
            action="draft_migrations",
            state=state,
            output_kind="backend_work_packet",
            notes=[
                "Inspect services, APIs, data models, migrations, and pipelines.",
                "Keep production data mutation out of scope without owner approval.",
            ],
        )

    def frontend_engineer(self, state: RagnarState) -> dict[str, Any]:
        return self._role_execution_artifact(
            role_id="frontend_engineer",
            action="run_frontend_build",
            state=state,
            output_kind="frontend_work_packet",
            notes=[
                "Inspect UI components, pages, client state, styling, and user-facing flows.",
                "Keep public branding changes behind owner approval.",
            ],
        )

    def workflow_engineer(self, state: RagnarState) -> dict[str, Any]:
        return self._role_execution_artifact(
            role_id="workflow_engineer",
            action="propose_integration_changes",
            state=state,
            output_kind="workflow_work_packet",
            notes=[
                "Inspect workflow configs, automation, webhooks, and system wiring.",
                "Do not enable external webhooks or production workflows without owner approval.",
            ],
        )

    def qa_gate(self, state: RagnarState) -> dict[str, Any]:
        qa = self.registry.get("qa_engineer")
        decision = self.approval_broker.decide(qa, "produce_qa_verdict")
        if decision.decision is not Decision.ALLOW:
            raise RuntimeError(f"QA gate is not allowed: {decision.reason}")
        build_packets = [artifact for artifact in state.get("artifacts", []) if artifact["kind"].endswith("_work_packet")]
        command_results = []
        denied_commands = []
        for command in state.get("qa_commands", []):
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
                    },
                )
            ],
            "audit_events": [_event("qa_gate", "produced QA verdict", verdict=verdict)],
        }

    def integrator_prepare(self, state: RagnarState) -> dict[str, Any]:
        integrator = self.registry.get("integrator")
        draft_decision = self.approval_broker.decide(integrator, "open_pull_request_draft")
        if draft_decision.decision is not Decision.ALLOW:
            raise RuntimeError(f"Integrator cannot prepare PR draft: {draft_decision.reason}")
        selected_roles = state.get("selected_build_roles", [])
        diff_reports = [self.workspaces.diff(state["run_id"], role_id).to_dict() for role_id in selected_roles]
        return {
            "phase": "integration_prepared",
            "artifacts": [
                _artifact(
                    "integration_packet",
                    integrator.role_id,
                    {
                        "summary": "Prepared integration packet from selected role outputs.",
                        "included_artifacts": [artifact["kind"] for artifact in state.get("artifacts", [])],
                        "role_diffs": diff_reports,
                        "workspace_policy_ok": all(report["policy_ok"] for report in diff_reports),
                        "next_outward_action": "open_pull_request",
                    },
                )
            ],
            "proposed_actions": [
                {
                    "role_id": integrator.role_id,
                    "action": "open_pull_request",
                    "reason": "Opening a PR is an outward action and must be approved by the owner.",
                }
            ],
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
            "audit_events": [_event("final_report", "created final orchestration report")],
        }

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
        if words & backend_terms or any(phrase in normalized for phrase in ("data model", "business logic", "rest api")):
            selected.append("backend_engineer")
        if words & frontend_terms or any(phrase in normalized for phrase in ("user interface", "design system", "landing page")):
            selected.append("frontend_engineer")
        if words & workflow_terms or any(phrase in normalized for phrase in ("third party", "third_party", "external system")):
            selected.append("workflow_engineer")
        if not selected:
            selected.append("backend_engineer")
        return selected

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
    ) -> dict[str, Any]:
        role = self.registry.get(role_id)
        decision = self.approval_broker.decide(role, action)
        if decision.decision is Decision.DENY:
            raise RuntimeError(f"{role_id} cannot run {action}: {decision.reason}")
        role_context = self._role_memory_context(state, role)
        runtime_result = self.role_runtime.run(role, state["objective"], action, role_context)
        workspace_report = self.workspaces.prepare(state["run_id"], role_id)
        diff_report = self.workspaces.diff(state["run_id"], role_id) if workspace_report.available else None
        return {
            "phase": f"{role_id}_complete",
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
                        "role_runtime": runtime_result.to_dict(),
                        "workspace": workspace_report.to_dict(),
                        "diff": diff_report.to_dict() if diff_report else None,
                        "notes": notes,
                    },
                )
            ],
            "audit_events": [_event(role_id, "completed bounded role node", action=action)],
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


def run_objective(
    objective: str,
    roles_path: Path | None = None,
    checkpoint_db: Path | None = None,
    run_id: str | None = None,
    memory_mode: MemoryMode = "off",
    qa_commands: list[list[str]] | None = None,
    record_runs: bool = False,
    prepare_worktrees: bool = False,
) -> RagnarState:
    registry = load_role_registry(roles_path or _default_roles_path())
    orchestrator = RagnarOrchestrator(
        registry,
        memory_mode=memory_mode,
        qa_commands=qa_commands,
        record_runs=record_runs,
        prepare_worktrees=prepare_worktrees,
    )
    return orchestrator.invoke(objective, checkpoint_db=checkpoint_db, run_id=run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the strict Ragnar LangGraph orchestrator.")
    parser.add_argument("objective")
    parser.add_argument("--roles", type=Path, default=_default_roles_path())
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
    )
    if args.json:
        print(json.dumps(state, indent=2, default=str))
        return
    print(state["final_report"])


if __name__ == "__main__":
    main()
