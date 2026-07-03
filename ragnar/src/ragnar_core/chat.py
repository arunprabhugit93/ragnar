from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from .approval_store import ApprovalStore, default_approvals_path
from .config import default_config_path
from .orchestrator import run_objective
from .role_runtime import RoleRuntimeMode


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_roles_path() -> Path:
    return _repo_root() / "roles" / "ragnar_roles.yaml"


def _render_role_packet(artifact: dict[str, Any]) -> str:
    body = artifact.get("body", {})
    workspace = body.get("workspace") or {}
    result = body.get("agent_result") or {}
    diff = body.get("diff") or {}
    patch_reports = body.get("patch_reports", [])
    rejected_actions = body.get("rejected_actions", [])
    lines = [
        f"- {artifact.get('owner_role')}: {artifact.get('kind')}",
        f"  action: {body.get('allowed_action')}",
        f"  workspace: {workspace.get('status', 'n/a')} {workspace.get('worktree_path') or ''}".rstrip(),
        f"  agent: {result.get('status', 'n/a')} schema={body.get('agent_invocation', {}).get('schema_version', 'n/a')}",
        f"  patches: proposed={len(result.get('proposed_patches', []))} applied={sum(1 for report in patch_reports if report.get('applied'))}",
        f"  handoffs: {len(body.get('handoffs', []))} memory_writebacks: {len(body.get('memory_writebacks', []))}",
    ]
    if rejected_actions:
        lines.append(f"  rejected_actions: {len(rejected_actions)}")
    if diff:
        lines.append(f"  diff: files={len(diff.get('changed_files', []))} policy_ok={diff.get('policy_ok')}")
    return "\n".join(lines)


def render_state(state: dict[str, Any], show_json: bool = False) -> str:
    if show_json:
        return json.dumps(state, indent=2, default=str)

    artifacts = state.get("artifacts", [])
    role_packets = [artifact for artifact in artifacts if str(artifact.get("kind", "")).endswith("_work_packet")]
    qa = next((artifact for artifact in artifacts if artifact.get("kind") == "qa_verdict"), None)
    integration = next((artifact for artifact in artifacts if artifact.get("kind") == "integration_packet"), None)

    lines = [
        f"run_id: {state.get('run_id')}",
        f"phase: {state.get('phase')}",
        f"roles: {', '.join(state.get('selected_build_roles', [])) or 'none'}",
    ]

    if role_packets:
        lines.append("")
        lines.append("role packets:")
        lines.extend(_render_role_packet(packet) for packet in role_packets)

    if qa:
        body = qa.get("body", {})
        lines.append("")
        lines.append(f"qa: {body.get('verdict')} commands={len(body.get('commands', []))} denied={len(body.get('denied_commands', []))}")

    if integration:
        body = integration.get("body", {})
        draft = body.get("pull_request_draft", {})
        lines.append("")
        lines.append(f"pr_draft: {draft.get('status', 'n/a')} changed_files={len(draft.get('changed_files', []))}")

    approval_requests = state.get("approval_requests", [])
    if approval_requests:
        lines.append("")
        lines.append("approval required:")
        for request in approval_requests:
            lines.append(
                f"- {request['role_id']} {request['action']} ({request['status']})"
            )
        lines.append("")
        if len(approval_requests) == 1:
            request = approval_requests[0]
            lines.append(f"approve with: /approve {state.get('run_id')} {request['role_id']} {request['action']}")
        else:
            lines.append("approve with:")
            for request in approval_requests:
                lines.append(f"/approve {state.get('run_id')} {request['role_id']} {request['action']}")
        lines.append("then rerun with: /rerun")

    return "\n".join(lines)


class ChatSession:
    def __init__(
        self,
        roles_path: Path,
        config_path: Path,
        memory_mode: str,
        qa_commands: list[list[str]],
        prepare_worktrees: bool,
        record_runs: bool,
        show_json: bool = False,
        role_runtime_mode: RoleRuntimeMode = "offline",
    ) -> None:
        self.roles_path = roles_path
        self.config_path = config_path
        self.memory_mode = memory_mode
        self.qa_commands = qa_commands
        self.prepare_worktrees = prepare_worktrees
        self.record_runs = record_runs
        self.show_json = show_json
        self.role_runtime_mode = role_runtime_mode
        self.last_objective: str | None = None
        self.last_run_id: str | None = None
        self.approvals = ApprovalStore(default_approvals_path(_repo_root()))

    def run_objective(self, objective: str, run_id: str | None = None) -> str:
        self.last_objective = objective
        state = run_objective(
            objective,
            roles_path=self.roles_path,
            checkpoint_db=None,
            run_id=run_id,
            memory_mode=self.memory_mode,  # type: ignore[arg-type]
            qa_commands=self.qa_commands,
            record_runs=self.record_runs,
            prepare_worktrees=self.prepare_worktrees,
            config_path=self.config_path,
            role_runtime_mode=self.role_runtime_mode,
        )
        self.last_run_id = str(state["run_id"])
        return render_state(state, self.show_json)

    def handle_command(self, line: str) -> str | None:
        parts = shlex.split(line)
        command = parts[0] if parts else ""
        if command in {"/quit", "/exit"}:
            raise EOFError
        if command == "/help":
            return HELP_TEXT
        if command == "/json":
            if len(parts) == 1:
                self.show_json = not self.show_json
            else:
                self.show_json = parts[1].lower() in {"on", "true", "1"}
            return f"json={str(self.show_json).lower()}"
        if command == "/approvals":
            run_id = self._resolve_run_id(parts[1]) if len(parts) > 1 else self.last_run_id
            records = [record.to_dict() for record in self.approvals.list(run_id)]
            return json.dumps(records, indent=2)
        if command == "/approve":
            if len(parts) != 4:
                return "usage: /approve <run_id|last> <role_id> <action>"
            run_id = self._resolve_run_id(parts[1])
            record = self.approvals.record(run_id, parts[2], parts[3], "approved", "Approved from ragnar-chat.")
            return json.dumps(record.to_dict(), indent=2)
        if command == "/deny":
            if len(parts) != 4:
                return "usage: /deny <run_id|last> <role_id> <action>"
            run_id = self._resolve_run_id(parts[1])
            record = self.approvals.record(run_id, parts[2], parts[3], "denied", "Denied from ragnar-chat.")
            return json.dumps(record.to_dict(), indent=2)
        if command == "/rerun":
            if not self.last_objective:
                return "No previous objective to rerun."
            return self.run_objective(self.last_objective, self.last_run_id)
        return f"Unknown command: {command}. Try /help."

    def _resolve_run_id(self, value: str) -> str:
        if value == "last":
            if not self.last_run_id:
                raise ValueError("No last run_id yet.")
            return self.last_run_id
        return value


HELP_TEXT = """Commands:
/help                         show this help
/json on|off                  toggle full JSON output
/approvals [run_id|last]      list approval ledger records
/approve <run_id|last> <role_id> <action>
/deny <run_id|last> <role_id> <action>
/rerun                        rerun the last objective with the same run_id
/quit                         exit

Any other line is treated as a Ragnar objective.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Ragnar from the terminal.")
    parser.add_argument("--roles", type=Path, default=_default_roles_path())
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--memory-mode", choices=["off", "auto", "pgvector", "graphiti", "all"], default="off")
    parser.add_argument("--qa-command", action="append", default=[])
    parser.add_argument("--no-worktrees", action="store_true")
    parser.add_argument("--no-record-runs", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--once", help="Run one objective and exit.")
    parser.add_argument(
        "--role-runtime",
        choices=["offline", "letta"],
        default="letta",
        help="offline: no LLM calls, stub packets only. letta: call each role's provisioned Letta agent.",
    )
    args = parser.parse_args()

    session = ChatSession(
        roles_path=args.roles,
        config_path=args.config,
        memory_mode=args.memory_mode,
        qa_commands=[shlex.split(command) for command in args.qa_command],
        prepare_worktrees=not args.no_worktrees,
        record_runs=not args.no_record_runs,
        show_json=args.json,
        role_runtime_mode=args.role_runtime,
    )

    if args.once:
        print(session.run_objective(args.once))
        return

    print("Ragnar terminal chat. Type /help for commands, /quit to exit.")
    while True:
        try:
            line = input("ragnar> ").strip()
            if not line:
                continue
            if line.startswith("/"):
                response = session.handle_command(line)
            else:
                response = session.run_objective(line)
            if response:
                print(response)
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        except Exception as exc:
            print(f"error: {exc}")


if __name__ == "__main__":
    main()
