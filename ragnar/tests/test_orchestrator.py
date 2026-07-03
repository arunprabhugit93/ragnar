from __future__ import annotations

import json
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ragnar_core.chat import ChatSession, render_state
from ragnar_core.config import load_config
from ragnar_core.edit_adapter import SafePatchAdapter, extract_changed_files
from ragnar_core.orchestrator import MAX_PLAN_REVISIONS, RagnarOrchestrator, run_objective
from ragnar_core.pr_adapter import build_pr_draft
from ragnar_core.role_registry import load_role_registry
from ragnar_core.role_runtime import RoleRuntimeResult
from ragnar_core.workspace import RoleWorkspaceManager


class _ScriptedRoleRuntime:
    """Test double: scripted raw_reply sequences for specific (role_id, action) pairs,
    offline-stub behavior (raw_reply=None) for everything else."""

    def __init__(self, scripted: dict[tuple[str, str], list[str]]) -> None:
        self.scripted = scripted
        self._call_counts: dict[tuple[str, str], int] = {}

    def run(self, role: Any, objective: str, action: str, context_hits: list[dict[str, Any]], invocation: Any = None) -> RoleRuntimeResult:
        used_context_hits = sum(len(item.get("hits", [])) for item in context_hits)
        key = (role.role_id, action)
        replies = self.scripted.get(key)
        if replies:
            index = self._call_counts.get(key, 0)
            self._call_counts[key] = index + 1
            reply = replies[min(index, len(replies) - 1)]
            return RoleRuntimeResult(
                role_id=role.role_id,
                runtime="scripted_test_double",
                letta_agent_id="test-agent",
                memory_namespace=role.private_memory_namespace,
                message="scripted reply for test",
                used_context_hits=used_context_hits,
                status="provider_responded",
                raw_reply=reply,
            )
        return RoleRuntimeResult(
            role_id=role.role_id,
            runtime="letta_manifest_stub",
            letta_agent_id=None,
            memory_namespace=role.private_memory_namespace,
            message="offline stub for test",
            used_context_hits=used_context_hits,
            status="no_real_letta_agent_manifest",
            raw_reply=None,
        )


def test_orchestrator_routes_all_build_roles() -> None:
    state = run_objective(
        "build a frontend settings page with backend API and webhook integration",
        checkpoint_db=None,
    )

    assert state["selected_build_roles"] == [
        "backend_engineer",
        "frontend_engineer",
        "workflow_engineer",
    ]
    assert state["blocked"] is True
    assert state["approval_requests"][0]["action"] == "open_pull_request"


def test_orchestrator_uses_backend_default_and_keeps_qa_gate() -> None:
    state = run_objective("fix the login bug", checkpoint_db=None)
    artifact_kinds = [artifact["kind"] for artifact in state["artifacts"]]

    assert state["selected_build_roles"] == ["backend_engineer"]
    assert "backend_work_packet" in artifact_kinds
    assert "qa_verdict" in artifact_kinds
    assert state["phase"] == "awaiting_owner_approval"


def test_orchestrator_runs_configured_qa_command() -> None:
    state = run_objective(
        "fix the database migration",
        checkpoint_db=None,
        qa_commands=[[sys.executable, "-m", "compileall", "-q", "src/ragnar_core"]],
    )
    qa_artifact = next(artifact for artifact in state["artifacts"] if artifact["kind"] == "qa_verdict")

    assert qa_artifact["body"]["verdict"] == "pass"
    assert qa_artifact["body"]["commands"][0]["exit_code"] == 0


def test_orchestrator_does_not_require_memory_provider() -> None:
    state = run_objective(
        "build dashboard UI",
        checkpoint_db=None,
        memory_mode="off",
    )

    assert state["selected_build_roles"] == ["frontend_engineer"]
    assert state.get("memory_context", []) == []


def test_orchestrator_includes_agent_contract_and_writeback_format() -> None:
    state = run_objective(
        "fix the login bug",
        checkpoint_db=None,
        memory_mode="off",
    )
    packet = next(artifact for artifact in state["artifacts"] if artifact["kind"] == "backend_work_packet")

    assert packet["body"]["agent_invocation"]["schema_version"] == "ragnar-contract/v1"
    assert packet["body"]["agent_invocation"]["role_id"] == "backend_engineer"
    assert packet["body"]["agent_result"]["status"] == "no_provider"
    assert packet["body"]["handoffs"][0]["to_role"] == "qa_engineer"
    assert packet["body"]["memory_writebacks"][0]["scope"] == "private"


def test_qa_rejects_disallowed_command() -> None:
    state = run_objective(
        "fix the database migration",
        checkpoint_db=None,
        qa_commands=[[sys.executable, "-c", "print('not allowed')"]],
    )
    qa_artifact = next(artifact for artifact in state["artifacts"] if artifact["kind"] == "qa_verdict")

    assert qa_artifact["body"]["verdict"] == "fail"
    assert qa_artifact["body"]["denied_commands"][0]["policy"]["allowed"] is False


def test_workspace_policy_checks_file_scopes_and_commands() -> None:
    manager = RoleWorkspaceManager(Path.cwd(), enabled=False)

    assert manager.check_path("backend_engineer", "src/service/users.py").allowed is True
    assert manager.check_path("backend_engineer", "components/Button.tsx").allowed is False
    assert manager.check_path("frontend_engineer", "components/Button.tsx").allowed is True
    assert manager.check_command("qa_engineer", [sys.executable, "-m", "compileall", "src"]).allowed is True
    assert manager.check_command("qa_engineer", [sys.executable, "-c", "print(1)"]).allowed is False


def test_workspace_manager_prepares_isolated_git_worktree() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "ragnar@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Ragnar"], cwd=root, check=True)
        (root / "README.md").write_text("test\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)

        manager = RoleWorkspaceManager(root)
        report = manager.prepare("run-1", "backend_engineer")

        assert report.available is True
        assert report.worktree_path is not None
        assert (Path(report.worktree_path) / ".git").exists()


def test_safe_patch_adapter_applies_allowed_patch_and_rejects_disallowed_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "ragnar@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Ragnar"], cwd=root, check=True)
        service_path = root / "src" / "service"
        service_path.mkdir(parents=True)
        (service_path / "users.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "src/service/users.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)

        patch = """diff --git a/src/service/users.py b/src/service/users.py
--- a/src/service/users.py
+++ b/src/service/users.py
@@ -1 +1 @@
-value = 1
+value = 2
"""
        adapter = SafePatchAdapter(RoleWorkspaceManager(root, enabled=False))
        report = adapter.apply("backend_engineer", root, patch)

        assert report.applied is True
        assert (service_path / "users.py").read_text(encoding="utf-8") == "value = 2\n"
        assert extract_changed_files(patch) == ["src/service/users.py"]

        disallowed_patch = """diff --git a/components/Button.tsx b/components/Button.tsx
--- a/components/Button.tsx
+++ b/components/Button.tsx
@@ -1 +1 @@
-old
+new
"""
        validation = adapter.validate("backend_engineer", disallowed_patch)
        assert validation.allowed is False
        assert validation.disallowed_files == ["components/Button.tsx"]


def test_config_and_pr_draft_shapes() -> None:
    config = load_config(Path("ragnar/ragnar.yaml"))
    model = config.role_model("backend_engineer")
    draft = build_pr_draft(
        "run-1",
        "fix auth API",
        [
            {
                "branch": "ragnar/run-1/backend_engineer",
                "changed_files": ["src/service/users.py"],
            }
        ],
    )

    assert model.provider == "openrouter"
    assert model.model.startswith("openrouter/")
    assert draft.status == "draft_only"
    assert draft.source_branches == ["ragnar/run-1/backend_engineer"]
    assert draft.changed_files == ["src/service/users.py"]


def test_chat_session_runs_objective_once() -> None:
    session = ChatSession(
        roles_path=Path("ragnar/roles/ragnar_roles.yaml"),
        config_path=Path("ragnar/ragnar.yaml"),
        memory_mode="off",
        qa_commands=[],
        prepare_worktrees=False,
        record_runs=False,
    )

    output = session.run_objective("build dashboard UI")

    assert "roles: frontend_engineer" in output
    assert "approval required:" in output
    assert session.last_run_id is not None


def test_chat_render_json_mode() -> None:
    rendered = render_state({"run_id": "run-1", "phase": "done", "selected_build_roles": []}, show_json=True)

    assert '"run_id": "run-1"' in rendered


def test_architect_plan_produces_real_agent_artifact_shape_offline() -> None:
    state = run_objective("fix the login bug", checkpoint_db=None)
    plan_artifact = next(a for a in state["artifacts"] if a["kind"] == "architecture_plan")

    assert plan_artifact["owner_role"] == "delivery_architect"
    assert plan_artifact["body"]["agent_invocation"]["role_id"] == "delivery_architect"
    assert plan_artifact["body"]["agent_result"]["status"] == "no_provider"


def test_conductor_review_plan_defaults_to_approve_offline() -> None:
    state = run_objective("fix the login bug", checkpoint_db=None)

    assert state["plan_review_verdict"] == "approve"
    assert state["plan_revision_count"] == 0
    artifact_kinds = [a["kind"] for a in state["artifacts"]]
    assert "conductor_plan_review" in artifact_kinds
    assert "backend_work_packet" in artifact_kinds


def test_conductor_review_qa_defaults_to_approve_offline() -> None:
    state = run_objective("fix the login bug", checkpoint_db=None)

    assert state["qa_review_verdict"] == "approve"
    assert state["qa_rework_roles"] == []
    artifact_kinds = [a["kind"] for a in state["artifacts"]]
    assert "conductor_qa_review" in artifact_kinds
    assert "integration_packet" in artifact_kinds


def test_plan_revision_loop_respects_retry_cap() -> None:
    registry = load_role_registry(Path("ragnar/roles/ragnar_roles.yaml"))
    orchestrator = RagnarOrchestrator(
        registry,
        config_path=Path("ragnar/ragnar.yaml"),
        prepare_worktrees=False,
        record_runs=False,
    )
    orchestrator.role_runtime = _ScriptedRoleRuntime(
        {
            ("conductor", "review_delivery_plan"): [
                '{"verdict":"revise","feedback":"needs more detail","rework_roles":[]}',
            ],
        }
    )

    state = orchestrator.invoke("fix the login bug")

    assert state["plan_revision_count"] == MAX_PLAN_REVISIONS
    assert state["plan_review_verdict"] == "escalated_to_owner"
    plan_artifacts = [a for a in state["artifacts"] if a["kind"] == "architecture_plan"]
    assert len(plan_artifacts) == MAX_PLAN_REVISIONS + 1
    assert state.get("final_report")


def test_qa_revision_loop_routes_back_to_named_role() -> None:
    registry = load_role_registry(Path("ragnar/roles/ragnar_roles.yaml"))
    orchestrator = RagnarOrchestrator(
        registry,
        config_path=Path("ragnar/ragnar.yaml"),
        prepare_worktrees=False,
        record_runs=False,
    )
    orchestrator.role_runtime = _ScriptedRoleRuntime(
        {
            ("conductor", "review_qa_verdict"): [
                '{"verdict":"revise","feedback":"fix the auth check","rework_roles":["backend_engineer"]}',
                '{"verdict":"approve","feedback":"looks good","rework_roles":[]}',
            ],
        }
    )

    state = orchestrator.invoke("fix the login bug")

    backend_packets = [a for a in state["artifacts"] if a["kind"] == "backend_work_packet"]
    assert len(backend_packets) == 2
    assert backend_packets[1]["body"]["agent_invocation"]["rework_feedback"] == "fix the auth check"
    assert state["qa_review_verdict"] == "approve"


def test_final_report_includes_owner_briefing_offline() -> None:
    state = run_objective("fix the login bug", checkpoint_db=None)

    assert isinstance(state.get("owner_briefing"), str) and state["owner_briefing"]
    parsed = json.loads(state["final_report"])
    assert parsed["run_id"] == state["run_id"]
    assert set(parsed.keys()) == {
        "run_id",
        "objective",
        "phase",
        "selected_build_roles",
        "artifact_count",
        "blocked",
        "approval_requests",
    }
