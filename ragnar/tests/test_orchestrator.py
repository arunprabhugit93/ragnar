from __future__ import annotations

import sys
import subprocess
import tempfile
from pathlib import Path

from ragnar_core.config import load_config
from ragnar_core.edit_adapter import SafePatchAdapter, extract_changed_files
from ragnar_core.orchestrator import run_objective
from ragnar_core.pr_adapter import build_pr_draft
from ragnar_core.workspace import RoleWorkspaceManager


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

    assert model.provider == "local"
    assert draft.status == "draft_only"
    assert draft.source_branches == ["ragnar/run-1/backend_engineer"]
    assert draft.changed_files == ["src/service/users.py"]
