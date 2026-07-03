from __future__ import annotations

import sys

from ragnar_core.orchestrator import run_objective


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
        qa_commands=[[sys.executable, "-c", "print('qa ok')"]],
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
