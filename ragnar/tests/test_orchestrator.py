from __future__ import annotations

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
