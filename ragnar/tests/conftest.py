from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_default_ragnar_state_paths(tmp_path, monkeypatch):
    """RagnarOrchestrator/ChatSession fall back to the real repo's .ragnar/
    directory (approvals.jsonl, run_ledger.jsonl, memory_writebacks.jsonl,
    agent_transcripts.jsonl, pr_drafts/, runs/) for any store path a test
    doesn't explicitly override. approval_store writes in particular are
    unconditional (not gated by record_runs), so any test whose objective
    reaches plan_approval_gate/approval_gate -- not just the ones that
    happen to be audited for it -- mutates real shared state on disk.

    Patches only the default_*_path functions each module calls, not the
    modules' own _repo_root() -- RoleWorkspaceManager/LocalExecutionAdapter
    still resolve the real repo root via repo_root, so tests that legitimately
    exercise real git worktrees (prepare_worktrees=True) are unaffected.
    """
    import ragnar_core.chat as chat_module
    import ragnar_core.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "default_approvals_path", lambda repo_root: tmp_path / "approvals.jsonl")
    monkeypatch.setattr(orchestrator_module, "default_run_ledger_path", lambda repo_root: tmp_path / "run_ledger.jsonl")
    monkeypatch.setattr(orchestrator_module, "default_writeback_path", lambda repo_root: tmp_path / "memory_writebacks.jsonl")
    monkeypatch.setattr(orchestrator_module, "default_transcripts_path", lambda repo_root: tmp_path / "agent_transcripts.jsonl")
    monkeypatch.setattr(orchestrator_module, "default_pr_draft_dir", lambda repo_root: tmp_path / "pr_drafts")
    monkeypatch.setattr(orchestrator_module, "default_runs_path", lambda repo_root: tmp_path / "runs")
    monkeypatch.setattr(chat_module, "default_approvals_path", lambda repo_root: tmp_path / "approvals.jsonl")
