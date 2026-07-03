from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PullRequestDraft:
    run_id: str
    title: str
    body: str
    source_branches: list[str]
    changed_files: list[str]
    status: str
    requires_approval_action: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PullRequestDraftStore:
    """Local PR draft store. It does not call GitHub."""

    def __init__(self, draft_dir: Path) -> None:
        self.draft_dir = draft_dir

    def save(self, draft: PullRequestDraft) -> Path:
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        path = self.draft_dir / f"{draft.run_id}.json"
        path.write_text(json.dumps(draft.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load(self, run_id: str) -> dict[str, Any] | None:
        path = self.draft_dir / f"{run_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def default_pr_draft_dir(repo_root: Path) -> Path:
    return repo_root / ".ragnar" / "pr_drafts"


def build_pr_draft(run_id: str, objective: str, role_diffs: list[dict[str, Any]]) -> PullRequestDraft:
    branches = [
        str(report["branch"])
        for report in role_diffs
        if report.get("branch")
    ]
    changed_files = sorted(
        {
            path
            for report in role_diffs
            for path in report.get("changed_files", [])
        }
    )
    body = "\n".join(
        [
            f"Objective: {objective}",
            "",
            "Ragnar prepared this draft from isolated role worktrees.",
            "No PR has been opened yet. The approval broker must approve open_pull_request first.",
            "",
            "Changed files:",
            *(f"- {path}" for path in changed_files),
        ]
    )
    return PullRequestDraft(
        run_id=run_id,
        title=f"Ragnar: {objective[:80]}",
        body=body,
        source_branches=branches,
        changed_files=changed_files,
        status="draft_only",
        requires_approval_action="open_pull_request",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local Ragnar PR drafts.")
    parser.add_argument("run_id")
    parser.add_argument("--draft-dir", type=Path, default=default_pr_draft_dir(Path.cwd()))
    args = parser.parse_args()
    draft = PullRequestDraftStore(args.draft_dir).load(args.run_id)
    print(json.dumps(draft or {}, indent=2))


if __name__ == "__main__":
    main()
