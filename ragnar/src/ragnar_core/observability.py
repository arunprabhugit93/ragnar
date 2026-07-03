from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunRecorder:
    """Local run recorder for provider-free observability."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, state: dict[str, Any]) -> dict[str, str]:
        run_id = str(state["run_id"])
        self.root.mkdir(parents=True, exist_ok=True)
        run_path = self.root / f"{run_id}.json"
        events_path = self.root / "events.jsonl"

        run_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        with events_path.open("a", encoding="utf-8") as handle:
            for event in state.get("audit_events", []):
                handle.write(json.dumps({"run_id": run_id, **event}, sort_keys=True, default=str) + "\n")

        return {"run": str(run_path), "events": str(events_path)}


def default_runs_path(repo_root: Path) -> Path:
    return repo_root / ".ragnar" / "runs"
