from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def default_transcripts_path(repo_root: Path) -> Path:
    return repo_root / ".ragnar" / "agent_transcripts.jsonl"


class AgentTranscriptStore:
    """Append-only ledger of real Letta call transcripts, for audit.

    Full per-turn message/tool-call detail lives here rather than in the
    per-run state JSON, since a single role turn can carry up to
    agent_max_steps messages once agent-to-agent messaging is enabled.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, run_id: str, role_id: str, action: str, transcript: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": run_id,
            "role_id": role_id,
            "action": action,
            **transcript,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def list(self, run_id: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if run_id is None or record.get("run_id") == run_id:
                records.append(record)
        return records


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local Ragnar agent-to-agent call transcripts.")
    parser.add_argument("--ledger", type=Path, default=default_transcripts_path(_repo_root()))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(json.dumps(AgentTranscriptStore(args.ledger).list(args.run_id), indent=2))


if __name__ == "__main__":
    main()
