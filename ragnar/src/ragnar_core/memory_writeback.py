from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .contracts import MemoryWriteback


def default_writeback_path(repo_root: Path) -> Path:
    return repo_root / ".ragnar" / "memory_writebacks.jsonl"


class MemoryWritebackStore:
    """Provider-free memory writeback ledger.

    This stores the exact records that later get promoted to pgvector and
    Graphiti once provider/runtime services are configured.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def append_many(self, records: Iterable[MemoryWriteback]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
                count += 1
        return count

    def list(self, run_id: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if run_id is None or record.get("source_run_id") == run_id:
                records.append(record)
        return records


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local Ragnar memory writeback records.")
    parser.add_argument("--ledger", type=Path, default=default_writeback_path(_repo_root()))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(json.dumps(MemoryWritebackStore(args.ledger).list(args.run_id), indent=2))


if __name__ == "__main__":
    main()
