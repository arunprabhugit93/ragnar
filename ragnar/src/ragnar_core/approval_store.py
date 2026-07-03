from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


ApprovalStatus = Literal["pending", "approved", "denied"]


@dataclass(frozen=True)
class ApprovalRecord:
    run_id: str
    role_id: str
    action: str
    status: ApprovalStatus
    reason: str
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApprovalStore:
    """Append-only local approval ledger for owner-gated actions."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def list(self, run_id: str | None = None) -> list[ApprovalRecord]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            record = ApprovalRecord(**raw)
            if run_id is None or record.run_id == run_id:
                records.append(record)
        return records

    def latest(self, run_id: str, role_id: str, action: str) -> ApprovalRecord | None:
        matches = [
            record
            for record in self.list(run_id)
            if record.role_id == role_id and record.action == action
        ]
        return matches[-1] if matches else None

    def record(self, run_id: str, role_id: str, action: str, status: ApprovalStatus, reason: str) -> ApprovalRecord:
        record = ApprovalRecord(
            run_id=run_id,
            role_id=role_id,
            action=action,
            status=status,
            reason=reason,
            recorded_at=_now(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        return record


def default_approvals_path(repo_root: Path) -> Path:
    return repo_root / ".ragnar" / "approvals.jsonl"
