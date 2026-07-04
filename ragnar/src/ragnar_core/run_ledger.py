from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_ledger_path(repo_root: Path) -> Path:
    return repo_root / ".ragnar" / "run_ledger.jsonl"


@dataclass(frozen=True)
class RunLedgerRecord:
    run_id: str
    role_id: str
    action: str
    status: str
    artifact_kind: str
    artifact_ref: str
    summary: str
    changed_files: list[str]
    handoff_to: list[str]
    fingerprint: str
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunLedgerStore:
    """Append-only run/task ledger used to avoid repeated work inside a run."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: RunLedgerRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if any(existing.get("fingerprint") == record.fingerprint for existing in self.list(record.run_id)):
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, default=str) + "\n")

    def append_many(self, records: Iterable[RunLedgerRecord]) -> int:
        count = 0
        for record in records:
            before = len(self.list(record.run_id))
            self.append(record)
            if len(self.list(record.run_id)) > before:
                count += 1
        return count

    def list(self, run_id: str | None = None) -> list[dict[str, Any]]:
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

    def latest_for(self, run_id: str, role_id: str, action: str) -> dict[str, Any] | None:
        for record in reversed(self.list(run_id)):
            if record.get("role_id") == role_id and record.get("action") == action:
                return record
        return None


def record_from_artifact(run_id: str, artifact: dict[str, Any]) -> RunLedgerRecord | None:
    body = artifact.get("body", {})
    result = body.get("agent_result", {}) or {}
    action = str(body.get("allowed_action") or "")
    role_id = str(artifact.get("owner_role") or result.get("role_id") or "")
    artifact_kind = str(artifact.get("kind") or "")
    if not action or not role_id or not artifact_kind:
        return None
    diff = body.get("diff") or {}
    changed_files = list(diff.get("changed_files") or [])
    summary = str(result.get("summary") or body.get("summary") or "")[:1000]
    handoff_to = [str(item.get("to_role")) for item in body.get("handoffs", []) if item.get("to_role")]
    fingerprint = _fingerprint(run_id, role_id, action, artifact_kind, changed_files, summary)
    return RunLedgerRecord(
        run_id=run_id,
        role_id=role_id,
        action=action,
        status=str(result.get("status") or "unknown"),
        artifact_kind=artifact_kind,
        artifact_ref=f"{artifact_kind}:{role_id}",
        summary=summary,
        changed_files=changed_files,
        handoff_to=handoff_to,
        fingerprint=fingerprint,
    )


def _fingerprint(run_id: str, role_id: str, action: str, artifact_kind: str, changed_files: list[str], summary: str) -> str:
    raw = json.dumps(
        {
            "run_id": run_id,
            "role_id": role_id,
            "action": action,
            "artifact_kind": artifact_kind,
            "changed_files": changed_files,
            "summary": summary,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Ragnar run ledger records.")
    parser.add_argument("--ledger", type=Path, default=default_run_ledger_path(_repo_root()))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(json.dumps(RunLedgerStore(args.ledger).list(args.run_id), indent=2))


if __name__ == "__main__":
    main()
