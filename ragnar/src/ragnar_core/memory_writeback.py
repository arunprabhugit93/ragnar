from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .contracts import MemoryWriteback
from .rag_memory import DEFAULT_DB_URL, DEFAULT_EMBEDDING_MODEL, MemoryRecord, upsert_records


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
        seen = {record.get("fingerprint") for record in self.list()}
        count = 0
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                curated = curate_writeback(record)
                if curated is None or curated["fingerprint"] in seen:
                    continue
                handle.write(json.dumps(curated, sort_keys=True) + "\n")
                seen.add(curated["fingerprint"])
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

    def promote_to_pgvector(self, db_url: str = DEFAULT_DB_URL, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> int:
        records = []
        for item in self.list():
            if item.get("promote") is False:
                continue
            records.append(
                MemoryRecord(
                    namespace=str(item["namespace"]),
                    scope=str(item["scope"]),
                    owner_role=str(item["role_id"]) if item.get("scope") == "private" else None,
                    text=str(item["text"]),
                    tags=list(item.get("tags", [])),
                    metadata={
                        "source_run_id": item.get("source_run_id"),
                        "source_artifact": item.get("source_artifact"),
                        "classification": item.get("classification"),
                    },
                )
            )
        return upsert_records(db_url, embedding_model, records)


def curate_writeback(record: MemoryWriteback) -> dict | None:
    data = record.to_dict()
    text = " ".join(str(data.get("text", "")).split())
    if len(text) < 24:
        return None
    classification = classify_writeback(text, data.get("tags", []), data.get("namespace"))
    data["text"] = text[:1200]
    data["classification"] = classification
    data["promote"] = classification not in {"run_noise", "provider_error"}
    data["fingerprint"] = _fingerprint(data["namespace"], data["scope"], data["role_id"], data["text"])
    return data


def classify_writeback(text: str, tags: list[str], namespace: str | None) -> str:
    lowered = text.lower()
    tag_text = " ".join(str(tag).lower() for tag in tags)
    if "provider_error" in tag_text or "provider-backed role call failed" in lowered:
        return "provider_error"
    if "run " in lowered and ("prepared" in lowered or "created" in lowered) and "lesson" not in lowered:
        return "run_noise"
    if namespace and str(namespace).startswith("roles/"):
        return "role_lesson"
    if "qa" in tag_text or "quality" in tag_text:
        return "qa_finding"
    if "decision" in tag_text or "decided" in lowered:
        return "decision"
    return "project_context"


def _fingerprint(namespace: str, scope: str, role_id: str, text: str) -> str:
    raw = json.dumps({"namespace": namespace, "scope": scope, "role_id": role_id, "text": text}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local Ragnar memory writeback records.")
    parser.add_argument("--ledger", type=Path, default=default_writeback_path(_repo_root()))
    parser.add_argument("--run-id")
    parser.add_argument("--promote-pgvector", action="store_true")
    parser.add_argument("--db-url", default=DEFAULT_DB_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()
    store = MemoryWritebackStore(args.ledger)
    if args.promote_pgvector:
        print(f"promoted={store.promote_to_pgvector(args.db_url, args.embedding_model)}")
        return
    print(json.dumps(store.list(args.run_id), indent=2))


if __name__ == "__main__":
    main()
