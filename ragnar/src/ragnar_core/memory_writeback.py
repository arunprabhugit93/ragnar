from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .contracts import MemoryWriteback
from .rag_memory import DEFAULT_DB_URL, DEFAULT_EMBEDDING_MODEL, MemoryRecord, upsert_records


# Near-duplicate consolidation: skip writing a new record whose text is this
# similar (difflib ratio, stdlib-only -- no embedding call needed at write time)
# to an existing record already in the same namespace/role/scope. Keeps the
# ledger from accumulating many near-identical restatements.
#
# Deliberately scoped to classifications where duplication really is just
# noise -- NOT "qa_finding" or "decision": for QA findings, repetition is
# itself the signal (a role failing the same way three times is more
# actionable than once, and silently dropping the 2nd/3rd occurrence would
# hide that pattern); decisions are rare and high-value enough that two
# textually-similar entries are more likely to be two genuinely distinct
# decisions than noise. "role_lesson" is excluded too, since
# classify_writeback() already routes the routine "X prepared Y for run Z"
# boilerplate to run_noise first -- what's left in role_lesson has already
# been filtered down to non-boilerplate, reflective content.
_NEAR_DUPLICATE_THRESHOLD = 0.87
_NEAR_DUPLICATE_WINDOW = 50
_NEAR_DUPLICATE_ELIGIBLE_CLASSIFICATIONS = {"run_noise", "provider_error", "project_context"}


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
        existing = self.list()
        seen = {record.get("fingerprint") for record in existing}
        # Keyed by classification too, not just (namespace, scope, role_id) -- otherwise
        # an excluded-classification entry (qa_finding/decision/role_lesson) sitting in
        # the same namespace/role/scope would still enter the comparison pool and could
        # silently suppress a genuinely new, eligible-classification writeback that
        # happens to be textually similar to it.
        recent_texts_by_scope: dict[tuple[str, str, str, str], list[str]] = {}
        for record in existing:
            key = (
                str(record.get("namespace")),
                str(record.get("scope")),
                str(record.get("role_id")),
                str(record.get("classification")),
            )
            recent_texts_by_scope.setdefault(key, []).append(str(record.get("text", "")))
        count = 0
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                curated = curate_writeback(record)
                if curated is None or curated["fingerprint"] in seen:
                    continue
                key = (str(curated["namespace"]), str(curated["scope"]), str(curated["role_id"]), str(curated["classification"]))
                recent_texts = recent_texts_by_scope.setdefault(key, [])
                if curated["classification"] in _NEAR_DUPLICATE_ELIGIBLE_CLASSIFICATIONS and _is_near_duplicate(
                    curated["text"], recent_texts
                ):
                    continue
                handle.write(json.dumps(curated, sort_keys=True) + "\n")
                seen.add(curated["fingerprint"])
                recent_texts.append(curated["text"])
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

    def promote_to_pgvector(
        self,
        db_url: str = DEFAULT_DB_URL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        run_id: str | None = None,
    ) -> int:
        """Upsert curated writebacks into rag_memory's pgvector table.

        Idempotent via upsert_records' fingerprint ON CONFLICT, so passing
        run_id to scope this to just the run that finished is a cost/latency
        optimization (skip re-embedding the whole history every time), not a
        correctness requirement -- omitting it re-promotes everything safely.
        """
        records = []
        for item in self.list(run_id=run_id):
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


def _is_near_duplicate(text: str, recent_texts: list[str]) -> bool:
    for other in recent_texts[-_NEAR_DUPLICATE_WINDOW:]:
        if difflib.SequenceMatcher(None, text, other).ratio() >= _NEAR_DUPLICATE_THRESHOLD:
            return True
    return False


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
