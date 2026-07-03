from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .graphiti_memory import DEFAULT_GROUP_ID, DEFAULT_MCP_URL, _result_to_jsonable, _search_facts
from .rag_memory import DEFAULT_DB_URL, DEFAULT_EMBEDDING_MODEL, search_records


MemoryMode = Literal["off", "auto", "pgvector", "graphiti", "all"]


@dataclass(frozen=True)
class MemoryHit:
    provider: str
    namespace: str | None
    scope: str | None
    text: str
    score: float | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryLookup:
    query: dict[str, Any]
    hits: list[MemoryHit]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "hits": [hit.to_dict() for hit in self.hits],
            "errors": self.errors,
        }


class ContextMemoryProvider:
    """Best-effort memory retrieval without requiring provider-backed services."""

    def __init__(
        self,
        mode: MemoryMode = "auto",
        db_url: str | None = None,
        embedding_model: str | None = None,
        graphiti_url: str | None = None,
        graphiti_group_id: str | None = None,
        limit_per_query: int = 3,
    ) -> None:
        self.mode = mode
        self.db_url = db_url or os.environ.get("RAGNAR_MEMORY_DB_URL", DEFAULT_DB_URL)
        self.embedding_model = embedding_model or os.environ.get("RAGNAR_MEMORY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.graphiti_url = graphiti_url or os.environ.get("RAGNAR_GRAPHITI_MCP_URL", DEFAULT_MCP_URL)
        self.graphiti_group_id = graphiti_group_id or os.environ.get("RAGNAR_GRAPHITI_GROUP_ID", DEFAULT_GROUP_ID)
        self.limit_per_query = limit_per_query

    def retrieve(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.mode == "off":
            return []
        return [self._retrieve_one(query).to_dict() for query in queries]

    def _retrieve_one(self, query: dict[str, Any]) -> MemoryLookup:
        hits: list[MemoryHit] = []
        errors: list[str] = []
        query_text = str(query.get("query", ""))
        namespace = str(query.get("namespace")) if query.get("namespace") else None

        if self.mode in {"auto", "pgvector", "all"}:
            try:
                records = search_records(
                    self.db_url,
                    self.embedding_model,
                    query_text,
                    namespace=namespace,
                    owner_role=None,
                    limit=self.limit_per_query,
                )
                hits.extend(
                    MemoryHit(
                        provider="pgvector",
                        namespace=record.get("namespace"),
                        scope=record.get("scope"),
                        text=str(record.get("text", "")),
                        score=float(record["score"]) if record.get("score") is not None else None,
                        metadata={
                            "owner_role": record.get("owner_role"),
                            "tags": record.get("tags", []),
                            "metadata": record.get("metadata", {}),
                        },
                    )
                    for record in records
                )
            except Exception as exc:
                errors.append(f"pgvector unavailable: {exc}")

        if self.mode in {"graphiti", "all"}:
            try:
                result = _result_to_jsonable(
                    asyncio.run(_search_facts(self.graphiti_url, self.graphiti_group_id, query_text, self.limit_per_query))
                )
                facts = result.get("content", result) if isinstance(result, dict) else result
                hits.extend(self._graphiti_hits(facts))
            except Exception as exc:
                errors.append(f"graphiti unavailable: {exc}")

        return MemoryLookup(query=query, hits=hits, errors=errors)

    def _graphiti_hits(self, facts: Any) -> list[MemoryHit]:
        items = facts if isinstance(facts, list) else []
        hits: list[MemoryHit] = []
        for item in items[: self.limit_per_query]:
            if isinstance(item, dict):
                text = str(item.get("fact") or item.get("text") or item)
                score = item.get("score")
                metadata = item
            else:
                text = str(item)
                score = None
                metadata = {}
            hits.append(
                MemoryHit(
                    provider="graphiti",
                    namespace=None,
                    scope="temporal",
                    text=text,
                    score=float(score) if score is not None else None,
                    metadata=metadata,
                )
            )
        return hits
