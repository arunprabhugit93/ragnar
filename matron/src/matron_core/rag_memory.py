from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .role_registry import RoleContract, load_role_registry


DEFAULT_DB_URL = "postgresql://letta:letta@localhost:55432/letta"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
BOOTSTRAP_VERSION = "matron-rag-memory/v1"


@dataclass(frozen=True)
class MemoryRecord:
    namespace: str
    scope: str
    owner_role: str | None
    text: str
    tags: list[str]
    metadata: dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_roles_path() -> Path:
    return _repo_root() / "roles" / "matron_roles.yaml"


def _embedder(model_name: str) -> Any:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("Missing dependency: fastembed. Run `pip install -e .` from the matron directory.") from exc
    return TextEmbedding(model_name=model_name)


def _connect(db_url: str) -> Any:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Missing dependency: psycopg. Run `pip install -e .` from the matron directory.") from exc
    return psycopg.connect(db_url)


def _vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def _fingerprint(namespace: str, scope: str, owner_role: str | None, text: str) -> str:
    raw = json.dumps(
        {"namespace": namespace, "scope": scope, "owner_role": owner_role, "text": text},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ensure_schema(db_url: str) -> None:
    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS matron_memory_passages (
                    id uuid PRIMARY KEY,
                    fingerprint text NOT NULL UNIQUE,
                    namespace text NOT NULL,
                    scope text NOT NULL,
                    owner_role text,
                    text text NOT NULL,
                    tags jsonb NOT NULL DEFAULT '[]'::jsonb,
                    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS matron_memory_namespace_idx ON matron_memory_passages (namespace)")
            cur.execute("CREATE INDEX IF NOT EXISTS matron_memory_owner_role_idx ON matron_memory_passages (owner_role)")
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS matron_memory_embedding_hnsw_idx
                ON matron_memory_passages USING hnsw (embedding vector_cosine_ops)
                """
            )


def upsert_records(db_url: str, model_name: str, records: list[MemoryRecord]) -> int:
    if not records:
        return 0
    ensure_schema(db_url)
    embeddings = list(_embedder(model_name).embed([record.text for record in records]))
    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            for record, embedding in zip(records, embeddings):
                cur.execute(
                    """
                    INSERT INTO matron_memory_passages (
                        id, fingerprint, namespace, scope, owner_role, text, tags, metadata, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::vector)
                    ON CONFLICT (fingerprint) DO UPDATE SET
                        tags = EXCLUDED.tags,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """,
                    (
                        str(uuid.uuid4()),
                        _fingerprint(record.namespace, record.scope, record.owner_role, record.text),
                        record.namespace,
                        record.scope,
                        record.owner_role,
                        record.text,
                        json.dumps(record.tags),
                        json.dumps(record.metadata),
                        _vector_literal(embedding),
                    ),
                )
    return len(records)


def search_records(
    db_url: str,
    model_name: str,
    query: str,
    namespace: str | None = None,
    owner_role: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ensure_schema(db_url)
    query_vector = _vector_literal(next(_embedder(model_name).embed([query])))
    clauses = []
    params: list[Any] = []
    if namespace:
        clauses.append("namespace = %s")
        params.append(namespace)
    if owner_role:
        clauses.append("(owner_role = %s OR owner_role IS NULL)")
        params.append(owner_role)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    sql_params = [query_vector] + params + [query_vector, limit]
    with _connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT namespace, scope, owner_role, text, tags, metadata, 1 - (embedding <=> %s::vector) AS score
                FROM matron_memory_passages
                {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                sql_params,
            )
            rows = cur.fetchall()
    return [
        {
            "namespace": row[0],
            "scope": row[1],
            "owner_role": row[2],
            "text": row[3],
            "tags": row[4],
            "metadata": row[5],
            "score": float(row[6]),
        }
        for row in rows
    ]


def _role_record(role: RoleContract) -> MemoryRecord:
    text = json.dumps(
        {
            "role_id": role.role_id,
            "display_name": role.display_name,
            "team": role.team,
            "responsibility": role.responsibility,
            "authority": role.authority,
            "handoffs": role.handoffs,
            "memory_policy": [
                "Search private memory before acting.",
                "Search relevant shared namespaces before planning.",
                "Write durable lessons as short passages with source and outcome.",
                "Do not rely on long chat history in prompt context.",
            ],
        },
        indent=2,
    )
    return MemoryRecord(
        namespace=role.private_memory_namespace,
        scope="private",
        owner_role=role.role_id,
        text=text,
        tags=["matron", BOOTSTRAP_VERSION, "role_contract", f"role:{role.role_id}", f"team:{role.team}"],
        metadata={"bootstrap_version": BOOTSTRAP_VERSION, "role_id": role.role_id},
    )


def _shared_records(roles: list[RoleContract]) -> list[MemoryRecord]:
    namespaces: dict[str, list[str]] = {}
    for role in roles:
        for namespace in role.memory.get("shared_namespaces", []):
            namespaces.setdefault(str(namespace), []).append(role.role_id)

    records = []
    for namespace, role_ids in sorted(namespaces.items()):
        text = json.dumps(
            {
                "namespace": namespace,
                "attached_roles": role_ids,
                "memory_policy": [
                    "This is shared vector memory to avoid repeating project context in every prompt.",
                    "Store concise facts, decisions, incidents, architecture notes, and conventions.",
                    "Retrieve by semantic search and pass only relevant snippets into working context.",
                ],
            },
            indent=2,
        )
        records.append(
            MemoryRecord(
                namespace=namespace,
                scope="shared",
                owner_role=None,
                text=text,
                tags=["matron", BOOTSTRAP_VERSION, "shared_namespace", f"namespace:{namespace}"],
                metadata={"bootstrap_version": BOOTSTRAP_VERSION, "attached_roles": role_ids},
            )
        )
    return records


def bootstrap(db_url: str, model_name: str, roles_path: Path) -> int:
    roles = load_role_registry(roles_path).all()
    return upsert_records(db_url, model_name, [_role_record(role) for role in roles] + _shared_records(roles))


def main() -> None:
    parser = argparse.ArgumentParser(description="Matron local RAG memory backed by pgvector and local embeddings.")
    parser.add_argument("--db-url", default=os.environ.get("MATRON_MEMORY_DB_URL", DEFAULT_DB_URL))
    parser.add_argument("--embedding-model", default=os.environ.get("MATRON_MEMORY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap")
    bootstrap_parser.add_argument("--roles", type=Path, default=_default_roles_path())

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--namespace")
    search_parser.add_argument("--role")
    search_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()
    if args.command == "bootstrap":
        print(f"upserted={bootstrap(args.db_url, args.embedding_model, args.roles)} model={args.embedding_model} db={args.db_url}")
    elif args.command == "search":
        print(
            json.dumps(
                search_records(args.db_url, args.embedding_model, args.query, args.namespace, args.role, args.limit),
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
