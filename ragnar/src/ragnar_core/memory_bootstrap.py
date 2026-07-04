from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .letta_provisioner import _client
from .role_registry import RoleContract, load_role_registry


# DEPRECATED / LEGACY: this seeds a SECOND, separate vector memory store --
# Letta's own native archival passages/archives, searched by the agent itself
# via its own tools, using whatever embedding model ragnar.yaml's `embedding:`
# names. It is invisible to and unbounded by Ragnar's own retrieval pipeline
# (ContextMemoryProvider / ContextBroker in context_memory.py / context_broker.py),
# which is the one actually feeding role_context/memory_context into every
# invocation contract, backed by rag_memory.py's own pgvector table with its
# own embedding model. The two never sync.
#
# rag_memory.py is the single source of truth for role/shared-namespace
# knowledge retrieval in this system. Prefer `python -m ragnar_core.rag_memory
# bootstrap` for seeding role knowledge; only run this module if you
# specifically want Letta's native archival search as an additional,
# Ragnar-invisible layer the agent can consult on its own.
BOOTSTRAP_VERSION = "vector-memory-bootstrap/v1"
DEFAULT_EMBEDDING = "letta/letta-free"


@dataclass(frozen=True)
class MemoryBootstrapResult:
    namespace: str
    archive_id: str | None
    attached_roles: list[str]
    seeded: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_roles_path() -> Path:
    return _repo_root() / "roles" / "ragnar_roles.yaml"


def _default_manifest_path() -> Path:
    return _repo_root() / ".ragnar" / "letta_agents.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Letta agent manifest: {path}. Run `ragnar-provision-letta` before bootstrapping memory."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _archive_name(namespace: str) -> str:
    safe = namespace.replace("/", "__").replace(" ", "_")
    return f"ragnar__shared__{safe}"


def _page_items(page: Any) -> list[Any]:
    if hasattr(page, "data"):
        return list(page.data)
    if hasattr(page, "items"):
        return list(page.items)
    try:
        return list(page)
    except TypeError:
        return []


def _archive_id(archive: Any) -> str:
    value = getattr(archive, "id", None)
    if not value:
        raise RuntimeError(f"Archive response did not include an id: {archive!r}")
    return str(value)


def _agent_id(manifest: dict[str, Any], role: RoleContract) -> str:
    try:
        return str(manifest["agents"][role.role_id]["letta_agent_id"])
    except KeyError as exc:
        raise KeyError(f"Manifest has no Letta agent for role {role.role_id}. Re-run ragnar-provision-letta.") from exc


def _role_seed_text(role: RoleContract) -> str:
    return json.dumps(
        {
            "bootstrap_version": BOOTSTRAP_VERSION,
            "memory_kind": "private_role_memory_policy",
            "role_id": role.role_id,
            "display_name": role.display_name,
            "team": role.team,
            "private_namespace": role.private_memory_namespace,
            "shared_namespaces": role.memory.get("shared_namespaces", []),
            "responsibility": role.responsibility,
            "memory_rules": [
                "Store durable lessons as archival passages, not long prompt text.",
                "Retrieve relevant private and shared memory before planning or acting.",
                "Write concise memories tied to observable project outcomes.",
                "Use tags for role, project, artifact, decision, incident, and source when known.",
                "Never use memory to bypass the approval broker.",
            ],
        },
        indent=2,
    )


def _shared_seed_text(namespace: str, roles: list[RoleContract]) -> str:
    return json.dumps(
        {
            "bootstrap_version": BOOTSTRAP_VERSION,
            "memory_kind": "shared_namespace_policy",
            "namespace": namespace,
            "attached_roles": [role.role_id for role in roles],
            "purpose": "Shared vector memory archive for Ragnar roles that need the same project context without duplicating prompt tokens.",
            "retrieval_policy": [
                "Search this archive only when the current task needs this namespace.",
                "Prefer small, source-linked passages over large summaries.",
                "Write new passages after decisions, incidents, QA findings, architecture changes, or repeated lessons.",
            ],
        },
        indent=2,
    )


def _shared_namespace_roles(roles: list[RoleContract]) -> dict[str, list[RoleContract]]:
    namespaces: dict[str, list[RoleContract]] = {}
    for role in roles:
        for namespace in role.memory.get("shared_namespaces", []):
            namespaces.setdefault(str(namespace), []).append(role)
    return namespaces


def _create_or_reuse_archive(client: Any, manifest: dict[str, Any], namespace: str, embedding: str) -> str:
    archives = manifest.setdefault("archives", {})
    existing = archives.get(namespace, {})
    if existing.get("archive_id"):
        return str(existing["archive_id"])

    name = _archive_name(namespace)
    matches = _page_items(client.archives.list(name=name, limit=10))
    archive = next((item for item in matches if getattr(item, "name", None) == name), None)
    if archive is None:
        archive = client.archives.create(
            name=name,
            description=f"Ragnar shared vector memory namespace: {namespace}",
            embedding=embedding,
        )

    archive_id = _archive_id(archive)
    archives[namespace] = {
        "archive_id": archive_id,
        "name": name,
        "embedding": embedding,
    }
    return archive_id


def bootstrap_memory(
    roles_path: Path,
    manifest_path: Path,
    base_url: str,
    api_key: str | None,
    embedding: str,
    dry_run: bool = False,
) -> list[MemoryBootstrapResult]:
    registry = load_role_registry(roles_path)
    roles = registry.all()
    manifest = _load_manifest(manifest_path)
    client = None if dry_run else _client(base_url, api_key)
    memory_state = manifest.setdefault("memory_bootstrap", {})
    private_seeded = memory_state.setdefault("private_seeded", {})
    shared_seeded = memory_state.setdefault("shared_seeded", {})
    results: list[MemoryBootstrapResult] = []

    for role in roles:
        agent_id = _agent_id(manifest, role)
        should_seed = private_seeded.get(role.role_id) != BOOTSTRAP_VERSION
        if should_seed and not dry_run:
            client.agents.passages.create(
                agent_id,
                text=_role_seed_text(role),
                tags=[
                    "ragnar",
                    BOOTSTRAP_VERSION,
                    "memory:private",
                    f"role:{role.role_id}",
                    f"namespace:{role.private_memory_namespace}",
                ],
            )
            private_seeded[role.role_id] = BOOTSTRAP_VERSION
        results.append(
            MemoryBootstrapResult(
                namespace=role.private_memory_namespace,
                archive_id=None,
                attached_roles=[role.role_id],
                seeded=should_seed,
            )
        )

    for namespace, attached_roles in sorted(_shared_namespace_roles(roles).items()):
        should_seed = shared_seeded.get(namespace) != BOOTSTRAP_VERSION
        archive_id = f"<dry-run:{namespace}>"
        if not dry_run:
            archive_id = _create_or_reuse_archive(client, manifest, namespace, embedding)
            for role in attached_roles:
                client.agents.archives.attach(archive_id, agent_id=_agent_id(manifest, role))
            if should_seed:
                client.archives.passages.create(
                    archive_id,
                    text=_shared_seed_text(namespace, attached_roles),
                    tags=[
                        "ragnar",
                        BOOTSTRAP_VERSION,
                        "memory:shared",
                        f"namespace:{namespace}",
                    ],
                    metadata={
                        "system": "ragnar",
                        "namespace": namespace,
                        "attached_roles": [role.role_id for role in attached_roles],
                    },
                )
                shared_seeded[namespace] = BOOTSTRAP_VERSION
        results.append(
            MemoryBootstrapResult(
                namespace=namespace,
                archive_id=archive_id,
                attached_roles=[role.role_id for role in attached_roles],
                seeded=should_seed,
            )
        )

    if not dry_run:
        memory_state["version"] = BOOTSTRAP_VERSION
        _save_manifest(manifest_path, manifest)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Ragnar vector memories in Letta archival memory and shared archives.")
    parser.add_argument("--roles", type=Path, default=_default_roles_path())
    parser.add_argument("--manifest", type=Path, default=_default_manifest_path())
    parser.add_argument("--base-url", default=os.environ.get("LETTA_BASE_URL", "http://localhost:8283"))
    parser.add_argument("--api-key", default=os.environ.get("LETTA_API_KEY"))
    parser.add_argument("--embedding", default=os.environ.get("RAGNAR_LETTA_EMBEDDING", DEFAULT_EMBEDDING))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(
        "warning: memory_bootstrap seeds Letta's own native archival memory, which "
        "Ragnar's retrieval pipeline (ContextMemoryProvider) never reads and never "
        "bounds. rag_memory.py's pgvector table is the source of truth for role "
        "knowledge retrieval -- run `python -m ragnar_core.rag_memory bootstrap` "
        "instead unless you specifically want this as an additional layer.\n"
    )

    results = bootstrap_memory(
        roles_path=args.roles,
        manifest_path=args.manifest,
        base_url=args.base_url,
        api_key=args.api_key,
        embedding=args.embedding,
        dry_run=args.dry_run,
    )

    for result in results:
        archive = result.archive_id or "<private-agent-archive>"
        seeded = "seeded" if result.seeded else "already-seeded"
        roles = ",".join(result.attached_roles)
        print(f"{result.namespace:32} archive={archive:48} roles={roles} {seeded}")


if __name__ == "__main__":
    main()
