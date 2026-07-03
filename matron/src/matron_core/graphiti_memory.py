from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from .role_registry import RoleContract, load_role_registry


DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp/"
DEFAULT_GROUP_ID = "matron"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_roles_path() -> Path:
    return _repo_root() / "roles" / "matron_roles.yaml"


def _episode_for_role(role: RoleContract) -> dict[str, Any]:
    return {
        "role_id": role.role_id,
        "display_name": role.display_name,
        "team": role.team,
        "responsibility": role.responsibility,
        "authority": role.authority,
        "private_memory_namespace": role.private_memory_namespace,
        "shared_memory_namespaces": role.memory.get("shared_namespaces", []),
        "handoffs": role.handoffs,
        "isolation": role.isolation,
        "memory_policy": [
            "This role has durable identity and private memory.",
            "Use pgvector RAG for chunk retrieval.",
            "Use Graphiti for temporal facts, ownership, decisions, incidents, and handoff relationships.",
            "Share only relevant retrieved context with other roles.",
        ],
    }


def _role_triplets(role: RoleContract) -> list[dict[str, str]]:
    role_name = f"role:{role.role_id}"
    triplets = [
        {
            "source_node_name": role_name,
            "edge_name": "HAS_PRIVATE_MEMORY",
            "fact": f"{role.role_id} owns private memory namespace {role.private_memory_namespace}.",
            "target_node_name": f"memory:{role.private_memory_namespace}",
        },
        {
            "source_node_name": role_name,
            "edge_name": "BELONGS_TO_TEAM",
            "fact": f"{role.role_id} belongs to the {role.team} team.",
            "target_node_name": f"team:{role.team}",
        },
    ]
    for namespace in role.memory.get("shared_namespaces", []):
        triplets.append(
            {
                "source_node_name": role_name,
                "edge_name": "USES_SHARED_MEMORY",
                "fact": f"{role.role_id} retrieves shared project context from {namespace}.",
                "target_node_name": f"memory:{namespace}",
            }
        )
    for target in role.handoffs.get("sends_to", []):
        triplets.append(
            {
                "source_node_name": role_name,
                "edge_name": "HANDOFFS_TO",
                "fact": f"{role.role_id} can hand work off to {target}.",
                "target_node_name": f"role:{target}",
            }
        )
    for source in role.handoffs.get("receives_from", []):
        triplets.append(
            {
                "source_node_name": f"role:{source}",
                "edge_name": "HANDOFFS_TO",
                "fact": f"{source} can hand work off to {role.role_id}.",
                "target_node_name": role_name,
            }
        )
    return triplets


class GraphitiClient:
    def __init__(self, url: str) -> None:
        self.url = url

    def _health_url(self) -> str:
        parts = urlsplit(self.url)
        return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))

    def _check_ready(self) -> None:
        health_url = self._health_url()
        try:
            with urlopen(health_url, timeout=3) as response:
                if response.status >= 500:
                    raise RuntimeError(f"health endpoint returned HTTP {response.status}")
        except Exception as exc:
            raise RuntimeError(
                f"Graphiti MCP is not reachable at {self.url} (health check {health_url} failed: {exc}). "
                "Start graphiti-mcp and make sure vendor/graphiti/mcp_server/.env contains a provider key."
            ) from exc

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:
            raise RuntimeError("Missing dependency: mcp. Run `pip install -e .` from the matron directory.") from exc

        self._check_ready()
        async with streamablehttp_client(self.url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await session.call_tool(name, arguments=arguments)


def _result_to_jsonable(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    if hasattr(result, "dict"):
        return result.dict()
    return result


async def _status(url: str) -> Any:
    return await GraphitiClient(url).call_tool("get_status", {})


async def _seed_roles(url: str, group_id: str, roles_path: Path, use_triplets: bool) -> dict[str, int]:
    client = GraphitiClient(url)
    roles = load_role_registry(roles_path).all()
    counts = {"episodes": 0, "triplets": 0}
    now = datetime.now(timezone.utc).isoformat()

    for role in roles:
        episode = _episode_for_role(role)
        await client.call_tool(
            "add_memory",
            {
                "name": f"Matron role contract: {role.role_id}",
                "episode_body": json.dumps(episode, indent=2),
                "group_id": group_id,
                "source": "json",
                "source_description": "Matron role registry",
                "reference_time": now,
                "saga": "matron-role-contracts",
                "custom_extraction_instructions": (
                    "Extract role ownership, responsibilities, permissions, memory namespaces, "
                    "and handoff relationships. Preserve role_id and namespace names exactly."
                ),
            },
        )
        counts["episodes"] += 1

        if use_triplets:
            for triplet in _role_triplets(role):
                await client.call_tool("add_triplet", {"group_id": group_id, **triplet})
                counts["triplets"] += 1

    return counts


async def _search_facts(url: str, group_id: str, query: str, limit: int) -> Any:
    return await GraphitiClient(url).call_tool(
        "search_memory_facts",
        {"query": query, "group_ids": [group_id], "max_facts": limit},
    )


async def _search_nodes(url: str, group_id: str, query: str, limit: int) -> Any:
    return await GraphitiClient(url).call_tool(
        "search_nodes",
        {"query": query, "group_ids": [group_id], "max_nodes": limit},
    )


def _print_result(result: Any) -> None:
    print(json.dumps(_result_to_jsonable(result), indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Matron Graphiti temporal memory bridge.")
    parser.add_argument("--url", default=os.environ.get("MATRON_GRAPHITI_MCP_URL", DEFAULT_MCP_URL))
    parser.add_argument("--group-id", default=os.environ.get("MATRON_GRAPHITI_GROUP_ID", DEFAULT_GROUP_ID))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status")

    seed_parser = subparsers.add_parser("seed-roles")
    seed_parser.add_argument("--roles", type=Path, default=_default_roles_path())
    seed_parser.add_argument("--skip-triplets", action="store_true")

    facts_parser = subparsers.add_parser("search-facts")
    facts_parser.add_argument("query")
    facts_parser.add_argument("--limit", type=int, default=5)

    nodes_parser = subparsers.add_parser("search-nodes")
    nodes_parser.add_argument("query")
    nodes_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()
    try:
        if args.command == "status":
            _print_result(asyncio.run(_status(args.url)))
        elif args.command == "seed-roles":
            counts = asyncio.run(_seed_roles(args.url, args.group_id, args.roles, not args.skip_triplets))
            print(f"queued_episodes={counts['episodes']} added_triplets={counts['triplets']} group={args.group_id}")
        elif args.command == "search-facts":
            _print_result(asyncio.run(_search_facts(args.url, args.group_id, args.query, args.limit)))
        elif args.command == "search-nodes":
            _print_result(asyncio.run(_search_nodes(args.url, args.group_id, args.query, args.limit)))
    except Exception as exc:
        raise SystemExit(
            f"Graphiti memory command failed: {exc}\n"
            f"Check Graphiti is running at {args.url} and that vendor/graphiti/mcp_server/.env has "
            "a valid LLM/embedding provider key."
        ) from exc


if __name__ == "__main__":
    main()
