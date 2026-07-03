from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import RagnarConfig, default_config_path, load_config
from .role_registry import RoleContract, load_role_registry


DEFAULT_MODEL = "letta/letta-free"
DEFAULT_EMBEDDING = "letta/letta-free"


@dataclass(frozen=True)
class ProvisionedAgent:
    role_id: str
    display_name: str
    letta_agent_id: str
    private_memory_namespace: str
    tags: list[str]
    model: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_roles_path() -> Path:
    return _repo_root() / "roles" / "ragnar_roles.yaml"


def _default_manifest_path() -> Path:
    return _repo_root() / ".ragnar" / "letta_agents.json"


def _client(base_url: str, api_key: str | None) -> Any:
    try:
        from letta_client import Letta
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: letta-client. Install with `pip install -e .` from the ragnar directory."
        ) from exc

    if api_key:
        try:
            return Letta(base_url=base_url, api_key=api_key)
        except TypeError:
            return Letta(base_url=base_url, token=api_key)
    return Letta(base_url=base_url)


def _role_instructions(role: RoleContract) -> str:
    responsibilities = "\n".join(f"- {item}" for item in role.responsibility)
    allowed = "\n".join(f"- {item}" for item in role.authority.get("can", [])) or "- none"
    approval = "\n".join(f"- {item}" for item in role.authority.get("requires_approval", [])) or "- none"
    denied = "\n".join(f"- {item}" for item in role.authority.get("cannot", [])) or "- none"
    receives = "\n".join(f"- {item}" for item in role.handoffs.get("receives_from", [])) or "- none"
    sends = "\n".join(f"- {item}" for item in role.handoffs.get("sends_to", [])) or "- none"

    return f"""You are a durable Ragnar role instance.

Role ID: {role.role_id}
Display label: {role.display_name}
Team: {role.team}

Responsibilities:
{responsibilities}

Allowed actions:
{allowed}

Actions requiring owner approval:
{approval}

Denied actions:
{denied}

Receives handoffs from:
{receives}

Sends handoffs to:
{sends}

Iron Rule:
Reads and research run free. Every outward action waits for owner approval.

You must preserve role-specific lessons in memory after useful work, especially:
- project conventions discovered
- repeated mistakes
- owner preferences
- acceptance criteria patterns
- QA findings
- tool-use lessons

Never claim an outward action happened unless the approval broker granted it.
"""


def _memory_blocks(role: RoleContract) -> list[dict[str, Any]]:
    return [
        {
            "label": "persona",
            "value": _role_instructions(role),
            "read_only": True,
        },
        {
            "label": "role_contract",
            "value": json.dumps(
                {
                    "role_id": role.role_id,
                    "display_name": role.display_name,
                    "team": role.team,
                    "responsibility": role.responsibility,
                    "authority": role.authority,
                    "handoffs": role.handoffs,
                    "isolation": role.isolation,
                },
                indent=2,
            ),
            "read_only": True,
        },
        {
            "label": "memory_scope",
            "value": json.dumps(role.memory, indent=2),
            "read_only": True,
        },
        {
            "label": "working_lessons",
            "value": (
                f"Private namespace: {role.private_memory_namespace}\n"
                "Use this block for durable role-specific lessons learned across runs.\n"
                "Keep entries concise, dated when useful, and tied to observable outcomes."
            ),
        },
    ]


def _tags(role: RoleContract) -> list[str]:
    return [
        "ragnar",
        f"role:{role.role_id}",
        f"team:{role.team}",
        f"memory:{role.private_memory_namespace}",
    ]


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"agents": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def create_or_reuse_agents(
    roles_path: Path,
    manifest_path: Path,
    base_url: str,
    api_key: str | None,
    model: str,
    embedding: str,
    communication_tools: bool = True,
    dry_run: bool = False,
    config: RagnarConfig | None = None,
) -> list[ProvisionedAgent]:
    """Provision one durable Letta agent per role.

    When `config` is given, each role's model comes from ragnar.yaml's
    per-role provider/model (already a Letta handle like
    "openrouter/anthropic/claude-haiku-4.5") instead of one fixed model
    for every role. `model`/`embedding` remain the fallback when a role
    has no config entry.
    """
    registry = load_role_registry(roles_path)
    manifest = _load_manifest(manifest_path)
    agents_manifest = manifest.setdefault("agents", {})

    def model_for(role_id: str) -> str:
        if config is None:
            return model
        return config.role_model(role_id).model

    embedding_for_run = config.embedding_model() if config is not None else embedding

    if dry_run:
        return [
            ProvisionedAgent(
                role_id=role.role_id,
                display_name=role.display_name,
                letta_agent_id=agents_manifest.get(role.role_id, {}).get("letta_agent_id", "<dry-run>"),
                private_memory_namespace=role.private_memory_namespace,
                tags=_tags(role),
                model=model_for(role.role_id),
            )
            for role in registry.all()
        ]

    client = _client(base_url, api_key)
    provisioned: list[ProvisionedAgent] = []

    for role in registry.all():
        existing = agents_manifest.get(role.role_id)
        if existing and existing.get("letta_agent_id"):
            provisioned.append(
                ProvisionedAgent(
                    role_id=role.role_id,
                    display_name=role.display_name,
                    letta_agent_id=existing["letta_agent_id"],
                    private_memory_namespace=role.private_memory_namespace,
                    tags=existing.get("tags", _tags(role)),
                    model=existing.get("model", model_for(role.role_id)),
                )
            )
            continue

        payload: dict[str, Any] = {
            "agent_type": "letta_v1_agent",
            "name": f"ragnar__{role.role_id}",
            "memory_blocks": _memory_blocks(role),
            "tags": _tags(role),
            "model": model_for(role.role_id),
            "embedding": embedding_for_run,
            "metadata": {
                "system": "ragnar",
                "role_id": role.role_id,
                "team": role.team,
                "private_memory_namespace": role.private_memory_namespace,
            },
        }
        if communication_tools:
            payload["tools"] = [
                "send_message_to_agent_and_wait_for_reply",
                "send_message_to_agents_matching_tags",
            ]

        create_signature = inspect.signature(client.agents.create)
        supported_payload = {
            key: value for key, value in payload.items() if key in create_signature.parameters
        }
        agent_state = client.agents.create(**supported_payload)

        agent_id = str(getattr(agent_state, "id"))
        record = ProvisionedAgent(
            role_id=role.role_id,
            display_name=role.display_name,
            letta_agent_id=agent_id,
            private_memory_namespace=role.private_memory_namespace,
            tags=_tags(role),
            model=model_for(role.role_id),
        )
        agents_manifest[role.role_id] = asdict(record)
        provisioned.append(record)
        _save_manifest(manifest_path, manifest)

    manifest["base_url"] = base_url
    manifest["embedding"] = embedding_for_run
    _save_manifest(manifest_path, manifest)
    return provisioned


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision durable Letta agents for Ragnar roles")
    parser.add_argument("--roles", type=Path, default=_default_roles_path())
    parser.add_argument("--manifest", type=Path, default=_default_manifest_path())
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--base-url", default=os.getenv("LETTA_SERVER_URL", "http://localhost:8283"))
    parser.add_argument("--api-key", default=os.getenv("LETTA_API_KEY"))
    parser.add_argument("--model", default=os.getenv("RAGNAR_LETTA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--embedding", default=os.getenv("RAGNAR_LETTA_EMBEDDING", DEFAULT_EMBEDDING))
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Ignore ragnar.yaml per-role models; use --model/--embedding for every role.",
    )
    parser.add_argument(
        "--no-communication-tools",
        action="store_true",
        help="Do not attach Letta multi-agent communication tools by name.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = None if args.no_config else load_config(args.config)

    try:
        records = create_or_reuse_agents(
            roles_path=args.roles,
            manifest_path=args.manifest,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            embedding=args.embedding,
            communication_tools=(not args.no_communication_tools) and (config is None or config.enable_agent_messaging()),
            dry_run=args.dry_run,
            config=config,
        )
    except Exception as exc:
        if "Connection error" in str(exc) or "Connection refused" in repr(exc):
            raise SystemExit(
                f"Could not connect to Letta at {args.base_url}. Start Letta first or pass --base-url."
            ) from exc
        raise
    for record in records:
        print(
            f"{record.role_id:24s} {record.display_name:12s} "
            f"letta_id={record.letta_agent_id} model={record.model} memory={record.private_memory_namespace}"
        )


if __name__ == "__main__":
    main()
