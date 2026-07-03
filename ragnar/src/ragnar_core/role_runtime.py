from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .role_registry import RoleContract


@dataclass(frozen=True)
class RoleRuntimeResult:
    role_id: str
    runtime: str
    letta_agent_id: str | None
    memory_namespace: str
    message: str
    used_context_hits: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RoleRuntime:
    """Provider-free role runtime envelope.

    This does not ask an LLM to think. It binds each graph node to the durable
    agent identity from the Letta manifest when present, and leaves a stable
    seam for provider-backed Letta calls later.
    """

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()

    def run(
        self,
        role: RoleContract,
        objective: str,
        action: str,
        context_hits: list[dict[str, Any]],
    ) -> RoleRuntimeResult:
        manifest_record = self.manifest.get("agents", {}).get(role.role_id, {})
        letta_agent_id = manifest_record.get("letta_agent_id")
        has_real_agent = bool(letta_agent_id and letta_agent_id != "<dry-run>")
        used_context_hits = sum(len(item.get("hits", [])) for item in context_hits)
        return RoleRuntimeResult(
            role_id=role.role_id,
            runtime="letta_manifest_stub",
            letta_agent_id=letta_agent_id if has_real_agent else None,
            memory_namespace=role.private_memory_namespace,
            message=(
                f"Prepared bounded role packet for {role.role_id} action {action}. "
                "Provider-backed Letta inference is not called in this mode."
            ),
            used_context_hits=used_context_hits,
            status="ready_for_provider_runtime" if has_real_agent else "no_real_letta_agent_manifest",
        )

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"agents": {}}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))


def default_manifest_path(repo_root: Path) -> Path:
    return repo_root / ".ragnar" / "letta_agents.json"
