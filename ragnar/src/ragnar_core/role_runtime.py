from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .contracts import RoleInvocationContract, contract_json
from .role_registry import RoleContract


RoleRuntimeMode = Literal["offline", "letta"]

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class RoleRuntimeResult:
    role_id: str
    runtime: str
    letta_agent_id: str | None
    memory_namespace: str
    message: str
    used_context_hits: int
    status: str
    raw_reply: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_json_object(text: str) -> str:
    """Best-effort extraction of a JSON object from a model reply.

    Models often wrap JSON in a markdown fence or add stray prose around it.
    """
    stripped = text.strip()
    fence_match = _JSON_FENCE.search(stripped)
    if fence_match:
        return fence_match.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


class RoleRuntime:
    """Binds each graph node to its durable Letta agent identity.

    In "offline" mode this never calls an LLM -- it only reports whether a
    real Letta agent exists for the role. In "letta" mode it sends the role's
    invocation contract to that agent and returns the raw reply for the
    orchestrator to parse into a RoleAgentResult.
    """

    def __init__(
        self,
        manifest_path: Path,
        mode: RoleRuntimeMode = "offline",
        base_url: str = "http://localhost:8283",
        api_key: str | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()
        self.mode = mode
        self.base_url = base_url
        self.api_key = api_key
        self._client: Any = None

    def run(
        self,
        role: RoleContract,
        objective: str,
        action: str,
        context_hits: list[dict[str, Any]],
        invocation: RoleInvocationContract | None = None,
    ) -> RoleRuntimeResult:
        manifest_record = self.manifest.get("agents", {}).get(role.role_id, {})
        letta_agent_id = manifest_record.get("letta_agent_id")
        has_real_agent = bool(letta_agent_id and letta_agent_id != "<dry-run>")
        used_context_hits = sum(len(item.get("hits", [])) for item in context_hits)

        if self.mode != "letta" or not has_real_agent or invocation is None:
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

        try:
            raw_reply = self._call_agent(letta_agent_id, invocation)
        except Exception as exc:
            return RoleRuntimeResult(
                role_id=role.role_id,
                runtime="letta_live",
                letta_agent_id=letta_agent_id,
                memory_namespace=role.private_memory_namespace,
                message=f"Letta call failed for {role.role_id}: {exc}",
                used_context_hits=used_context_hits,
                status="provider_error",
                raw_reply=None,
            )

        return RoleRuntimeResult(
            role_id=role.role_id,
            runtime="letta_live",
            letta_agent_id=letta_agent_id,
            memory_namespace=role.private_memory_namespace,
            message=f"Letta agent {letta_agent_id} responded for {role.role_id} action {action}.",
            used_context_hits=used_context_hits,
            status="provider_responded",
            raw_reply=raw_reply,
        )

    def _client_instance(self) -> Any:
        if self._client is None:
            from letta_client import Letta

            self._client = Letta(base_url=self.base_url, token=self.api_key) if self.api_key else Letta(base_url=self.base_url)
        return self._client

    def _call_agent(self, agent_id: str, invocation: RoleInvocationContract) -> str:
        client = self._client_instance()
        prompt = (
            "Here is your bounded task packet for this run. Respond with JSON ONLY, "
            "no prose outside the JSON object, matching the expected_output_schema "
            "in this packet exactly (required_fields and any *_values/rules it lists). "
            "If rework_feedback is not null, prioritize addressing it directly.\n\n"
            f"{contract_json(invocation)}"
        )
        response = client.agents.messages.create(agent_id=agent_id, messages=[{"role": "user", "content": prompt}])
        texts = [
            str(message.content)
            for message in response.messages
            if type(message).__name__ == "AssistantMessage" and getattr(message, "content", None)
        ]
        if not texts:
            raise RuntimeError("Letta agent returned no assistant message")
        return "\n".join(texts)

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"agents": {}}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))


def default_manifest_path(repo_root: Path) -> Path:
    return repo_root / ".ragnar" / "letta_agents.json"
