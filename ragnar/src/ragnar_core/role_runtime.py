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
    transcript: dict[str, Any] | None = None

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
        enable_agent_messaging: bool = False,
        agent_max_steps: int = 12,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()
        self.mode = mode
        self.base_url = base_url
        self.api_key = api_key
        self.enable_agent_messaging = enable_agent_messaging
        self.agent_max_steps = agent_max_steps
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
            raw_reply, transcript = self._call_agent(letta_agent_id, invocation)
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
            transcript=transcript,
        )

    def _client_instance(self) -> Any:
        if self._client is None:
            from letta_client import Letta

            self._client = Letta(base_url=self.base_url, token=self.api_key) if self.api_key else Letta(base_url=self.base_url)
        return self._client

    def _peer_messaging_block(self, invocation: RoleInvocationContract) -> str:
        handoffs = invocation.role_contract.get("handoffs", {}) or {}
        sends_to = handoffs.get("sends_to", []) or []
        receives_from = handoffs.get("receives_from", []) or []
        peers = sorted(set(sends_to) | set(receives_from))
        if not peers:
            return ""
        peer_lines = "\n".join(f"- role:{peer}" for peer in peers)
        return (
            "\n\nYou have access to Letta's native agent-to-agent messaging tools "
            "(send_message_to_agent_and_wait_for_reply, send_message_to_agents_matching_tags). "
            "You may use them to directly consult peer role agents you hand off to or receive "
            "handoffs from, when their input would materially improve this task's outcome "
            "(e.g. clarifying an ambiguous interface, confirming an assumption before you commit "
            "to a design). To reach a specific peer role, call "
            'send_message_to_agents_matching_tags(message=..., match_all=["ragnar"], '
            'match_some=["role:<target_role_id>"]) using one of the role ids below -- never '
            "invent a raw agent_id.\n"
            "Roles you hand off to or receive handoffs from in this run:\n"
            f"{peer_lines}\n"
            "This is optional context-gathering, not a substitute for your own bounded task output: "
            "you must still return the required JSON object as your final reply to THIS message, "
            "regardless of any side conversations you have with peers. Outward actions must still "
            "go through proposed_actions -- peer chat does not grant new authority."
        )

    def _call_agent(self, agent_id: str, invocation: RoleInvocationContract) -> tuple[str, dict[str, Any]]:
        client = self._client_instance()
        prompt = (
            "Here is your bounded task packet for this run. Respond with JSON ONLY, "
            "no prose outside the JSON object, matching the expected_output_schema "
            "in this packet exactly (required_fields and any *_values/rules it lists). "
            "If rework_feedback is not null, prioritize addressing conductor_feedback and pay "
            "close attention to any raw command output (stdout/stderr) it carries for exact "
            "error detail."
        )
        if self.enable_agent_messaging:
            prompt += self._peer_messaging_block(invocation)
        prompt += f"\n\n{contract_json(invocation)}"

        response = client.agents.messages.create(
            agent_id=agent_id,
            messages=[{"role": "user", "content": prompt}],
            max_steps=self.agent_max_steps,
        )

        texts = [
            str(message.content)
            for message in response.messages
            if getattr(message, "message_type", None) == "assistant_message" and getattr(message, "content", None)
        ]
        if not texts:
            raise RuntimeError("Letta agent returned no assistant message")

        transcript = self._build_transcript(agent_id, response.messages)
        return "\n".join(texts), transcript

    def _build_transcript(self, agent_id: str, messages: list[Any]) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for message in messages:
            message_type = getattr(message, "message_type", type(message).__name__)
            entry: dict[str, Any] = {
                "message_id": getattr(message, "id", None),
                "message_type": message_type,
                "sender_id": getattr(message, "sender_id", None),
                "at": str(getattr(message, "date", None)),
            }
            content = getattr(message, "content", None)
            if content is not None:
                entry["content"] = str(content)
            tool_call = getattr(message, "tool_call", None)
            if tool_call is not None:
                entry["tool_name"] = getattr(tool_call, "name", None)
                entry["tool_arguments"] = getattr(tool_call, "arguments", None)
            tool_return = getattr(message, "tool_return", None)
            if tool_return is not None:
                entry["tool_return"] = str(tool_return)
                entry["tool_is_err"] = getattr(message, "is_err", None)
            entries.append(entry)
        return {
            "agent_id": agent_id,
            "max_steps": self.agent_max_steps,
            "messages": entries,
        }

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"agents": {}}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))


def default_manifest_path(repo_root: Path) -> Path:
    return repo_root / ".ragnar" / "letta_agents.json"
