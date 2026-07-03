from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from .config import ModelConfig
from .role_registry import RoleContract
from .workspace import RoleWorkspacePolicy


SCHEMA_VERSION = "ragnar-contract/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class HandoffArtifact:
    from_role: str
    to_role: str
    kind: str
    summary: str
    artifact_refs: list[str]
    needs_response: bool
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryWriteback:
    role_id: str
    namespace: str
    scope: Literal["private", "shared", "temporal"]
    text: str
    tags: list[str]
    source_run_id: str
    source_artifact: str
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProposedPatch:
    patch_id: str
    summary: str
    unified_diff: str
    target_role: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoleInvocationContract:
    schema_version: str
    run_id: str
    role_id: str
    objective: str
    action: str
    model: dict[str, Any]
    role_contract: dict[str, Any]
    workspace: dict[str, Any]
    file_policy: dict[str, Any]
    command_policy: dict[str, Any]
    memory_context: list[dict[str, Any]]
    handoff_inputs: list[dict[str, Any]]
    expected_output_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoleAgentResult:
    schema_version: str
    run_id: str
    role_id: str
    status: Literal["no_provider", "proposed", "completed", "blocked", "failed"]
    summary: str
    proposed_patches: list[ProposedPatch]
    handoffs: list[HandoffArtifact]
    memory_writebacks: list[MemoryWriteback]
    proposed_actions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "proposed_patches": [patch.to_dict() for patch in self.proposed_patches],
            "handoffs": [handoff.to_dict() for handoff in self.handoffs],
            "memory_writebacks": [writeback.to_dict() for writeback in self.memory_writebacks],
        }


def role_contract_dict(role: RoleContract) -> dict[str, Any]:
    return {
        "role_id": role.role_id,
        "display_name": role.display_name,
        "team": role.team,
        "responsibility": role.responsibility,
        "authority": role.authority,
        "memory": role.memory,
        "handoffs": role.handoffs,
        "isolation": role.isolation,
    }


def expected_role_output_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "required_fields": [
            "run_id",
            "role_id",
            "status",
            "summary",
            "proposed_patches",
            "handoffs",
            "memory_writebacks",
            "proposed_actions",
        ],
        "status_values": ["proposed", "completed", "blocked", "failed"],
        "rules": [
            "Return JSON only.",
            "Do not claim edits were applied; return proposed_patches for the edit adapter.",
            "Every proposed patch must be a unified diff.",
            "Every outward action must appear in proposed_actions, not as completed work.",
            "Memory writebacks must be concise and sourced to the run/artifact.",
        ],
    }


def build_invocation_contract(
    run_id: str,
    objective: str,
    action: str,
    role: RoleContract,
    model: ModelConfig,
    workspace: dict[str, Any],
    policy: RoleWorkspacePolicy,
    memory_context: list[dict[str, Any]],
    handoff_inputs: list[dict[str, Any]] | None = None,
) -> RoleInvocationContract:
    return RoleInvocationContract(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        role_id=role.role_id,
        objective=objective,
        action=action,
        model=model.to_dict(),
        role_contract=role_contract_dict(role),
        workspace=workspace,
        file_policy={"allowed_path_globs": policy.allowed_path_globs},
        command_policy={"allowed_command_families": policy.allowed_command_families},
        memory_context=memory_context,
        handoff_inputs=handoff_inputs or [],
        expected_output_schema=expected_role_output_schema(),
    )


def provider_free_result(
    run_id: str,
    role: RoleContract,
    artifact_kind: str,
    summary: str,
) -> RoleAgentResult:
    handoffs = [
        HandoffArtifact(
            from_role=role.role_id,
            to_role=target,
            kind=f"{artifact_kind}_handoff",
            summary=summary,
            artifact_refs=[artifact_kind],
            needs_response=target == "qa_engineer",
        )
        for target in role.handoffs.get("sends_to", [])
    ]
    writebacks = [
        MemoryWriteback(
            role_id=role.role_id,
            namespace=role.private_memory_namespace,
            scope="private",
            text=f"{role.role_id} prepared {artifact_kind} for run {run_id}. Summary: {summary}",
            tags=["ragnar", "role_result", f"role:{role.role_id}", f"artifact:{artifact_kind}"],
            source_run_id=run_id,
            source_artifact=artifact_kind,
        )
    ]
    return RoleAgentResult(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        role_id=role.role_id,
        status="no_provider",
        summary=summary,
        proposed_patches=[],
        handoffs=handoffs,
        memory_writebacks=writebacks,
        proposed_actions=[],
    )


def contract_json(contract: RoleInvocationContract) -> str:
    return json.dumps(contract.to_dict(), indent=2, sort_keys=True)
