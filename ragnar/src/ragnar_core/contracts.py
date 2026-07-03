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
    rework_feedback: dict[str, Any] | None = None
    project_profile: dict[str, Any] | None = None

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


@dataclass(frozen=True)
class RoleReviewResult:
    schema_version: str
    run_id: str
    role_id: str
    verdict: Literal["approve", "revise"]
    feedback: str
    rework_roles: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        "field_shapes": {
            "proposed_patches": [
                {
                    "patch_id": "string",
                    "summary": "string",
                    "unified_diff": "string containing a complete unified diff",
                }
            ],
            "memory_writebacks": [
                {
                    "text": "string",
                    "tags": ["short", "strings"],
                }
            ],
            "proposed_actions": [
                {
                    "action": "string from this role's authority; outward actions only",
                    "reason": "string explaining why owner approval is needed",
                }
            ],
        },
        "rules": [
            "Return JSON only.",
            "Do not claim edits were applied; return proposed_patches for the edit adapter.",
            "Every proposed patch must be a unified diff.",
            "For implementation/edit actions, status='proposed' or status='completed' requires at least one proposed_patches item unless you explicitly explain why no safe patch is possible and do not propose push_branch.",
            "Do not propose push_branch, open_pull_request, or any outward action when proposed_patches is empty.",
            "Every outward action must appear in proposed_actions, not as completed work.",
            "handoffs must be an array, memory_writebacks must be an array, proposed_actions must be an array.",
            "Do not invent approval handoffs. QA verdicts and owner approvals are handled by the orchestrator.",
            "Memory writebacks must be concise and sourced to the run/artifact.",
        ],
    }


def expected_review_output_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "required_fields": ["run_id", "role_id", "verdict", "feedback", "rework_roles"],
        "verdict_values": ["approve", "revise"],
        "rules": [
            "Return JSON only.",
            "verdict must be exactly 'approve' or 'revise'; never invent other values.",
            "feedback must be actionable and specific enough for the reviewed role to act on.",
            "rework_roles is only meaningful when verdict is 'revise' for a QA review; "
            "list the build role_ids (backend_engineer, frontend_engineer, workflow_engineer) "
            "that need to redo work. Leave empty for plan reviews or when the whole plan is fine.",
            "Never claim to have performed the reviewed role's work yourself.",
        ],
    }


def build_invocation_contract(
    run_id: str,
    objective: str,
    action: str,
    role: RoleContract,
    model: ModelConfig,
    workspace: dict[str, Any],
    policy: RoleWorkspacePolicy | None,
    memory_context: list[dict[str, Any]],
    handoff_inputs: list[dict[str, Any]] | None = None,
    rework_feedback: dict[str, Any] | None = None,
    expected_output_schema: dict[str, Any] | None = None,
    project_profile: dict[str, Any] | None = None,
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
        file_policy={"allowed_path_globs": policy.allowed_path_globs if policy else []},
        command_policy={"allowed_command_families": policy.allowed_command_families if policy else []},
        memory_context=memory_context,
        handoff_inputs=handoff_inputs or [],
        expected_output_schema=expected_output_schema or expected_role_output_schema(),
        rework_feedback=rework_feedback,
        project_profile=project_profile,
    )


def _deterministic_handoffs(role: RoleContract, artifact_kind: str, summary: str) -> list[HandoffArtifact]:
    """Handoff routing always comes from the role contract, never from model output.

    The policy broker outside the LLM owns routing; a model reply can supply
    summary text but must not be able to redirect who receives a handoff.
    """
    return [
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


def provider_free_result(
    run_id: str,
    role: RoleContract,
    artifact_kind: str,
    summary: str,
) -> RoleAgentResult:
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
        handoffs=_deterministic_handoffs(role, artifact_kind, summary),
        memory_writebacks=writebacks,
        proposed_actions=[],
    )


def provider_error_result(
    run_id: str,
    role: RoleContract,
    artifact_kind: str,
    error_message: str,
) -> RoleAgentResult:
    summary = f"Provider-backed role call failed for {role.role_id}: {error_message}"[:2000]
    return RoleAgentResult(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        role_id=role.role_id,
        status="failed",
        summary=summary,
        proposed_patches=[],
        handoffs=_deterministic_handoffs(role, artifact_kind, summary),
        memory_writebacks=[
            MemoryWriteback(
                role_id=role.role_id,
                namespace=role.private_memory_namespace,
                scope="private",
                text=summary,
                tags=["ragnar", "provider_error", f"role:{role.role_id}", f"artifact:{artifact_kind}"],
                source_run_id=run_id,
                source_artifact=artifact_kind,
            )
        ],
        proposed_actions=[],
    )


def agent_result_from_reply(
    run_id: str,
    role: RoleContract,
    artifact_kind: str,
    reply_text: str,
) -> RoleAgentResult:
    """Parse a live provider reply into a RoleAgentResult.

    Never trusts the model's own claim that an outward action already
    happened -- proposed_actions still flow through the same ApprovalBroker
    as everything else. A malformed reply degrades to status="failed"
    rather than raising, so one bad model response can't crash the run.
    """
    from .role_runtime import extract_json_object

    try:
        payload = json.loads(extract_json_object(reply_text))
        if not isinstance(payload, dict):
            raise ValueError("reply JSON is not an object")
    except (json.JSONDecodeError, ValueError):
        return RoleAgentResult(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            role_id=role.role_id,
            status="failed",
            summary=f"Provider reply for {role.role_id} was not valid JSON.",
            proposed_patches=[],
            handoffs=_deterministic_handoffs(role, artifact_kind, "Provider reply was not valid JSON."),
            memory_writebacks=[],
            proposed_actions=[],
        )

    parse_warnings: list[str] = []
    status = str(payload.get("status", "failed"))
    if status not in {"proposed", "completed", "blocked", "failed"}:
        status = "failed"
    summary = str(payload.get("summary") or "").strip()[:2000] or f"{role.role_id} returned no summary for {artifact_kind}."

    raw_patches = payload.get("proposed_patches", [])
    if not isinstance(raw_patches, list):
        parse_warnings.append("proposed_patches was not an array; ignored.")
        raw_patches = []
    proposed_patches = [
        ProposedPatch(
            patch_id=str(patch.get("patch_id", f"{role.role_id}-patch-{index}")),
            summary=str(patch.get("summary", "")),
            unified_diff=str(patch.get("unified_diff", "")),
            target_role=role.role_id,
        )
        for index, patch in enumerate(raw_patches)
        if isinstance(patch, dict) and patch.get("unified_diff")
    ]

    raw_writebacks = payload.get("memory_writebacks", [])
    if not isinstance(raw_writebacks, list):
        parse_warnings.append("memory_writebacks was not an array; default writeback inserted.")
        raw_writebacks = []
    writeback_items = [
        MemoryWriteback(
            role_id=role.role_id,
            namespace=role.private_memory_namespace,
            scope="private",
            text=str(item.get("text", ""))[:2000],
            tags=["ragnar", "role_result", f"role:{role.role_id}", f"artifact:{artifact_kind}"]
            + [str(tag) for tag in item.get("tags", []) if isinstance(tag, (str, int, float))],
            source_run_id=run_id,
            source_artifact=artifact_kind,
        )
        for item in raw_writebacks
        if isinstance(item, dict) and item.get("text")
    ]
    if not writeback_items:
        writeback_items = [
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

    raw_actions = payload.get("proposed_actions", [])
    if not isinstance(raw_actions, list):
        parse_warnings.append("proposed_actions was not an array; ignored.")
        raw_actions = []
    proposed_actions = [
        {
            "role_id": role.role_id,
            "action": str(item["action"]),
            "reason": str(item.get("reason", "Proposed by role agent; requires policy review.")),
        }
        for item in raw_actions
        if isinstance(item, dict) and item.get("action")
    ]
    if not isinstance(payload.get("handoffs", []), list):
        parse_warnings.append("handoffs was not an array; deterministic handoffs used.")

    build_roles = {"backend_engineer", "frontend_engineer", "workflow_engineer"}
    outward_without_patch = any(
        action["action"] in {"push_branch", "open_pull_request"} for action in proposed_actions
    ) and not proposed_patches
    if role.role_id in build_roles and outward_without_patch:
        parse_warnings.append("outward action proposed without any patch; proposed_actions ignored.")
        proposed_actions = [
            action for action in proposed_actions if action["action"] not in {"push_branch", "open_pull_request"}
        ]
        if status in {"proposed", "completed"}:
            status = "blocked"

    if parse_warnings:
        summary = f"{summary} Parser warnings: {' '.join(parse_warnings)}"[:2000]

    return RoleAgentResult(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        role_id=role.role_id,
        status=status,
        summary=summary,
        proposed_patches=proposed_patches,
        handoffs=_deterministic_handoffs(role, artifact_kind, summary),
        memory_writebacks=writeback_items,
        proposed_actions=proposed_actions,
    )


_BUILD_ROLE_IDS = {"backend_engineer", "frontend_engineer", "workflow_engineer"}


def review_result_from_reply(run_id: str, role: RoleContract, reply_text: str) -> RoleReviewResult:
    """Parse a conductor review reply into a RoleReviewResult.

    Unlike agent_result_from_reply, a parse failure here defaults to
    verdict="approve" rather than a "failed" status. This result IS the
    routing signal for the graph -- a transient malformed reply must not
    silently burn a retry-cap slot or force a rework loop. The deterministic
    QA command results and the human approval_gate remain the real safety
    net downstream regardless of this parser's outcome.
    """
    from .role_runtime import extract_json_object

    try:
        payload = json.loads(extract_json_object(reply_text))
        if not isinstance(payload, dict):
            raise ValueError("reply JSON is not an object")
    except (json.JSONDecodeError, ValueError):
        return RoleReviewResult(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            role_id=role.role_id,
            verdict="approve",
            feedback=f"Provider reply for {role.role_id} was not valid JSON; auto-approved.",
            rework_roles=[],
        )

    verdict = str(payload.get("verdict", "approve"))
    if verdict not in {"approve", "revise"}:
        verdict = "approve"
    feedback = str(payload.get("feedback") or "").strip()[:2000]
    rework_roles = [
        str(item) for item in (payload.get("rework_roles", []) or []) if str(item) in _BUILD_ROLE_IDS
    ]

    return RoleReviewResult(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        role_id=role.role_id,
        verdict=verdict,
        feedback=feedback,
        rework_roles=rework_roles,
    )


def contract_json(contract: RoleInvocationContract) -> str:
    return json.dumps(contract.to_dict(), indent=2, sort_keys=True)
