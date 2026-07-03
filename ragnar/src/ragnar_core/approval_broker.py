from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .role_registry import RoleContract


class Decision(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyDecision:
    decision: Decision
    reason: str


class ApprovalBroker:
    """Structural policy gate for the Iron Rule."""

    def decide(self, role: RoleContract, action: str) -> PolicyDecision:
        if role.denies(action):
            return PolicyDecision(Decision.DENY, f"{role.role_id} is explicitly denied {action}")
        if role.requires_approval(action):
            return PolicyDecision(
                Decision.REQUIRE_APPROVAL,
                f"{action} is an outward action and requires owner approval",
            )
        if role.allows(action):
            return PolicyDecision(Decision.ALLOW, f"{role.role_id} may perform {action}")
        return PolicyDecision(Decision.DENY, f"{role.role_id} has no grant for {action}")

