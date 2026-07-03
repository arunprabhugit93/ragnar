from __future__ import annotations

import argparse
from pathlib import Path

from .approval_broker import ApprovalBroker
from .role_registry import load_role_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Ragnar role contracts")
    parser.add_argument(
        "--roles",
        default=str(Path(__file__).resolve().parents[2] / "roles" / "ragnar_roles.yaml"),
        help="Path to ragnar_roles.yaml",
    )
    parser.add_argument("--check-action", nargs=2, metavar=("ROLE_ID", "ACTION"))
    args = parser.parse_args()

    registry = load_role_registry(args.roles)

    if args.check_action:
        role_id, action = args.check_action
        decision = ApprovalBroker().decide(registry.get(role_id), action)
        print(f"{role_id}:{action} -> {decision.decision.value} ({decision.reason})")
        return

    for role in registry.all():
        print(f"{role.role_id:24s} {role.display_name:12s} team={role.team:10s} memory={role.private_memory_namespace}")


if __name__ == "__main__":
    main()

