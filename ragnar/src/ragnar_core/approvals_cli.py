from __future__ import annotations

import argparse
import json
from pathlib import Path

from .approval_store import ApprovalStore, default_approvals_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Ragnar owner approvals.")
    parser.add_argument("--ledger", type=Path, default=default_approvals_path(_repo_root()))
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--run-id")

    approve_parser = subparsers.add_parser("approve")
    approve_parser.add_argument("run_id")
    approve_parser.add_argument("role_id")
    approve_parser.add_argument("action")
    approve_parser.add_argument("--reason", default="Approved by owner.")

    deny_parser = subparsers.add_parser("deny")
    deny_parser.add_argument("run_id")
    deny_parser.add_argument("role_id")
    deny_parser.add_argument("action")
    deny_parser.add_argument("--reason", default="Denied by owner.")

    args = parser.parse_args()
    store = ApprovalStore(args.ledger)

    if args.command == "list":
        print(json.dumps([record.to_dict() for record in store.list(args.run_id)], indent=2))
        return

    if args.command == "approve":
        record = store.record(args.run_id, args.role_id, args.action, "approved", args.reason)
        print(json.dumps(record.to_dict(), indent=2))
        return

    if args.command == "deny":
        record = store.record(args.run_id, args.role_id, args.action, "denied", args.reason)
        print(json.dumps(record.to_dict(), indent=2))


if __name__ == "__main__":
    main()
