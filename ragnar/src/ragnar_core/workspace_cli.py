from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from .workspace import RoleWorkspaceManager, policy_json


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Ragnar role workspace safety policy.")
    parser.add_argument("--anchor", type=Path, default=_repo_root())
    parser.add_argument("--no-worktrees", action="store_true")
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("policy")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("run_id")
    prepare_parser.add_argument("role_id")

    diff_parser = subparsers.add_parser("diff")
    diff_parser.add_argument("run_id")
    diff_parser.add_argument("role_id")

    path_parser = subparsers.add_parser("check-path")
    path_parser.add_argument("role_id")
    path_parser.add_argument("path")

    command_parser = subparsers.add_parser("check-command")
    command_parser.add_argument("role_id")
    command_parser.add_argument("command_text")

    args = parser.parse_args()
    manager = RoleWorkspaceManager(args.anchor, enabled=not args.no_worktrees)

    if args.action == "policy":
        print(policy_json())
        return
    if args.action == "prepare":
        print(json.dumps(manager.prepare(args.run_id, args.role_id).to_dict(), indent=2))
        return
    if args.action == "diff":
        print(json.dumps(manager.diff(args.run_id, args.role_id).to_dict(), indent=2))
        return
    if args.action == "check-path":
        print(json.dumps(manager.check_path(args.role_id, args.path).to_dict(), indent=2))
        return
    if args.action == "check-command":
        print(json.dumps(manager.check_command(args.role_id, shlex.split(args.command_text)).to_dict(), indent=2))


if __name__ == "__main__":
    main()
