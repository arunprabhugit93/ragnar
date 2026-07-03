from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .workspace import RoleWorkspaceManager


@dataclass(frozen=True)
class PatchValidation:
    allowed: bool
    changed_files: list[str]
    disallowed_files: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatchApplyReport:
    role_id: str
    applied: bool
    validation: PatchValidation
    exit_code: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "applied": self.applied,
            "validation": self.validation.to_dict(),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class SafePatchAdapter:
    """Applies unified diffs only after role file-scope and git checks pass."""

    def __init__(self, workspace_manager: RoleWorkspaceManager) -> None:
        self.workspace_manager = workspace_manager

    def validate(self, role_id: str, unified_diff: str) -> PatchValidation:
        changed_files = extract_changed_files(unified_diff)
        if not changed_files:
            return PatchValidation(False, [], [], "patch contains no changed files")
        disallowed = [path for path in changed_files if not self.workspace_manager.check_path(role_id, path).allowed]
        if disallowed:
            return PatchValidation(False, changed_files, disallowed, "patch touches files outside role scope")
        return PatchValidation(True, changed_files, [], "patch file scope is allowed")

    def apply(self, role_id: str, worktree_path: Path, unified_diff: str) -> PatchApplyReport:
        validation = self.validate(role_id, unified_diff)
        if not validation.allowed:
            return PatchApplyReport(role_id, False, validation, 1, "", validation.reason)

        check = self._git_apply(worktree_path, unified_diff, check_only=True)
        if check.returncode != 0:
            return PatchApplyReport(role_id, False, validation, check.returncode, check.stdout, check.stderr)

        applied = self._git_apply(worktree_path, unified_diff, check_only=False)
        return PatchApplyReport(
            role_id,
            applied.returncode == 0,
            validation,
            applied.returncode,
            applied.stdout,
            applied.stderr,
        )

    def _git_apply(self, worktree_path: Path, unified_diff: str, check_only: bool) -> subprocess.CompletedProcess[str]:
        command = ["git", "-C", str(worktree_path), "apply", "--recount", f"-p{_patch_strip_level(unified_diff)}"]
        if check_only:
            command.append("--check")
        return subprocess.run(command, input=unified_diff, text=True, capture_output=True, check=False)


def extract_changed_files(unified_diff: str) -> list[str]:
    files: set[str] = set()
    for line in unified_diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                files.add(_strip_diff_path(parts[3]))
        elif line.startswith("+++ "):
            path = line[4:].strip()
            if path != "/dev/null":
                files.add(_strip_diff_path(path))
    return sorted(path for path in files if path and not _is_unsafe_path(path))


def _patch_strip_level(unified_diff: str) -> int:
    for line in unified_diff.splitlines():
        if line.startswith("diff --git "):
            return 1
        if line.startswith("--- ") or line.startswith("+++ "):
            path = line[4:].strip()
            if path == "/dev/null":
                continue
            return 1 if path.startswith(("a/", "b/")) else 0
    return 1


def _strip_diff_path(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path.replace("\\", "/").lstrip("/")


def _is_unsafe_path(path: str) -> bool:
    parts = Path(path).parts
    return path.startswith("/") or ".." in parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a proposed Ragnar patch through role file-scope policy.")
    parser.add_argument("role_id")
    parser.add_argument("worktree_path", type=Path)
    parser.add_argument("patch_file", type=Path)
    parser.add_argument("--anchor", type=Path, default=Path.cwd())
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    adapter = SafePatchAdapter(RoleWorkspaceManager(args.anchor, enabled=False))
    patch_text = args.patch_file.read_text(encoding="utf-8")
    if args.check_only:
        print(json.dumps(adapter.validate(args.role_id, patch_text).to_dict(), indent=2))
        return
    print(json.dumps(adapter.apply(args.role_id, args.worktree_path, patch_text).to_dict(), indent=2))


if __name__ == "__main__":
    main()
