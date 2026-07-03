from __future__ import annotations

import fnmatch
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BUILD_ROLES = {"backend_engineer", "frontend_engineer", "workflow_engineer"}


@dataclass(frozen=True)
class PolicyCheck:
    allowed: bool
    reason: str
    matched_rule: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkspaceReport:
    role_id: str
    enabled: bool
    available: bool
    git_root: str | None
    branch: str | None
    worktree_path: str | None
    status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiffReport:
    role_id: str
    available: bool
    branch: str | None
    worktree_path: str | None
    changed_files: list[str]
    disallowed_files: list[str]
    status_short: str
    diff_stat: str
    policy_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoleWorkspacePolicy:
    role_id: str
    allowed_path_globs: list[str]
    allowed_command_families: list[str]

    def check_path(self, path: str) -> PolicyCheck:
        normalized = path.replace("\\", "/").lstrip("/")
        for pattern in self.allowed_path_globs:
            if fnmatch.fnmatch(normalized, pattern):
                return PolicyCheck(True, f"{normalized} is allowed for {self.role_id}", pattern)
        return PolicyCheck(False, f"{normalized} is outside {self.role_id} file scope")

    def check_command(self, command: list[str]) -> PolicyCheck:
        if not command:
            return PolicyCheck(False, "command is empty")
        family = command_family(command)
        if family in self.allowed_command_families:
            return PolicyCheck(True, f"{family} is allowed for {self.role_id}", family)
        return PolicyCheck(False, f"{family} is not allowed for {self.role_id}")


def command_family(command: list[str]) -> str:
    executable = Path(command[0]).name
    if executable.startswith("python") and len(command) >= 3 and command[1] == "-m":
        return f"python -m {command[2]}"
    if executable in {"npm", "pnpm", "yarn"} and len(command) >= 2:
        return f"{executable} {command[1]}"
    if executable in {"pytest", "ruff", "mypy"}:
        return executable
    if executable == "git" and len(command) >= 2:
        return f"git {command[1]}"
    return executable


def default_workspace_policies() -> dict[str, RoleWorkspacePolicy]:
    return {
        "backend_engineer": RoleWorkspacePolicy(
            role_id="backend_engineer",
            allowed_path_globs=[
                "backend/**",
                "server/**",
                "services/**",
                "api/**",
                "apis/**",
                "app/api/**",
                "src/**/*.py",
                "src/**/api/**",
                "src/**/service/**",
                "src/**/services/**",
                "src/**/models/**",
                "src/**/schemas/**",
                "migrations/**",
                "alembic/**",
                "db/**",
                "database/**",
                "prisma/**",
                "package.json",
                "pyproject.toml",
            ],
            allowed_command_families=[
                "python -m compileall",
                "python -m pytest",
                "python -m unittest",
                "pytest",
                "ruff",
                "mypy",
                "npm test",
                "pnpm test",
                "yarn test",
            ],
        ),
        "frontend_engineer": RoleWorkspacePolicy(
            role_id="frontend_engineer",
            allowed_path_globs=[
                "frontend/**",
                "web/**",
                "ui/**",
                "app/**/page.*",
                "app/**/layout.*",
                "pages/**",
                "components/**",
                "src/**/*.ts",
                "src/**/*.tsx",
                "src/**/*.js",
                "src/**/*.jsx",
                "src/**/*.css",
                "src/**/*.scss",
                "styles/**",
                "public/**",
                "package.json",
                "vite.config.*",
                "next.config.*",
            ],
            allowed_command_families=[
                "npm test",
                "npm run",
                "npm build",
                "pnpm test",
                "pnpm run",
                "pnpm build",
                "yarn test",
                "yarn run",
                "yarn build",
            ],
        ),
        "workflow_engineer": RoleWorkspacePolicy(
            role_id="workflow_engineer",
            allowed_path_globs=[
                ".github/workflows/**",
                "workflows/**",
                "workflow/**",
                "automation/**",
                "automations/**",
                "integrations/**",
                "connectors/**",
                "webhooks/**",
                "n8n/**",
                "configs/**",
                "config/**",
                "**/*.yml",
                "**/*.yaml",
                "**/*.json",
            ],
            allowed_command_families=[
                "python -m compileall",
                "npm test",
                "pnpm test",
                "yarn test",
            ],
        ),
        "qa_engineer": RoleWorkspacePolicy(
            role_id="qa_engineer",
            allowed_path_globs=["**"],
            allowed_command_families=[
                "python -m compileall",
                "python -m pytest",
                "python -m unittest",
                "pytest",
                "ruff",
                "mypy",
                "npm test",
                "npm run",
                "npm build",
                "pnpm test",
                "pnpm run",
                "pnpm build",
                "yarn test",
                "yarn run",
                "yarn build",
                "go",
                "cargo",
                "mvn",
                "gradle",
                "bundle",
                "rake",
                "composer",
            ],
        ),
        "integrator": RoleWorkspacePolicy(
            role_id="integrator",
            allowed_path_globs=["**"],
            allowed_command_families=["git status", "git diff", "git merge"],
        ),
    }


class RoleWorkspaceManager:
    """Creates isolated role worktrees and validates role file/command scope."""

    def __init__(self, anchor: Path, enabled: bool = True, policies: dict[str, RoleWorkspacePolicy] | None = None) -> None:
        self.anchor = anchor
        self.enabled = enabled
        self.policies = policies or default_workspace_policies()
        self._git_root: Path | None | bool = None

    def prepare(self, run_id: str, role_id: str) -> WorkspaceReport:
        git_root = self.git_root()
        if not self.enabled:
            return WorkspaceReport(role_id, False, False, None, None, None, "disabled", "worktree preparation disabled")
        if role_id not in BUILD_ROLES:
            return WorkspaceReport(role_id, True, False, str(git_root) if git_root else None, None, None, "skipped", "role does not own a build worktree")
        if git_root is None:
            return WorkspaceReport(role_id, True, False, None, None, None, "unavailable", "not running inside a Git checkout")

        worktree_path = git_root / ".ragnar" / "worktrees" / self._safe(run_id) / role_id
        branch = f"ragnar/{self._safe(run_id)}/{role_id}"
        if (worktree_path / ".git").exists():
            return WorkspaceReport(role_id, True, True, str(git_root), branch, str(worktree_path), "ready", "existing isolated worktree")

        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        result = self._run(["git", "-C", str(git_root), "worktree", "add", "-B", branch, str(worktree_path), "HEAD"])
        if result.returncode != 0:
            return WorkspaceReport(
                role_id,
                True,
                False,
                str(git_root),
                branch,
                str(worktree_path),
                "failed",
                (result.stderr or result.stdout).strip(),
            )
        return WorkspaceReport(role_id, True, True, str(git_root), branch, str(worktree_path), "ready", "created isolated worktree")

    def diff(self, run_id: str, role_id: str) -> DiffReport:
        report = self.prepare(run_id, role_id)
        if not report.available or report.worktree_path is None:
            return DiffReport(role_id, False, report.branch, report.worktree_path, [], [], "", "", False)
        worktree_path = Path(report.worktree_path)
        status = self._run(["git", "-C", str(worktree_path), "status", "--short"])
        diff_stat = self._run(["git", "-C", str(worktree_path), "diff", "--stat"])
        changed = self._changed_files(worktree_path)
        disallowed = [path for path in changed if not self.check_path(role_id, path).allowed]
        return DiffReport(
            role_id=role_id,
            available=True,
            branch=report.branch,
            worktree_path=report.worktree_path,
            changed_files=changed,
            disallowed_files=disallowed,
            status_short=status.stdout,
            diff_stat=diff_stat.stdout,
            policy_ok=not disallowed,
        )

    def check_path(self, role_id: str, path: str) -> PolicyCheck:
        policy = self.policies.get(role_id)
        if policy is None:
            return PolicyCheck(False, f"no file policy for {role_id}")
        return policy.check_path(path)

    def check_command(self, role_id: str, command: list[str]) -> PolicyCheck:
        policy = self.policies.get(role_id)
        if policy is None:
            return PolicyCheck(False, f"no command policy for {role_id}")
        return policy.check_command(command)

    def git_root(self) -> Path | None:
        if self._git_root is False:
            return None
        if isinstance(self._git_root, Path):
            return self._git_root
        result = self._run(["git", "-C", str(self.anchor), "rev-parse", "--show-toplevel"])
        if result.returncode != 0:
            self._git_root = False
            return None
        self._git_root = Path(result.stdout.strip()).resolve()
        return self._git_root

    def _changed_files(self, worktree_path: Path) -> list[str]:
        result = self._run(["git", "-C", str(worktree_path), "diff", "--name-only"])
        staged = self._run(["git", "-C", str(worktree_path), "diff", "--cached", "--name-only"])
        untracked = self._run(["git", "-C", str(worktree_path), "ls-files", "--others", "--exclude-standard"])
        files = set()
        for output in (result.stdout, staged.stdout, untracked.stdout):
            files.update(line.strip() for line in output.splitlines() if line.strip())
        return sorted(files)

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def _safe(self, value: str) -> str:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
        stem = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)
        return f"{stem[:48]}-{digest}"


def policies_as_dict(policies: dict[str, RoleWorkspacePolicy] | None = None) -> dict[str, Any]:
    return {
        role_id: {
            "allowed_path_globs": policy.allowed_path_globs,
            "allowed_command_families": policy.allowed_command_families,
        }
        for role_id, policy in (policies or default_workspace_policies()).items()
    }


def policy_json() -> str:
    return json.dumps(policies_as_dict(), indent=2, sort_keys=True)
