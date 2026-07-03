from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_NOISE_DIRS = {
    ".git", "node_modules", "vendor", ".venv", "venv", "__pycache__",
    ".next", "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "target", ".gradle", ".ragnar",
}

_ENTRYPOINT_NAMES = {
    "main.py", "app.py", "manage.py", "wsgi.py", "asgi.py",
    "index.js", "index.ts", "server.js", "server.ts", "main.js", "main.ts",
    "main.go", "Main.java", "Main.kt",
}

_CI_TOP_LEVEL = {".gitlab-ci.yml", "Jenkinsfile", ".circleci/config.yml"}
_DEPLOYMENT_FILES = {"Dockerfile", "docker-compose.yml", "docker-compose.yaml", "Procfile"}
_MIGRATION_DIR_NAMES = {"migrations", "alembic", "migrate"}

# Recognized test command strings -> safe argv lists. Only exact matches from
# this allowlist ever become an executable command -- a detected string that
# doesn't match is surfaced in the profile for a human/model to read, never
# shell-parsed or run.
QA_COMMAND_ALLOWLIST: dict[str, list[str]] = {
    "pytest": [sys.executable, "-m", "pytest"],
    "python -m unittest": [sys.executable, "-m", "unittest"],
    "npm test": ["npm", "test"],
    "yarn test": ["yarn", "test"],
    "pnpm test": ["pnpm", "test"],
    "go test ./...": ["go", "test", "./..."],
    "cargo test": ["cargo", "test"],
    "mvn test": ["mvn", "test"],
    "gradle test": ["gradle", "test"],
    "bundle exec rspec": ["bundle", "exec", "rspec"],
    "rake test": ["rake", "test"],
    "composer test": ["composer", "test"],
}


@dataclass(frozen=True)
class ProjectProfile:
    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)
    build_commands: list[str] = field(default_factory=list)
    entrypoints: list[str] = field(default_factory=list)
    ci_files: list[str] = field(default_factory=list)
    deployment_files: list[str] = field(default_factory=list)
    database_or_migration_paths: list[str] = field(default_factory=list)
    domain_hints: list[str] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _walk_bounded(root: Path, max_depth: int = 4):
    def _walk(path: Path, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir())
        except (PermissionError, OSError):
            return
        for entry in entries:
            if entry.name in _NOISE_DIRS:
                continue
            yield entry
            if entry.is_dir():
                yield from _walk(entry, depth + 1)

    yield from _walk(root, 0)


def _read_text(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


def _relpath(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _detect_ecosystems_at(scan_root: Path) -> dict[str, list[str]]:
    """Manifest-file-based ecosystem detection at a single directory (non-recursive)."""
    languages: list[str] = []
    frameworks: list[str] = []
    package_managers: list[str] = []
    test_commands: list[str] = []
    build_commands: list[str] = []
    domain_hints: list[str] = []

    pyproject = scan_root / "pyproject.toml"
    requirements = scan_root / "requirements.txt"
    pipfile = scan_root / "Pipfile"
    setup_py = scan_root / "setup.py"
    if pyproject.exists() or requirements.exists() or pipfile.exists() or setup_py.exists():
        languages.append("python")
        manifest_text = _read_text(pyproject) + _read_text(requirements) + _read_text(pipfile)
        if (scan_root / "poetry.lock").exists():
            package_managers.append("poetry")
        elif pipfile.exists():
            package_managers.append("pipenv")
        else:
            package_managers.append("pip")
        for name in ("django", "flask", "fastapi"):
            if re.search(rf"\b{name}\b", manifest_text, re.IGNORECASE):
                frameworks.append(name)
        test_commands.append("pytest" if "pytest" in manifest_text.lower() else "python -m unittest")
        description_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', _read_text(pyproject))
        if description_match:
            domain_hints.append(description_match.group(1))

    package_json = scan_root / "package.json"
    if package_json.exists():
        languages.append("typescript" if (scan_root / "tsconfig.json").exists() else "javascript")
        if (scan_root / "pnpm-lock.yaml").exists():
            package_managers.append("pnpm")
        elif (scan_root / "yarn.lock").exists():
            package_managers.append("yarn")
        else:
            package_managers.append("npm")
        try:
            package_data = json.loads(_read_text(package_json, limit=20000) or "{}")
        except json.JSONDecodeError:
            package_data = {}
        deps = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}
        for name in ("react", "next", "vue", "express", "@nestjs/core"):
            if name in deps:
                frameworks.append("nestjs" if name == "@nestjs/core" else name)
        scripts = package_data.get("scripts", {})
        if "test" in scripts:
            test_commands.append("npm test")
        if "build" in scripts:
            build_commands.append("npm run build")
        description = package_data.get("description")
        if description:
            domain_hints.append(str(description))

    if (scan_root / "go.mod").exists():
        languages.append("go")
        package_managers.append("go modules")
        test_commands.append("go test ./...")
        build_commands.append("go build ./...")

    if (scan_root / "Cargo.toml").exists():
        languages.append("rust")
        package_managers.append("cargo")
        test_commands.append("cargo test")
        build_commands.append("cargo build")

    if (scan_root / "pom.xml").exists():
        languages.append("java")
        package_managers.append("maven")
        test_commands.append("mvn test")
    if (scan_root / "build.gradle").exists() or (scan_root / "build.gradle.kts").exists():
        languages.append("kotlin" if (scan_root / "build.gradle.kts").exists() else "java")
        package_managers.append("gradle")
        test_commands.append("gradle test")

    if (scan_root / "Gemfile").exists():
        languages.append("ruby")
        package_managers.append("bundler")
        gemfile_text = _read_text(scan_root / "Gemfile")
        test_commands.append("bundle exec rspec" if "rspec" in gemfile_text.lower() else "rake test")

    if (scan_root / "composer.json").exists():
        languages.append("php")
        package_managers.append("composer")
        test_commands.append("composer test")

    return {
        "languages": languages,
        "frameworks": frameworks,
        "package_managers": package_managers,
        "test_commands": test_commands,
        "build_commands": build_commands,
        "domain_hints": domain_hints,
    }


def build_project_profile(root: Path) -> ProjectProfile:
    """Detect language/framework/tooling evidence from real repo files.

    Pure filesystem detection -- no LLM call, no shell execution. Mirrors
    the deterministic-ground-truth pattern qa_gate already uses for its
    command verdict: agents get evidence, not a guess.
    """
    entrypoints: list[str] = []
    ci_files: list[str] = []
    deployment_files: list[str] = []
    database_or_migration_paths: list[str] = []
    confidence: dict[str, float] = {}

    def mark(field_name: str, value: float) -> None:
        confidence[field_name] = max(confidence.get(field_name, 0.0), value)

    root_detected = _detect_ecosystems_at(root)
    languages = list(root_detected["languages"])
    frameworks = list(root_detected["frameworks"])
    package_managers = list(root_detected["package_managers"])
    test_commands = list(root_detected["test_commands"])
    build_commands = list(root_detected["build_commands"])
    domain_hints = list(root_detected["domain_hints"])
    if languages:
        mark("languages", 1.0)
        if domain_hints:
            mark("domain_hints", 0.5)
    else:
        # Monorepo fallback: many real repos nest the actual project one level
        # down (e.g. a top-level README/vendor/ alongside the real project
        # directory) rather than putting manifests at the git root. Lower
        # confidence than a root-level hit since it's not the conventional spot.
        for candidate in sorted(root.iterdir()) if root.is_dir() else []:
            if not candidate.is_dir() or candidate.name in _NOISE_DIRS:
                continue
            nested = _detect_ecosystems_at(candidate)
            if nested["languages"]:
                languages.extend(nested["languages"])
                frameworks.extend(nested["frameworks"])
                package_managers.extend(nested["package_managers"])
                test_commands.extend(nested["test_commands"])
                build_commands.extend(nested["build_commands"])
                domain_hints.extend(nested["domain_hints"])
                mark("languages", 0.7)
                if nested["domain_hints"]:
                    mark("domain_hints", 0.4)

    # Entrypoints (shallow: root + one level down)
    for candidate in _walk_bounded(root, max_depth=1):
        if candidate.is_file() and candidate.name in _ENTRYPOINT_NAMES:
            entrypoints.append(_relpath(root, candidate))

    # CI files
    workflows_dir = root / ".github" / "workflows"
    if workflows_dir.is_dir():
        for workflow in sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml")):
            ci_files.append(_relpath(root, workflow))
    for name in _CI_TOP_LEVEL:
        if (root / name).exists():
            ci_files.append(name)
    if ci_files:
        mark("ci_files", 1.0)

    # Deployment files
    for name in _DEPLOYMENT_FILES:
        if (root / name).exists():
            deployment_files.append(name)
    if deployment_files:
        mark("deployment_files", 1.0)

    # Database/migration paths (bounded, noise-dir-excluded walk)
    for candidate in _walk_bounded(root, max_depth=4):
        if candidate.is_dir() and candidate.name in _MIGRATION_DIR_NAMES:
            database_or_migration_paths.append(_relpath(root, candidate))
    if database_or_migration_paths:
        mark("database_or_migration_paths", 0.8)

    # README domain hint
    for readme_name in ("README.md", "README.rst", "README.txt", "README"):
        readme_path = root / readme_name
        if readme_path.exists():
            text = _read_text(readme_path, limit=1000).strip()
            if text:
                domain_hints.append(text.splitlines()[0][:200])
                mark("domain_hints", max(confidence.get("domain_hints", 0.0), 0.3))
            break

    return ProjectProfile(
        languages=sorted(set(languages)),
        frameworks=sorted(set(frameworks)),
        package_managers=sorted(set(package_managers)),
        test_commands=sorted(set(test_commands)),
        build_commands=sorted(set(build_commands)),
        entrypoints=sorted(set(entrypoints)),
        ci_files=sorted(set(ci_files)),
        deployment_files=sorted(set(deployment_files)),
        database_or_migration_paths=sorted(set(database_or_migration_paths)),
        domain_hints=domain_hints,
        confidence=confidence,
    )


def qa_commands_from_profile(profile: dict[str, Any]) -> list[list[str]]:
    """Map recognized test_commands strings to safe argv lists.

    Only exact matches in QA_COMMAND_ALLOWLIST become runnable commands --
    an unrecognized detected string is silently skipped here (it's still
    visible to roles/humans in the profile itself), never shell-parsed.
    """
    commands = []
    for detected in profile.get("test_commands", []) or []:
        argv = QA_COMMAND_ALLOWLIST.get(detected)
        if argv:
            commands.append(argv)
    return commands
