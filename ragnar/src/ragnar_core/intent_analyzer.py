from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any


_PATH_PATTERN = re.compile(
    r"(?P<path>(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]+|[\w.-]+\.(?:html?|tsx?|jsx?|css|md|py|json|ya?ml|txt))"
)


@dataclass(frozen=True)
class IntentAnalysis:
    intent: str
    target_project: str | None = None
    target_location: str | None = None
    persistence_mode: str | None = None
    output_type: str | None = None
    risk_level: str = "low"
    missing_slots: list[str] = field(default_factory=list)
    question: str | None = None
    assumptions: list[str] = field(default_factory=list)
    confidence: float = 0.8

    @property
    def needs_clarification(self) -> bool:
        return bool(self.missing_slots)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["needs_clarification"] = self.needs_clarification
        return data


def analyze_intent(objective: str) -> IntentAnalysis:
    normalized = " ".join(objective.lower().split())
    words = _tokens(normalized)

    intent = _classify_intent(normalized, words)
    target_project = _target_project(normalized)
    target_location = _target_location(objective)
    persistence_mode = _persistence_mode(normalized, target_project, target_location)
    output_type = _output_type(normalized, target_location)
    risk_level = _risk_level(words)
    assumptions: list[str] = []
    missing_slots: list[str] = []

    if _should_clarify_file_destination(intent, normalized, words, target_project, target_location):
        missing_slots.extend(["target_project", "target_location", "persistence_mode"])

    if intent in {"deploy", "external_change"} and not _has_environment(normalized):
        missing_slots.append("target_environment")

    if target_project == "current_repo" and not target_location and _is_file_creation(normalized, words):
        assumptions.append("Use the current git repository as the target project.")

    question = _question_for(missing_slots, output_type, risk_level)
    confidence = 0.92
    if missing_slots:
        confidence = 0.45
    elif not target_location and intent == "create_file":
        confidence = 0.72

    return IntentAnalysis(
        intent=intent,
        target_project=target_project,
        target_location=target_location,
        persistence_mode=persistence_mode,
        output_type=output_type,
        risk_level=risk_level,
        missing_slots=_dedupe(missing_slots),
        question=question,
        assumptions=assumptions,
        confidence=confidence,
    )


def merge_clarification(objective: str, answer: str) -> str:
    cleaned_answer = answer.strip()
    if not cleaned_answer:
        return objective
    return f"{objective.strip()}\nUser clarification: {cleaned_answer}"


def _classify_intent(normalized: str, words: set[str]) -> str:
    if words & {"deploy", "release", "publish", "ship"}:
        return "deploy"
    if words & {"research", "compare", "recommend", "investigate"}:
        return "research"
    if words & {"plan", "design", "architect", "architecture", "strategy"} and not words & {
        "build",
        "create",
        "implement",
        "fix",
        "write",
        "add",
        "update",
    }:
        return "planning"
    if words & {"create", "build", "write", "generate", "add", "make"} and _looks_like_file_output(normalized):
        return "create_file"
    if words & {"fix", "update", "change", "edit", "implement", "refactor", "add", "build", "create"}:
        return "modify_code"
    if words & {"delete", "remove", "drop", "wipe", "truncate"}:
        return "destructive_change"
    return "unknown"


def _target_project(normalized: str) -> str | None:
    if "clarification" in normalized:
        if any(phrase in normalized for phrase in ("current repo", "this repo", "git repo", "repository", "project")):
            return "current_repo"
        if any(phrase in normalized for phrase in ("local file", "temporary file", "temp file", "outside repo")):
            return "local_file"
    if any(phrase in normalized for phrase in ("current repo", "this repo", "this project", "git repo", "repository")):
        return "current_repo"
    if any(phrase in normalized for phrase in ("local file", "temporary file", "temp file", "outside repo")):
        return "local_file"
    return None


def _target_location(objective: str) -> str | None:
    match = _PATH_PATTERN.search(objective)
    if not match:
        return None
    path = match.group("path").strip("`'\"")
    normalized_path = str(PurePosixPath(path))
    if normalized_path in {".", ""}:
        return None
    return normalized_path


def _persistence_mode(normalized: str, target_project: str | None, target_location: str | None) -> str | None:
    if any(phrase in normalized for phrase in ("pull request", "pr", "commit", "branch", "git repo", "repository", "current repo", "this repo")):
        return "repo_patch"
    if target_project == "local_file":
        return "local_file"
    if target_location:
        return "repo_patch"
    return None


def _output_type(normalized: str, target_location: str | None) -> str | None:
    if target_location:
        suffix = PurePosixPath(target_location).suffix.lower().lstrip(".")
        if suffix:
            return suffix
    if "html" in normalized or "web page" in normalized or "webpage" in normalized:
        return "html"
    if "readme" in normalized or "markdown" in normalized:
        return "markdown"
    if "api" in normalized:
        return "api"
    return None


def _risk_level(words: set[str]) -> str:
    high_risk = {"auth", "oauth", "payment", "billing", "migration", "database", "deploy", "production", "secret"}
    destructive = {"delete", "remove", "drop", "wipe", "truncate"}
    if words & destructive:
        return "critical"
    if words & high_risk:
        return "high"
    return "low"


def _question_for(missing_slots: list[str], output_type: str | None, risk_level: str) -> str | None:
    missing = set(missing_slots)
    if {"target_project", "target_location", "persistence_mode"} & missing:
        noun = f" {output_type.upper()}" if output_type else ""
        return (
            f"Where should I create or change the{noun} output: in the current git repo, "
            "at a specific path, or as a temporary local file? Reply with the repo/path choice."
        )
    if "target_environment" in missing:
        return f"Which environment should this {risk_level}-risk action target: local, staging, or production?"
    return None


def _should_clarify_file_destination(
    intent: str,
    normalized: str,
    words: set[str],
    target_project: str | None,
    target_location: str | None,
) -> bool:
    if intent != "create_file" or target_project or target_location:
        return False
    standalone_output_terms = {"file", "html", "htm", "readme", "markdown", "txt"}
    if "hello world" in normalized:
        return True
    if words & standalone_output_terms:
        return True
    return False


def _is_file_creation(normalized: str, words: set[str]) -> bool:
    return bool(words & {"create", "write", "generate", "make", "build"} and _looks_like_file_output(normalized))


def _looks_like_file_output(normalized: str) -> bool:
    return any(
        term in normalized
        for term in (
            "html",
            "htm",
            "page",
            "file",
            "readme",
            "markdown",
            "css",
            "component",
            "screen",
        )
    )


def _has_environment(normalized: str) -> bool:
    return any(term in normalized for term in ("local", "dev", "development", "staging", "prod", "production"))


def _tokens(text: str) -> set[str]:
    return {token.strip(".,:;()[]{}<>!?").lower() for token in text.split()}


def _dedupe(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
