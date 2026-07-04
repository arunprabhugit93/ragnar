from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .execution_profiles import ExecutionProfileName, profile_by_name


BUILD_ROLE_IDS = {"backend_engineer", "frontend_engineer", "workflow_engineer"}


@dataclass(frozen=True)
class ConductorDecision:
    execution_profile: ExecutionProfileName
    selected_build_roles: list[str]
    selected_roles: list[str]
    risk_level: str
    complexity_level: str
    knowledge_gaps: list[str]
    research_required: bool
    architect_required: bool
    review_required: bool
    inter_agent_comm_required: bool
    confidence: float
    needs_llm_decision: bool
    reason: str
    budgets: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_conductor_decision(objective: str, selected_build_roles: list[str], project_profile: dict[str, Any] | None = None) -> ConductorDecision:
    normalized = objective.lower().replace("-", " ").replace("_", " ").replace("/", " ")
    words = _tokens(normalized)
    profile_data = project_profile or {}

    planning_terms = {"plan", "design", "architecture", "architect", "approach", "strategy", "roadmap"}
    implementation_terms = {"build", "create", "implement", "fix", "add", "change", "update", "edit", "develop"}
    research_terms = {
        "research",
        "latest",
        "current",
        "best",
        "compare",
        "comparison",
        "which",
        "choose",
        "recommend",
        "pricing",
        "docs",
        "documentation",
        "library",
        "framework",
        "integrate",
        "integration",
        "unknown",
    }
    risky_terms = {
        "auth",
        "login",
        "oauth",
        "security",
        "payment",
        "billing",
        "database",
        "migration",
        "schema",
        "deploy",
        "deployment",
        "production",
        "permission",
        "secret",
        "token",
        "webhook",
        "external",
        "destructive",
        "rollback",
    }
    trivial_terms = {"hello", "world", "simple", "static", "html", "page", "pag", "readme", "copy", "text", "css"}
    external_products = {"stripe", "twilio", "sendgrid", "firebase", "supabase", "aws", "azure", "gcp", "openai", "anthropic", "openrouter"}

    research_required = bool(words & research_terms or words & external_products)
    high_risk = bool(words & risky_terms)
    planning_only = bool(words & planning_terms) and not bool(words & implementation_terms)
    trivial = len(selected_build_roles) == 1 and bool(words & trivial_terms) and not high_risk and not research_required
    multi_role = len(selected_build_roles) > 1
    profiler_unknown = not profile_data.get("languages") and not profile_data.get("frameworks")

    if planning_only:
        selected_build_roles = []
        profile_name: ExecutionProfileName = "planning_only"
        reason = "Objective asks for planning/design without implementation verbs."
    elif trivial:
        profile_name = "fast_path"
        reason = "Low-risk one-role task with trivial/static edit signals."
    elif research_required:
        profile_name = "research_first"
        reason = "Objective has external/current/unknown knowledge requirements."
    elif high_risk or multi_role:
        profile_name = "governed_path"
        reason = "Task is high-risk or spans multiple build roles."
    else:
        profile_name = "standard_path"
        reason = "Known implementation task with bounded risk."

    if high_risk:
        risk_level = "high"
    elif multi_role or research_required:
        risk_level = "medium"
    else:
        risk_level = "low"

    if profile_name in {"governed_path", "research_first"} or multi_role:
        complexity_level = "high" if high_risk and multi_role else "medium"
    else:
        complexity_level = "low"

    knowledge_gaps = []
    if research_required:
        knowledge_gaps.append("External or current information should be researched before implementation.")
    if profiler_unknown:
        knowledge_gaps.append("Project profiler did not identify language/framework evidence.")
    if high_risk:
        knowledge_gaps.append("Risk-sensitive area requires stronger validation.")

    profile = profile_by_name(profile_name)
    selected_roles = []
    if profile_name == "research_first":
        selected_roles.append("researcher")
    if profile.use_architect:
        selected_roles.append("delivery_architect")
    selected_roles.extend(selected_build_roles)
    selected_roles.append("qa_engineer")
    if profile_name not in {"fast_path", "planning_only"}:
        selected_roles.append("integrator")

    confidence = 0.92
    if research_required or profiler_unknown:
        confidence -= 0.12
    if not selected_build_roles and not planning_only:
        confidence -= 0.25
    confidence = max(0.45, min(confidence, 0.98))

    return ConductorDecision(
        execution_profile=profile_name,
        selected_build_roles=selected_build_roles,
        selected_roles=selected_roles,
        risk_level=risk_level,
        complexity_level=complexity_level,
        knowledge_gaps=knowledge_gaps,
        research_required=profile_name == "research_first",
        architect_required=profile.use_architect,
        review_required=profile.use_conductor_plan_review or profile.use_conductor_qa_review == "always",
        inter_agent_comm_required=profile.allow_inter_agent_chat and (multi_role or profile_name in {"research_first", "governed_path"}),
        confidence=confidence,
        needs_llm_decision=confidence < 0.65,
        reason=reason,
        budgets={
            "max_agent_calls": profile.max_agent_calls,
            "max_review_loops": profile.max_review_loops,
            "max_memory_hits": profile.max_memory_hits,
        },
    )


def _tokens(text: str) -> set[str]:
    return {token.strip(".,:;()[]{}<>!?").lower() for token in text.split()}
