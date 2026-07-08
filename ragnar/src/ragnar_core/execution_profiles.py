from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


ExecutionProfileName = Literal["fast_path", "standard_path", "research_first", "governed_path", "planning_only"]


@dataclass(frozen=True)
class ExecutionProfile:
    name: ExecutionProfileName
    max_agent_calls: int
    max_review_loops: int
    max_memory_hits: int
    allow_inter_agent_chat: bool
    use_architect: bool
    use_conductor_plan_review: bool
    use_qa_agent: bool
    use_conductor_qa_review: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_EXECUTION_PROFILES: dict[str, ExecutionProfile] = {
    "fast_path": ExecutionProfile(
        name="fast_path",
        max_agent_calls=1,
        max_review_loops=0,
        max_memory_hits=3,
        allow_inter_agent_chat=True,
        use_architect=False,
        use_conductor_plan_review=False,
        use_qa_agent=False,
        use_conductor_qa_review="never",
    ),
    "standard_path": ExecutionProfile(
        name="standard_path",
        max_agent_calls=4,
        max_review_loops=1,
        max_memory_hits=5,
        allow_inter_agent_chat=True,
        use_architect=False,
        use_conductor_plan_review=False,
        use_qa_agent=False,
        use_conductor_qa_review="on_failure",
    ),
    "research_first": ExecutionProfile(
        name="research_first",
        max_agent_calls=6,
        max_review_loops=1,
        max_memory_hits=8,
        allow_inter_agent_chat=True,
        use_architect=True,
        use_conductor_plan_review=False,
        use_qa_agent=True,
        use_conductor_qa_review="on_failure",
    ),
    "governed_path": ExecutionProfile(
        name="governed_path",
        max_agent_calls=10,
        max_review_loops=2,
        max_memory_hits=10,
        allow_inter_agent_chat=True,
        use_architect=True,
        use_conductor_plan_review=True,
        use_qa_agent=True,
        use_conductor_qa_review="always",
    ),
    "planning_only": ExecutionProfile(
        name="planning_only",
        max_agent_calls=3,
        max_review_loops=1,
        max_memory_hits=6,
        allow_inter_agent_chat=True,
        use_architect=True,
        use_conductor_plan_review=True,
        use_qa_agent=False,
        use_conductor_qa_review="never",
    ),
}


def profile_by_name(name: str) -> ExecutionProfile:
    return DEFAULT_EXECUTION_PROFILES.get(name, DEFAULT_EXECUTION_PROFILES["standard_path"])
