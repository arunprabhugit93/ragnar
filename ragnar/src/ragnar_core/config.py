from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RagnarConfig:
    version: str
    raw: dict[str, Any]

    def role_model(self, role_id: str) -> ModelConfig:
        default = self.raw.get("models", {}).get("default", {})
        role_config = self.raw.get("models", {}).get("roles", {}).get(role_id, {})
        merged = {**default, **role_config}
        return ModelConfig(
            provider=str(merged.get("provider", "local")),
            model=str(merged.get("model", "provider-not-configured")),
            temperature=float(merged.get("temperature", 0.1)),
            max_tokens=int(merged.get("max_tokens", 4096)),
        )

    def memory_mode(self) -> str:
        return str(self.raw.get("memory", {}).get("mode", "auto"))

    def embedding_model(self) -> str:
        return str(self.raw.get("models", {}).get("embedding", "letta/letta-free"))

    def allow_agent_edits(self) -> bool:
        return bool(self.raw.get("execution", {}).get("allow_agent_edits", False))

    def enable_agent_messaging(self) -> bool:
        return bool(self.raw.get("execution", {}).get("enable_agent_messaging", False))

    def agent_max_steps(self) -> int:
        return int(self.raw.get("execution", {}).get("agent_max_steps", 12))

    def max_plan_revisions(self, default: int = 2) -> int:
        return int(self.raw.get("execution", {}).get("max_plan_revisions", default))

    def max_qa_revisions(self, default: int = 2) -> int:
        return int(self.raw.get("execution", {}).get("max_qa_revisions", default))

    def enable_qa_profile_discovery(self) -> bool:
        return bool(self.raw.get("execution", {}).get("enable_qa_profile_discovery", False))

    def trivial_fast_path(self) -> bool:
        return bool(self.raw.get("execution", {}).get("trivial_fast_path", True))

    def trivial_skip_qa_agent(self) -> bool:
        return bool(self.raw.get("execution", {}).get("trivial_skip_qa_agent", True))

    def compact_letta_invocations(self) -> bool:
        return bool(self.raw.get("execution", {}).get("compact_letta_invocations", True))

    def to_dict(self) -> dict[str, Any]:
        return self.raw


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return repo_root() / "ragnar.yaml"


def load_config(path: Path | None = None) -> RagnarConfig:
    config_path = path or default_config_path()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return RagnarConfig(version=str(raw.get("version", "0.1")), raw=dict(raw))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Ragnar provider/runtime config.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--role")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.role:
        print(json.dumps(config.role_model(args.role).to_dict(), indent=2))
        return
    print(json.dumps(config.to_dict(), indent=2))


if __name__ == "__main__":
    main()
