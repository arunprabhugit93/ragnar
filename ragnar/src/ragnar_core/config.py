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
