from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RoleContract:
    role_id: str
    display_name: str
    team: str
    responsibility: list[str]
    authority: dict[str, list[str]]
    memory: dict[str, Any]
    handoffs: dict[str, list[str]]
    isolation: dict[str, str]

    @property
    def private_memory_namespace(self) -> str:
        return str(self.memory["private_namespace"])

    def requires_approval(self, action: str) -> bool:
        return action in self.authority.get("requires_approval", [])

    def denies(self, action: str) -> bool:
        return action in self.authority.get("cannot", [])

    def allows(self, action: str) -> bool:
        return action in self.authority.get("can", [])


class RoleRegistry:
    def __init__(self, roles: list[RoleContract]) -> None:
        self._roles = {role.role_id: role for role in roles}
        if len(self._roles) != len(roles):
            raise ValueError("Duplicate role_id in role registry")

    def all(self) -> list[RoleContract]:
        return list(self._roles.values())

    def get(self, role_id: str) -> RoleContract:
        try:
            return self._roles[role_id]
        except KeyError as exc:
            raise KeyError(f"Unknown role_id: {role_id}") from exc

    def by_team(self, team: str) -> list[RoleContract]:
        return [role for role in self._roles.values() if role.team == team]


def load_role_registry(path: str | Path) -> RoleRegistry:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    roles = []
    for item in raw.get("roles", []):
        roles.append(
            RoleContract(
                role_id=item["role_id"],
                display_name=item.get("display_name", item["role_id"]),
                team=item["team"],
                responsibility=list(item.get("responsibility", [])),
                authority=dict(item.get("authority", {})),
                memory=dict(item.get("memory", {})),
                handoffs=dict(item.get("handoffs", {})),
                isolation=dict(item.get("isolation", {})),
            )
        )
    return RoleRegistry(roles)

