from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalExecutionAdapter:
    """Runs bounded local commands and captures a compact report."""

    def __init__(self, cwd: Path, timeout_seconds: int = 120, output_limit: int = 12000) -> None:
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds
        self.output_limit = output_limit

    def run(self, command: list[str]) -> CommandResult:
        if not command:
            raise ValueError("Command cannot be empty")
        try:
            completed = subprocess.run(
                command,
                cwd=self.cwd,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return CommandResult(
                command=command,
                cwd=str(self.cwd),
                exit_code=completed.returncode,
                stdout=self._clip(completed.stdout),
                stderr=self._clip(completed.stderr),
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=command,
                cwd=str(self.cwd),
                exit_code=124,
                stdout=self._clip(exc.stdout or ""),
                stderr=self._clip(exc.stderr or f"Timed out after {self.timeout_seconds}s"),
                timed_out=True,
            )

    def _clip(self, value: str) -> str:
        if len(value) <= self.output_limit:
            return value
        return value[: self.output_limit] + "\n...[truncated]"
