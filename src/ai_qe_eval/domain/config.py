"""EvaluationConfig — P2-07.

Selected evaluation capability identifiers for a run.
Answers which capabilities should run; does not execute them,
configure how they run, or apply quality policy.

Capability names are stored without consulting the registry.
Registry validation is deferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationConfig:
    evaluations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        selected = list(self.evaluations)
        seen: set[str] = set()
        for name in selected:
            if name in seen:
                raise ValueError(
                    f"Duplicate evaluation capability in configuration: {name!r}"
                )
            seen.add(name)
        self.evaluations = selected

    def to_dict(self) -> dict[str, Any]:
        return {"evaluations": list(self.evaluations)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationConfig:
        return cls(evaluations=list(data.get("evaluations") or []))
