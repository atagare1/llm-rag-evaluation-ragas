"""Evaluation capability registry — P2-06.

Catalog of available evaluation capabilities (name, evaluator, category).
Does not execute evaluators, own thresholds, or select what should run.

Lookup:
- get(name) raises KeyError if the capability is not registered
- contains(name) reports presence
- list() returns an insertion-order snapshot; mutating it does not
  change registry state
- register() rejects duplicate names with ValueError
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationCapability:
    name: str
    evaluator: str
    category: str


class EvaluationRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[str, EvaluationCapability] = {}

    def register(self, capability: EvaluationCapability) -> None:
        if capability.name in self._capabilities:
            raise ValueError(
                f"Evaluation capability already registered: {capability.name!r}"
            )
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> EvaluationCapability:
        try:
            return self._capabilities[name]
        except KeyError:
            raise KeyError(f"Unknown evaluation capability: {name!r}") from None

    def contains(self, name: str) -> bool:
        return name in self._capabilities

    def list(self) -> list[EvaluationCapability]:
        return list(self._capabilities.values())
