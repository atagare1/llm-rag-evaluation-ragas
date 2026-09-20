"""EvaluationResult — P2-04 domain model.

Canonical, evaluator-agnostic outcome of one metric evaluation.
Describes what an evaluator determined; it does not apply thresholds
or decide PASS/FAIL (QualityPolicy / QualityGate are later phases).

Lineage, metadata, raw_key, nullable score, and severity are deferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    metric: str
    evaluator: str
    score: int | float
    reason: str | None = None
    raw_result: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "evaluator": self.evaluator,
            "score": self.score,
            "reason": self.reason,
            "raw_result": self.raw_result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        return cls(
            metric=data["metric"],
            evaluator=data["evaluator"],
            score=data["score"],
            reason=data.get("reason"),
            raw_result=data.get("raw_result"),
        )
