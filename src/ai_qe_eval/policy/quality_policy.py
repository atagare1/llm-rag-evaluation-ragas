"""P2-12 Quality Policy.

Applies a metric threshold rule to one EvaluationResult.
Does not execute evaluators, mutate scores, or make release decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ai_qe_eval.domain.result import EvaluationResult

SUPPORTED_OPERATORS = (">", ">=", "<", "<=", "==", "!=")

_OPERATOR_FUNCTIONS: dict[str, Callable[[int | float, int | float], bool]] = {
    ">": lambda score, threshold: score > threshold,
    ">=": lambda score, threshold: score >= threshold,
    "<": lambda score, threshold: score < threshold,
    "<=": lambda score, threshold: score <= threshold,
    "==": lambda score, threshold: score == threshold,
    "!=": lambda score, threshold: score != threshold,
}


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass
class QualityPolicy:
    metric: str
    operator: str
    threshold: int | float

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("QualityPolicy metric must be a non-empty string")
        if self.operator not in _OPERATOR_FUNCTIONS:
            raise ValueError(
                f"Unsupported QualityPolicy operator: {self.operator!r}. "
                f"Supported operators: {', '.join(SUPPORTED_OPERATORS)}"
            )
        if not _is_numeric(self.threshold):
            raise TypeError("QualityPolicy threshold must be a numeric int or float")

    def apply(self, result: EvaluationResult) -> PolicyDecision:
        if not isinstance(result, EvaluationResult):
            raise TypeError(
                "QualityPolicy.apply() requires an EvaluationResult, "
                f"got {type(result).__name__}"
            )
        if result.metric != self.metric:
            raise ValueError(
                f"QualityPolicy metric {self.metric!r} does not match "
                f"EvaluationResult metric {result.metric!r}"
            )
        if not _is_numeric(result.score):
            raise TypeError(
                "EvaluationResult.score must be a numeric int or float "
                f"for policy comparison, got {type(result.score).__name__}"
            )
        passed = _OPERATOR_FUNCTIONS[self.operator](result.score, self.threshold)
        return PolicyDecision(
            metric=self.metric,
            score=result.score,
            operator=self.operator,
            threshold=self.threshold,
            passed=passed,
            reason=(
                f"{result.score} {self.operator} {self.threshold} "
                f"is {'true' if passed else 'false'}"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityPolicy:
        return cls(
            metric=data["metric"],
            operator=data["operator"],
            threshold=data["threshold"],
        )


@dataclass
class PolicyDecision:
    metric: str
    score: int | float
    operator: str
    threshold: int | float
    passed: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "score": self.score,
            "operator": self.operator,
            "threshold": self.threshold,
            "passed": self.passed,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyDecision:
        return cls(
            metric=data["metric"],
            score=data["score"],
            operator=data["operator"],
            threshold=data["threshold"],
            passed=data["passed"],
            reason=data.get("reason"),
        )
