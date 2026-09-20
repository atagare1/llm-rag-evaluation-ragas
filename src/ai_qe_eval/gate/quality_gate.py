"""P2-13 Quality Gate.

Aggregates PolicyDecision outcomes into one release-level GateDecision.
Does not execute evaluators, apply thresholds, or aggregate scores.

Empty-input behavior: an empty list of decisions is a PASS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ai_qe_eval.policy.quality_policy import PolicyDecision

_REASON_EMPTY = "No policy decisions were supplied; the gate passes by default."
_REASON_PASS = "All quality policies passed."
_REASON_FAIL = "One or more quality policies failed."


@dataclass
class GateDecision:
    passed: bool
    decisions: list[PolicyDecision]
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateDecision:
        raw_decisions = data.get("decisions") or []
        return cls(
            passed=data["passed"],
            decisions=[PolicyDecision.from_dict(item) for item in raw_decisions],
            reason=data.get("reason"),
        )


class QualityGate:
    """Release-level aggregation of policy outcomes.

    The gate trusts PolicyDecision.passed and does not re-evaluate
    score, operator, or threshold.
    """

    def evaluate(self, decisions: Sequence[PolicyDecision]) -> GateDecision:
        if decisions is None:
            raise TypeError("QualityGate.evaluate() requires a list of PolicyDecision, got None")
        if not isinstance(decisions, (list, tuple)):
            raise TypeError(
                "QualityGate.evaluate() requires a list of PolicyDecision, "
                f"got {type(decisions).__name__}"
            )
        for index, item in enumerate(decisions):
            if not isinstance(item, PolicyDecision):
                raise TypeError(
                    "QualityGate.evaluate() requires PolicyDecision items, "
                    f"got {type(item).__name__} at index {index}"
                )

        preserved = list(decisions)
        passed = all(decision.passed for decision in preserved)
        if not preserved:
            reason = _REASON_EMPTY
        elif passed:
            reason = _REASON_PASS
        else:
            reason = _REASON_FAIL
        return GateDecision(passed=passed, decisions=preserved, reason=reason)
