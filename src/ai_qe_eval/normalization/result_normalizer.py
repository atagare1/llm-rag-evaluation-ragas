"""P2-11 Result Normalizer.

Canonicalizes EvaluationResult structure without changing meaning.
Does not recalculate scores, apply thresholds, or decide PASS/FAIL.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ai_qe_eval.domain.result import EvaluationResult


def _copy_raw(raw_result: Any) -> Any:
    if raw_result is None:
        return None
    return deepcopy(raw_result)


def normalize(result: EvaluationResult) -> EvaluationResult:
    if not isinstance(result, EvaluationResult):
        raise TypeError(
            "normalize() requires an EvaluationResult, "
            f"got {type(result).__name__}"
        )
    return EvaluationResult(
        metric=result.metric,
        evaluator=result.evaluator,
        score=result.score,
        reason=result.reason,
        raw_result=_copy_raw(result.raw_result),
    )


def normalize_many(results: list[EvaluationResult]) -> list[EvaluationResult]:
    if not isinstance(results, list):
        raise TypeError(
            "normalize_many() requires a list of EvaluationResult, "
            f"got {type(results).__name__}"
        )
    return [normalize(item) for item in results]
