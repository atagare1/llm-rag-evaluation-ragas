"""P2-08 deterministic exact-match evaluator.

Compares trace.output == trace.expected with ordinary Python equality.
Produces one EvaluationResult. Does not apply thresholds or PASS/FAIL.
Does not auto-register capabilities.
"""

from __future__ import annotations

from typing import Any

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.trace import EvaluationTrace

EXACT_MATCH_METRIC = "exact_match"
DETERMINISTIC_EVALUATOR_NAME = "deterministic"
EXACT_MATCH_SUCCEEDED_REASON = "Exact match succeeded."
EXACT_MATCH_FAILED_REASON = "Exact match failed."


class DeterministicEvaluator:
    def evaluate(
        self,
        trace: EvaluationTrace,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        matched = trace.output == trace.expected
        score = 1.0 if matched else 0.0
        return [
            EvaluationResult(
                metric=EXACT_MATCH_METRIC,
                evaluator=DETERMINISTIC_EVALUATOR_NAME,
                score=score,
                reason=(
                    EXACT_MATCH_SUCCEEDED_REASON
                    if matched
                    else EXACT_MATCH_FAILED_REASON
                ),
                raw_result={
                    "output": trace.output,
                    "expected": trace.expected,
                    "matched": matched,
                },
            )
        ]
