"""Evaluator contract — P2-05.

Synchronous, evaluator-agnostic invocation:

    evaluate(trace, configuration=None) -> list[EvaluationResult]

Configuration is opaque. Thresholds, PASS/FAIL, registry, adapters,
and concrete evaluators are deferred.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.trace import EvaluationTrace


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(
        self,
        trace: EvaluationTrace,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        ...
