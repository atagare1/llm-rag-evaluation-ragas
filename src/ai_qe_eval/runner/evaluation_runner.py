"""P2-14 Thin Evaluation Runner.

Coordinates frozen evaluation components. Does not evaluate, normalize,
apply thresholds, or decide the gate outcome itself.
"""

from __future__ import annotations

from typing import Any, Mapping

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationRegistry
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.gate.quality_gate import GateDecision, QualityGate
from ai_qe_eval.normalization.result_normalizer import normalize_many
from ai_qe_eval.policy.quality_policy import PolicyDecision, QualityPolicy


class EvaluationRunner:
    """Synchronous orchestrator for one EvaluationTrace.

    Dependencies are injected. The registry catalogs what can run;
    EvaluationConfig selects what should run; evaluators and policies
    are explicit instance maps keyed by capability name and metric.
    """

    def __init__(
        self,
        registry: EvaluationRegistry,
        evaluators: Mapping[str, Any],
        policies: Mapping[str, QualityPolicy],
        normalizer: Any | None = None,
        gate: QualityGate | None = None,
    ) -> None:
        self._registry = registry
        self._evaluators = dict(evaluators)
        self._policies = dict(policies)
        self._normalizer = normalizer
        self._gate = gate if gate is not None else QualityGate()

    def run(
        self,
        trace: EvaluationTrace,
        configuration: EvaluationConfig,
    ) -> GateDecision:
        collected: list[EvaluationResult] = []
        for capability_name in configuration.evaluations:
            self._registry.get(capability_name)
            evaluator = self._resolve_evaluator(capability_name)
            collected.extend(evaluator.evaluate(trace))

        normalized = self._normalize(collected)
        decisions: list[PolicyDecision] = []
        for result in normalized:
            policy = self._resolve_policy(result.metric)
            decisions.append(policy.apply(result))

        return self._gate.evaluate(decisions)

    def _resolve_evaluator(self, capability_name: str) -> Any:
        try:
            return self._evaluators[capability_name]
        except KeyError:
            raise KeyError(
                f"No evaluator instance wired for capability {capability_name!r}"
            ) from None

    def _resolve_policy(self, metric: str) -> QualityPolicy:
        try:
            return self._policies[metric]
        except KeyError:
            raise KeyError(
                f"No quality policy configured for metric {metric!r}"
            ) from None

    def _normalize(self, results: list[EvaluationResult]) -> list[EvaluationResult]:
        if self._normalizer is None:
            return normalize_many(results)
        return self._normalizer.normalize_many(results)
