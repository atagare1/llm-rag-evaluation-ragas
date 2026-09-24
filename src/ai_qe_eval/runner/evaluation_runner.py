"""P2-14 Thin Evaluation Runner.

Coordinates frozen evaluation components. Does not evaluate, normalize,
apply thresholds, or decide the gate outcome itself.
"""

from __future__ import annotations

from typing import Any, Mapping

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationRegistry
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.run import EvaluationRun, TraceEvaluation
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.gate.quality_gate import GateDecision, QualityGate
from ai_qe_eval.normalization.result_normalizer import normalize_many
from ai_qe_eval.policy.quality_policy import PolicyDecision, QualityPolicy


class EvaluationRunner:
    """Synchronous orchestrator.

    run() evaluates one trace. run_many() evaluates several traces in order
    and records one EvaluationRun. Dependencies are injected.
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
        self.last_run: EvaluationRun | None = None

    def run(
        self,
        trace: EvaluationTrace,
        configuration: EvaluationConfig,
        *,
        run_id: str = "",
    ) -> GateDecision:
        return self.run_many([trace], configuration, run_id=run_id)

    def run_many(
        self,
        traces: list[EvaluationTrace],
        configuration: EvaluationConfig,
        *,
        run_id: str = "",
    ) -> GateDecision:
        if not isinstance(traces, list):
            raise TypeError(
                "EvaluationRunner.run_many() requires a list of EvaluationTrace, "
                f"got {type(traces).__name__}"
            )
        if not traces:
            raise ValueError("EvaluationRunner.run_many() requires at least one EvaluationTrace")
        for index, trace in enumerate(traces):
            if not isinstance(trace, EvaluationTrace):
                raise TypeError(
                    "EvaluationRunner.run_many() requires EvaluationTrace items, "
                    f"got {type(trace).__name__} at index {index}"
                )

        self.last_run = None
        groups: list[TraceEvaluation] = []
        flat_results: list[EvaluationResult] = []
        flat_decisions: list[PolicyDecision] = []
        for trace in traces:
            normalized, decisions = self._evaluate_trace(trace, configuration)
            groups.append(TraceEvaluation(results=normalized, decisions=decisions))
            flat_results.extend(normalized)
            flat_decisions.extend(decisions)

        gate_decision = self._gate.evaluate(flat_decisions)
        self.last_run = EvaluationRun(
            run_id=run_id,
            traces=list(traces),
            results=flat_results,
            decisions=flat_decisions,
            gate_decision=gate_decision,
            trace_evaluations=groups,
        )
        return gate_decision

    def _evaluate_trace(
        self,
        trace: EvaluationTrace,
        configuration: EvaluationConfig,
    ) -> tuple[list[EvaluationResult], list[PolicyDecision]]:
        collected: list[EvaluationResult] = []
        for capability_name in configuration.evaluations:
            self._registry.get(capability_name)
            evaluator = self._resolve_evaluator(capability_name)
            collected.extend(evaluator.evaluate(trace))
        normalized = self._normalize(collected)
        decisions = [
            self._resolve_policy(result.metric).apply(result) for result in normalized
        ]
        return normalized, decisions

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
