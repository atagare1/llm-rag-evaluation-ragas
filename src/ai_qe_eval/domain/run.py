"""EvaluationRun — P2-03 domain model; P3-02 execution record; P3-03 multi-trace.

Container for one logical evaluation execution of one or more traces.
Does not evaluate, apply thresholds, decide the gate, or aggregate scores.

Association is index-aligned:
    traces[i] <-> trace_evaluations[i]

EvaluationResult has no trace id. TraceEvaluation keeps that trace's
normalized results and policy decisions together.

results and decisions on the run are the same objects in trace order,
then evaluator/result order. They are a flat view, not a second scoring model.

The gate decision is whatever QualityGate returned for that flat decision
list. EvaluationRun does not invent a separate multi-trace rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.gate.quality_gate import GateDecision
from ai_qe_eval.policy.quality_policy import PolicyDecision


@dataclass
class TraceEvaluation:
    """Normalized results and policy decisions for one trace. Not an executor."""

    results: list[EvaluationResult] = field(default_factory=list)
    decisions: list[PolicyDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEvaluation:
        raw_results = data.get("results") or []
        raw_decisions = data.get("decisions") or []
        return cls(
            results=[EvaluationResult.from_dict(item) for item in raw_results],
            decisions=[PolicyDecision.from_dict(item) for item in raw_decisions],
        )


@dataclass
class EvaluationRun:
    run_id: str
    traces: list[EvaluationTrace] = field(default_factory=list)
    results: list[EvaluationResult] = field(default_factory=list)
    decisions: list[PolicyDecision] = field(default_factory=list)
    gate_decision: GateDecision | None = None
    trace_evaluations: list[TraceEvaluation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "traces": [trace.to_dict() for trace in self.traces],
            "results": [result.to_dict() for result in self.results],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "gate_decision": (
                None if self.gate_decision is None else self.gate_decision.to_dict()
            ),
            "trace_evaluations": [item.to_dict() for item in self.trace_evaluations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRun:
        raw_traces = data.get("traces") or []
        raw_results = data.get("results") or []
        raw_decisions = data.get("decisions") or []
        raw_gate = data.get("gate_decision")
        raw_groups = data.get("trace_evaluations") or []
        return cls(
            run_id=data["run_id"],
            traces=[EvaluationTrace.from_dict(item) for item in raw_traces],
            results=[EvaluationResult.from_dict(item) for item in raw_results],
            decisions=[PolicyDecision.from_dict(item) for item in raw_decisions],
            gate_decision=None if raw_gate is None else GateDecision.from_dict(raw_gate),
            trace_evaluations=[TraceEvaluation.from_dict(item) for item in raw_groups],
        )
