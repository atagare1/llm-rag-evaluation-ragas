"""PV-B2: one exact_match FAIL through the real evaluation runner.

Same wiring as PV-B1. The trace output does not equal expected.
"""

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.deterministic import DeterministicEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner


def test_pv_b2_deterministic_exact_match_fail_through_runner():
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="exact_match",
            evaluator="deterministic",
            category="deterministic",
        )
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={"exact_match": DeterministicEvaluator()},
        policies={
            "exact_match": QualityPolicy(
                metric="exact_match",
                operator=">=",
                threshold=1.0,
            )
        },
        gate=QualityGate(),
    )
    trace = EvaluationTrace(
        trace_id="pv-b2",
        scenario_type="qa",
        input="2+2",
        output="5",
        expected="4",
    )
    config = EvaluationConfig(evaluations=["exact_match"])

    decision = runner.run(trace, config)

    assert decision.passed is False
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert runner.last_run.results[0].score == 0.0
    assert runner.last_run.results[0].metric == "exact_match"
