"""PV-C3: two traces, one exact_match FAIL, through run_many.

Same wiring as PV-B3. The failing policy decision stays on the second trace.
The run gate fails when any policy decision fails.
"""

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.deterministic import DeterministicEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner


def test_pv_c3_multi_trace_one_fail_through_runner():
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
    trace1 = EvaluationTrace(
        trace_id="pv-c3-1",
        scenario_type="qa",
        input="2+2",
        output="4",
        expected="4",
    )
    trace2 = EvaluationTrace(
        trace_id="pv-c3-2",
        scenario_type="qa",
        input="2+2",
        output="5",
        expected="4",
    )
    config = EvaluationConfig(evaluations=["exact_match"])

    decision = runner.run_many([trace1, trace2], config)

    assert decision.passed is False
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert runner.last_run.results[0].score == 1.0
    assert runner.last_run.results[1].score == 0.0
    assert runner.last_run.trace_evaluations[0].decisions[0].passed is True
    assert runner.last_run.trace_evaluations[1].decisions[0].passed is False
