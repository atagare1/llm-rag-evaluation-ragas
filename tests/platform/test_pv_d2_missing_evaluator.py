"""PV-D2: a registered capability with no wired evaluator fails before a run is recorded.

Uses the real EvaluationRunner and a real EvaluationRegistry.
The capability is registered. The runner is given no evaluator instance for it.
"""

import pytest

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner


def test_pv_d2_missing_evaluator_instance_raises_and_does_not_record_run():
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
        evaluators={},
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
        trace_id="pv-d2",
        scenario_type="qa",
        input="2+2",
        output="4",
        expected="4",
    )
    config = EvaluationConfig(evaluations=["exact_match"])

    with pytest.raises(KeyError, match="No evaluator instance wired for capability"):
        runner.run(trace, config)

    assert runner.last_run is None
