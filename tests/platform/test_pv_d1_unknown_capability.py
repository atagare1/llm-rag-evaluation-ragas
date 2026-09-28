"""PV-D1: an unregistered capability fails before a run is recorded.

Uses the real EvaluationRunner and an empty real EvaluationRegistry.
"""

import pytest

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationRegistry
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner


def test_pv_d1_unknown_capability_raises_and_does_not_record_run():
    runner = EvaluationRunner(
        registry=EvaluationRegistry(),
        evaluators={},
        policies={},
        gate=QualityGate(),
    )
    trace = EvaluationTrace(
        trace_id="pv-d1",
        scenario_type="qa",
        input="2+2",
        output="4",
        expected="4",
    )
    config = EvaluationConfig(evaluations=["unknown_capability"])

    with pytest.raises(KeyError, match="Unknown evaluation capability"):
        runner.run(trace, config)

    assert runner.last_run is None
