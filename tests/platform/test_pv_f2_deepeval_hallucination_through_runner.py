"""PV-F2: one live DeepEval hallucination result through the real EvaluationRunner.

Uses the known-good Paris trace from the hallucination smoke and the same
Together LocalModel. DeepEval 4.2.6 scores alignment, so higher is better.
The policy is hallucination >= metric_threshold("hallucination"), default 0.8.
DeepEval's own threshold is not the platform gate.
"""

import math
import os

import pytest
from deepeval.models.llms.local_model import LocalModel

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.deepeval import DeepEvalHallucinationEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner
from utils import metric_threshold

SMOKE_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


@pytest.mark.live
def test_pv_f2_deepeval_hallucination_through_runner():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        pytest.skip("OPENAI_API_KEY or OPENAI_BASE_URL is not set")

    trace = EvaluationTrace(
        trace_id="pv-f2-hallucination",
        scenario_type="rag",
        input="What is the capital of France?",
        output="Paris.",
        expected="Paris.",
        retrieval=["Paris is the capital and largest city of France."],
    )
    model = LocalModel(
        model=SMOKE_JUDGE_MODEL,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="hallucination",
            evaluator="deepeval",
            category="rag",
        )
    )
    policy = QualityPolicy(
        metric="hallucination",
        operator=">=",
        threshold=metric_threshold("hallucination"),
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={"hallucination": DeepEvalHallucinationEvaluator(model=model)},
        policies={"hallucination": policy},
        gate=QualityGate(),
    )
    config = EvaluationConfig(evaluations=["hallucination"])
    print("provider_model", SMOKE_JUDGE_MODEL)
    print("policy_threshold", policy.threshold)
    print("policy_operator", policy.operator)

    decision = runner.run(trace, config)

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "hallucination"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.score >= policy.threshold
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    assert len(runner.last_run.decisions) == 1
    assert runner.last_run.decisions[0].passed is True
    assert runner.last_run.decisions[0].operator == ">="
    assert runner.last_run.decisions[0].threshold == policy.threshold
    print("hallucination_score", result.score)
    print("hallucination_reason", result.reason)
    print("gate_passed", decision.passed)
