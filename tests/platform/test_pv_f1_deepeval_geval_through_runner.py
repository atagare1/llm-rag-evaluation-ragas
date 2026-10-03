"""PV-F1: one live G-Eval correctness result through the real EvaluationRunner.

Uses the known-good exact case from the Layer F smoke and the same Together
LocalModel. The policy is the existing correctness rule used by the Phase 2
tests: correctness >= 0.80.
"""

import math
import os

import pytest
from deepeval.models.llms.local_model import LocalModel

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deepeval import DeepEvalGEvalCorrectnessEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

SMOKE_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
CORRECTNESS_THRESHOLD = 0.80


@pytest.mark.live
def test_pv_f1_deepeval_geval_through_runner():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        pytest.skip("OPENAI_API_KEY or OPENAI_BASE_URL is not set")

    request = {
        "correctness": {"args": ["What is 2 + 2?", "4", "4"], "kwargs": {}},
    }
    model = LocalModel(
        model=SMOKE_JUDGE_MODEL,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="correctness",
            evaluator="deepeval",
            category="semantic",
        )
    )
    policy = QualityPolicy(
        metric="correctness",
        operator=">=",
        threshold=CORRECTNESS_THRESHOLD,
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={
            "correctness": DeepEvalGEvalCorrectnessEvaluator(model=model)
        },
        policies={"correctness": policy},
        gate=QualityGate(),
    )
    config = EvaluationConfig(evaluations=["correctness"])
    print("provider_model", SMOKE_JUDGE_MODEL)
    print("policy_threshold", policy.threshold)

    decision = runner.run(request, config)

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "correctness"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.score == 1.0
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    assert len(runner.last_run.decisions) == 1
    assert runner.last_run.decisions[0].passed is True
    assert runner.last_run.decisions[0].threshold == CORRECTNESS_THRESHOLD
    print("geval_score", result.score)
    print("geval_reason", result.reason)
    print("gate_passed", decision.passed)
