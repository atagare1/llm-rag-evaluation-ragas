"""PV-F4: one live tool correctness result through QualityPolicy and QualityGate.

Uses the same Together LocalModel as the tool correctness smoke. The platform
gate is an explicit QualityPolicy of tool_correctness >= 0.80. DeepEval's own
threshold is not the gate. tool_correctness is not added to the historical
RAG threshold map. EvaluationRunner is not used because ToolCorrectness no
longer accepts EvaluationTrace.
"""

import math
import os

import pytest
from deepeval.models.llms.local_model import LocalModel

from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.evaluators.deepeval_tool_correctness import DeepEvalToolCorrectnessEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy

SMOKE_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
TOOL_CORRECTNESS_THRESHOLD = 0.80


def _weather() -> ToolInvocation:
    return ToolInvocation(
        name="weather",
        arguments={"location": "Pune"},
        result="Temperature is 28 C and conditions are clear.",
    )


@pytest.mark.live
def test_pv_f4_deepeval_tool_correctness_through_runner():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        pytest.skip("OPENAI_API_KEY or OPENAI_BASE_URL is not set")

    model = LocalModel(
        model=SMOKE_JUDGE_MODEL,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    policy = QualityPolicy(
        metric="tool_correctness",
        operator=">=",
        threshold=TOOL_CORRECTNESS_THRESHOLD,
    )
    print("provider_model", SMOKE_JUDGE_MODEL)
    print("policy_threshold", policy.threshold)

    result = DeepEvalToolCorrectnessEvaluator(model=model).evaluate(
        [_weather()],
        [_weather()],
        input="Find the weather for Pune.",
    )[0]
    policy_decision = policy.apply(result)
    decision = QualityGate().evaluate([policy_decision])

    assert decision.passed is True
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.score >= TOOL_CORRECTNESS_THRESHOLD
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    assert policy_decision.passed is True
    assert policy_decision.threshold == TOOL_CORRECTNESS_THRESHOLD
    print("tool_correctness_score", result.score)
    print("tool_correctness_reason", result.reason)
    print("gate_passed", decision.passed)
