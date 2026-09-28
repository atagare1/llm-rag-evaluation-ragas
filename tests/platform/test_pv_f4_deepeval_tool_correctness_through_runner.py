"""PV-F4: one live tool correctness result through the real EvaluationRunner.

Uses the same Together LocalModel as the tool correctness smoke. The platform
gate is an explicit QualityPolicy of tool_correctness >= 0.80. DeepEval's own
threshold is not the gate. tool_correctness is not added to the historical
RAG threshold map.
"""

import math
import os

import pytest
from deepeval.models.llms.local_model import LocalModel

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.conversation import ConversationTurn, ToolInvocation
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.deepeval_tool_correctness import DeepEvalToolCorrectnessEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

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

    trace = EvaluationTrace(
        trace_id="pv-f4-tool-correctness",
        scenario_type="agent",
        input="Find the weather for Pune.",
        output="Temperature is 28 C and conditions are clear.",
        expected="Temperature is 28 C and conditions are clear.",
        turns=[
            ConversationTurn(role="user", content="Find the weather for Pune."),
            ConversationTurn(
                role="assistant",
                content="Temperature is 28 C and conditions are clear.",
                tool_calls=[_weather()],
            ),
        ],
        expected_tool_calls=[_weather()],
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
            name="tool_correctness",
            evaluator="deepeval",
            category="agent",
        )
    )
    policy = QualityPolicy(
        metric="tool_correctness",
        operator=">=",
        threshold=TOOL_CORRECTNESS_THRESHOLD,
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={
            "tool_correctness": DeepEvalToolCorrectnessEvaluator(model=model)
        },
        policies={"tool_correctness": policy},
        gate=QualityGate(),
    )
    print("provider_model", SMOKE_JUDGE_MODEL)
    print("policy_threshold", policy.threshold)

    decision = runner.run(
        trace,
        EvaluationConfig(evaluations=["tool_correctness"]),
        run_id="pv-f4-tool-correctness",
    )

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.score >= TOOL_CORRECTNESS_THRESHOLD
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    assert runner.last_run.decisions[0].passed is True
    assert runner.last_run.decisions[0].threshold == TOOL_CORRECTNESS_THRESHOLD
    print("tool_correctness_score", result.score)
    print("tool_correctness_reason", result.reason)
    print("gate_passed", decision.passed)
