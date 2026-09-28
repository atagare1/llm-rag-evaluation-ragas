"""PV-F3: one live turn relevancy result through the real EvaluationRunner.

Uses the same Together LocalModel as the turn relevancy smoke. The platform
gate is an explicit QualityPolicy of turn_relevancy >= 0.80, matching the
numeric convention used for correctness and answer relevancy. DeepEval's own
threshold is not the gate. This key is not added to the historical RAG
threshold map.
"""

import math
import os

import pytest
from deepeval.models.llms.local_model import LocalModel

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.conversation import ConversationTurn
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.deepeval_turn_relevancy import DeepEvalTurnRelevancyEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

SMOKE_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
TURN_RELEVANCY_THRESHOLD = 0.80


@pytest.mark.live
def test_pv_f3_deepeval_turn_relevancy_through_runner():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        pytest.skip("OPENAI_API_KEY or OPENAI_BASE_URL is not set")

    trace = EvaluationTrace(
        trace_id="pv-f3-turn-relevancy",
        scenario_type="chat",
        input="What is 2 + 2?",
        output="3 + 3 is 6.",
        expected="3 + 3 is 6.",
        turns=[
            ConversationTurn(role="user", content="What is 2 + 2?"),
            ConversationTurn(role="assistant", content="2 + 2 is 4."),
            ConversationTurn(role="user", content="What is 3 + 3?"),
            ConversationTurn(role="assistant", content="3 + 3 is 6."),
        ],
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
            name="turn_relevancy",
            evaluator="deepeval",
            category="semantic",
        )
    )
    policy = QualityPolicy(
        metric="turn_relevancy",
        operator=">=",
        threshold=TURN_RELEVANCY_THRESHOLD,
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={"turn_relevancy": DeepEvalTurnRelevancyEvaluator(model=model)},
        policies={"turn_relevancy": policy},
        gate=QualityGate(),
    )
    config = EvaluationConfig(evaluations=["turn_relevancy"])
    print("provider_model", SMOKE_JUDGE_MODEL)
    print("policy_threshold", policy.threshold)

    decision = runner.run(trace, config, run_id="pv-f3-turn-relevancy")

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "turn_relevancy"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.score >= TURN_RELEVANCY_THRESHOLD
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    assert len(runner.last_run.decisions) == 1
    assert runner.last_run.decisions[0].passed is True
    assert runner.last_run.decisions[0].threshold == TURN_RELEVANCY_THRESHOLD
    print("turn_relevancy_score", result.score)
    print("turn_relevancy_reason", result.reason)
    print("gate_passed", decision.passed)
