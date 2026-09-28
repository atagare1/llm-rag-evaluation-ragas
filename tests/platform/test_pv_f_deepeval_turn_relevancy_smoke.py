"""PV-F: one live DeepEval turn relevancy call.

Uses the installed DeepEval LocalModel and Together Llama 3.3 70B Instruct Turbo.
Credentials come from OPENAI_API_KEY and OPENAI_BASE_URL.
The conversation ends on an assistant turn so DeepEval's unit grouping keeps it.
"""

import math
import os

import pytest
from deepeval.models.llms.local_model import LocalModel

from ai_qe_eval.domain.conversation import ConversationTurn
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.deepeval_turn_relevancy import DeepEvalTurnRelevancyEvaluator

SMOKE_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


@pytest.mark.live
def test_pv_f_deepeval_turn_relevancy_smoke_returns_numeric_score():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        pytest.skip("OPENAI_API_KEY or OPENAI_BASE_URL is not set")

    trace = EvaluationTrace(
        trace_id="pv-f-turn-relevancy",
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
    print("provider_model", SMOKE_JUDGE_MODEL)

    results = DeepEvalTurnRelevancyEvaluator(model=model).evaluate(trace)

    assert len(results) == 1
    result = results[0]
    assert result.metric == "turn_relevancy"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    print("turn_relevancy_score", result.score)
    print("turn_relevancy_reason", result.reason)
