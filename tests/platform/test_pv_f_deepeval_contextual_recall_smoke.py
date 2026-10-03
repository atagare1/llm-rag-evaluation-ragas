"""Live DeepEval Contextual Recall smoke.

Uses the same Together LocalModel as the other DeepEval smokes. Calls
DeepEvalContextualRecallEvaluator directly. Does not apply a quality threshold.
"""

import math
import os

import pytest
from deepeval.models.llms.local_model import LocalModel

from ai_qe_eval.evaluators.deepeval import DeepEvalContextualRecallEvaluator

SMOKE_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


@pytest.mark.live
def test_pv_f_deepeval_contextual_recall_smoke_returns_numeric_score():
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
    print("provider_model", SMOKE_JUDGE_MODEL)

    results = DeepEvalContextualRecallEvaluator(model=model).evaluate(
        "What is 2 + 2?",
        "4",
        ["2 + 2 = 4"],
    )

    assert len(results) == 1
    result = results[0]
    assert result.metric == "contextual_recall"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    print("contextual_recall_score", result.score)
    print("contextual_recall_reason", result.reason)
