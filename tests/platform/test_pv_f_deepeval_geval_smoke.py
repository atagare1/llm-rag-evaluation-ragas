"""PV-F: one live DeepEval G-Eval correctness call.

Uses the installed DeepEval 4.2.6 LocalModel, which is the OpenAI-compatible
client that accepts an explicit model name and base URL. The evaluator's
model= argument receives that object. Credentials come from the existing
OPENAI_API_KEY and OPENAI_BASE_URL environment values.
"""

import math
import os

import pytest
from deepeval.models.llms.local_model import LocalModel

from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.deepeval import DeepEvalGEvalCorrectnessEvaluator

SMOKE_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


@pytest.mark.live
def test_pv_f_deepeval_geval_smoke_returns_numeric_score():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        pytest.skip("OPENAI_API_KEY or OPENAI_BASE_URL is not set")

    trace = EvaluationTrace(
        trace_id="pv-f",
        scenario_type="llm",
        input="What is 2 + 2?",
        output="4",
        expected="4",
    )
    model = LocalModel(
        model=SMOKE_JUDGE_MODEL,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    print("provider_model", SMOKE_JUDGE_MODEL)

    results = DeepEvalGEvalCorrectnessEvaluator(model=model).evaluate(trace)

    assert len(results) == 1
    result = results[0]
    assert result.metric == "correctness"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    print("geval_score", result.score)
    print("geval_reason", result.reason)
