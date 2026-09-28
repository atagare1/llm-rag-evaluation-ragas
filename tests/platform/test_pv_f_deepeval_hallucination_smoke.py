"""Live DeepEval Hallucination smoke.

Uses the same Together LocalModel as the other DeepEval smokes. Calls
DeepEvalHallucinationEvaluator directly. Does not apply a quality threshold.

DeepEval 4.2.6 scores alignment: 1 means the output agrees with the context.
"""

import math
import os

import pytest
from deepeval.models.llms.local_model import LocalModel

from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.deepeval import DeepEvalHallucinationEvaluator

SMOKE_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


@pytest.mark.live
def test_pv_f_deepeval_hallucination_smoke_returns_numeric_score():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        pytest.skip("OPENAI_API_KEY or OPENAI_BASE_URL is not set")

    trace = EvaluationTrace(
        trace_id="pv-f-hallucination",
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
    print("provider_model", SMOKE_JUDGE_MODEL)

    results = DeepEvalHallucinationEvaluator(model=model).evaluate(trace)

    assert len(results) == 1
    result = results[0]
    assert result.metric == "hallucination"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    print("hallucination_score", result.score)
    print("hallucination_reason", result.reason)
