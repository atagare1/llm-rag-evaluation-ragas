"""PV-F: one live DeepEval tool correctness call.

Uses the installed DeepEval LocalModel and Together Llama 3.3 70B Instruct Turbo.
The case is a provider-neutral tool call. No MCP server is required.
Exact match does not need the judge when available tools are omitted, but the
metric constructor still receives the LocalModel.
"""

import math
import os

import pytest
from deepeval.models.llms.local_model import LocalModel

from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.evaluators.deepeval_tool_correctness import DeepEvalToolCorrectnessEvaluator

SMOKE_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


def _weather() -> ToolInvocation:
    return ToolInvocation(
        name="weather",
        arguments={"location": "Pune"},
        result="Temperature is 28 C and conditions are clear.",
    )


@pytest.mark.live
def test_pv_f_deepeval_tool_correctness_smoke_returns_numeric_score():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        pytest.skip("OPENAI_API_KEY or OPENAI_BASE_URL is not set")

    call = _weather()
    model = LocalModel(
        model=SMOKE_JUDGE_MODEL,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    print("provider_model", SMOKE_JUDGE_MODEL)

    results = DeepEvalToolCorrectnessEvaluator(model=model).evaluate(
        [call],
        [_weather()],
        input="Find the weather for Pune.",
    )

    assert len(results) == 1
    result = results[0]
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    print("tool_correctness_score", result.score)
    print("tool_correctness_reason", result.reason)
