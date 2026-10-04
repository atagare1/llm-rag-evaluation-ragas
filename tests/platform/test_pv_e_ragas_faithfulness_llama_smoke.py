"""Second live RAGAS faithfulness smoke with an injected judge model.

Reuses Phase 1 faithfulness test data and the get_test_data fixture.
The judge is RAGAS_LLM_MODEL or the OpenRouter Llama default, passed only
into this test's RAGASFaithfulnessEvaluator. File defaults stay Mixtral.
"""

import math

import pytest
from ragas.llms import LangchainLLMWrapper

from ai_qe_eval.evaluators.ragas import (
    FAITHFULNESS_METRIC,
    RAGAS_EVALUATOR_NAME,
    RAGASFaithfulnessEvaluator,
)
from ragas_live import live_ragas_llama_chat, ragas_llama_judge_model_name
from utils import read_test_data


@pytest.mark.live
@pytest.mark.parametrize(
    "get_test_data",
    [read_test_data("rag_test_data_faithfulness.json")],
    indirect=True,
)
def test_ragas_faithfulness_smoke_with_llama_judge(get_test_data):
    sample = get_test_data
    llm = live_ragas_llama_chat()
    wrapper = LangchainLLMWrapper(llm)
    print("provider_model", ragas_llama_judge_model_name())

    results = RAGASFaithfulnessEvaluator(llm=wrapper).evaluate(
        sample.user_input,
        sample.response,
        list(sample.retrieved_contexts or []),
    )

    assert len(results) == 1
    result = results[0]
    assert result.metric == FAITHFULNESS_METRIC
    assert result.evaluator == RAGAS_EVALUATOR_NAME
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    print("faithfulness_score", result.score)
