"""Second live RAGAS faithfulness smoke with an injected judge model.

Reuses Phase 1 faithfulness test data and the get_test_data fixture.
The judge is the previously measured process-env model, passed only into
this test's RAGASFaithfulnessEvaluator. File defaults stay Mixtral.
"""

import math

import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.ragas import (
    FAITHFULNESS_METRIC,
    RAGAS_EVALUATOR_NAME,
    RAGASFaithfulnessEvaluator,
)
from utils import read_test_data

SMOKE_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


@pytest.mark.live
@pytest.mark.parametrize(
    "get_test_data",
    [read_test_data("rag_test_data_faithfulness.json")],
    indirect=True,
)
def test_ragas_faithfulness_smoke_with_llama_judge(get_test_data):
    sample = get_test_data
    trace = EvaluationTrace(
        trace_id="pv-e-faithfulness-llama-smoke",
        scenario_type="rag",
        input=sample.user_input,
        output=sample.response,
        expected=sample.reference,
        retrieval=list(sample.retrieved_contexts or []),
    )
    llm = ChatOpenAI(model=SMOKE_JUDGE_MODEL, temperature=0)
    wrapper = LangchainLLMWrapper(llm)
    print("provider_model", SMOKE_JUDGE_MODEL)

    results = RAGASFaithfulnessEvaluator(llm=wrapper).evaluate(trace)

    assert len(results) == 1
    result = results[0]
    assert result.metric == FAITHFULNESS_METRIC
    assert result.evaluator == RAGAS_EVALUATOR_NAME
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    print("faithfulness_score", result.score)
