"""One live RAGAS faithfulness smoke through RAGASFaithfulnessEvaluator.

Reuses Phase 1 faithfulness test data and the get_test_data / llm_wrapper
fixtures. Does not apply a quality threshold.
"""

import math

import pytest

from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.ragas import (
    FAITHFULNESS_METRIC,
    RAGAS_EVALUATOR_NAME,
    RAGASFaithfulnessEvaluator,
)
from utils import llm_model_name, read_test_data


@pytest.mark.live
@pytest.mark.parametrize(
    "get_test_data",
    [read_test_data("rag_test_data_faithfulness.json")],
    indirect=True,
)
def test_ragas_faithfulness_smoke_returns_numeric_score(get_test_data, llm_wrapper):
    sample = get_test_data
    trace = EvaluationTrace(
        trace_id="pv-e-faithfulness-smoke",
        scenario_type="rag",
        input=sample.user_input,
        output=sample.response,
        expected=sample.reference,
        retrieval=list(sample.retrieved_contexts or []),
    )
    print("provider_model", llm_model_name())

    results = RAGASFaithfulnessEvaluator(llm=llm_wrapper).evaluate(trace)

    assert len(results) == 1
    result = results[0]
    assert result.metric == FAITHFULNESS_METRIC
    assert result.evaluator == RAGAS_EVALUATOR_NAME
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    print("faithfulness_score", result.score)
