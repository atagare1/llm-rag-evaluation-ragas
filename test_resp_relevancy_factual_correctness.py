import math

import pytest
from ragas import EvaluationDataset, evaluate
from ragas.metrics import FactualCorrectness, ResponseRelevancy
from ragas.metrics.base import ModeMetric

from utils import metric_threshold, read_test_data

# RAGAS 0.2.15 evaluate() emits ModeMetric scores as "{name}(mode={mode})".
ANSWER_RELEVANCY_RESULT_KEY = "answer_relevancy"
FACTUAL_CORRECTNESS_RESULT_KEY = "factual_correctness(mode=f1)"


def _first_score(results, metric_name):
    values = results[metric_name]
    if not values:
        raise AssertionError(f"No scores returned for metric '{metric_name}'")
    score = values[0]
    if score is None or (isinstance(score, float) and math.isnan(score)):
        raise AssertionError(f"Metric '{metric_name}' returned a missing/NaN score")
    return float(score)


@pytest.mark.parametrize(
    "get_test_data", [read_test_data("rag_test_data_faithfulness.json")], indirect=True
)
def test_resp_relevancy_and_factual_correctness(llm_wrapper, embeddings_wrapper, get_test_data):
    metrics = [
        ResponseRelevancy(llm=llm_wrapper, embeddings=embeddings_wrapper),
        FactualCorrectness(llm=llm_wrapper),
    ]
    eval_dataset = EvaluationDataset([get_test_data])
    results = evaluate(dataset=eval_dataset, metrics=metrics)

    relevancy_metric, factual_metric = metrics
    relevancy_key = (
        f"{relevancy_metric.name}(mode={relevancy_metric.mode})"
        if isinstance(relevancy_metric, ModeMetric)
        else relevancy_metric.name
    )
    factual_key = (
        f"{factual_metric.name}(mode={factual_metric.mode})"
        if isinstance(factual_metric, ModeMetric)
        else factual_metric.name
    )
    print("evaluate_result_keys", list(getattr(results, "_scores_dict", {}).keys()))
    print("relevancy_key", relevancy_key)
    print("factual_key", factual_key)
    if relevancy_key != ANSWER_RELEVANCY_RESULT_KEY or factual_key != FACTUAL_CORRECTNESS_RESULT_KEY:
        raise AssertionError(
            f"Unexpected RAGAS result keys: relevancy={relevancy_key!r}, factual={factual_key!r}"
        )

    relevancy_score = _first_score(results, relevancy_key)
    print("answer_relevancy", relevancy_score)
    factual_score = _first_score(results, factual_key)
    print("factual_correctness", factual_score)
    context_count = len(get_test_data.retrieved_contexts or [])
    print("retrieved_context_count", context_count)

    # Experimental POC thresholds. Not production-validated.
    assert relevancy_score > metric_threshold("answer_relevancy")
    assert factual_score > metric_threshold("factual_correctness")
