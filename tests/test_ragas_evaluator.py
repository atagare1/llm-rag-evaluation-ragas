"""Focused tests for P2-09 RAGAS Faithfulness evaluator.

Mocks the RAGAS metric boundary. Does not call Together or the RAG API.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_qe_eval.domain.evaluator import Evaluator
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.ragas import (
    FAITHFULNESS_METRIC,
    RAGAS_EVALUATOR_NAME,
    RAGASFaithfulnessEvaluator,
)


class RecordingFaithfulnessMetric:
    def __init__(self, score=0.92) -> None:
        self.score = score
        self.sample = None

    def single_turn_score(self, sample):
        self.sample = sample
        return self.score


class FailingFaithfulnessMetric:
    def single_turn_score(self, sample):
        raise RuntimeError("RAGAS faithfulness execution failed")


def _trace() -> EvaluationTrace:
    return EvaluationTrace(
        trace_id="trace-faithfulness",
        scenario_type="rag",
        input="How many articles are there?",
        output="There are 23 articles.",
        expected="23",
        retrieval=[
            {"file_name": "course.docx", "page_content": "23 articles"},
            {"file_name": "outline.docx", "page_content": "Selenium WebDriver Python"},
        ],
    )


def test_evaluator_conforms_to_protocol():
    evaluator = RAGASFaithfulnessEvaluator(
        faithfulness_metric=RecordingFaithfulnessMetric()
    )
    assert isinstance(evaluator, Evaluator)


def test_trace_maps_to_response_and_retrieved_contexts():
    metric = RecordingFaithfulnessMetric()
    evaluator = RAGASFaithfulnessEvaluator(faithfulness_metric=metric)
    trace = _trace()
    evaluator.evaluate(trace)
    assert metric.sample is not None
    assert metric.sample.user_input == trace.input
    assert metric.sample.response == trace.output
    assert metric.sample.retrieved_contexts == [
        "23 articles",
        "Selenium WebDriver Python",
    ]
    assert trace.retrieval[0]["page_content"] == "23 articles"


def test_successful_ragas_score_is_preserved_exactly():
    metric = RecordingFaithfulnessMetric(score=0.92)
    results = RAGASFaithfulnessEvaluator(faithfulness_metric=metric).evaluate(_trace())
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EvaluationResult)
    assert result.metric == FAITHFULNESS_METRIC
    assert result.evaluator == RAGAS_EVALUATOR_NAME
    assert result.score == 0.92
    assert result.score is metric.score


def test_score_is_not_rounded_clamped_or_thresholded():
    metric = RecordingFaithfulnessMetric(score=0.823456789)
    result = RAGASFaithfulnessEvaluator(faithfulness_metric=metric).evaluate(_trace())[0]
    assert result.score == 0.823456789
    assert "threshold" not in result.__dataclass_fields__
    assert "passed" not in result.__dataclass_fields__
    assert "severity" not in result.__dataclass_fields__
    assert "quality_policy" not in result.__dataclass_fields__
    assert "threshold" not in result.raw_result
    assert "passed" not in result.raw_result


def test_raw_result_retains_ragas_inputs_and_score():
    metric = RecordingFaithfulnessMetric(score=0.92)
    result = RAGASFaithfulnessEvaluator(faithfulness_metric=metric).evaluate(_trace())[0]
    assert result.raw_result["score"] == 0.92
    assert result.raw_result["response"] == "There are 23 articles."
    assert result.raw_result["retrieved_contexts"] == [
        "23 articles",
        "Selenium WebDriver Python",
    ]
    assert result.raw_result["user_input"] == "How many articles are there?"
    assert result.reason == "RAGAS faithfulness score=0.92."


def test_evaluate_with_configuration_none():
    metric = RecordingFaithfulnessMetric(score=0.5)
    results = RAGASFaithfulnessEvaluator(faithfulness_metric=metric).evaluate(
        _trace(), None
    )
    assert len(results) == 1
    assert results[0].score == 0.5


def test_ragas_errors_propagate():
    evaluator = RAGASFaithfulnessEvaluator(
        faithfulness_metric=FailingFaithfulnessMetric()
    )
    try:
        evaluator.evaluate(_trace())
    except RuntimeError as exc:
        assert "RAGAS faithfulness execution failed" in str(exc)
    else:
        raise AssertionError("expected RAGAS execution error to propagate")


def test_string_retrieval_is_passed_through():
    metric = RecordingFaithfulnessMetric()
    trace = EvaluationTrace(
        trace_id="trace-string-contexts",
        scenario_type="rag",
        input="q",
        output="a",
        expected="a",
        retrieval=["ctx-1", "ctx-2"],
    )
    RAGASFaithfulnessEvaluator(faithfulness_metric=metric).evaluate(trace)
    assert metric.sample.retrieved_contexts == ["ctx-1", "ctx-2"]


def test_domain_package_does_not_import_ragas():
    domain_dir = Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "domain"
    forbidden = {"ragas", "langchain", "langchain_openai", "openai"}
    for path in domain_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        assert forbidden.isdisjoint(imported), path.name
