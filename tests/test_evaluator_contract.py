"""Focused tests for P2-05 Evaluator contract.

Uses in-process fakes only. No RAGAS, DeepEval, or Phase 1 adapters.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_qe_eval.domain.evaluator import Evaluator
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.trace import EvaluationTrace

_EVALUATOR_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "domain" / "evaluator.py"
)


def _trace() -> EvaluationTrace:
    return EvaluationTrace(
        trace_id="trace-001",
        scenario_type="rag",
        input="question-1",
        output="answer-1",
        expected="expected-1",
    )


class FakeEvaluator:
    def __init__(self) -> None:
        self.received_trace = None
        self.received_configuration = None

    def evaluate(self, trace, configuration=None):
        self.received_trace = trace
        self.received_configuration = configuration
        return [
            EvaluationResult(metric="fake_metric", evaluator="fake", score=1.0),
        ]


class MultiMetricFakeEvaluator:
    def evaluate(self, trace, configuration=None):
        return [
            EvaluationResult(metric="context_precision", evaluator="fake", score=0.95),
            EvaluationResult(metric="faithfulness", evaluator="fake", score=0.91),
        ]


class DeterministicStyleEvaluator:
    def evaluate(self, trace, configuration=None):
        score = 1.0 if trace.output == trace.expected else 0.0
        return [
            EvaluationResult(metric="exact_match", evaluator="deterministic", score=score),
        ]


class SemanticStyleEvaluator:
    def evaluate(self, trace, configuration=None):
        return [
            EvaluationResult(metric="correctness", evaluator="semantic", score=0.91),
        ]


def test_fake_evaluator_conforms_to_evaluator_protocol():
    evaluator = FakeEvaluator()
    assert isinstance(evaluator, Evaluator)


def test_evaluator_receives_the_actual_evaluation_trace():
    evaluator = FakeEvaluator()
    trace = _trace()
    evaluator.evaluate(trace)
    assert evaluator.received_trace is trace
    assert evaluator.received_trace.trace_id == "trace-001"
    assert evaluator.received_trace.input == "question-1"


def test_configuration_is_passed_through_unchanged():
    evaluator = FakeEvaluator()
    configuration = {"mode": "strict", "custom_value": 123}
    evaluator.evaluate(_trace(), configuration)
    assert evaluator.received_configuration is configuration
    assert evaluator.received_configuration["mode"] == "strict"
    assert evaluator.received_configuration["custom_value"] == 123


def test_evaluate_returns_a_list_of_evaluation_results():
    results = FakeEvaluator().evaluate(_trace())
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], EvaluationResult)
    assert results[0].metric == "fake_metric"
    assert results[0].evaluator == "fake"
    assert results[0].score == 1.0


def test_evaluator_can_return_multiple_evaluation_results():
    results = MultiMetricFakeEvaluator().evaluate(_trace())
    assert isinstance(MultiMetricFakeEvaluator(), Evaluator)
    assert [item.metric for item in results] == ["context_precision", "faithfulness"]
    assert [item.score for item in results] == [0.95, 0.91]


def test_deterministic_style_evaluator_conforms_to_the_same_contract():
    evaluator = DeterministicStyleEvaluator()
    assert isinstance(evaluator, Evaluator)
    results = evaluator.evaluate(_trace())
    assert results[0].metric == "exact_match"
    assert results[0].evaluator == "deterministic"
    assert results[0].score == 0.0


def test_semantic_style_evaluator_conforms_to_the_same_contract():
    evaluator = SemanticStyleEvaluator()
    assert isinstance(evaluator, Evaluator)
    results = evaluator.evaluate(_trace())
    assert results[0].metric == "correctness"
    assert results[0].evaluator == "semantic"
    assert results[0].score == 0.91


def test_contract_does_not_require_threshold_passed_or_severity():
    results = FakeEvaluator().evaluate(_trace())
    fields = results[0].__dataclass_fields__
    assert "threshold" not in fields
    assert "passed" not in fields
    assert "severity" not in fields


def test_evaluate_may_be_called_without_configuration():
    evaluator = FakeEvaluator()
    evaluator.evaluate(_trace())
    assert evaluator.received_configuration is None


def test_evaluator_module_does_not_import_vendor_or_transport_packages():
    tree = ast.parse(_EVALUATOR_SOURCE.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    forbidden = {
        "ragas",
        "deepeval",
        "langchain",
        "langchain_openai",
        "openai",
        "requests",
        "httpx",
        "pytest",
        "mcp",
        "langfuse",
        "langgraph",
        "opentelemetry",
        "abc",
    }
    assert forbidden.isdisjoint(imported_roots)
