"""Focused tests for P2-08 DeterministicEvaluator exact_match.

No RAGAS, DeepEval, network, or registry auto-registration.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_qe_eval.domain.evaluator import Evaluator
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.deterministic import (
    DETERMINISTIC_EVALUATOR_NAME,
    EXACT_MATCH_FAILED_REASON,
    EXACT_MATCH_METRIC,
    EXACT_MATCH_SUCCEEDED_REASON,
    DeterministicEvaluator,
)

_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "evaluators"
    / "deterministic.py"
)


def _trace(output, expected) -> EvaluationTrace:
    return EvaluationTrace(
        trace_id="trace-exact-match",
        scenario_type="llm",
        input="question",
        output=output,
        expected=expected,
    )


def _evaluate(output, expected, configuration=None) -> EvaluationResult:
    results = DeterministicEvaluator().evaluate(
        _trace(output, expected),
        configuration,
    )
    assert len(results) == 1
    return results[0]


def test_exact_match_succeeds_for_identical_strings():
    result = _evaluate("23", "23")
    assert result.score == 1.0
    assert result.reason == EXACT_MATCH_SUCCEEDED_REASON


def test_exact_match_fails_for_different_strings():
    result = _evaluate("23", "24")
    assert result.score == 0.0
    assert result.reason == EXACT_MATCH_FAILED_REASON


def test_exact_match_is_case_sensitive():
    result = _evaluate("Hello", "hello")
    assert result.score == 0.0


def test_exact_match_treats_whitespace_as_significant():
    result = _evaluate("hello", " hello")
    assert result.score == 0.0


def test_exact_match_treats_type_differences_as_significant():
    result = _evaluate("23", 23)
    assert result.score == 0.0


def test_none_equals_none_is_an_exact_match():
    result = _evaluate(None, None)
    assert result.score == 1.0


def test_result_contract_for_canonical_trace_flow():
    trace = _trace("23", "23")
    results = DeterministicEvaluator().evaluate(trace)
    assert isinstance(DeterministicEvaluator(), Evaluator)
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EvaluationResult)
    assert result.metric == EXACT_MATCH_METRIC
    assert result.evaluator == DETERMINISTIC_EVALUATOR_NAME
    assert result.score == 1.0
    assert isinstance(result.score, float)


def test_raw_result_preserves_comparison_without_policy_fields():
    result = _evaluate("23", "24")
    assert result.raw_result == {
        "output": "23",
        "expected": "24",
        "matched": False,
    }
    assert "threshold" not in result.raw_result
    assert "passed" not in result.raw_result
    assert "severity" not in result.raw_result
    assert "quality_policy" not in result.raw_result
    assert "threshold" not in result.__dataclass_fields__
    assert "passed" not in result.__dataclass_fields__


def test_configuration_is_optional_and_ignored():
    evaluator = DeterministicEvaluator()
    trace = _trace("ok", "ok")
    without_config = evaluator.evaluate(trace)
    with_none = evaluator.evaluate(trace, None)
    with_ignored = evaluator.evaluate(trace, {"case_sensitive": False})
    assert without_config == with_none == with_ignored
    assert without_config[0].score == 1.0


def test_same_trace_produces_identical_results():
    evaluator = DeterministicEvaluator()
    trace = _trace("23", "24")
    first = evaluator.evaluate(trace)
    second = evaluator.evaluate(trace)
    assert first == second


def test_evaluator_does_not_auto_register_or_import_vendors():
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    forbidden = {
        "ragas",
        "deepeval",
        "openai",
        "anthropic",
        "langchain",
        "langchain_openai",
        "requests",
        "httpx",
        "pytest",
        "mcp",
        "langfuse",
        "langgraph",
        "os",
    }
    assert forbidden.isdisjoint(imported_roots)
    assert "ai_qe_eval.domain.registry" not in {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported_roots <= {"ai_qe_eval", "typing", "annotations", "__future__"}
