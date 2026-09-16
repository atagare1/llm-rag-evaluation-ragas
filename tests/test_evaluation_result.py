"""Focused tests for P2-04 EvaluationResult.

Does not import RAGAS, DeepEval, LangChain, HTTP helpers, or Phase 1 mapping.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_qe_eval.domain.result import EvaluationResult

_RESULT_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "domain" / "result.py"
)


def test_evaluation_result_can_be_created_with_metric_evaluator_and_score():
    result = EvaluationResult(metric="faithfulness", evaluator="ragas", score=0.95)
    assert result.metric == "faithfulness"
    assert result.evaluator == "ragas"
    assert result.score == 0.95


def test_score_is_preserved_without_coercion():
    float_result = EvaluationResult(metric="faithfulness", evaluator="ragas", score=0.92)
    int_result = EvaluationResult(metric="schema_valid", evaluator="deterministic", score=1)
    assert float_result.score == 0.92
    assert isinstance(float_result.score, float)
    assert int_result.score == 1
    assert isinstance(int_result.score, int)
    assert int_result.score is not True


def test_metric_and_evaluator_identity_are_preserved():
    ragas_result = EvaluationResult(metric="faithfulness", evaluator="ragas", score=0.95)
    deepeval_result = EvaluationResult(metric="correctness", evaluator="deepeval", score=0.91)
    deterministic_result = EvaluationResult(
        metric="schema_valid", evaluator="deterministic", score=1.0
    )
    assert ragas_result.metric == "faithfulness"
    assert ragas_result.evaluator == "ragas"
    assert deepeval_result.metric == "correctness"
    assert deepeval_result.evaluator == "deepeval"
    assert deterministic_result.metric == "schema_valid"
    assert deterministic_result.evaluator == "deterministic"


def test_optional_reason_is_preserved():
    with_reason = EvaluationResult(
        metric="faithfulness",
        evaluator="ragas",
        score=0.95,
        reason="Claims are supported by retrieved context.",
    )
    without_reason = EvaluationResult(metric="faithfulness", evaluator="ragas", score=0.95)
    assert with_reason.reason == "Claims are supported by retrieved context."
    assert without_reason.reason is None


def test_optional_raw_result_is_preserved_without_interpretation():
    raw = {"factual_correctness(mode=f1)": 0.0, "extra": {"n": 1}}
    result = EvaluationResult(
        metric="factual_correctness",
        evaluator="ragas",
        score=0.0,
        raw_result=raw,
    )
    assert result.raw_result is raw
    assert result.raw_result["factual_correctness(mode=f1)"] == 0.0
    assert result.raw_result["extra"] == {"n": 1}


def test_result_does_not_calculate_pass_fail_or_require_threshold():
    result = EvaluationResult(metric="faithfulness", evaluator="ragas", score=0.82)
    assert not hasattr(result, "threshold") or "threshold" not in result.__dataclass_fields__
    assert "threshold" not in result.__dataclass_fields__
    assert "passed" not in result.__dataclass_fields__
    assert "pass_fail" not in result.__dataclass_fields__
    assert "severity" not in result.__dataclass_fields__
    payload = result.to_dict()
    assert "threshold" not in payload
    assert "passed" not in payload
    assert "severity" not in payload


def test_serialization_round_trip_preserves_result():
    raw = {"answer_relevancy": 0.929, "factual_correctness(mode=f1)": 0.0}
    result = EvaluationResult(
        metric="answer_relevancy",
        evaluator="ragas",
        score=0.929,
        reason="Relevant to the question.",
        raw_result=raw,
    )
    restored = EvaluationResult.from_dict(result.to_dict())
    assert restored.metric == "answer_relevancy"
    assert restored.evaluator == "ragas"
    assert restored.score == 0.929
    assert restored.reason == "Relevant to the question."
    assert restored.raw_result == raw


def test_result_instances_do_not_share_mutable_state():
    first = EvaluationResult(metric="faithfulness", evaluator="ragas", score=0.95)
    second = EvaluationResult(metric="correctness", evaluator="deepeval", score=0.91)
    first.raw_result = {"vendor": "a"}
    first.reason = "first"
    assert second.raw_result is None
    assert second.reason is None
    assert first.raw_result is not second.raw_result


def test_result_module_does_not_import_vendor_or_transport_packages():
    tree = ast.parse(_RESULT_SOURCE.read_text(encoding="utf-8"))
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
    }
    assert forbidden.isdisjoint(imported_roots)
