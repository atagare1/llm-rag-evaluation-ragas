"""Focused tests for P2-07 EvaluationConfig.

Selects capability identifiers only. No registry lookup, evaluators, or policy.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai_qe_eval.domain.config import EvaluationConfig

_CONFIG_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "domain" / "config.py"
)


def test_empty_configuration_is_valid():
    config = EvaluationConfig(evaluations=[])
    assert config.evaluations == []
    defaulted = EvaluationConfig()
    assert defaulted.evaluations == []


def test_one_selected_evaluation():
    config = EvaluationConfig(evaluations=["faithfulness"])
    assert config.evaluations == ["faithfulness"]


def test_multiple_selected_evaluations_preserve_caller_order():
    config = EvaluationConfig(
        evaluations=["faithfulness", "context_precision", "answer_relevancy"]
    )
    assert config.evaluations == [
        "faithfulness",
        "context_precision",
        "answer_relevancy",
    ]


def test_duplicate_evaluations_are_rejected():
    with pytest.raises(ValueError, match="faithfulness"):
        EvaluationConfig(evaluations=["faithfulness", "faithfulness"])


def test_constructor_copies_the_caller_list():
    names = ["faithfulness", "context_precision"]
    config = EvaluationConfig(evaluations=names)
    names.append("answer_relevancy")
    assert config.evaluations == ["faithfulness", "context_precision"]


def test_default_lists_are_not_shared_across_instances():
    first = EvaluationConfig()
    second = EvaluationConfig()
    first.evaluations.append("faithfulness")
    assert first.evaluations == ["faithfulness"]
    assert second.evaluations == []
    assert first.evaluations is not second.evaluations


def test_serialization_round_trip_preserves_evaluations():
    config = EvaluationConfig(
        evaluations=["faithfulness", "context_precision", "answer_relevancy"]
    )
    payload = config.to_dict()
    assert payload == {
        "evaluations": ["faithfulness", "context_precision", "answer_relevancy"]
    }
    assert "threshold" not in payload
    restored = EvaluationConfig.from_dict(payload)
    assert restored.evaluations == config.evaluations
    payload["evaluations"].append("geval")
    assert restored.evaluations == [
        "faithfulness",
        "context_precision",
        "answer_relevancy",
    ]


def test_from_dict_treats_missing_evaluations_as_empty():
    assert EvaluationConfig.from_dict({}).evaluations == []


def test_config_has_no_policy_or_evaluator_execution_fields():
    field_names = set(EvaluationConfig.__dataclass_fields__)
    assert field_names == {"evaluations"}
    forbidden = {
        "threshold",
        "score",
        "passed",
        "severity",
        "quality_policy",
        "model",
        "embedding_model",
        "judge_model",
        "prompt",
        "temperature",
        "max_tokens",
    }
    assert forbidden.isdisjoint(field_names)


def test_config_does_not_require_or_import_the_registry():
    source = _CONFIG_SOURCE.read_text(encoding="utf-8")
    assert "EvaluationRegistry" not in source
    assert "EvaluationCapability" not in source
    EvaluationConfig(evaluations=["unknown_capability"])


def test_config_module_does_not_import_vendor_packages():
    tree = ast.parse(_CONFIG_SOURCE.read_text(encoding="utf-8"))
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
    }
    assert forbidden.isdisjoint(imported_roots)
