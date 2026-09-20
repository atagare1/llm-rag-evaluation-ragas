"""Focused tests for P2-06 Evaluation Registry.

Capability metadata only. No evaluator implementations or vendor SDKs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry

_REGISTRY_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "domain" / "registry.py"
)


def _capability(
    name: str,
    evaluator: str = "ragas",
    category: str = "semantic",
) -> EvaluationCapability:
    return EvaluationCapability(name=name, evaluator=evaluator, category=category)


def test_empty_registry_contains_no_capabilities():
    registry = EvaluationRegistry()
    assert registry.list() == []
    assert registry.contains("faithfulness") is False


def test_register_and_resolve_one_capability():
    registry = EvaluationRegistry()
    capability = _capability("faithfulness", evaluator="ragas", category="semantic")
    registry.register(capability)
    resolved = registry.get("faithfulness")
    assert resolved is capability
    assert resolved.name == "faithfulness"
    assert resolved.evaluator == "ragas"
    assert resolved.category == "semantic"


def test_unknown_capability_raises_key_error():
    registry = EvaluationRegistry()
    with pytest.raises(KeyError, match="does_not_exist"):
        registry.get("does_not_exist")


def test_contains_identifies_registered_and_unregistered_capabilities():
    registry = EvaluationRegistry()
    registry.register(_capability("faithfulness"))
    assert registry.contains("faithfulness") is True
    assert registry.contains("geval") is False


def test_multiple_capabilities_from_different_categories():
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(name="exact_match", evaluator="deterministic", category="deterministic")
    )
    registry.register(
        EvaluationCapability(name="faithfulness", evaluator="ragas", category="semantic")
    )
    registry.register(
        EvaluationCapability(name="geval", evaluator="deepeval", category="semantic")
    )
    registry.register(
        EvaluationCapability(name="trajectory", evaluator="agent", category="agent")
    )
    assert registry.get("exact_match").evaluator == "deterministic"
    assert registry.get("faithfulness").evaluator == "ragas"
    assert registry.get("geval").evaluator == "deepeval"
    assert registry.get("trajectory").evaluator == "agent"
    assert [item.name for item in registry.list()] == [
        "exact_match",
        "faithfulness",
        "geval",
        "trajectory",
    ]


def test_duplicate_registration_raises_value_error():
    registry = EvaluationRegistry()
    registry.register(_capability("faithfulness"))
    with pytest.raises(ValueError, match="faithfulness"):
        registry.register(_capability("faithfulness", evaluator="deepeval"))
    assert registry.get("faithfulness").evaluator == "ragas"


def test_list_is_insertion_ordered_and_does_not_mutate_registry():
    registry = EvaluationRegistry()
    registry.register(_capability("exact_match", evaluator="deterministic", category="deterministic"))
    registry.register(_capability("faithfulness"))
    snapshot = registry.list()
    snapshot.append(_capability("injected"))
    assert [item.name for item in registry.list()] == ["exact_match", "faithfulness"]
    snapshot.clear()
    assert len(registry.list()) == 2


def test_registry_instances_do_not_share_state():
    first = EvaluationRegistry()
    second = EvaluationRegistry()
    first.register(_capability("faithfulness"))
    assert first.contains("faithfulness") is True
    assert second.contains("faithfulness") is False
    assert second.list() == []


def test_capability_descriptor_has_no_policy_semantics():
    field_names = set(EvaluationCapability.__dataclass_fields__)
    assert field_names == {"name", "evaluator", "category"}
    assert "threshold" not in field_names
    assert "passed" not in field_names
    assert "severity" not in field_names
    assert "quality_policy" not in field_names
    registry_fields = set(EvaluationRegistry.__dict__)
    assert "evaluate" not in registry_fields


def test_registry_module_does_not_import_vendor_packages():
    tree = ast.parse(_REGISTRY_SOURCE.read_text(encoding="utf-8"))
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
        "pathlib",
    }
    assert forbidden.isdisjoint(imported_roots)


def test_register_and_lookup_do_not_execute_evaluators():
    registry = EvaluationRegistry()
    registry.register(_capability("faithfulness", evaluator="ragas", category="semantic"))
    resolved = registry.get("faithfulness")
    assert not hasattr(resolved, "evaluate")
    assert callable(registry.register)
    assert callable(registry.get)
    assert not hasattr(registry, "evaluate")
    assert not hasattr(registry, "run")
