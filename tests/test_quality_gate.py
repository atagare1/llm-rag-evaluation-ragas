"""Focused tests for P2-13 Quality Gate.

Multiple PolicyDecision objects → one GateDecision.
No live evaluators, RAGAS, DeepEval, or QualityPolicy invocation.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from ai_qe_eval.gate.quality_gate import GateDecision, QualityGate
from ai_qe_eval.policy.quality_policy import PolicyDecision

_GATE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "gate"
    / "quality_gate.py"
)


def _decision(
    metric: str,
    *,
    passed: bool,
    score: float = 1.0,
    operator: str = ">=",
    threshold: float = 0.80,
) -> PolicyDecision:
    return PolicyDecision(
        metric=metric,
        score=score,
        operator=operator,
        threshold=threshold,
        passed=passed,
    )


def test_gate_construction():
    gate = QualityGate()
    assert isinstance(gate, QualityGate)


def test_empty_decisions_pass():
    decision = QualityGate().evaluate([])
    assert decision.passed is True
    assert decision.decisions == []
    assert "empty" in decision.reason.lower() or "no policy" in decision.reason.lower()


def test_one_passing_decision_passes():
    decision = QualityGate().evaluate([_decision("faithfulness", passed=True)])
    assert decision.passed is True
    assert len(decision.decisions) == 1


def test_one_failing_decision_fails():
    decision = QualityGate().evaluate([_decision("faithfulness", passed=False)])
    assert decision.passed is False
    assert len(decision.decisions) == 1


def test_multiple_passing_decisions_pass():
    decision = QualityGate().evaluate(
        [
            _decision("faithfulness", passed=True),
            _decision("correctness", passed=True),
            _decision("exact_match", passed=True),
        ]
    )
    assert decision.passed is True


def test_mixed_decisions_fail():
    decision = QualityGate().evaluate(
        [
            _decision("faithfulness", passed=True),
            _decision("correctness", passed=True),
            _decision("exact_match", passed=False),
        ]
    )
    assert decision.passed is False


def test_failure_followed_by_passes_fails():
    decision = QualityGate().evaluate(
        [
            _decision("security", passed=False),
            _decision("faithfulness", passed=True),
            _decision("correctness", passed=True),
        ]
    )
    assert decision.passed is False


def test_multiple_failures_fail():
    decision = QualityGate().evaluate(
        [
            _decision("faithfulness", passed=False),
            _decision("correctness", passed=False),
        ]
    )
    assert decision.passed is False


def test_decision_order_and_count_preserved():
    inputs = [
        _decision("faithfulness", passed=True),
        _decision("correctness", passed=False),
        _decision("exact_match", passed=True),
    ]
    decision = QualityGate().evaluate(inputs)
    assert [item.metric for item in decision.decisions] == [
        "faithfulness",
        "correctness",
        "exact_match",
    ]
    assert len(decision.decisions) == 3
    assert decision.decisions[0] is inputs[0]
    assert decision.decisions[1] is inputs[1]
    assert decision.decisions[2] is inputs[2]


def test_duplicates_are_not_deduplicated():
    inputs = [
        _decision("faithfulness", passed=True, score=0.95),
        _decision("faithfulness", passed=False, score=0.10),
    ]
    decision = QualityGate().evaluate(inputs)
    assert len(decision.decisions) == 2
    assert decision.passed is False
    assert [item.passed for item in decision.decisions] == [True, False]


def test_original_policy_decisions_are_not_mutated():
    original = _decision("faithfulness", passed=True, score=0.91, threshold=0.80)
    snapshot = copy.deepcopy(original)
    gate_decision = QualityGate().evaluate([original])
    assert original == snapshot
    original.passed = False
    original.score = 0.0
    assert gate_decision.decisions[0] is original


def test_input_list_is_not_aliased():
    inputs = [_decision("faithfulness", passed=True)]
    gate_decision = QualityGate().evaluate(inputs)
    inputs.append(_decision("correctness", passed=False))
    assert len(gate_decision.decisions) == 1
    assert gate_decision.passed is True


def test_gate_has_no_score_aggregation_fields():
    decision = QualityGate().evaluate(
        [
            _decision("faithfulness", passed=True, score=0.95),
            _decision("correctness", passed=False, score=0.10),
        ]
    )
    for name in (
        "average_score",
        "weighted_score",
        "overall_score",
        "composite_score",
        "minimum_score",
        "maximum_score",
    ):
        assert not hasattr(decision, name)


def test_gate_does_not_re_evaluate_thresholds():
    inconsistent = PolicyDecision(
        metric="faithfulness",
        score=0.95,
        operator=">=",
        threshold=0.80,
        passed=False,
    )
    decision = QualityGate().evaluate([inconsistent])
    assert decision.passed is False
    assert decision.decisions[0].score == 0.95
    assert decision.decisions[0].threshold == 0.80
    assert decision.decisions[0].passed is False


def test_invalid_input_is_rejected():
    gate = QualityGate()
    with pytest.raises(TypeError):
        gate.evaluate(None)
    with pytest.raises(TypeError):
        gate.evaluate({"passed": True})
    with pytest.raises(TypeError):
        gate.evaluate([_decision("faithfulness", passed=True), "not-a-decision"])


def test_gate_is_deterministic():
    inputs = [
        _decision("faithfulness", passed=True, score=0.95),
        _decision("correctness", passed=False, score=0.70),
    ]
    first = QualityGate().evaluate(inputs)
    second = QualityGate().evaluate(inputs)
    assert first.passed == second.passed
    assert first.reason == second.reason
    assert [item.metric for item in first.decisions] == [
        item.metric for item in second.decisions
    ]
    assert [item.passed for item in first.decisions] == [
        item.passed for item in second.decisions
    ]


def test_gate_decision_contains_supplied_decisions():
    inputs = [
        _decision("faithfulness", passed=True),
        _decision("correctness", passed=False),
    ]
    decision = QualityGate().evaluate(inputs)
    assert isinstance(decision, GateDecision)
    assert decision.decisions == inputs


def test_serialization_round_trip():
    original = QualityGate().evaluate(
        [
            PolicyDecision(
                metric="faithfulness",
                score=0.72,
                operator=">=",
                threshold=0.80,
                passed=False,
            )
        ]
    )
    restored = GateDecision.from_dict(original.to_dict())
    assert restored.passed is False
    assert restored.reason == original.reason
    assert len(restored.decisions) == 1
    assert restored.decisions[0].metric == "faithfulness"
    assert restored.decisions[0].score == 0.72
    assert restored.decisions[0].passed is False


def test_synthetic_all_pass_run_allows_release():
    decision = QualityGate().evaluate(
        [
            _decision("faithfulness", passed=True, score=0.95, threshold=0.80),
            _decision("correctness", passed=True, score=0.88, threshold=0.80),
            _decision("exact_match", passed=True, score=1.0, threshold=1.0),
        ]
    )
    assert decision.passed is True


def test_synthetic_mixed_run_blocks_release():
    decision = QualityGate().evaluate(
        [
            _decision("faithfulness", passed=True, score=0.95, threshold=0.80),
            _decision("correctness", passed=False, score=0.70, threshold=0.80),
            _decision("exact_match", passed=True, score=1.0, threshold=1.0),
        ]
    )
    assert decision.passed is False


def test_gate_module_has_no_vendor_or_evaluator_imports():
    tree = ast.parse(_GATE_SOURCE.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
            imported_modules.add(node.module)
    forbidden = {"ragas", "deepeval", "langchain", "openai", "pytest"}
    assert forbidden.isdisjoint(imported_roots)
    assert "ai_qe_eval.evaluators" not in imported_modules
    assert "ai_qe_eval.normalization" not in imported_modules
    assert "ai_qe_eval.domain" not in imported_modules
