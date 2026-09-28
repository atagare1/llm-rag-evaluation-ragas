"""Focused tests for DeepEvalToolCorrectnessEvaluator.

Injects a ToolCorrectnessMetric stub. Does not call an external LLM.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.conversation import ConversationTurn, ToolInvocation
from ai_qe_eval.domain.evaluator import Evaluator
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.deepeval_tool_correctness import (
    DEEPEVAL_EVALUATOR_NAME,
    TOOL_CORRECTNESS_METRIC,
    DeepEvalToolCorrectnessEvaluator,
)
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

_ADAPTER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "evaluators"
    / "deepeval_tool_correctness.py"
)


class RecordingToolCorrectnessMetric:
    def __init__(self, score=1.0, reason="Exact match.") -> None:
        self.score = score
        self.reason = reason
        self.test_case = None
        self.threshold = 0.5
        self.success = True
        self.measure_calls = 0

    def measure(self, test_case):
        self.measure_calls += 1
        self.test_case = test_case
        return self.score


def _weather() -> ToolInvocation:
    return ToolInvocation(
        name="weather",
        arguments={"location": "Pune"},
        result="Temperature is 28 C and conditions are clear.",
    )


def _trace(**overrides) -> EvaluationTrace:
    values = {
        "trace_id": "trace-tool-correctness",
        "scenario_type": "agent",
        "input": "Find the weather for Pune.",
        "output": "SHOULD_NOT_BE_USED_AS_TOOLS",
        "expected": "SHOULD_NOT_BE_USED_AS_TOOLS",
        "retrieval": ["SHOULD_NOT_BE_SENT"],
        "events": [{"type": "tool_call", "name": "SHOULD_NOT_BE_SENT"}],
        "turns": [
            ConversationTurn(role="user", content="Find the weather for Pune."),
            ConversationTurn(
                role="assistant",
                content="It is 28 C and clear in Pune.",
                tool_calls=[_weather()],
            ),
        ],
        "expected_tool_calls": [_weather()],
    }
    values.update(overrides)
    return EvaluationTrace(**values)


def test_evaluator_conforms_to_protocol():
    evaluator = DeepEvalToolCorrectnessEvaluator(
        tool_correctness_metric=RecordingToolCorrectnessMetric()
    )
    assert isinstance(evaluator, Evaluator)


def test_single_tool_call_maps_name_arguments_and_result():
    metric = RecordingToolCorrectnessMetric()
    DeepEvalToolCorrectnessEvaluator(tool_correctness_metric=metric).evaluate(_trace())
    called = metric.test_case.tools_called
    expected = metric.test_case.expected_tools
    assert metric.test_case.input == "Find the weather for Pune."
    assert type(metric.test_case).__name__ == "LLMTestCase"
    assert len(called) == 1
    assert called[0].name == "weather"
    assert called[0].input_parameters == {"location": "Pune"}
    assert called[0].output == "Temperature is 28 C and conditions are clear."
    assert len(expected) == 1
    assert expected[0].name == "weather"
    assert expected[0].input_parameters == {"location": "Pune"}
    assert expected[0].output == called[0].output
    assert "SHOULD_NOT_BE_USED_AS_TOOLS" not in repr(metric.test_case)
    assert "SHOULD_NOT_BE_SENT" not in repr(metric.test_case)


def test_multiple_tool_calls_preserve_conversation_order():
    first = ToolInvocation(name="lookup_city", arguments={"q": "Pune"}, result="in")
    second = ToolInvocation(name="weather", arguments={"location": "Pune"}, result="28 C")
    metric = RecordingToolCorrectnessMetric()
    DeepEvalToolCorrectnessEvaluator(tool_correctness_metric=metric).evaluate(
        _trace(
            turns=[
                ConversationTurn(role="user", content="Find the weather for Pune."),
                ConversationTurn(role="assistant", content="Looking up the city.", tool_calls=[first]),
                ConversationTurn(role="assistant", content="It is 28 C.", tool_calls=[second]),
            ],
            expected_tool_calls=[first, second],
        )
    )
    assert [call.name for call in metric.test_case.tools_called] == ["lookup_city", "weather"]
    assert [call.name for call in metric.test_case.expected_tools] == ["lookup_city", "weather"]
    assert metric.test_case.tools_called[0].input_parameters == {"q": "Pune"}
    assert metric.test_case.tools_called[1].output == "28 C"


def test_none_arguments_are_not_rewritten_to_an_empty_dict():
    metric = RecordingToolCorrectnessMetric()
    DeepEvalToolCorrectnessEvaluator(tool_correctness_metric=metric).evaluate(
        _trace(
            turns=[
                ConversationTurn(
                    role="assistant",
                    content="Called.",
                    tool_calls=[ToolInvocation(name="weather", arguments=None, result=None)],
                )
            ],
            expected_tool_calls=[ToolInvocation(name="weather", arguments={}, result=None)],
        )
    )
    actual = metric.test_case.tools_called[0]
    expected = metric.test_case.expected_tools[0]
    assert actual.input_parameters is None
    assert expected.input_parameters == {}
    assert actual != expected


def test_result_identity_score_and_reason():
    metric = RecordingToolCorrectnessMetric(score=0.91, reason="Names and arguments match.")
    result = DeepEvalToolCorrectnessEvaluator(tool_correctness_metric=metric).evaluate(
        _trace()
    )[0]
    assert isinstance(result, EvaluationResult)
    assert result.metric == TOOL_CORRECTNESS_METRIC
    assert result.metric == "tool_correctness"
    assert result.evaluator == DEEPEVAL_EVALUATOR_NAME
    assert result.evaluator == "deepeval"
    assert result.score == 0.91
    assert result.score is metric.score
    assert result.reason == "Names and arguments match."
    assert result.reason is metric.reason
    assert result.raw_result["tools_called"][0]["arguments"] == {"location": "Pune"}
    assert "threshold" not in result.raw_result
    assert "success" not in result.raw_result
    assert "passed" not in result.raw_result


def test_mismatched_tools_are_passed_through_without_being_rewritten():
    metric = RecordingToolCorrectnessMetric(score=0.0, reason="Not an exact match.")
    DeepEvalToolCorrectnessEvaluator(tool_correctness_metric=metric).evaluate(
        _trace(
            expected_tool_calls=[
                ToolInvocation(name="calendar", arguments={"location": "Pune"}, result="none")
            ]
        )
    )
    assert metric.test_case.tools_called[0].name == "weather"
    assert metric.test_case.expected_tools[0].name == "calendar"


def test_turns_without_tool_calls_map_to_an_empty_called_list():
    metric = RecordingToolCorrectnessMetric()
    DeepEvalToolCorrectnessEvaluator(tool_correctness_metric=metric).evaluate(
        _trace(
            turns=[
                ConversationTurn(role="user", content="Find the weather for Pune."),
                ConversationTurn(role="assistant", content="I cannot look that up.", tool_calls=None),
            ]
        )
    )
    assert metric.test_case.tools_called == []
    assert len(metric.test_case.expected_tools) == 1
    assert metric.measure_calls == 1


def test_empty_expected_tool_calls_are_not_fabricated():
    metric = RecordingToolCorrectnessMetric()
    DeepEvalToolCorrectnessEvaluator(tool_correctness_metric=metric).evaluate(
        _trace(
            turns=[ConversationTurn(role="assistant", content="No tools.", tool_calls=[])],
            expected_tool_calls=[],
        )
    )
    assert metric.test_case.tools_called == []
    assert metric.test_case.expected_tools == []
    assert metric.measure_calls == 1


def test_missing_turns_raise_before_the_metric_is_called():
    metric = RecordingToolCorrectnessMetric()
    with pytest.raises(ValueError, match="turns"):
        DeepEvalToolCorrectnessEvaluator(tool_correctness_metric=metric).evaluate(
            _trace(turns=None)
        )
    assert metric.measure_calls == 0


def test_missing_expected_tool_calls_raise_before_the_metric_is_called():
    metric = RecordingToolCorrectnessMetric()
    with pytest.raises(ValueError, match="expected_tool_calls"):
        DeepEvalToolCorrectnessEvaluator(tool_correctness_metric=metric).evaluate(
            _trace(expected_tool_calls=None)
        )
    assert metric.measure_calls == 0


def test_empty_turn_list_is_an_observed_conversation_with_no_calls():
    metric = RecordingToolCorrectnessMetric()
    DeepEvalToolCorrectnessEvaluator(tool_correctness_metric=metric).evaluate(
        _trace(turns=[], expected_tool_calls=[])
    )
    assert metric.test_case.tools_called == []
    assert metric.test_case.expected_tools == []
    assert metric.measure_calls == 1


def test_default_metric_enables_exact_match_of_arguments_and_output():
    from deepeval.metrics.tool_correctness import tool_correctness as metric_module
    from deepeval.test_case import ToolCallParams

    captured: dict = {}

    class CapturingMetric:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.score = 1.0
            self.reason = "captured"

        def measure(self, test_case):
            self.test_case = test_case

    original = metric_module.ToolCorrectnessMetric
    metric_module.ToolCorrectnessMetric = CapturingMetric
    try:
        model = object()
        DeepEvalToolCorrectnessEvaluator(model=model).evaluate(_trace())
    finally:
        metric_module.ToolCorrectnessMetric = original

    assert captured["should_exact_match"] is True
    assert captured["available_tools"] is None
    assert captured["include_reason"] is True
    assert captured["model"] is model
    assert ToolCallParams.INPUT_PARAMETERS in captured["evaluation_params"]
    assert ToolCallParams.OUTPUT in captured["evaluation_params"]


def test_adapter_does_not_import_mcp():
    tree = ast.parse(_ADAPTER_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert "mcp" not in imported


def test_runner_applies_policy_and_gate_to_tool_correctness():
    metric = RecordingToolCorrectnessMetric(score=0.91, reason="Exact match.")
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="tool_correctness",
            evaluator="deepeval",
            category="agent",
        )
    )
    policy = QualityPolicy(metric="tool_correctness", operator=">=", threshold=0.8)
    runner = EvaluationRunner(
        registry=registry,
        evaluators={
            "tool_correctness": DeepEvalToolCorrectnessEvaluator(
                tool_correctness_metric=metric
            )
        },
        policies={"tool_correctness": policy},
        gate=QualityGate(),
    )

    decision = runner.run(
        _trace(),
        EvaluationConfig(evaluations=["tool_correctness"]),
        run_id="run-tool-correctness",
    )

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.run_id == "run-tool-correctness"
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert result.score == 0.91
    assert result.reason == "Exact match."
    assert runner.last_run.decisions[0].passed is True
    assert runner.last_run.decisions[0].threshold == 0.8
    assert metric.measure_calls == 1
    assert metric.test_case.expected_tools[0].name == "weather"
