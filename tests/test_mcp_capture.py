"""Focused tests for deterministic MCP observation capture.

Does not import the MCP SDK, DeepEval, Runner, Policy, or Gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai_qe_eval.capture.mcp_trace import (
    MCP_SCENARIO_TYPE,
    build_mcp_evaluation_trace,
    capture_executed_tool,
    tool_invocation_from_observation,
)
from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.domain.trace import EvaluationTrace

_CAPTURE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "capture"
    / "mcp_trace.py"
)


def test_single_capture_preserves_name_arguments_and_result():
    call = tool_invocation_from_observation(
        name="weather",
        arguments={"location": "Pune"},
        result="Temperature is 28 C and conditions are clear.",
    )
    assert isinstance(call, ToolInvocation)
    assert call.name == "weather"
    assert call.arguments == {"location": "Pune"}
    assert call.result == "Temperature is 28 C and conditions are clear."


def test_multiple_captures_preserve_order():
    first = tool_invocation_from_observation(
        name="lookup_city",
        arguments={"q": "Pune"},
        result="in",
    )
    second = tool_invocation_from_observation(
        name="weather",
        arguments={"location": "Pune"},
        result="28 C",
    )
    observed = [first, second]
    assert [call.name for call in observed] == ["lookup_city", "weather"]
    assert observed[0].arguments == {"q": "Pune"}
    assert observed[1].result == "28 C"


def test_none_arguments_remain_none_and_empty_dict_remains_empty():
    none_args = tool_invocation_from_observation(
        name="weather",
        arguments=None,
        result=None,
    )
    empty_args = tool_invocation_from_observation(
        name="weather",
        arguments={},
        result=None,
    )
    assert none_args.arguments is None
    assert empty_args.arguments == {}
    assert none_args.arguments is not empty_args.arguments
    assert none_args != empty_args


def test_result_is_preserved_exactly():
    nested = {"status": "ok", "rows": [{"id": 1}]}
    call = tool_invocation_from_observation(
        name="query",
        arguments={"sql": "select 1"},
        result=nested,
    )
    assert call.result is nested
    assert call.result["rows"][0]["id"] == 1


def test_capture_executed_tool_propagates_failure_without_fabricating_calls():
    def boom():
        raise RuntimeError("tool transport failed")

    with pytest.raises(RuntimeError, match="tool transport failed"):
        capture_executed_tool(
            name="weather",
            arguments={"location": "Pune"},
            execute=boom,
        )


def test_capture_executed_tool_records_successful_result():
    call = capture_executed_tool(
        name="weather",
        arguments={"location": "Pune"},
        execute=lambda: "Temperature is 28 C and conditions are clear.",
    )
    assert call.name == "weather"
    assert call.arguments == {"location": "Pune"}
    assert call.result == "Temperature is 28 C and conditions are clear."


def test_build_mcp_evaluation_trace_constructs_expected_shape():
    actual = tool_invocation_from_observation(
        name="weather",
        arguments={"location": "Pune"},
        result="Temperature is 28 C and conditions are clear.",
    )
    expected_call = tool_invocation_from_observation(
        name="weather",
        arguments={"location": "Pune"},
        result="Temperature is 28 C and conditions are clear.",
    )
    trace = build_mcp_evaluation_trace(
        trace_id="trace-mcp-capture-1",
        input="Find the weather for Pune.",
        output="Temperature is 28 C and conditions are clear.",
        expected="Temperature is 28 C and conditions are clear.",
        observed_tool_calls=[actual],
        expected_tool_calls=[expected_call],
    )
    assert isinstance(trace, EvaluationTrace)
    assert trace.scenario_type == MCP_SCENARIO_TYPE
    assert trace.scenario_type == "mcp"
    assert trace.input == "Find the weather for Pune."
    assert trace.output == "Temperature is 28 C and conditions are clear."
    assert trace.expected == "Temperature is 28 C and conditions are clear."
    assert len(trace.turns) == 2
    assert trace.turns[0].role == "user"
    assert trace.turns[0].content == "Find the weather for Pune."
    assert trace.turns[0].tool_calls is None
    assert trace.turns[1].role == "assistant"
    assert trace.turns[1].content == "Temperature is 28 C and conditions are clear."
    assert trace.turns[1].tool_calls == [actual]
    assert trace.expected_tool_calls == [expected_call]
    assert trace.events is None


def test_actual_and_expected_tool_calls_remain_separate():
    actual = tool_invocation_from_observation(
        name="weather",
        arguments={"location": "Pune"},
        result="28 C",
    )
    expected_call = tool_invocation_from_observation(
        name="calendar",
        arguments={"location": "Pune"},
        result="none",
    )
    trace = build_mcp_evaluation_trace(
        trace_id="trace-mcp-separate",
        input="Find the weather for Pune.",
        output="28 C",
        expected="SHOULD_NOT_BECOME_A_TOOL_CALL",
        observed_tool_calls=[actual],
        expected_tool_calls=[expected_call],
    )
    assert [call.name for call in trace.turns[1].tool_calls] == ["weather"]
    assert [call.name for call in trace.expected_tool_calls] == ["calendar"]
    assert trace.expected == "SHOULD_NOT_BECOME_A_TOOL_CALL"
    assert "calendar" not in [call.name for call in trace.turns[1].tool_calls]
    assert "weather" not in [call.name for call in trace.expected_tool_calls]


def test_multiple_observed_calls_preserve_order_on_the_trace():
    observed = [
        tool_invocation_from_observation(name="a", arguments={"n": 1}, result=1),
        tool_invocation_from_observation(name="b", arguments={"n": 2}, result=2),
    ]
    expected_calls = [
        tool_invocation_from_observation(name="a", arguments={"n": 1}, result=1),
        tool_invocation_from_observation(name="b", arguments={"n": 2}, result=2),
    ]
    trace = build_mcp_evaluation_trace(
        trace_id="trace-mcp-order",
        input="goal",
        output="done",
        expected="done",
        observed_tool_calls=observed,
        expected_tool_calls=expected_calls,
    )
    assert [call.name for call in trace.turns[1].tool_calls] == ["a", "b"]
    assert [call.name for call in trace.expected_tool_calls] == ["a", "b"]


def test_capture_module_does_not_import_mcp_or_deepeval():
    tree = ast.parse(_CAPTURE_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert "mcp" not in imported
    assert "deepeval" not in imported
