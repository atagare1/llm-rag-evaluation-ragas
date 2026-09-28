"""Deterministic MCP observation capture.

Converts plain observed tool data into ToolInvocation values and builds an
EvaluationTrace for an MCP scenario. Does not execute MCP transports, evaluate
metrics, or apply quality gates.

Does not import the MCP SDK or DeepEval.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ai_qe_eval.domain.conversation import ConversationTurn, ToolInvocation
from ai_qe_eval.domain.trace import EvaluationTrace

MCP_SCENARIO_TYPE = "mcp"


def tool_invocation_from_observation(
    *,
    name: str,
    arguments: dict[str, Any] | None = None,
    result: Any | None = None,
) -> ToolInvocation:
    """Map one observed tool call into a domain ToolInvocation.

    Values are preserved exactly. None stays None. {} stays {}.
    """
    return ToolInvocation(name=name, arguments=arguments, result=result)


def capture_executed_tool(
    *,
    name: str,
    arguments: dict[str, Any] | None = None,
    execute: Callable[[], Any],
) -> ToolInvocation:
    """Execute a tool call and capture the result.

    If execute raises, the exception propagates. No ToolInvocation is returned
    and the failure is not rewritten as an empty tool-call list.
    """
    if not callable(execute):
        raise TypeError(
            "capture_executed_tool requires a callable execute, "
            f"got {type(execute).__name__}"
        )
    result = execute()
    return tool_invocation_from_observation(
        name=name,
        arguments=arguments,
        result=result,
    )


def _as_tool_invocations(
    calls: Sequence[ToolInvocation],
    *,
    field_name: str,
) -> list[ToolInvocation]:
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        raise TypeError(
            f"{field_name} must be a sequence of ToolInvocation, "
            f"got {type(calls).__name__}"
        )
    materialized = list(calls)
    for index, call in enumerate(materialized):
        if not isinstance(call, ToolInvocation):
            raise TypeError(
                f"{field_name} items must be ToolInvocation, "
                f"got {type(call).__name__} at index {index}"
            )
    return materialized


def build_mcp_evaluation_trace(
    *,
    trace_id: str,
    input: Any,
    output: Any,
    expected: Any,
    observed_tool_calls: Sequence[ToolInvocation],
    expected_tool_calls: Sequence[ToolInvocation],
    application_id: str | None = None,
) -> EvaluationTrace:
    """Build an EvaluationTrace for an MCP tool scenario.

    Actual calls come only from observed_tool_calls. Expected calls come only
    from expected_tool_calls. Neither is derived from expected or events.
    """
    actual = _as_tool_invocations(
        observed_tool_calls,
        field_name="observed_tool_calls",
    )
    expected_calls = _as_tool_invocations(
        expected_tool_calls,
        field_name="expected_tool_calls",
    )
    return EvaluationTrace(
        trace_id=trace_id,
        scenario_type=MCP_SCENARIO_TYPE,
        input=input,
        output=output,
        expected=expected,
        application_id=application_id,
        turns=[
            ConversationTurn(role="user", content=str(input)),
            ConversationTurn(
                role="assistant",
                content=str(output),
                tool_calls=list(actual),
            ),
        ],
        expected_tool_calls=list(expected_calls),
    )
