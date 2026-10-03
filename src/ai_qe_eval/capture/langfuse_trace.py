"""Langfuse TOOL observation capture.

Converts retrieved Langfuse observation dicts into ToolInvocation values and
Design A request maps. Does not call the Langfuse API, run an agent, or
evaluate metrics.

Does not import the Langfuse SDK, OpenAI Agents SDK, or DeepEval.

Observed tool calls come only from TOOL observations. Expected tool calls are
supplied by the caller and are never derived from telemetry.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ai_qe_eval.domain.conversation import ToolInvocation


def parse_observation_io(value: Any) -> Any:
    """Parse Observations API v2 I/O strings. Non-JSON values are unchanged."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "{[":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _row_type(row: Mapping[str, Any]) -> str:
    return str(row.get("type") or "").upper()


def _row_trace_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("trace_id")
    if value is None:
        value = row.get("traceId")
    return str(value) if value is not None else None


def _row_start_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    start = row.get("start_time")
    if start is None:
        start = row.get("startTime")
    if hasattr(start, "isoformat"):
        start_key = start.isoformat()
    else:
        start_key = str(start or "")
    return (start_key, str(row.get("id") or ""))


def _status_message(row: Mapping[str, Any]) -> Any:
    if row.get("status_message") is not None:
        return row.get("status_message")
    return row.get("statusMessage")


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


def tool_invocation_from_langfuse_observation(
    row: Mapping[str, Any],
) -> ToolInvocation:
    """Map one Langfuse TOOL observation into a domain ToolInvocation.

    Retries stay as separate rows. Errors are preserved on result when Langfuse
    reports level=ERROR or a status_message; ToolInvocation has no error field.
    """
    if not isinstance(row, Mapping):
        raise TypeError(
            "Langfuse observation must be a mapping, "
            f"got {type(row).__name__}"
        )
    name = row.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Langfuse TOOL observation is missing a string name")
    parsed_input = parse_observation_io(row.get("input"))
    arguments: dict[str, Any] | None
    if parsed_input is None:
        arguments = None
    elif isinstance(parsed_input, dict):
        nested = parsed_input.get("arguments")
        arguments = nested if isinstance(nested, dict) else parsed_input
    else:
        arguments = None
    parsed_output = parse_observation_io(row.get("output"))
    level = row.get("level")
    status = _status_message(row)
    if str(level or "").upper() == "ERROR" or status:
        result: Any = {
            "output": parsed_output,
            "level": level,
            "status_message": status,
        }
    else:
        result = parsed_output
    return ToolInvocation(name=name, arguments=arguments, result=result)


def langfuse_tool_correctness_request(
    *,
    observations: Sequence[Mapping[str, Any]],
    trace_id: str,
    expected_tool_calls: Sequence[ToolInvocation],
) -> dict[str, dict[str, list[ToolInvocation]]]:
    """Build a Design A ToolCorrectness request from Langfuse observations.

    Observed calls come only from TOOL observations. expected_tool_calls is
    a QE specification input and is never derived from telemetry.
    """
    observed = observed_tool_invocations(observations, trace_id=trace_id)
    expected = _as_tool_invocations(
        expected_tool_calls,
        field_name="expected_tool_calls",
    )
    return {
        "tool_correctness": {
            "args": [observed, expected],
        }
    }


def observed_tool_invocations(
    observations: Sequence[Mapping[str, Any]],
    *,
    trace_id: str,
) -> list[ToolInvocation]:
    """Return TOOL observations for one trace, ordered by start_time then id."""
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise TypeError(
            "observations must be a sequence of mappings, "
            f"got {type(observations).__name__}"
        )
    tool_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(observations):
        if not isinstance(row, Mapping):
            raise TypeError(
                "observations items must be mappings, "
                f"got {type(row).__name__} at index {index}"
            )
        row_trace = _row_trace_id(row)
        if row_trace not in (None, trace_id):
            raise ValueError(
                f"observation {row.get('id')!r} has trace_id {row_trace!r}, "
                f"expected {trace_id!r}"
            )
        if _row_type(row) != "TOOL":
            continue
        if row_trace is None:
            raise ValueError(
                f"TOOL observation {row.get('id')!r} is missing trace_id"
            )
        tool_rows.append(row)
    tool_rows.sort(key=_row_start_sort_key)
    return [tool_invocation_from_langfuse_observation(row) for row in tool_rows]
