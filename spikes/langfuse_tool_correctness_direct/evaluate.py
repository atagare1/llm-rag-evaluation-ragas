"""Spike: ToolCorrectness from settled Langfuse TOOL rows, no EvaluationTrace.

Minimum evaluator input is two ordered lists of tool calls (observed vs
independently supplied expected). This module does not import EvaluationTrace,
EvaluationRunner, QualityPolicy, QualityGate, or DeepEvalToolCorrectnessEvaluator.

Not a production adapter.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TOOL_A = "fetch_timezone_clock"
TOOL_B = "fetch_public_uuid"
TOOL_C = "fetch_httpbin_json"
ORDER_ABC = (TOOL_A, TOOL_B, TOOL_C)
ORDER_ACB = (TOOL_A, TOOL_C, TOOL_B)

# DeepEval LLMTestCase still requires an input string. This is not a scored
# EvaluationTrace field.
_DEEPEVAL_INPUT_PLACEHOLDER = (
    "spike-local ToolCorrectness input placeholder; not EvaluationTrace.input"
)


@dataclass(frozen=True)
class ToolCallInput:
    """Evaluator-specific tool-call evidence. Not an EvaluationTrace."""

    name: str
    arguments: dict[str, Any] | None = None


def names_only(names: Sequence[str]) -> list[ToolCallInput]:
    return [ToolCallInput(name=name) for name in names]


def _parse_io(value: Any) -> Any:
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


def _start_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    start = row.get("start_time")
    if start is None:
        start = row.get("startTime")
    if hasattr(start, "isoformat"):
        start_key = start.isoformat()
    else:
        start_key = str(start or "")
    return (start_key, str(row.get("id") or ""))


def _arguments_from_input(value: Any) -> dict[str, Any] | None:
    parsed = _parse_io(value)
    if parsed is None:
        return None
    if isinstance(parsed, dict):
        nested = parsed.get("arguments")
        return nested if isinstance(nested, dict) else parsed
    return None


def observed_tool_calls_from_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    trace_id: str,
) -> list[ToolCallInput]:
    """Keep TOOL rows for one trace_id, ordered by start_time then id."""
    tool_rows: list[Mapping[str, Any]] = []
    for row in observations:
        if not isinstance(row, Mapping):
            raise TypeError(
                "observations items must be mappings, "
                f"got {type(row).__name__}"
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
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Langfuse TOOL observation is missing a string name")
        tool_rows.append(row)
    tool_rows.sort(key=_start_sort_key)
    return [
        ToolCallInput(
            name=str(row.get("name")),
            arguments=_arguments_from_input(row.get("input")),
        )
        for row in tool_rows
    ]


def score_tool_correctness(
    *,
    observed: Sequence[ToolCallInput],
    expected: Sequence[ToolCallInput],
    include_arguments: bool = False,
) -> tuple[float, str | None]:
    """Call DeepEval ToolCorrectnessMetric.measure on evaluator-specific input."""
    from deepeval.metrics.tool_correctness.tool_correctness import ToolCorrectnessMetric
    from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

    evaluation_params = (
        [ToolCallParams.INPUT_PARAMETERS] if include_arguments else []
    )
    metric = ToolCorrectnessMetric(
        should_exact_match=True,
        available_tools=None,
        evaluation_params=evaluation_params,
        include_reason=True,
        async_mode=False,
        model=None,
    )

    def _to_tool_call(item: ToolCallInput) -> Any:
        return ToolCall(
            name=item.name,
            input_parameters=item.arguments,
            output=None,
        )

    metric.measure(
        LLMTestCase(
            input=_DEEPEVAL_INPUT_PLACEHOLDER,
            tools_called=[_to_tool_call(item) for item in observed],
            expected_tools=[_to_tool_call(item) for item in expected],
        )
    )
    score = metric.score
    if score is None:
        raise RuntimeError("ToolCorrectnessMetric.measure returned no score")
    return float(score), getattr(metric, "reason", None)
