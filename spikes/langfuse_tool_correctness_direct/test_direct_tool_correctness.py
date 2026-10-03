"""Deterministic ToolCorrectness without EvaluationTrace.

Uses Langfuse-shaped observation dicts. Does not call Langfuse, OpenAI, or
the Agents SDK. Does not import EvaluationTrace, Runner, Policy, or Gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SPIKE_DIR = Path(__file__).resolve().parent
if str(_SPIKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPIKE_DIR))

from evaluate import (
    ORDER_ABC,
    ORDER_ACB,
    names_only,
    observed_tool_calls_from_observations,
    score_tool_correctness,
)

TRACE_ID = "spike-trace-abc"


def _row(
    *,
    obs_id: str,
    obs_type: str,
    name: str,
    start: str,
    input_value: object = "{}",
) -> dict:
    return {
        "id": obs_id,
        "type": obs_type,
        "name": name,
        "trace_id": TRACE_ID,
        "start_time": start,
        "input": input_value,
        "output": '{"ok": true}',
    }


SETTLED_OBSERVATIONS = [
    _row(
        obs_id="gen-1",
        obs_type="GENERATION",
        name="generation",
        start="2026-10-03T03:00:00Z",
    ),
    _row(
        obs_id="tool-c",
        obs_type="TOOL",
        name="fetch_httpbin_json",
        start="2026-10-03T03:00:06Z",
    ),
    _row(
        obs_id="tool-a",
        obs_type="TOOL",
        name="fetch_timezone_clock",
        start="2026-10-03T03:00:02Z",
        input_value={"iana_timezone": "UTC"},
    ),
    _row(
        obs_id="tool-b",
        obs_type="TOOL",
        name="fetch_public_uuid",
        start="2026-10-03T03:00:04Z",
    ),
    _row(
        obs_id="span-1",
        obs_type="SPAN",
        name="wrap",
        start="2026-10-03T03:00:01Z",
    ),
]


def test_observed_abc_vs_expected_abc_passes():
    observed = observed_tool_calls_from_observations(
        SETTLED_OBSERVATIONS, trace_id=TRACE_ID
    )
    assert [call.name for call in observed] == list(ORDER_ABC)
    score, _reason = score_tool_correctness(
        observed=observed,
        expected=names_only(ORDER_ABC),
    )
    assert score == 1.0


def test_observed_abc_vs_expected_acb_fails():
    observed = observed_tool_calls_from_observations(
        SETTLED_OBSERVATIONS, trace_id=TRACE_ID
    )
    assert [call.name for call in observed] == list(ORDER_ABC)
    score, _reason = score_tool_correctness(
        observed=observed,
        expected=names_only(ORDER_ACB),
    )
    assert score == 0.0


def test_generation_and_span_rows_are_not_tool_calls():
    observed = observed_tool_calls_from_observations(
        SETTLED_OBSERVATIONS, trace_id=TRACE_ID
    )
    assert [call.name for call in observed] == list(ORDER_ABC)
