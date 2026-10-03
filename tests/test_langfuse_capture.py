"""Focused tests for Langfuse observation capture.

Uses recorded observation-shaped dicts. Does not call Langfuse, OpenAI, or
the Agents SDK. Does not import those packages.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ai_qe_eval.capture.langfuse_trace import (
    langfuse_tool_correctness_request,
    observed_tool_invocations,
    parse_observation_io,
    tool_invocation_from_langfuse_observation,
)
from ai_qe_eval.domain.conversation import ToolInvocation

_CAPTURE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "capture"
    / "langfuse_trace.py"
)


def _tool_row(
    *,
    obs_id: str,
    name: str,
    start: str,
    trace_id: str = "trace-1",
    input_value: object = "{}",
    output_value: object = '{"ok": true}',
    level: str = "DEFAULT",
    status_message: str | None = None,
    parent: str = "parent-1",
) -> dict:
    return {
        "id": obs_id,
        "type": "TOOL",
        "name": name,
        "trace_id": trace_id,
        "parent_observation_id": parent,
        "start_time": start,
        "input": input_value,
        "output": output_value,
        "level": level,
        "status_message": status_message,
    }


def test_parse_observation_io_loads_json_object_strings():
    assert parse_observation_io('{"iana_timezone": "UTC"}') == {"iana_timezone": "UTC"}
    assert parse_observation_io({}) == {}
    assert parse_observation_io("not-json") == "not-json"


def test_tool_invocation_parses_string_input_as_arguments_dict():
    call = tool_invocation_from_langfuse_observation(
        _tool_row(
            obs_id="1",
            name="fetch_timezone_clock",
            start="2026-10-02T15:12:10Z",
            input_value='{"iana_timezone": "UTC"}',
            output_value='{"http_status": 200}',
        )
    )
    assert call.name == "fetch_timezone_clock"
    assert call.arguments == {"iana_timezone": "UTC"}
    assert call.result == {"http_status": 200}


def test_object_input_maps_iana_timezone_argument():
    call = tool_invocation_from_langfuse_observation(
        _tool_row(
            obs_id="1b",
            name="fetch_timezone_clock",
            start="2026-10-02T15:12:10Z",
            input_value={"iana_timezone": "UTC"},
            output_value={"http_status": 200},
        )
    )
    assert call.name == "fetch_timezone_clock"
    assert call.arguments == {"iana_timezone": "UTC"}


def test_httpbin_tool_result_preserves_slideshow_title_in_body():
    call = tool_invocation_from_langfuse_observation(
        _tool_row(
            obs_id="json-1",
            name="fetch_httpbin_json",
            start="2026-10-02T15:12:16Z",
            input_value={},
            output_value={
                "http_status": 200,
                "source_url": "https://httpbin.org/json",
                "body": '{"slideshow": {"title": "Sample Slide Show"}}',
            },
        )
    )
    assert call.name == "fetch_httpbin_json"
    assert call.result["http_status"] == 200
    assert "Sample Slide Show" in call.result["body"]


def test_empty_object_arguments_remain_empty_dict():
    call = tool_invocation_from_langfuse_observation(
        _tool_row(
            obs_id="2",
            name="fetch_public_uuid",
            start="2026-10-02T15:12:13Z",
            input_value="{}",
        )
    )
    assert call.arguments == {}


def test_error_observation_preserves_status_on_result():
    call = tool_invocation_from_langfuse_observation(
        _tool_row(
            obs_id="3",
            name="fetch_github_zen",
            start="2026-10-02T15:12:15Z",
            output_value=None,
            level="ERROR",
            status_message="Client error '415 Unsupported Media Type'",
        )
    )
    assert call.name == "fetch_github_zen"
    assert isinstance(call.result, dict)
    assert call.result["level"] == "ERROR"
    assert "415" in call.result["status_message"]
    assert call.result["output"] is None


def test_retries_remain_separate_ordered_entries():
    rows = [
        _tool_row(
            obs_id="c",
            name="fetch_httpbin_json",
            start="2026-10-02T15:12:16Z",
            parent="t3",
        ),
        _tool_row(
            obs_id="a",
            name="fetch_timezone_clock",
            start="2026-10-02T15:12:10Z",
            parent="t1",
            input_value='{"iana_timezone": "UTC"}',
        ),
        _tool_row(
            obs_id="b",
            name="fetch_public_uuid",
            start="2026-10-02T15:12:13Z",
            parent="t2",
        ),
        _tool_row(
            obs_id="c2",
            name="fetch_httpbin_json",
            start="2026-10-02T15:12:19Z",
            parent="t4",
            level="ERROR",
            status_message="retry",
            output_value=None,
        ),
    ]
    calls = observed_tool_invocations(rows, trace_id="trace-1")
    assert [call.name for call in calls] == [
        "fetch_timezone_clock",
        "fetch_public_uuid",
        "fetch_httpbin_json",
        "fetch_httpbin_json",
    ]
    assert calls[3].result["status_message"] == "retry"


def test_generation_rows_are_not_treated_as_tool_calls():
    rows = [
        {
            "id": "g",
            "type": "GENERATION",
            "name": "generation",
            "trace_id": "trace-1",
            "start_time": "2026-10-02T15:12:00Z",
            "input": "[]",
            "output": '{"tool_calls": [{"function": {"name": "ignored"}}]}',
        },
        _tool_row(
            obs_id="a",
            name="fetch_timezone_clock",
            start="2026-10-02T15:12:10Z",
        ),
    ]
    calls = observed_tool_invocations(rows, trace_id="trace-1")
    assert [call.name for call in calls] == ["fetch_timezone_clock"]


def test_foreign_trace_id_is_rejected():
    rows = [
        _tool_row(obs_id="a", name="a", start="2026-10-02T15:12:10Z", trace_id="other"),
    ]
    with pytest.raises(ValueError, match="trace_id"):
        observed_tool_invocations(rows, trace_id="trace-1")


def test_tool_correctness_request_keeps_expected_independent_of_observed():
    observed_rows = [
        _tool_row(obs_id="a", name="fetch_timezone_clock", start="2026-10-02T15:12:10Z"),
        _tool_row(obs_id="b", name="fetch_public_uuid", start="2026-10-02T15:12:13Z"),
        _tool_row(obs_id="c", name="fetch_httpbin_json", start="2026-10-02T15:12:16Z"),
        {
            "id": "g",
            "type": "GENERATION",
            "name": "generation",
            "trace_id": "trace-1",
            "start_time": "2026-10-02T15:12:00Z",
        },
    ]
    expected = [
        ToolInvocation(name="fetch_httpbin_json"),
        ToolInvocation(name="fetch_timezone_clock"),
        ToolInvocation(name="fetch_public_uuid"),
    ]
    request = langfuse_tool_correctness_request(
        observations=observed_rows,
        trace_id="trace-1",
        expected_tool_calls=expected,
    )
    observed, expected_args = request["tool_correctness"]["args"]
    assert [call.name for call in observed] == [
        "fetch_timezone_clock",
        "fetch_public_uuid",
        "fetch_httpbin_json",
    ]
    assert [call.name for call in expected_args] == [
        "fetch_httpbin_json",
        "fetch_timezone_clock",
        "fetch_public_uuid",
    ]
    assert expected_args is not observed
    assert "kwargs" not in request["tool_correctness"]


def test_datetime_start_times_sort_with_strings():
    rows = [
        _tool_row(
            obs_id="b",
            name="second",
            start="2026-10-02T15:12:13.000000+00:00",
        ),
        {
            **_tool_row(obs_id="a", name="first", start="ignored"),
            "start_time": datetime(2026, 10, 2, 15, 12, 10, tzinfo=timezone.utc),
        },
    ]
    calls = observed_tool_invocations(rows, trace_id="trace-1")
    assert [call.name for call in calls] == ["first", "second"]


def test_capture_module_does_not_import_vendor_sdks():
    tree = ast.parse(_CAPTURE_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert "langfuse" not in imported
    assert "agents" not in imported
    assert "openai" not in imported
    assert "deepeval" not in imported
