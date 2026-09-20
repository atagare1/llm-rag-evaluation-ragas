"""Focused tests for P2-02 trace events.

Events remain untyped dicts with a `type` discriminator.
Does not import RAGAS, LangChain, HTTP helpers, or Phase 1 mapping.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_qe_eval.domain.events import EVENT_TYPE_KEY, make_trace_event
from ai_qe_eval.domain.trace import EvaluationTrace

_EVENTS_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "domain" / "events.py"
)


def _minimal_trace(**kwargs) -> EvaluationTrace:
    defaults = {
        "trace_id": "trace-events",
        "scenario_type": "agent",
        "input": "goal",
        "output": "done",
        "expected": "done",
    }
    defaults.update(kwargs)
    return EvaluationTrace(**defaults)


def test_trace_can_contain_zero_events_as_absent_or_empty_list():
    absent = _minimal_trace()
    assert absent.events is None

    empty = _minimal_trace(events=[])
    assert empty.events == []


def test_trace_can_contain_multiple_events():
    events = [
        {"type": "llm_call", "model": "example-model"},
        {"type": "tool_call", "tool": "example_tool", "arguments": {"x": 1}},
        {"type": "tool_result", "tool": "example_tool", "output": {"ok": True}},
        {"type": "observation", "text": "note"},
    ]
    trace = _minimal_trace(events=events)
    assert len(trace.events) == 4
    assert trace.events is events


def test_each_event_preserves_type_discriminator():
    events = [
        {"type": "llm_call", "model": "example-model"},
        make_trace_event("tool_call", tool="example_tool", arguments={"x": 1}),
    ]
    trace = _minimal_trace(events=events)
    assert [event[EVENT_TYPE_KEY] for event in trace.events] == ["llm_call", "tool_call"]
    assert trace.events[0]["model"] == "example-model"


def test_arbitrary_event_specific_fields_are_preserved():
    event = {
        "type": "tool_call",
        "tool": "example_tool",
        "arguments": {"x": 1, "nested": {"y": [2, 3]}},
        "vendor_raw": {"id": "call-1"},
    }
    trace = _minimal_trace(events=[event])
    stored = trace.events[0]
    assert stored is event
    assert stored["tool"] == "example_tool"
    assert stored["arguments"] == {"x": 1, "nested": {"y": [2, 3]}}
    assert stored["vendor_raw"] == {"id": "call-1"}


def test_event_ordering_is_preserved():
    events = [
        {"type": "llm_call", "step": 1},
        {"type": "tool_call", "step": 2},
        {"type": "tool_result", "step": 3},
    ]
    trace = _minimal_trace(events=events)
    assert [event["type"] for event in trace.events] == [
        "llm_call",
        "tool_call",
        "tool_result",
    ]
    assert [event["step"] for event in trace.events] == [1, 2, 3]


def test_serialization_round_trip_preserves_events():
    events = [
        {"type": "llm_call", "model": "example-model"},
        {"type": "tool_call", "tool": "example_tool", "arguments": {"x": 1}},
        {"type": "tool_result", "output": "ok"},
    ]
    trace = _minimal_trace(events=events)
    restored = EvaluationTrace.from_dict(trace.to_dict())
    assert restored.events == events
    assert [event["type"] for event in restored.events] == [
        "llm_call",
        "tool_call",
        "tool_result",
    ]


def test_rag_trace_does_not_require_events_or_duplicate_retrieval():
    retrieval = [{"page_content": "23 articles"}]
    trace = EvaluationTrace(
        trace_id="trace-rag-events",
        scenario_type="rag",
        input="How many articles?",
        output="There are 23 articles.",
        expected="23",
        retrieval=retrieval,
    )
    assert trace.events is None
    assert trace.retrieval == retrieval
    assert trace.expected == "23"


def test_make_trace_event_does_not_interpret_payload():
    payload = {"tool": "search", "arguments": {"q": "x"}, "raw": {"vendor": "none"}}
    event = make_trace_event("tool_call", **payload)
    assert event["type"] == "tool_call"
    assert event["arguments"] == {"q": "x"}
    assert event["raw"] == {"vendor": "none"}
    assert event["raw"] is payload["raw"]


def test_events_module_does_not_import_vendor_or_transport_packages():
    tree = ast.parse(_EVENTS_SOURCE.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    forbidden = {
        "ragas",
        "deepeval",
        "langchain",
        "langchain_openai",
        "requests",
        "httpx",
        "pytest",
        "mcp",
        "langfuse",
        "langgraph",
        "opentelemetry",
    }
    assert forbidden.isdisjoint(imported_roots)
