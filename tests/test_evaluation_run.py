"""Focused tests for P2-03 EvaluationRun.

Does not import RAGAS, LangChain, HTTP helpers, or Phase 1 mapping.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_qe_eval.domain.run import EvaluationRun
from ai_qe_eval.domain.trace import EvaluationTrace

_RUN_SOURCE = Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "domain" / "run.py"


def _trace(
    trace_id: str,
    *,
    input_value: str = "question",
    output: str = "answer",
    expected: str = "expected",
    events=None,
    **kwargs,
) -> EvaluationTrace:
    return EvaluationTrace(
        trace_id=trace_id,
        scenario_type="rag",
        input=input_value,
        output=output,
        expected=expected,
        events=events,
        **kwargs,
    )


def test_evaluation_run_can_be_created_with_run_id():
    run = EvaluationRun(run_id="run-001")
    assert run.run_id == "run-001"


def test_evaluation_run_can_contain_zero_traces():
    run = EvaluationRun(run_id="run-001", traces=[])
    assert run.traces == []
    defaulted = EvaluationRun(run_id="run-002")
    assert defaulted.traces == []


def test_evaluation_run_can_contain_one_trace():
    trace = _trace("trace-001", input_value="question-1", output="answer-1", expected="expected-1")
    run = EvaluationRun(run_id="run-001", traces=[trace])
    assert len(run.traces) == 1
    assert run.traces[0] is trace
    assert run.traces[0].trace_id == "trace-001"


def test_evaluation_run_can_contain_multiple_traces_in_order():
    trace1 = _trace("trace-001", input_value="question-1", output="answer-1", expected="expected-1")
    trace2 = _trace("trace-002", input_value="question-2", output="answer-2", expected="expected-2")
    run = EvaluationRun(run_id="run-001", traces=[trace1, trace2])
    assert run.run_id == "run-001"
    assert len(run.traces) == 2
    assert run.traces[0].trace_id == "trace-001"
    assert run.traces[1].trace_id == "trace-002"
    assert [t.trace_id for t in run.traces] == ["trace-001", "trace-002"]


def test_grouped_traces_retain_identity_and_values():
    events = [{"type": "llm_call", "model": "example-model"}]
    trace = _trace(
        "trace-001",
        input_value="question-1",
        output="answer-1",
        expected="expected-1",
        retrieval=[{"page_content": "chunk"}],
        events=events,
        raw={"answer": "answer-1"},
    )
    run = EvaluationRun(run_id="run-001", traces=[trace])
    grouped = run.traces[0]
    assert grouped is trace
    assert grouped.input == "question-1"
    assert grouped.output == "answer-1"
    assert grouped.expected == "expected-1"
    assert grouped.retrieval == [{"page_content": "chunk"}]
    assert grouped.events == events
    assert grouped.raw == {"answer": "answer-1"}


def test_serialization_preserves_nested_traces_and_events():
    traces = [
        _trace(
            "trace-001",
            input_value="question-1",
            output="answer-1",
            expected="expected-1",
            events=[{"type": "llm_call", "model": "example-model"}],
        ),
        _trace(
            "trace-002",
            input_value="question-2",
            output="answer-2",
            expected="expected-2",
            events=[
                {"type": "tool_call", "tool": "example_tool", "arguments": {"x": 1}},
                {"type": "tool_result", "output": "ok"},
            ],
        ),
    ]
    run = EvaluationRun(run_id="run-001", traces=traces)
    restored = EvaluationRun.from_dict(run.to_dict())
    assert restored.run_id == "run-001"
    assert [t.trace_id for t in restored.traces] == ["trace-001", "trace-002"]
    assert restored.traces[0].input == "question-1"
    assert restored.traces[0].output == "answer-1"
    assert restored.traces[0].expected == "expected-1"
    assert restored.traces[1].input == "question-2"
    assert restored.traces[0].events == [{"type": "llm_call", "model": "example-model"}]
    assert [event["type"] for event in restored.traces[1].events] == [
        "tool_call",
        "tool_result",
    ]
    assert restored.traces[1].events[0]["arguments"] == {"x": 1}


def test_default_trace_lists_are_not_shared_across_runs():
    run_a = EvaluationRun(run_id="run-a")
    run_b = EvaluationRun(run_id="run-b")
    run_a.traces.append(_trace("trace-001"))
    assert [t.trace_id for t in run_a.traces] == ["trace-001"]
    assert run_b.traces == []
    assert run_a.traces is not run_b.traces


def test_run_does_not_duplicate_trace_semantic_fields():
    payload = EvaluationRun(run_id="run-001", traces=[_trace("trace-001")]).to_dict()
    assert "input" not in payload
    assert "output" not in payload
    assert "expected" not in payload
    assert "retrieval" not in payload
    assert "events" not in payload
    assert payload["traces"][0]["input"] == "question"


def test_run_module_does_not_import_vendor_or_transport_packages():
    tree = ast.parse(_RUN_SOURCE.read_text(encoding="utf-8"))
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
        "uuid",
    }
    assert forbidden.isdisjoint(imported_roots)
