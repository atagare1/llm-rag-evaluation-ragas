"""Focused unit tests for EvaluationTrace (P2-01).

Does not import Phase 1 mapping, RAGAS, LangChain, or HTTP helpers.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from ai_qe_eval.domain.trace import EvaluationTrace

_TRACE_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "domain" / "trace.py"
)


def test_minimal_trace_can_be_constructed():
    trace = EvaluationTrace(
        trace_id="trace-1",
        scenario_type="llm",
        input="What is RAG?",
        output="Retrieval-augmented generation.",
        expected="Retrieval-augmented generation.",
    )
    assert trace.trace_id == "trace-1"
    assert trace.scenario_type == "llm"
    assert trace.input == "What is RAG?"
    assert trace.output == "Retrieval-augmented generation."
    assert trace.expected == "Retrieval-augmented generation."


def test_rag_trace_preserves_question_answer_reference_retrieval_and_raw():
    raw_response = {
        "answer": "There are **23 articles** in the Selenium WebDriver Python course.",
        "retrieved_docs": [
            {"file_name": "course.docx", "page_content": "23 articles"},
            {"file_name": "outline.docx", "page_content": "Selenium WebDriver Python"},
        ],
    }
    trace = EvaluationTrace(
        trace_id="trace-rag-1",
        scenario_type="rag",
        input="How many articles are there in the Selenium webdriver python course?",
        output=raw_response["answer"],
        expected="23",
        retrieval=raw_response["retrieved_docs"],
        raw=raw_response,
    )
    assert trace.input.startswith("How many articles")
    assert trace.output == raw_response["answer"]
    assert trace.expected == "23"
    assert trace.retrieval == raw_response["retrieved_docs"]
    assert trace.raw is raw_response
    assert trace.raw["retrieved_docs"][0]["page_content"] == "23 articles"
    assert trace.events is None
    assert trace.turns is None


def test_optional_fields_may_be_absent():
    trace = EvaluationTrace(
        trace_id="trace-2",
        scenario_type="llm",
        input="q",
        output="a",
        expected="a",
    )
    assert trace.application_id is None
    assert trace.retrieval is None
    assert trace.turns is None
    assert trace.events is None


def test_events_preserve_untyped_dicts_with_type_discriminator():
    events = [
        {"type": "llm", "model": "example-model"},
        {"type": "tool_call", "name": "search"},
    ]
    trace = EvaluationTrace(
        trace_id="trace-3",
        scenario_type="agent",
        input="goal",
        output="done",
        expected="done",
        events=events,
    )
    assert trace.events == events
    assert trace.events[0]["type"] == "llm"
    assert trace.retrieval is None


def test_raw_payload_is_preserved_without_transformation():
    raw = {"answer": "ok", "retrieved_docs": [{"page_content": "ctx"}], "extra": {"n": 1}}
    trace = EvaluationTrace(
        trace_id="trace-4",
        scenario_type="rag",
        input="q",
        output="ok",
        expected="ok",
        raw=raw,
    )
    assert trace.raw == raw
    assert trace.raw is raw
    assert "extra" in trace.raw


def test_dict_and_json_round_trip_preserve_semantic_fields():
    trace = EvaluationTrace(
        trace_id="trace-5",
        scenario_type="rag",
        application_id="demo-rag",
        input="question",
        output="answer",
        expected="23",
        retrieval=[{"page_content": "chunk"}],
        turns=None,
        events=[{"type": "note", "text": "x"}],
        raw={"answer": "answer"},
    )
    as_dict = trace.to_dict()
    restored = EvaluationTrace.from_dict(as_dict)
    assert restored == trace

    json_payload = json.dumps(as_dict)
    restored_json = EvaluationTrace.from_dict(json.loads(json_payload))
    assert restored_json.trace_id == trace.trace_id
    assert restored_json.input == trace.input
    assert restored_json.output == trace.output
    assert restored_json.expected == "23"
    assert restored_json.retrieval == [{"page_content": "chunk"}]
    assert restored_json.raw == {"answer": "answer"}
    assert restored_json.events == [{"type": "note", "text": "x"}]


def test_trace_module_does_not_import_vendor_or_transport_packages():
    tree = ast.parse(_TRACE_SOURCE.read_text(encoding="utf-8"))
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
    }
    assert forbidden.isdisjoint(imported_roots)
