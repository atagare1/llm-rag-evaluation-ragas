"""Domain tests for conversation turns and tool invocations.

Does not import DeepEval, MCP, RAGAS, or live providers.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from ai_qe_eval.domain.conversation import ConversationTurn, ToolInvocation
from ai_qe_eval.domain.events import make_trace_event
from ai_qe_eval.domain.trace import EvaluationTrace

_DOMAIN_DIR = Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "domain"
_CONVERSATION_SOURCE = _DOMAIN_DIR / "conversation.py"
_TRACE_SOURCE = _DOMAIN_DIR / "trace.py"


def _rag_trace(**kwargs) -> EvaluationTrace:
    defaults = {
        "trace_id": "trace-rag-1",
        "scenario_type": "rag",
        "input": "How many articles are there in the Selenium webdriver python course?",
        "output": "There are **23 articles** in the Selenium WebDriver Python course.",
        "expected": "23",
        "retrieval": [{"file_name": "course.docx", "page_content": "23 articles"}],
    }
    defaults.update(kwargs)
    return EvaluationTrace(**defaults)


def test_conversation_turn_construction():
    turn = ConversationTurn(role="user", content="What is your return window?")
    assert turn.role == "user"
    assert turn.content == "What is your return window?"
    assert turn.retrieval is None
    assert turn.tool_calls is None

    assistant = ConversationTurn(
        role="assistant",
        content="Returns are accepted within 30 days.",
        retrieval=["30 day return policy"],
    )
    assert assistant.role == "assistant"
    assert assistant.retrieval == ["30 day return policy"]


def test_tool_invocation_construction():
    call = ToolInvocation(name="get_order")
    assert call.name == "get_order"
    assert call.arguments is None
    assert call.result is None

    detailed = ToolInvocation(
        name="get_order",
        arguments={"order_id": "A100"},
        result={"status": "shipped"},
    )
    assert detailed.arguments == {"order_id": "A100"}
    assert detailed.result == {"status": "shipped"}


def test_evaluation_trace_with_turns():
    turns = [
        ConversationTurn(role="user", content="What is your return window?"),
        ConversationTurn(role="assistant", content="30 days from delivery."),
    ]
    trace = EvaluationTrace(
        trace_id="trace-chat-1",
        scenario_type="chat",
        input="What is your return window?",
        output="30 days from delivery.",
        expected="Returns are accepted within 30 days.",
        turns=turns,
    )
    assert trace.turns == turns
    assert [turn.role for turn in trace.turns] == ["user", "assistant"]
    assert trace.chatbot_role is None
    assert trace.expected_outcome is None
    assert trace.expected_tool_calls is None


def test_evaluation_trace_with_chatbot_role():
    trace = EvaluationTrace(
        trace_id="trace-chat-role",
        scenario_type="chat",
        input="Hello",
        output="Hello, I can help with returns.",
        expected="Hello, I can help with returns.",
        chatbot_role="retail returns agent",
    )
    assert trace.chatbot_role == "retail returns agent"
    assert trace.turns is None


def test_evaluation_trace_with_expected_outcome():
    trace = EvaluationTrace(
        trace_id="trace-chat-outcome",
        scenario_type="chat",
        input="What is the return window?",
        output="30 days.",
        expected="30 days.",
        expected_outcome="The user learns the 30-day return window.",
    )
    assert trace.expected_outcome == "The user learns the 30-day return window."


def test_evaluation_trace_with_expected_tool_calls():
    expected = [
        ToolInvocation(
            name="get_order",
            arguments={"order_id": "A100"},
            result={"status": "shipped"},
        )
    ]
    trace = EvaluationTrace(
        trace_id="trace-agent-1",
        scenario_type="agent",
        input="What is the status of order A100?",
        output="Order A100 has shipped.",
        expected="Order A100 has shipped.",
        expected_tool_calls=expected,
    )
    assert trace.expected_tool_calls == expected
    assert trace.expected_tool_calls[0].name == "get_order"


def test_nested_tool_invocation_inside_conversation_turn():
    call = ToolInvocation(
        name="get_order",
        arguments={"order_id": "A100"},
        result={"status": "shipped"},
    )
    turn = ConversationTurn(
        role="assistant",
        content="Order A100 has shipped.",
        tool_calls=[call],
    )
    trace = EvaluationTrace(
        trace_id="trace-mcp-1",
        scenario_type="agent",
        input="What is the status of order A100?",
        output="Order A100 has shipped.",
        expected="Order A100 has shipped.",
        turns=[
            ConversationTurn(role="user", content="What is the status of order A100?"),
            turn,
        ],
    )
    stored = trace.turns[1].tool_calls[0]
    assert stored is call
    assert stored.arguments == {"order_id": "A100"}
    assert stored.result == {"status": "shipped"}


def test_to_dict_serializes_turns_and_tool_calls():
    trace = EvaluationTrace(
        trace_id="trace-serial",
        scenario_type="agent",
        input="goal",
        output="done",
        expected="done",
        chatbot_role="order agent",
        expected_outcome="Order status is reported.",
        turns=[
            ConversationTurn(
                role="assistant",
                content="Order A100 has shipped.",
                retrieval=["order record"],
                tool_calls=[
                    ToolInvocation(
                        name="get_order",
                        arguments={"order_id": "A100"},
                        result={"status": "shipped"},
                    )
                ],
            )
        ],
        expected_tool_calls=[
            ToolInvocation(name="get_order", arguments={"order_id": "A100"})
        ],
    )
    payload = trace.to_dict()
    assert payload["chatbot_role"] == "order agent"
    assert payload["expected_outcome"] == "Order status is reported."
    assert payload["turns"][0]["role"] == "assistant"
    assert payload["turns"][0]["retrieval"] == ["order record"]
    assert payload["turns"][0]["tool_calls"][0] == {
        "name": "get_order",
        "arguments": {"order_id": "A100"},
        "result": {"status": "shipped"},
    }
    assert payload["expected_tool_calls"][0]["name"] == "get_order"
    assert payload["expected_tool_calls"][0]["result"] is None


def test_from_dict_rebuilds_turns_and_tool_calls():
    payload = {
        "trace_id": "trace-rebuild",
        "scenario_type": "chat",
        "input": "question",
        "output": "answer",
        "expected": "answer",
        "turns": [
            {
                "role": "user",
                "content": "What is the return window?",
            },
            {
                "role": "assistant",
                "content": "30 days.",
                "retrieval": ["policy"],
                "tool_calls": [
                    {
                        "name": "lookup_policy",
                        "arguments": {"topic": "returns"},
                        "result": "30 days",
                    }
                ],
            },
        ],
        "expected_tool_calls": [
            {"name": "lookup_policy", "arguments": {"topic": "returns"}}
        ],
        "chatbot_role": "returns agent",
        "expected_outcome": "The return window is stated.",
    }
    trace = EvaluationTrace.from_dict(payload)
    assert isinstance(trace.turns[0], ConversationTurn)
    assert trace.turns[0].tool_calls is None
    assert isinstance(trace.turns[1].tool_calls[0], ToolInvocation)
    assert trace.turns[1].tool_calls[0].result == "30 days"
    assert trace.turns[1].retrieval == ["policy"]
    assert isinstance(trace.expected_tool_calls[0], ToolInvocation)
    assert trace.expected_tool_calls[0].arguments == {"topic": "returns"}
    assert trace.expected_tool_calls[0].result is None
    assert trace.chatbot_role == "returns agent"
    assert trace.expected_outcome == "The return window is stated."


def test_round_trip_preserves_conversation_trace():
    trace = EvaluationTrace(
        trace_id="trace-round",
        scenario_type="agent",
        application_id="demo",
        input="What is the status of order A100?",
        output="Order A100 has shipped.",
        expected="Order A100 has shipped.",
        retrieval=None,
        turns=[
            ConversationTurn(role="user", content="What is the status of order A100?"),
            ConversationTurn(
                role="assistant",
                content="Order A100 has shipped.",
                tool_calls=[
                    ToolInvocation(
                        name="get_order",
                        arguments={"order_id": "A100"},
                        result={"status": "shipped"},
                    )
                ],
            ),
        ],
        events=[{"type": "note", "text": "audit"}],
        raw={"vendor": "none"},
        chatbot_role="order agent",
        expected_outcome="The order status is reported.",
        expected_tool_calls=[
            ToolInvocation(
                name="get_order",
                arguments={"order_id": "A100"},
                result={"status": "shipped"},
            )
        ],
    )
    restored = EvaluationTrace.from_dict(trace.to_dict())
    assert restored == trace

    json_restored = EvaluationTrace.from_dict(json.loads(json.dumps(trace.to_dict())))
    assert json_restored == trace


def test_old_style_trace_without_new_fields_remains_valid():
    payload = {
        "trace_id": "trace-old",
        "scenario_type": "llm",
        "input": "q",
        "output": "a",
        "expected": "a",
    }
    trace = EvaluationTrace.from_dict(payload)
    assert trace.turns is None
    assert trace.events is None
    assert trace.chatbot_role is None
    assert trace.expected_outcome is None
    assert trace.expected_tool_calls is None
    assert trace.retrieval is None
    assert trace.raw is None


def test_turns_none_remains_valid():
    trace = EvaluationTrace(
        trace_id="trace-none",
        scenario_type="llm",
        input="q",
        output="a",
        expected="a",
        turns=None,
    )
    assert trace.turns is None
    restored = EvaluationTrace.from_dict(trace.to_dict())
    assert restored.turns is None


def test_turns_empty_list_remains_valid():
    trace = EvaluationTrace(
        trace_id="trace-empty",
        scenario_type="chat",
        input="q",
        output="a",
        expected="a",
        turns=[],
        expected_tool_calls=[],
    )
    assert trace.turns == []
    assert trace.expected_tool_calls == []
    restored = EvaluationTrace.from_dict(trace.to_dict())
    assert restored.turns == []
    assert restored.expected_tool_calls == []


def test_existing_events_continue_to_work_unchanged():
    events = [
        make_trace_event("tool_call", tool="example_tool", arguments={"x": 1}),
        {"type": "tool_result", "tool": "example_tool", "output": {"ok": True}},
    ]
    trace = EvaluationTrace(
        trace_id="trace-events",
        scenario_type="agent",
        input="goal",
        output="done",
        expected="done",
        events=events,
    )
    assert trace.events is events
    assert trace.events[0]["type"] == "tool_call"
    assert trace.events[0]["arguments"] == {"x": 1}
    restored = EvaluationTrace.from_dict(trace.to_dict())
    assert restored.events == events
    assert restored.turns is None


def test_rag_trace_without_turns_continues_to_work():
    trace = _rag_trace()
    assert trace.retrieval[0]["page_content"] == "23 articles"
    assert trace.turns is None
    assert trace.events is None
    assert trace.expected_tool_calls is None
    restored = EvaluationTrace.from_dict(trace.to_dict())
    assert restored == trace
    assert restored.retrieval == trace.retrieval


def test_minimal_positional_construction_still_works():
    trace = EvaluationTrace("trace-pos", "llm", "q", "a", "a")
    assert trace.trace_id == "trace-pos"
    assert trace.input == "q"
    assert trace.output == "a"
    assert trace.expected == "a"
    assert trace.turns is None
    assert trace.chatbot_role is None


def test_conversation_and_trace_modules_do_not_import_deepeval_or_mcp():
    forbidden = {"deepeval", "mcp", "ragas", "langchain"}
    for source in (_CONVERSATION_SOURCE, _TRACE_SOURCE):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        assert forbidden.isdisjoint(imported_roots)
