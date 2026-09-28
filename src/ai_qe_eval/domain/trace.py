"""EvaluationTrace — P2-01 domain model; P2-02 event representation.

Stable semantic representation of one evaluation execution.
Does not depend on RAGAS, DeepEval, HTTP, pytest, or other transports.

Ownership:
- input / output / expected: core evaluated semantics
- retrieval: scenario-specific structured field (absent when unused)
- turns: optional ordered ConversationTurn list (absent when unused)
- chatbot_role / expected_outcome / expected_tool_calls: optional
  conversation and tool ground truth (absent when unused)
- events: optional ordered list of dicts with a `type` discriminator (no typed event classes)
- raw: original source payload for audit, not a second semantic model
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ai_qe_eval.domain.conversation import (
    ConversationTurn,
    ToolInvocation,
    conversation_turns_from_payload,
    tool_invocations_from_payload,
)
from ai_qe_eval.domain.events import TraceEvent


@dataclass
class EvaluationTrace:
    trace_id: str
    scenario_type: str
    input: Any
    output: Any
    expected: Any
    application_id: str | None = None
    retrieval: list[Any] | None = None
    turns: list[ConversationTurn] | None = None
    events: list[TraceEvent] | None = None
    raw: Any = None
    chatbot_role: str | None = None
    expected_outcome: str | None = None
    expected_tool_calls: list[ToolInvocation] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationTrace:
        return cls(
            trace_id=data["trace_id"],
            scenario_type=data["scenario_type"],
            input=data["input"],
            output=data["output"],
            expected=data["expected"],
            application_id=data.get("application_id"),
            retrieval=data.get("retrieval"),
            turns=conversation_turns_from_payload(data.get("turns")),
            events=data.get("events"),
            raw=data.get("raw"),
            chatbot_role=data.get("chatbot_role"),
            expected_outcome=data.get("expected_outcome"),
            expected_tool_calls=tool_invocations_from_payload(
                data.get("expected_tool_calls")
            ),
        )
