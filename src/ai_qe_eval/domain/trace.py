"""EvaluationTrace — P2-01 domain model; P2-02 event representation.

Stable semantic representation of one evaluation execution.
Does not depend on RAGAS, DeepEval, HTTP, pytest, or other transports.

Ownership:
- input / output / expected: core evaluated semantics
- retrieval / turns: scenario-specific structured fields (absent when unused)
- events: optional ordered list of dicts with a `type` discriminator (no typed event classes)
- raw: original source payload for audit, not a second semantic model
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

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
    turns: list[dict[str, Any]] | None = None
    events: list[TraceEvent] | None = None
    raw: Any = None

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
            turns=data.get("turns"),
            events=data.get("events"),
            raw=data.get("raw"),
        )
