"""EvaluationRun — P2-03 domain model.

Container that groups EvaluationTrace instances for one evaluation run.
Does not own trace semantics (input/output/expected/retrieval/events).
Does not emit scores, thresholds, or PASS/FAIL.

Lineage fields (config, models, git, policy versions) are deferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_qe_eval.domain.trace import EvaluationTrace


@dataclass
class EvaluationRun:
    run_id: str
    traces: list[EvaluationTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "traces": [trace.to_dict() for trace in self.traces],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRun:
        raw_traces = data.get("traces") or []
        return cls(
            run_id=data["run_id"],
            traces=[EvaluationTrace.from_dict(item) for item in raw_traces],
        )
