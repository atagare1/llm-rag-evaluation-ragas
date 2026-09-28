"""DeepEval Turn Relevancy adapter.

Maps EvaluationTrace.turns into a DeepEval ConversationalTestCase and
TurnRelevancyMetric.measure, then returns one EvaluationResult.

Turn relevancy is the DeepEval mechanism; evaluator identity is "deepeval".
Does not apply DeepEval's threshold or metric.success. Does not auto-register.
Does not read retrieval, expected, events, or tool calls.
"""

from __future__ import annotations

from typing import Any

from ai_qe_eval.domain.conversation import ConversationTurn
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.trace import EvaluationTrace

TURN_RELEVANCY_METRIC = "turn_relevancy"
DEEPEVAL_EVALUATOR_NAME = "deepeval"


def _conversation_turns(trace: EvaluationTrace) -> list[ConversationTurn]:
    turns = trace.turns
    if not turns:
        raise ValueError(
            "DeepEvalTurnRelevancyEvaluator requires EvaluationTrace.turns "
            "to be a non-empty conversation"
        )
    for index, turn in enumerate(turns):
        if not isinstance(turn, ConversationTurn):
            raise TypeError(
                "DeepEvalTurnRelevancyEvaluator requires ConversationTurn items, "
                f"got {type(turn).__name__} at index {index}"
            )
    if not any(turn.role == "user" for turn in turns):
        raise ValueError(
            "DeepEvalTurnRelevancyEvaluator requires at least one user turn"
        )
    if turns[-1].role != "assistant":
        raise ValueError(
            "DeepEvalTurnRelevancyEvaluator requires the conversation to end "
            "with an assistant turn"
        )
    return turns


def _trace_to_conversational_test_case(trace: EvaluationTrace) -> Any:
    from deepeval.test_case import ConversationalTestCase, Turn

    turns = _conversation_turns(trace)
    return ConversationalTestCase(
        turns=[Turn(role=turn.role, content=turn.content) for turn in turns]
    )


def _default_turn_relevancy_metric(*, model: Any | None) -> Any:
    from deepeval.metrics.turn_relevancy.turn_relevancy import TurnRelevancyMetric

    return TurnRelevancyMetric(model=model, include_reason=True)


class DeepEvalTurnRelevancyEvaluator:
    """DeepEval Turn Relevancy adapter.

    Returns metric="turn_relevancy" and evaluator="deepeval".
    Does not apply DeepEval's threshold or metric.success.
    """

    def __init__(
        self,
        *,
        turn_relevancy_metric: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self._turn_relevancy_metric = turn_relevancy_metric
        self._model = model

    def _metric(self) -> Any:
        if self._turn_relevancy_metric is not None:
            return self._turn_relevancy_metric
        return _default_turn_relevancy_metric(model=self._model)

    def evaluate(
        self,
        trace: EvaluationTrace,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        test_case = _trace_to_conversational_test_case(trace)
        metric = self._metric()
        metric.measure(test_case)
        score = metric.score
        reason = getattr(metric, "reason", None)
        return [
            EvaluationResult(
                metric=TURN_RELEVANCY_METRIC,
                evaluator=DEEPEVAL_EVALUATOR_NAME,
                score=score,
                reason=reason,
                raw_result={
                    "score": score,
                    "reason": reason,
                    "turns": [
                        {"role": turn.role, "content": turn.content}
                        for turn in test_case.turns
                    ],
                },
            )
        ]
