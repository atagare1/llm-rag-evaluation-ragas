"""Focused tests for DeepEvalTurnRelevancyEvaluator.

Injects a TurnRelevancyMetric stub. Does not call an external LLM.
"""

from __future__ import annotations

import pytest

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.conversation import ConversationTurn, ToolInvocation
from ai_qe_eval.domain.evaluator import Evaluator
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.evaluators.deepeval_turn_relevancy import (
    DEEPEVAL_EVALUATOR_NAME,
    TURN_RELEVANCY_METRIC,
    DeepEvalTurnRelevancyEvaluator,
)
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner


class RecordingTurnRelevancyMetric:
    def __init__(self, score=0.93, reason="Each assistant reply addresses the preceding user turn.") -> None:
        self.score = score
        self.reason = reason
        self.test_case = None
        self.threshold = 0.5
        self.success = True
        self.measure_calls = 0

    def measure(self, test_case):
        self.measure_calls += 1
        self.test_case = test_case
        return self.score


def _turns() -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user", content="What is your return window?"),
        ConversationTurn(role="assistant", content="Returns are accepted within 30 days of delivery."),
        ConversationTurn(role="user", content="Does that include sale items?"),
        ConversationTurn(role="assistant", content="Yes. Sale items use the same 30-day window."),
    ]


def _trace(**overrides) -> EvaluationTrace:
    values = {
        "trace_id": "trace-turn-relevancy",
        "scenario_type": "chat",
        "input": "SHOULD_NOT_BE_SENT",
        "output": "SHOULD_NOT_BE_SENT",
        "expected": "SHOULD_NOT_BE_SENT",
        "retrieval": ["SHOULD_NOT_BE_SENT"],
        "events": [{"type": "note", "text": "SHOULD_NOT_BE_SENT"}],
        "turns": _turns(),
        "chatbot_role": "SHOULD_NOT_BE_SENT",
        "expected_outcome": "SHOULD_NOT_BE_SENT",
        "expected_tool_calls": [
            ToolInvocation(name="SHOULD_NOT_BE_SENT", arguments={"q": "x"}, result="no")
        ],
    }
    values.update(overrides)
    return EvaluationTrace(**values)


def test_evaluator_conforms_to_protocol():
    evaluator = DeepEvalTurnRelevancyEvaluator(
        turn_relevancy_metric=RecordingTurnRelevancyMetric()
    )
    assert isinstance(evaluator, Evaluator)


def test_evaluate_returns_one_evaluation_result():
    results = DeepEvalTurnRelevancyEvaluator(
        turn_relevancy_metric=RecordingTurnRelevancyMetric()
    ).evaluate(_trace())
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], EvaluationResult)


def test_metric_and_evaluator_identity():
    result = DeepEvalTurnRelevancyEvaluator(
        turn_relevancy_metric=RecordingTurnRelevancyMetric()
    ).evaluate(_trace())[0]
    assert result.metric == TURN_RELEVANCY_METRIC
    assert result.metric == "turn_relevancy"
    assert result.evaluator == DEEPEVAL_EVALUATOR_NAME
    assert result.evaluator == "deepeval"


def test_turns_map_to_conversational_test_case_in_order():
    metric = RecordingTurnRelevancyMetric()
    DeepEvalTurnRelevancyEvaluator(turn_relevancy_metric=metric).evaluate(_trace())
    mapped = metric.test_case.turns
    assert [(turn.role, turn.content) for turn in mapped] == [
        ("user", "What is your return window?"),
        ("assistant", "Returns are accepted within 30 days of delivery."),
        ("user", "Does that include sale items?"),
        ("assistant", "Yes. Sale items use the same 30-day window."),
    ]
    assert type(metric.test_case).__name__ == "ConversationalTestCase"
    assert type(mapped[0]).__name__ == "Turn"


def test_mapping_ignores_retrieval_expected_events_and_tools():
    metric = RecordingTurnRelevancyMetric()
    DeepEvalTurnRelevancyEvaluator(turn_relevancy_metric=metric).evaluate(_trace())
    assert metric.test_case.scenario is None
    assert metric.test_case.expected_outcome is None
    assert metric.test_case.chatbot_role is None
    for turn in metric.test_case.turns:
        assert turn.retrieval_context is None
        assert turn.tools_called is None
        assert "SHOULD_NOT_BE_SENT" not in turn.content
    rendered = repr(metric.test_case)
    assert "SHOULD_NOT_BE_SENT" not in rendered


def test_score_and_reason_are_copied_without_policy_fields():
    metric = RecordingTurnRelevancyMetric(
        score=0.93,
        reason="Each assistant reply addresses the preceding user turn.",
    )
    result = DeepEvalTurnRelevancyEvaluator(turn_relevancy_metric=metric).evaluate(_trace())[0]
    assert result.score == 0.93
    assert result.score is metric.score
    assert result.reason == "Each assistant reply addresses the preceding user turn."
    assert result.reason is metric.reason
    assert result.raw_result["turns"][0] == {
        "role": "user",
        "content": "What is your return window?",
    }
    assert "threshold" not in result.raw_result
    assert "success" not in result.raw_result
    assert "passed" not in result.raw_result
    assert "passed" not in result.__dataclass_fields__


def test_evaluate_accepts_optional_configuration():
    metric = RecordingTurnRelevancyMetric(score=0.7)
    evaluator = DeepEvalTurnRelevancyEvaluator(turn_relevancy_metric=metric)
    assert evaluator.evaluate(_trace())[0].score == 0.7
    assert evaluator.evaluate(_trace(), None)[0].score == 0.7
    assert evaluator.evaluate(_trace(), {"threshold": 0.99})[0].score == 0.7


@pytest.mark.parametrize("turns", [None, []])
def test_empty_turns_raise_before_the_metric_is_called(turns):
    metric = RecordingTurnRelevancyMetric()
    with pytest.raises(ValueError, match="non-empty"):
        DeepEvalTurnRelevancyEvaluator(turn_relevancy_metric=metric).evaluate(
            _trace(turns=turns)
        )
    assert metric.measure_calls == 0


def test_trailing_user_turn_is_rejected():
    metric = RecordingTurnRelevancyMetric()
    turns = _turns() + [ConversationTurn(role="user", content="And opened items?")]
    with pytest.raises(ValueError, match="assistant turn"):
        DeepEvalTurnRelevancyEvaluator(turn_relevancy_metric=metric).evaluate(
            _trace(turns=turns)
        )
    assert metric.measure_calls == 0


def test_conversation_without_a_user_turn_is_rejected():
    metric = RecordingTurnRelevancyMetric()
    with pytest.raises(ValueError, match="user turn"):
        DeepEvalTurnRelevancyEvaluator(turn_relevancy_metric=metric).evaluate(
            _trace(turns=[ConversationTurn(role="assistant", content="Hello.")])
        )
    assert metric.measure_calls == 0


def test_runner_applies_policy_and_gate_to_turn_relevancy():
    metric = RecordingTurnRelevancyMetric(score=0.91, reason="Relevant throughout.")
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="turn_relevancy",
            evaluator="deepeval",
            category="semantic",
        )
    )
    policy = QualityPolicy(metric="turn_relevancy", operator=">=", threshold=0.8)
    runner = EvaluationRunner(
        registry=registry,
        evaluators={
            "turn_relevancy": DeepEvalTurnRelevancyEvaluator(turn_relevancy_metric=metric)
        },
        policies={"turn_relevancy": policy},
        gate=QualityGate(),
    )

    decision = runner.run(
        _trace(),
        EvaluationConfig(evaluations=["turn_relevancy"]),
        run_id="run-turn-relevancy",
    )

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.run_id == "run-turn-relevancy"
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "turn_relevancy"
    assert result.evaluator == "deepeval"
    assert result.score == 0.91
    assert result.reason == "Relevant throughout."
    assert runner.last_run.decisions[0].passed is True
    assert runner.last_run.decisions[0].operator == ">="
    assert runner.last_run.decisions[0].threshold == 0.8
    assert metric.measure_calls == 1
    assert metric.test_case.turns[-1].role == "assistant"
