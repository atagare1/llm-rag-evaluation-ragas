"""DeepEval Tool Correctness adapter.

Maps flattened EvaluationTrace tool calls into an LLMTestCase and
ToolCorrectnessMetric.measure, then returns one EvaluationResult.

Tool correctness is the DeepEval mechanism; evaluator identity is "deepeval".
Does not apply DeepEval's threshold or metric.success. Does not auto-register.
Does not read events, trace.expected, or MCP SDK types.

Exact match compares arguments and results only when ToolCallParams
INPUT_PARAMETERS and OUTPUT are selected. Those parameters are enabled so
ToolInvocation.arguments and ToolInvocation.result participate in the score.
None and {} are passed through unchanged. DeepEval treats them as different.
"""

from __future__ import annotations

from typing import Any

from ai_qe_eval.domain.conversation import ConversationTurn, ToolInvocation
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.trace import EvaluationTrace

TOOL_CORRECTNESS_METRIC = "tool_correctness"
DEEPEVAL_EVALUATOR_NAME = "deepeval"


def _require_tool_correctness_inputs(
    trace: EvaluationTrace,
) -> tuple[list[ConversationTurn], list[ToolInvocation]]:
    if trace.input is None:
        raise ValueError(
            "DeepEvalToolCorrectnessEvaluator requires EvaluationTrace.input"
        )
    if trace.turns is None:
        raise ValueError(
            "DeepEvalToolCorrectnessEvaluator requires EvaluationTrace.turns "
            "so actual tool calls can be observed"
        )
    if trace.expected_tool_calls is None:
        raise ValueError(
            "DeepEvalToolCorrectnessEvaluator requires "
            "EvaluationTrace.expected_tool_calls"
        )
    for index, turn in enumerate(trace.turns):
        if not isinstance(turn, ConversationTurn):
            raise TypeError(
                "DeepEvalToolCorrectnessEvaluator requires ConversationTurn items, "
                f"got {type(turn).__name__} at index {index}"
            )
    for index, call in enumerate(trace.expected_tool_calls):
        if not isinstance(call, ToolInvocation):
            raise TypeError(
                "DeepEvalToolCorrectnessEvaluator requires ToolInvocation "
                "expected_tool_calls, "
                f"got {type(call).__name__} at index {index}"
            )
    return list(trace.turns), list(trace.expected_tool_calls)


def _flatten_tool_calls(turns: list[ConversationTurn]) -> list[ToolInvocation]:
    calls: list[ToolInvocation] = []
    for turn_index, turn in enumerate(turns):
        if turn.tool_calls is None:
            continue
        for call_index, call in enumerate(turn.tool_calls):
            if not isinstance(call, ToolInvocation):
                raise TypeError(
                    "DeepEvalToolCorrectnessEvaluator requires ToolInvocation "
                    "tool_calls, "
                    f"got {type(call).__name__} at turn {turn_index} "
                    f"index {call_index}"
                )
            calls.append(call)
    return calls


def _to_tool_call(invocation: ToolInvocation) -> Any:
    from deepeval.test_case import ToolCall

    return ToolCall(
        name=invocation.name,
        input_parameters=invocation.arguments,
        output=invocation.result,
    )


def _trace_to_llm_test_case(trace: EvaluationTrace) -> Any:
    from deepeval.test_case import LLMTestCase

    turns, expected = _require_tool_correctness_inputs(trace)
    return LLMTestCase(
        input=trace.input,
        tools_called=[_to_tool_call(call) for call in _flatten_tool_calls(turns)],
        expected_tools=[_to_tool_call(call) for call in expected],
    )


def _default_evaluation_params() -> list[Any]:
    from deepeval.test_case import ToolCallParams

    return [
        ToolCallParams.INPUT_PARAMETERS,
        ToolCallParams.OUTPUT,
    ]


def _default_tool_correctness_metric(
    *,
    model: Any | None,
    evaluation_params: list[Any] | None = None,
) -> Any:
    from deepeval.metrics.tool_correctness.tool_correctness import ToolCorrectnessMetric

    return ToolCorrectnessMetric(
        should_exact_match=True,
        available_tools=None,
        evaluation_params=(
            list(evaluation_params)
            if evaluation_params is not None
            else _default_evaluation_params()
        ),
        include_reason=True,
        model=model,
    )


def _tool_record(invocation: ToolInvocation) -> dict[str, Any]:
    return {
        "name": invocation.name,
        "arguments": invocation.arguments,
        "result": invocation.result,
    }


class DeepEvalToolCorrectnessEvaluator:
    """DeepEval Tool Correctness adapter.

    Returns metric="tool_correctness" and evaluator="deepeval".
    Does not apply DeepEval's threshold or metric.success.

    Default evaluation_params compare INPUT_PARAMETERS and OUTPUT.
    Pass evaluation_params to override that default when constructing the
    metric. An injected tool_correctness_metric takes precedence.
    """

    def __init__(
        self,
        *,
        tool_correctness_metric: Any | None = None,
        model: Any | None = None,
        evaluation_params: list[Any] | None = None,
    ) -> None:
        self._tool_correctness_metric = tool_correctness_metric
        self._model = model
        self._evaluation_params = (
            None if evaluation_params is None else list(evaluation_params)
        )

    def _metric(self) -> Any:
        if self._tool_correctness_metric is not None:
            return self._tool_correctness_metric
        return _default_tool_correctness_metric(
            model=self._model,
            evaluation_params=self._evaluation_params,
        )

    def evaluate(
        self,
        trace: EvaluationTrace,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        turns, expected = _require_tool_correctness_inputs(trace)
        actual = _flatten_tool_calls(turns)
        test_case = _trace_to_llm_test_case(trace)
        metric = self._metric()
        metric.measure(test_case)
        score = metric.score
        reason = getattr(metric, "reason", None)
        return [
            EvaluationResult(
                metric=TOOL_CORRECTNESS_METRIC,
                evaluator=DEEPEVAL_EVALUATOR_NAME,
                score=score,
                reason=reason,
                raw_result={
                    "score": score,
                    "reason": reason,
                    "input": test_case.input,
                    "tools_called": [_tool_record(call) for call in actual],
                    "expected_tools": [_tool_record(call) for call in expected],
                },
            )
        ]
