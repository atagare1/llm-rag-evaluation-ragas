"""C.2 deterministic Runner integration for MCP capture.

Proves:
capture → EvaluationTrace → EvaluationRunner
→ DeepEvalToolCorrectnessEvaluator → EvaluationResult
→ QualityPolicy → QualityGate

Uses a mocked ToolCorrectnessMetric. No live provider. No capture-layer scoring.
"""

from __future__ import annotations

from ai_qe_eval.capture.mcp_trace import (
    build_mcp_evaluation_trace,
    tool_invocation_from_observation,
)
from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deepeval_tool_correctness import DeepEvalToolCorrectnessEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

TOOL_CORRECTNESS_THRESHOLD = 0.80


class RecordingToolCorrectnessMetric:
    def __init__(self, score: float, reason: str) -> None:
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


def _captured_mcp_trace():
    actual = tool_invocation_from_observation(
        name="weather",
        arguments={"location": "Pune"},
        result="Temperature is 28 C and conditions are clear.",
    )
    expected = tool_invocation_from_observation(
        name="weather",
        arguments={"location": "Pune"},
        result="Temperature is 28 C and conditions are clear.",
    )
    return build_mcp_evaluation_trace(
        trace_id="trace-mcp-runner",
        input="Find the weather for Pune.",
        output="Temperature is 28 C and conditions are clear.",
        expected="Temperature is 28 C and conditions are clear.",
        observed_tool_calls=[actual],
        expected_tool_calls=[expected],
    )


def _runner(metric: RecordingToolCorrectnessMetric) -> EvaluationRunner:
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="tool_correctness",
            evaluator="deepeval",
            category="agent",
        )
    )
    return EvaluationRunner(
        registry=registry,
        evaluators={
            "tool_correctness": DeepEvalToolCorrectnessEvaluator(
                tool_correctness_metric=metric
            )
        },
        policies={
            "tool_correctness": QualityPolicy(
                metric="tool_correctness",
                operator=">=",
                threshold=TOOL_CORRECTNESS_THRESHOLD,
            )
        },
        gate=QualityGate(),
    )


def test_mcp_capture_through_runner_passes_when_score_meets_policy():
    metric = RecordingToolCorrectnessMetric(score=0.91, reason="Exact match.")
    runner = _runner(metric)

    decision = runner.run(
        _captured_mcp_trace(),
        EvaluationConfig(evaluations=["tool_correctness"]),
        run_id="run-mcp-capture-pass",
    )

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.run_id == "run-mcp-capture-pass"
    assert runner.last_run.gate_decision is decision
    assert runner.last_run.traces[0].scenario_type == "mcp"
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert result.score == 0.91
    assert result.reason == "Exact match."
    assert runner.last_run.decisions[0].passed is True
    assert runner.last_run.decisions[0].operator == ">="
    assert runner.last_run.decisions[0].threshold == TOOL_CORRECTNESS_THRESHOLD
    assert metric.measure_calls == 1
    assert metric.test_case.tools_called[0].name == "weather"
    assert metric.test_case.expected_tools[0].name == "weather"


def test_mcp_capture_through_runner_fails_when_score_misses_policy():
    metric = RecordingToolCorrectnessMetric(score=0.40, reason="Not an exact match.")
    runner = _runner(metric)

    decision = runner.run(
        _captured_mcp_trace(),
        EvaluationConfig(evaluations=["tool_correctness"]),
        run_id="run-mcp-capture-fail",
    )

    assert decision.passed is False
    assert runner.last_run is not None
    assert runner.last_run.run_id == "run-mcp-capture-fail"
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert result.score == 0.40
    assert result.reason == "Not an exact match."
    assert runner.last_run.decisions[0].passed is False
    assert runner.last_run.decisions[0].threshold == TOOL_CORRECTNESS_THRESHOLD
    assert metric.measure_calls == 1
