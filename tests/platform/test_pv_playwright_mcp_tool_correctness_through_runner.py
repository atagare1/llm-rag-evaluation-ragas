"""PV: real Playwright MCP → capture → ToolCorrectness through EvaluationRunner.

Uses INPUT_PARAMETERS-only exact match so dynamic browser_navigate output is
ignored. Compares tool name and URL arguments. Platform gate is an explicit
QualityPolicy of tool_correctness >= 0.80.
"""

from __future__ import annotations

import math
import shutil

import pytest
from deepeval.metrics.tool_correctness.tool_correctness import ToolCorrectnessMetric
from deepeval.test_case import ToolCallParams
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from ai_qe_eval.capture.mcp_trace import (
    build_mcp_evaluation_trace,
    tool_invocation_from_observation,
)
from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deepeval_tool_correctness import DeepEvalToolCorrectnessEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.integrations.playwright_mcp import (
    PLAYWRIGHT_MCP_PACKAGE,
    TODO_MVC_URL,
    playwright_mcp_stdio_parameters,
    serialize_call_tool_result,
)
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

TOOL_CORRECTNESS_THRESHOLD = 0.80


@pytest.mark.live
@pytest.mark.asyncio
async def test_pv_playwright_mcp_tool_correctness_through_runner():
    if shutil.which("npx") is None and shutil.which("npx.cmd") is None:
        pytest.skip("npx is not available on PATH")

    goal = "Open the Playwright TodoMVC demo."
    arguments = {"url": TODO_MVC_URL}

    async with stdio_client(playwright_mcp_stdio_parameters()) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            mcp_result = await session.call_tool("browser_navigate", arguments)
            print("playwright_mcp_package", PLAYWRIGHT_MCP_PACKAGE)
            print("navigate_is_error", mcp_result.isError)
            assert mcp_result.isError is False

            observed = tool_invocation_from_observation(
                name="browser_navigate",
                arguments=arguments,
                result=serialize_call_tool_result(mcp_result),
            )
            expected_call = tool_invocation_from_observation(
                name="browser_navigate",
                arguments={"url": TODO_MVC_URL},
                result=None,
            )
            trace = build_mcp_evaluation_trace(
                trace_id="pv-playwright-mcp-tool-correctness",
                input=goal,
                output="Navigated to the Playwright TodoMVC demo.",
                expected="Navigated to the Playwright TodoMVC demo.",
                observed_tool_calls=[observed],
                expected_tool_calls=[expected_call],
            )

    metric = ToolCorrectnessMetric(
        should_exact_match=True,
        available_tools=None,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        include_reason=True,
        async_mode=False,
        model=None,
    )
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="tool_correctness",
            evaluator="deepeval",
            category="agent",
        )
    )
    policy = QualityPolicy(
        metric="tool_correctness",
        operator=">=",
        threshold=TOOL_CORRECTNESS_THRESHOLD,
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={
            "tool_correctness": DeepEvalToolCorrectnessEvaluator(
                tool_correctness_metric=metric,
            )
        },
        policies={"tool_correctness": policy},
        gate=QualityGate(),
    )
    print("policy_threshold", policy.threshold)
    print("evaluation_params", ["INPUT_PARAMETERS"])

    decision = runner.run(
        trace,
        EvaluationConfig(evaluations=["tool_correctness"]),
        run_id="pv-playwright-mcp-tool-correctness",
    )

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert runner.last_run.traces[0].scenario_type == "mcp"
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.score >= TOOL_CORRECTNESS_THRESHOLD
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    assert runner.last_run.decisions[0].passed is True
    assert runner.last_run.decisions[0].threshold == TOOL_CORRECTNESS_THRESHOLD
    print("tool_correctness_score", result.score)
    print("tool_correctness_reason", result.reason)
    print("gate_passed", decision.passed)
