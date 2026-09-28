"""PV P0.3b: final state ok but wrong tool order → scenario fails.

Negative counterpart to the flagship multi-step Playwright MCP scenario.

Proves the scenario AND:
- TodoMVC final state intentionally contains "Buy milk" (final_state_ok=True)
- observed tool order differs from expected → ToolCorrectness gate fails
- scenario_task_succeeded is therefore False

ToolCorrectness and final-state remain separate evidence. No combined score
or new evaluator.
"""

from __future__ import annotations

import math
import shutil

import pytest
from deepeval.metrics.tool_correctness.tool_correctness import ToolCorrectnessMetric
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

from test_pv_playwright_mcp_multistep_tool_order_through_runner import (  # noqa: E402
    EXPECTED_TODO_TEXT,
    EXPECTED_TOOL_ORDER,
    TOOL_CORRECTNESS_THRESHOLD,
    _ref_for_label,
    _resolve_snapshot_body,
    _snapshot_text,
)

# Same tools as flagship, but snapshot/click swapped after type.
WRONG_TOOL_ORDER = [
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_snapshot",
    "browser_click",
]


@pytest.mark.live
@pytest.mark.asyncio
async def test_pv_playwright_mcp_multistep_wrong_tool_order_final_state_ok_scenario_fails():
    if shutil.which("npx") is None and shutil.which("npx.cmd") is None:
        pytest.skip("npx is not available on PATH")

    goal = "Add a todo item on the Playwright TodoMVC demo."
    observed = []
    final_state_ok = False

    async with stdio_client(playwright_mcp_stdio_parameters()) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("playwright_mcp_package", PLAYWRIGHT_MCP_PACKAGE)
            print("wrong_observed_order", WRONG_TOOL_ORDER)
            print("expected_tool_order", EXPECTED_TOOL_ORDER)

            async def _call(name: str, arguments: dict | None) -> dict:
                result = await session.call_tool(name, arguments)
                plain = serialize_call_tool_result(result)
                print("tool_call", name, "isError", plain["isError"])
                assert plain["isError"] is False, plain
                observed.append(
                    tool_invocation_from_observation(
                        name=name,
                        arguments=arguments,
                        result=plain,
                    )
                )
                return plain

            await _call("browser_navigate", {"url": TODO_MVC_URL})
            first_snapshot = await _call("browser_snapshot", {})
            snapshot_text = _snapshot_text(first_snapshot)
            textbox_ref = _ref_for_label(snapshot_text, 'textbox "What needs to be done?"')
            heading_ref = _ref_for_label(snapshot_text, 'heading "todos"')

            await _call(
                "browser_click",
                {
                    "element": "Todo input textbox",
                    "target": textbox_ref,
                },
            )
            await _call(
                "browser_type",
                {
                    "element": "Todo input textbox",
                    "target": textbox_ref,
                    "text": EXPECTED_TODO_TEXT,
                    "submit": True,
                },
            )
            # Wrong order vs flagship: snapshot before the second click.
            post_type_snapshot = await _call("browser_snapshot", {})
            await _call(
                "browser_click",
                {
                    "element": "Todos heading",
                    "target": heading_ref,
                },
            )

            # Evidence 1: application outcome is still correct.
            final_body = _resolve_snapshot_body(post_type_snapshot)
            final_state_ok = EXPECTED_TODO_TEXT in final_body
            print("final_state_ok", final_state_ok)
            assert final_state_ok is True, (
                f"Negative wrong-order case requires {EXPECTED_TODO_TEXT!r} present; "
                f"resolved snapshot body starts with: {final_body[:500]!r}"
            )

    captured_names = [call.name for call in observed]
    print("captured_tool_names", captured_names)
    assert captured_names == WRONG_TOOL_ORDER
    assert captured_names != EXPECTED_TOOL_ORDER

    expected_tool_calls = [
        tool_invocation_from_observation(name=name, arguments=None, result=None)
        for name in EXPECTED_TOOL_ORDER
    ]
    trace = build_mcp_evaluation_trace(
        trace_id="pv-playwright-mcp-multistep-wrong-tool-order",
        input=goal,
        output="Multi-step Playwright MCP flow with intentional wrong tool order.",
        expected="Multi-step Playwright MCP flow with intentional wrong tool order.",
        observed_tool_calls=observed,
        expected_tool_calls=expected_tool_calls,
    )
    assert [call.name for call in trace.turns[1].tool_calls] == WRONG_TOOL_ORDER

    metric = ToolCorrectnessMetric(
        should_exact_match=True,
        available_tools=None,
        evaluation_params=[],
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

    decision = runner.run(
        trace,
        EvaluationConfig(evaluations=["tool_correctness"]),
        run_id="pv-playwright-mcp-multistep-wrong-tool-order",
    )

    # Evidence 2: ToolCorrectness gate fails on wrong order.
    tool_correctness_gate_passed = decision.passed is True
    assert runner.last_run is not None
    result = runner.last_run.results[0]
    assert result.metric == "tool_correctness"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert result.score < TOOL_CORRECTNESS_THRESHOLD
    assert tool_correctness_gate_passed is False
    print("tool_correctness_score", result.score)
    print("tool_correctness_reason", result.reason)
    print("tool_correctness_gate_passed", tool_correctness_gate_passed)

    scenario_task_succeeded = tool_correctness_gate_passed and final_state_ok
    print("scenario_task_succeeded", scenario_task_succeeded)
    assert scenario_task_succeeded is False, (
        "Wrong-order scenario must fail overall despite correct final state: "
        f"tool_correctness_gate_passed={tool_correctness_gate_passed}, "
        f"final_state_ok={final_state_ok}"
    )
