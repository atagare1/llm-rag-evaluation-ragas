"""PV P0.3a: ToolCorrectness passes but wrong final state → scenario fails.

Negative counterpart to the flagship multi-step Playwright MCP scenario.

Proves the scenario AND:
- ordered ToolCorrectness gate can still pass
- final TodoMVC state intentionally lacks "Buy milk"
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

# Reuse flagship helpers/constants (same directory; not a platform abstraction).
from test_pv_playwright_mcp_multistep_tool_order_through_runner import (  # noqa: E402
    EXPECTED_TODO_TEXT,
    EXPECTED_TOOL_ORDER,
    TOOL_CORRECTNESS_THRESHOLD,
    _ref_for_label,
    _resolve_snapshot_body,
    _snapshot_text,
)

# Intentionally different from EXPECTED_TODO_TEXT so final state fails.
TYPED_TODO_TEXT = "Buy bread"


@pytest.mark.live
@pytest.mark.asyncio
async def test_pv_playwright_mcp_multistep_tool_ok_final_state_mismatch_scenario_fails():
    if shutil.which("npx") is None and shutil.which("npx.cmd") is None:
        pytest.skip("npx is not available on PATH")

    goal = "Add a todo item on the Playwright TodoMVC demo."
    observed = []
    final_state_ok = True  # flipped after intentional mismatch

    async with stdio_client(playwright_mcp_stdio_parameters()) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("playwright_mcp_package", PLAYWRIGHT_MCP_PACKAGE)
            print("typed_todo_text", TYPED_TODO_TEXT)
            print("expected_todo_text", EXPECTED_TODO_TEXT)

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
            # Same tool trajectory; wrong application outcome on purpose.
            await _call(
                "browser_type",
                {
                    "element": "Todo input textbox",
                    "target": textbox_ref,
                    "text": TYPED_TODO_TEXT,
                    "submit": True,
                },
            )
            await _call(
                "browser_click",
                {
                    "element": "Todos heading",
                    "target": heading_ref,
                },
            )
            final_snapshot = await _call("browser_snapshot", {})
            # Evidence 1: deterministic final-state mismatch (test-local).
            final_body = _resolve_snapshot_body(final_snapshot)
            final_state_ok = EXPECTED_TODO_TEXT in final_body
            print("final_state_ok", final_state_ok)
            print("typed_todo_present", TYPED_TODO_TEXT in final_body)
            assert final_state_ok is False, (
                f"Negative test requires missing {EXPECTED_TODO_TEXT!r}; "
                f"resolved snapshot body starts with: {final_body[:500]!r}"
            )
            assert TYPED_TODO_TEXT in final_body

    captured_names = [call.name for call in observed]
    print("captured_tool_names", captured_names)
    assert captured_names == EXPECTED_TOOL_ORDER

    expected_tool_calls = [
        tool_invocation_from_observation(name=name, arguments=None, result=None)
        for name in EXPECTED_TOOL_ORDER
    ]
    trace = build_mcp_evaluation_trace(
        trace_id="pv-playwright-mcp-multistep-final-state-mismatch",
        input=goal,
        output="Multi-step Playwright MCP flow with intentional final-state mismatch.",
        expected="Multi-step Playwright MCP flow with intentional final-state mismatch.",
        observed_tool_calls=observed,
        expected_tool_calls=expected_tool_calls,
    )
    assert [call.name for call in trace.turns[1].tool_calls] == EXPECTED_TOOL_ORDER

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
        run_id="pv-playwright-mcp-multistep-final-state-mismatch",
    )

    # Evidence 2: ordered ToolCorrectness still passes.
    tool_correctness_gate_passed = decision.passed is True
    assert tool_correctness_gate_passed
    assert runner.last_run is not None
    result = runner.last_run.results[0]
    assert result.metric == "tool_correctness"
    assert result.score == 1.0
    assert result.score >= TOOL_CORRECTNESS_THRESHOLD
    assert math.isfinite(result.score)
    print("tool_correctness_score", result.score)
    print("tool_correctness_gate_passed", tool_correctness_gate_passed)

    # Scenario AND: trajectory ok + wrong state ⇒ overall failure.
    scenario_task_succeeded = tool_correctness_gate_passed and final_state_ok
    print("scenario_task_succeeded", scenario_task_succeeded)
    assert scenario_task_succeeded is False, (
        "Negative scenario must fail overall when final state is wrong: "
        f"tool_correctness_gate_passed={tool_correctness_gate_passed}, "
        f"final_state_ok={final_state_ok}"
    )
