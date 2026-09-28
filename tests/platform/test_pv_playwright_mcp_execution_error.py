"""PV P0.3c: MCP tool isError=True is preserved; scenario is not successful.

Focused live negative: force a real Playwright MCP tool failure, capture it
through serialize → ToolInvocation → EvaluationTrace, and prove the scenario
is not reported as successful.

Does not introduce a new evaluator, combined score, or platform abstraction.
"""

from __future__ import annotations

import shutil

import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from ai_qe_eval.capture.mcp_trace import (
    build_mcp_evaluation_trace,
    tool_invocation_from_observation,
)
from ai_qe_eval.integrations.playwright_mcp import (
    PLAYWRIGHT_MCP_PACKAGE,
    TODO_MVC_URL,
    playwright_mcp_stdio_parameters,
    serialize_call_tool_result,
)

from test_pv_playwright_mcp_multistep_tool_order_through_runner import (  # noqa: E402
    EXPECTED_TODO_TEXT,
    _resolve_snapshot_body,
)


@pytest.mark.live
@pytest.mark.asyncio
async def test_pv_playwright_mcp_execution_error_preserved_scenario_not_successful():
    if shutil.which("npx") is None and shutil.which("npx.cmd") is None:
        pytest.skip("npx is not available on PATH")

    goal = "Add Buy milk on TodoMVC (execution-error negative path)."
    observed = []

    async with stdio_client(playwright_mcp_stdio_parameters()) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("playwright_mcp_package", PLAYWRIGHT_MCP_PACKAGE)

            async def _call_allow_error(name: str, arguments: dict | None) -> dict:
                result = await session.call_tool(name, arguments)
                plain = serialize_call_tool_result(result)
                print("tool_call", name, "isError", plain["isError"])
                observed.append(
                    tool_invocation_from_observation(
                        name=name,
                        arguments=arguments,
                        result=plain,
                    )
                )
                return plain

            nav = await _call_allow_error("browser_navigate", {"url": TODO_MVC_URL})
            assert nav["isError"] is False

            # Invalid target after a real page load → MCP tool-level error.
            err = await _call_allow_error(
                "browser_click",
                {
                    "element": "Nonexistent control",
                    "target": "e99999",
                },
            )
            assert err["isError"] is True

            # Optional post-error snapshot for final-state evidence only.
            snap = await _call_allow_error("browser_snapshot", {})
            final_body = (
                _resolve_snapshot_body(snap) if snap["isError"] is False else ""
            )
            final_state_ok = EXPECTED_TODO_TEXT in final_body
            print("final_state_ok", final_state_ok)

    assert any(
        isinstance(call.result, dict) and call.result.get("isError") is True
        for call in observed
    )

    expected_tool_calls = [
        tool_invocation_from_observation(name=call.name, arguments=None, result=None)
        for call in observed
    ]
    trace = build_mcp_evaluation_trace(
        trace_id="pv-playwright-mcp-execution-error",
        input=goal,
        output="Playwright MCP flow stopped after a tool execution error.",
        expected="Playwright MCP flow stopped after a tool execution error.",
        observed_tool_calls=observed,
        expected_tool_calls=expected_tool_calls,
    )

    # Failure preserved in existing trace / tool evidence.
    error_calls = [
        call
        for call in trace.turns[1].tool_calls
        if isinstance(call.result, dict) and call.result.get("isError") is True
    ]
    assert len(error_calls) >= 1
    assert error_calls[0].name == "browser_click"
    assert error_calls[0].result["isError"] is True
    assert error_calls[0].result["content"]
    print("preserved_error_tool", error_calls[0].name)
    print("preserved_is_error", error_calls[0].result["isError"])

    mcp_execution_ok = not any(
        isinstance(call.result, dict) and call.result.get("isError") is True
        for call in observed
    )
    # Scenario success requires clean MCP execution and correct final state.
    # ToolCorrectness is not claimed here; evidence is the captured isError.
    scenario_task_succeeded = mcp_execution_ok and final_state_ok
    print("mcp_execution_ok", mcp_execution_ok)
    print("scenario_task_succeeded", scenario_task_succeeded)
    assert mcp_execution_ok is False
    assert scenario_task_succeeded is False
