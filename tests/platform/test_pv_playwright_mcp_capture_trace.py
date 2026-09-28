"""PV: real Playwright MCP call captured into EvaluationTrace.

Validates capture/trace fidelity only. Does not run ToolCorrectnessMetric.
"""

from __future__ import annotations

import json
import shutil

import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from ai_qe_eval.capture.mcp_trace import (
    build_mcp_evaluation_trace,
    tool_invocation_from_observation,
)
from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.domain.trace import EvaluationTrace
from ai_qe_eval.integrations.playwright_mcp import (
    PLAYWRIGHT_MCP_PACKAGE,
    TODO_MVC_URL,
    playwright_mcp_stdio_parameters,
    serialize_call_tool_result,
)


@pytest.mark.live
@pytest.mark.asyncio
async def test_pv_playwright_mcp_browser_navigate_builds_evaluation_trace():
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

            plain_result = serialize_call_tool_result(mcp_result)
            observed = tool_invocation_from_observation(
                name="browser_navigate",
                arguments=arguments,
                result=plain_result,
            )
            # Ground truth for later scoring; result is intentionally omitted
            # because the live browser payload is dynamic.
            expected_call = tool_invocation_from_observation(
                name="browser_navigate",
                arguments={"url": TODO_MVC_URL},
                result=None,
            )
            trace = build_mcp_evaluation_trace(
                trace_id="pv-playwright-mcp-capture",
                input=goal,
                output="Navigated to the Playwright TodoMVC demo.",
                expected="Navigated to the Playwright TodoMVC demo.",
                observed_tool_calls=[observed],
                expected_tool_calls=[expected_call],
            )

    assert isinstance(trace, EvaluationTrace)
    assert trace.scenario_type == "mcp"
    assert trace.input == goal
    assert len(trace.turns) == 2
    assert trace.turns[0].role == "user"
    assert trace.turns[0].content == goal
    assert trace.turns[0].tool_calls is None
    assert trace.turns[1].role == "assistant"
    assert isinstance(trace.turns[1].tool_calls, list)
    assert len(trace.turns[1].tool_calls) == 1

    actual = trace.turns[1].tool_calls[0]
    assert isinstance(actual, ToolInvocation)
    assert actual.name == "browser_navigate"
    assert actual.arguments == {"url": TODO_MVC_URL}
    assert actual.arguments is not None
    assert actual.result is not None
    assert actual.result["isError"] is False
    assert actual.result["content"]
    json.dumps(actual.result)

    assert trace.expected_tool_calls is not None
    assert len(trace.expected_tool_calls) == 1
    assert trace.expected_tool_calls[0].name == "browser_navigate"
    assert trace.expected_tool_calls[0].arguments == {"url": TODO_MVC_URL}
    assert trace.expected_tool_calls[0].result is None
    assert actual.result is not trace.expected_tool_calls[0].result

    print("captured_tool_name", actual.name)
    print("captured_arguments", actual.arguments)
    print("captured_result_keys", sorted(actual.result.keys()))
    print("session_closed", True)
