"""PV P0 flagship e2e (PASS): NL task → Playwright MCP → Trace → gate → state.

Natural-language task → real Playwright MCP multi-step TodoMVC flow →
EvaluationTrace → ordered ToolCorrectness through Runner/Policy/Gate →
deterministic final-state validation → scenario_task_succeeded=True.

Ordered tool names only:
browser_navigate → browser_snapshot → browser_click → browser_type
→ browser_click → browser_snapshot

Final browser_snapshot is checked for "Buy milk" (inline text or
[Snapshot](path) sidecar). That assertion is test-local and is not part
of ToolCorrectness.

Scenario success requires BOTH separate evidence signals:
1. ordered ToolCorrectness gate passes
2. final TodoMVC state contains "Buy milk"
No combined score or new evaluator is introduced.

Dynamic snapshot refs are used for live calls but are NOT copied into
expected_tool_calls. ToolCorrectness uses evaluation_params=[].
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

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

TOOL_CORRECTNESS_THRESHOLD = 0.80
EXPECTED_TODO_TEXT = "Buy milk"
EXPECTED_TOOL_ORDER = [
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_click",
    "browser_snapshot",
]
_SNAPSHOT_SIDECAR_RE = re.compile(r"\[Snapshot\]\(([^)]+)\)")


def _snapshot_text(plain_result: dict) -> str:
    for item in plain_result.get("content") or []:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            return item["text"]
    return ""


def _resolve_snapshot_body(plain_result: dict) -> str:
    """Return inline snapshot text, or sidecar file contents when referenced."""
    text = _snapshot_text(plain_result)
    match = _SNAPSHOT_SIDECAR_RE.search(text)
    if match is None:
        return text
    path = Path(match.group(1))
    if not path.is_file():
        raise AssertionError(f"Snapshot sidecar not found: {path}")
    return path.read_text(encoding="utf-8")


def _ref_for_label(snapshot_text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}.*?\[ref=(e\d+)\]"
    match = re.search(pattern, snapshot_text, flags=re.DOTALL)
    if match is None:
        raise AssertionError(f"Could not find ref for label {label!r} in snapshot")
    return match.group(1)


@pytest.mark.live
@pytest.mark.asyncio
async def test_pv_playwright_mcp_flagship_todomvc_e2e_scenario_passes():
    if shutil.which("npx") is None and shutil.which("npx.cmd") is None:
        pytest.skip("npx is not available on PATH")

    goal = "Add 'Buy milk' to the Playwright TodoMVC demo."
    observed = []
    final_state_ok = False

    async with stdio_client(playwright_mcp_stdio_parameters()) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("playwright_mcp_package", PLAYWRIGHT_MCP_PACKAGE)

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
            # Second click uses a stable ref from the first snapshot so the
            # requested tool order remains executable without an extra snapshot.
            await _call(
                "browser_click",
                {
                    "element": "Todos heading",
                    "target": heading_ref,
                },
            )
            final_snapshot = await _call("browser_snapshot", {})
            # Evidence 1: deterministic final-state (test-local; not DeepEval).
            final_body = _resolve_snapshot_body(final_snapshot)
            final_state_ok = EXPECTED_TODO_TEXT in final_body
            print("final_state_ok", final_state_ok)
            assert final_state_ok, (
                f"Expected TodoMVC state to contain {EXPECTED_TODO_TEXT!r}; "
                f"resolved snapshot body starts with: {final_body[:500]!r}"
            )

    captured_names = [call.name for call in observed]
    print("captured_tool_names", captured_names)
    assert captured_names == EXPECTED_TOOL_ORDER

    # Ground truth is ordered tool names only. Do not copy dynamic args/refs.
    expected_tool_calls = [
        tool_invocation_from_observation(name=name, arguments=None, result=None)
        for name in EXPECTED_TOOL_ORDER
    ]
    trace = build_mcp_evaluation_trace(
        trace_id="pv-playwright-mcp-flagship-todomvc-e2e",
        input=goal,
        output="Added 'Buy milk' via the multi-step Playwright MCP flow.",
        expected="Added 'Buy milk' via the multi-step Playwright MCP flow.",
        observed_tool_calls=observed,
        expected_tool_calls=expected_tool_calls,
    )
    assert [call.name for call in trace.turns[1].tool_calls] == EXPECTED_TOOL_ORDER
    assert all(call.arguments is None for call in trace.expected_tool_calls)
    assert all(call.result is None for call in trace.expected_tool_calls)

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
    print("evaluation_params", [])
    print("policy_threshold", policy.threshold)

    decision = runner.run(
        trace,
        EvaluationConfig(evaluations=["tool_correctness"]),
        run_id="pv-playwright-mcp-flagship-todomvc-e2e",
    )

    # Evidence 2: ordered ToolCorrectness through Runner / QualityGate.
    tool_correctness_gate_passed = decision.passed is True
    assert tool_correctness_gate_passed
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert runner.last_run.traces[0].scenario_type == "mcp"
    result = runner.last_run.results[0]
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert result.score == 1.0
    assert result.score >= TOOL_CORRECTNESS_THRESHOLD
    assert result.reason is not None
    assert runner.last_run.decisions[0].passed is True
    print("tool_correctness_score", result.score)
    print("tool_correctness_reason", result.reason)
    print("tool_correctness_gate_passed", tool_correctness_gate_passed)

    # Scenario-level AND of the two separate evidence signals (no combined score).
    scenario_task_succeeded = tool_correctness_gate_passed and final_state_ok
    print("scenario_task_succeeded", scenario_task_succeeded)
    assert scenario_task_succeeded, (
        "Flagship task succeeds only when BOTH are true: "
        f"tool_correctness_gate_passed={tool_correctness_gate_passed}, "
        f"final_state_ok={final_state_ok}"
    )
