"""Playwright MCP stdio integration boundary.

Starts/uses the official @playwright/mcp package through the installed Python
mcp ClientSession transport helpers. Serializes MCP SDK result objects into
plain Python values for the domain capture layer.

Does not import EvaluationTrace builders, evaluators, Runner, Policy, or Gate.
"""

from __future__ import annotations

from typing import Any

from mcp import StdioServerParameters
from mcp.types import CallToolResult

PLAYWRIGHT_MCP_PACKAGE = "@playwright/mcp@0.0.82"
TODO_MVC_URL = "https://demo.playwright.dev/todomvc"


def playwright_mcp_stdio_parameters(
    *,
    headless: bool = True,
    isolated: bool = True,
) -> StdioServerParameters:
    """Return stdio launch parameters for the official Playwright MCP server."""
    args = ["--yes", PLAYWRIGHT_MCP_PACKAGE]
    if headless:
        args.append("--headless")
    if isolated:
        args.append("--isolated")
    return StdioServerParameters(command="npx", args=args)


def serialize_call_tool_result(result: CallToolResult) -> dict[str, Any]:
    """Convert an MCP CallToolResult into plain JSON-friendly data.

    Domain ToolInvocation.result must not hold MCP SDK objects.
    """
    if not isinstance(result, CallToolResult):
        raise TypeError(
            "serialize_call_tool_result requires CallToolResult, "
            f"got {type(result).__name__}"
        )
    content: list[Any] = []
    for item in result.content or []:
        if hasattr(item, "model_dump"):
            content.append(item.model_dump(mode="json"))
        else:
            content.append({"type": type(item).__name__, "repr": repr(item)})
    return {
        "isError": bool(result.isError),
        "content": content,
        "structuredContent": result.structuredContent,
    }
