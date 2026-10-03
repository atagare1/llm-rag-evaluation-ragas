"""Optional live path: existing SUT spike → settle → spike-local ToolCorrectness.

Does not construct EvaluationTrace or call EvaluationRunner.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

SPIKE_DIR = Path(__file__).resolve().parent
PLATFORM_ROOT = SPIKE_DIR.parent.parent
if str(SPIKE_DIR) not in sys.path:
    sys.path.insert(0, str(SPIKE_DIR))
_SRC = PLATFORM_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evaluate import (
    ORDER_ABC,
    ORDER_ACB,
    names_only,
    observed_tool_calls_from_observations,
    score_tool_correctness,
)

SUT_DIR = PLATFORM_ROOT / "spikes" / "langfuse_openai_agents"
SUT_SCRIPT = SUT_DIR / "run_spike.py"
TRACE_ID_PATH = SUT_DIR / "artifacts" / "last_trace_id.txt"
SHORT_VENV_PYTHON = Path(r"C:\Users\aarya\.venvs\lf-oa-spike\Scripts\python.exe")
LOCAL_SUT_PYTHON = SUT_DIR / ".venv" / "Scripts" / "python.exe"


def _sut_python() -> Path:
    if SHORT_VENV_PYTHON.exists():
        return SHORT_VENV_PYTHON
    if LOCAL_SUT_PYTHON.exists():
        return LOCAL_SUT_PYTHON
    return Path(sys.executable)


def main() -> int:
    load_dotenv(PLATFORM_ROOT / ".env")
    required = (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "OPENROUTER_API_KEY",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        print("Missing live credentials:", ", ".join(missing), file=sys.stderr)
        return 2
    if not SUT_SCRIPT.exists():
        print(f"SUT spike not found: {SUT_SCRIPT}", file=sys.stderr)
        return 2

    from langfuse import get_client

    from ai_qe_eval.integrations.langfuse_observations import (
        flush_langfuse_client,
        wait_for_settled_observations,
    )

    started_at = datetime.now(timezone.utc)
    completed = subprocess.run(
        [str(_sut_python()), str(SUT_SCRIPT)],
        cwd=str(SUT_DIR),
        check=False,
    )
    if completed.returncode != 0:
        print("SUT spike failed", file=sys.stderr)
        return completed.returncode

    trace_id = TRACE_ID_PATH.read_text(encoding="utf-8").strip()
    client = get_client()
    if not client.auth_check():
        print("Langfuse auth_check() failed", file=sys.stderr)
        return 2
    flush_langfuse_client(client)

    def is_complete(rows: list) -> bool:
        names = [
            str(row.get("name") or "")
            for row in rows
            if str(row.get("type") or "").upper() == "TOOL"
        ]
        return all(name in names for name in ORDER_ABC)

    observations = wait_for_settled_observations(
        client,
        trace_id,
        from_start_time=started_at - timedelta(minutes=5),
        is_complete=is_complete,
        timeout_s=45.0,
        poll_interval_s=0.5,
        page_limit=2,
    )
    observed = observed_tool_calls_from_observations(
        observations, trace_id=trace_id
    )
    print("langfuse_trace_id", trace_id)
    print("observed_names", [call.name for call in observed])

    pass_score, pass_reason = score_tool_correctness(
        observed=observed,
        expected=names_only(ORDER_ABC),
    )
    fail_score, fail_reason = score_tool_correctness(
        observed=observed,
        expected=names_only(ORDER_ACB),
    )
    print("abc_score", pass_score)
    print("abc_reason", pass_reason)
    print("acb_score", fail_score)
    print("acb_reason", fail_reason)
    if pass_score != 1.0 or fail_score != 0.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
