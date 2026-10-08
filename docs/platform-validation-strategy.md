# Platform Validation Test Strategy

**Repository:** `llm-rag-evaluation-ragas`  
**Document type:** validation strategy (implemented platform)  
**Status:** platform validation exists under `tests/` and `tests/platform/`. Live paths are opt-in (`@pytest.mark.live`).

This document describes how to **validate** the platform that already exists. Counts and live outcomes below are copied from `README.md`. Do not invent additional results.

Current Phase 2 regression baseline (`pytest tests -q`; excludes root Phase 1 files; `pytest.ini` applies `-m "not live"` and `-p no:deepeval`):

**392 passed, 35 deselected, 1 warning** in 19.30s.

`python -m pytest` also collects four unmarked Phase 1 root tests. Those are not `@pytest.mark.live` and they are not `EvaluationRunner` tests. With the current OpenRouter `.env` they fail 401 (`Missing Authentication header`). Last measured full collection: **395 passed, 4 failed, 35 deselected, 1 warning** in 33.67s. That is a Phase 1 provider-auth issue, not a Phase 2 regression.

---

## 1. Purpose

Prove, with evidence classes kept distinct, that:

1. Phase 2 contracts still hold.
2. Capture helpers compose with `EvaluationRunner` across the four SUT paths.
3. The runner pipeline works end-to-end.
4. Deterministic evaluators work without a network.
5. Provider adapters work **when** they are live-validated; stub tests are not live proof.
6. Both policy PASS and FAIL, and gate pass/fail, are observable **per path**.
7. Request maps stay aligned with `trace_evaluations` on `EvaluationRun`. There is no combined platform score.

Do **not** treat stub tests as live provider proof. Do **not** treat Phase 1 root tests as proof that `EvaluationRunner` works. Do **not** treat a judge exception as an SUT failure.

---

## 2. What exists today (source of truth: README)

### 2.1 Execution path (actual)

Four SUT paths produce Design A request maps. `EvaluationRunner` is the only executor. `QualityGate` is fail-closed **per path**.

```text
External RAG / Langfuse Agent / Playwright MCP / Langfuse Chatbot
        ↓
Design A request map + EvaluationConfig
        ↓
EvaluationRunner.run | run_many
        ↓
Evaluator.evaluate(*args, **kwargs) → EvaluationResult[]
        ↓
normalize_many
        ↓
QualityPolicy.apply → PolicyDecision[]
        ↓
QualityGate.evaluate → GateDecision   (per path; any-fail)
```

`requests[i]` aligns with `trace_evaluations[i]`. One `QualityGate.evaluate` call uses the flat decision list (any `passed is False` fails **that** run). No average, pass rate, or combined platform score.

Playwright MCP is the interactive CLI/web flagship. External RAG, Langfuse Agent, and the Chatbot are validated through integration and live tests, not CLI scenarios.

### 2.2 Two evaluation paths

| Path | Location | Enters runner? |
|---|---|---|
| Phase 1 live RAGAS POC | `utils.py`, `conftest.py`, root `test_*.py` | **No** |
| Phase 2+ platform | `src/ai_qe_eval/` | **Yes** |

Phase 1 thresholds live in `utils.py` (`RAGAS_THRESHOLD_*`). Phase 2+ thresholds live only on `QualityPolicy`. They are not the same system.

### 2.3 Implemented evaluators

Adapters return `EvaluationResult`. They do not apply `QualityPolicy` or `QualityGate`.

**Deterministic:** `exact_match`, `mcp_execution_health`, `final_state`.

**RAGAS:** `faithfulness` only (`RAGASFaithfulnessEvaluator`). Other Phase 1 RAGAS metrics are not Phase 2 adapters.

**DeepEval:** `correctness`, `faithfulness`, `answer_relevancy`, `contextual_relevancy`, `contextual_precision`, `contextual_recall`, `hallucination`, `tool_correctness`, `turn_relevancy`.

The default CLI runs only the three P0 evaluators: `tool_correctness`, `mcp_execution_health`, `final_state`.

### 2.4 Current test organization

| Layer | Files |
|---|---|
| Domain contracts | `tests/test_evaluation_*.py`, `test_evaluator_contract.py` |
| Deterministic adapters | `tests/test_deterministic_evaluator.py` |
| Provider adapters (stubs) | `tests/test_ragas_evaluator.py`, `tests/test_deepeval_*.py` |
| Runner / policy / gate | `tests/test_evaluation_runner.py`, `test_quality_policy.py`, `test_quality_gate.py` |
| CLI / MCP demo | `tests/test_cli.py`, `tests/test_cli_mcp_p0_demo.py` |
| Platform validation | `tests/platform/test_pv_*.py` (includes `@pytest.mark.live` MCP / Langfuse / RAG / provider tests) |
| Phase 1 live (historical) | root `test_context_precision.py`, `test_context_recall.py`, `test_faithfulness.py`, `test_resp_relevancy_factual_correctness.py` |

`pytest.ini`: `testpaths = .`, `pythonpath = src`, `addopts` includes `-p no:deepeval` and `-m "not live"`. `pytest tests -q` does not collect root Phase 1 files. `python -m pytest` does.

### 2.5 Coverage that already exists (do not duplicate blindly)

Already proven with doubles or local deterministic code (do not treat as live-provider proof):

- Config selects capabilities; order preserved.
- Multiple `EvaluationResult`s from one evaluator.
- Normalize before policy; policy owns threshold; gate uses `passed` only.
- Missing capability / missing instance / missing policy → `KeyError`.
- Empty `EvaluationConfig` → no evaluators, `gate.evaluate([])` → PASS.
- `run()` records `last_run`. `run_many` keeps per-trace groups; empty list → `ValueError`.
- Synthetic PASS and FAIL for policy and gate.

Live through `EvaluationRunner` (README final live MVP matrix):

| Path | Live through EvaluationRunner | Result |
|---|---|---|
| Langfuse OpenAI Agent | Tool Correctness; Final State | **Pass**, including expected-negative cases |
| Playwright MCP | Tool Correctness; MCP Execution Health; Final State | **Pass**, including expected-negative cases |
| Langfuse user-feedback chatbot | Turn Relevancy; per-turn G-Eval | Turn Relevancy **1.0**; G-Eval **0.8 / 0.8**; QualityGate **pass** |
| External RAG Faithfulness | SUT extraction, then RAGAS Faithfulness | **JUDGE_BLOCKED / NOT SCORED** |

Agent GENERATION G-Eval and non-faithfulness RAG maps remain unit-tested only.

---

## 3. Validation principles

1. **Classify evidence:** `UNIT` | `INTEGRATION (doubles)` | `DETERMINISTIC SMOKE` | `LIVE PROVIDER` | `JUDGE_BLOCKED / NOT SCORED` | `NOT RUN`.
2. **Do not convert** a stub score into a live score claim.
3. **Do not change** Phase 1 live tests, Mixtral default, or gold `"23"` to make a validation suite green.
4. **Do not add** score aggregation or a combined platform score.
5. Live tests stay **opt-in** (`pytest -m live --override-ini="addopts=-p no:deepeval"`) so `pytest tests -q` stays deterministic.
6. Preserve evaluator vs policy vs gate: a live score is still only an `EvaluationResult`; PASS/FAIL is still `PolicyDecision` / `GateDecision`.
7. Isolate failure domains: a judge exception is not an SUT or extraction failure.

---

## 4. Validation layers

### Layer A — Phase 2 regression (implemented)

Command: `pytest tests -q`  
Purpose: no regression of Phase 2 contracts.  
Measured baseline: **392 passed, 35 deselected, 1 warning** in 19.30s. The warning is a DeepEval `HallucinationMetric` score-direction notice; platform hallucination scoring is already higher-is-better with policy `>= 0.8`.

### Layer B — Cross-boundary composition (implemented)

Real `EvaluationRunner` + real policy/gate + capture helpers. Highest-value **non-network** proof remains deterministic `exact_match` / MCP scripted paths. Live composition is the four SUT paths in Layer E/F below.

### Layer C — PASS / FAIL matrix (implemented)

Policy and gate cover:

- Boundary `score == threshold` with `>=` → policy PASS.
- All policies PASS → gate PASS.
- One policy FAIL → gate FAIL (any-fail), including MCP/Agent expected-negative cases.

Do not add severity, blocking, or pass-rate rules.

### Layer D — Negative / failure transparency (implemented)

Re-assert, do not redesign:

- Unknown capability, unwired evaluator, missing policy → `KeyError`.
- Evaluator exception → no successful scored run for that call.
- `run_many([])` → `ValueError`.

### Layer E — Live RAGAS Faithfulness through the runner (attempted)

Live DeepEval and RAGAS evaluation paths are configured for OpenRouter. The final RAGAS Faithfulness run was judge-blocked by `LLMDidNotFinishException`.

**RAG diagnostic lesson.** The external RAG API returned **HTTP 200** with a valid answer and **4** retrieved contexts. The downstream RAGAS judge then failed with `LLMDidNotFinishException` (`max_tokens`). That is failure-domain isolation: the SUT succeeded; the judge did not. No score and no QualityGate. This is not an RAG application failure and not a Faithfulness score.

Do **not** assert production quality of the RAG demo. Do **not** claim this replaces Phase 1’s five metrics.

### Layer F — Live DeepEval (implemented for the measured paths)

Pin: `deepeval==4.2.6`. Pytest plugin remains disabled (`-p no:deepeval`). Live tests stay `live`-marked. Judge config: OpenRouter `meta-llama/llama-3.3-70b-instruct` via `DEEPEVAL_JUDGE_MODEL` / `OPENROUTER_API_KEY` / `OPENAI_BASE_URL`.

Measured live through the runner: Agent Tool Correctness + Final State; MCP Tool Correctness + Execution Health + Final State; chatbot Turn Relevancy + per-turn G-Eval. Agent GENERATION G-Eval is unit-tested only.

### Layer G — Phase 1 live suite (historical)

Keep as-is. Unmarked root files. Do not migrate into `EvaluationRun`. Current measured failure is 401 (`Missing Authentication header`) on OpenRouter with the Together-era Mixtral + `OPENAI_API_KEY` combination. Historically, Together returned `model_not_available` for the same Mixtral id. That Together wording is history; the current measured failure is the 401.

---

## 5. PASS / FAIL and single / multi-request matrix

| | Single `run()` | `run_many` |
|---|---|---|
| All policies PASS | Deterministic / live path gate PASS | Same per request; run gate PASS only if every decision passes |
| One policy FAIL | Gate FAIL (any-fail) | Gate FAIL; failing decision stays on the correct `TraceEvaluation` |
| Empty config | Existing: evaluators not called; gate `[]` PASS | Same if config empty |
| Empty request list | N/A | Existing: `ValueError` |

There is no combined score across the four SUT paths.

---

## 6. What validation will **not** prove

- Remaining Phase 1 RAGAS metrics as Phase 2 adapters.
- Live Langfuse G-Eval for the OpenAI Agent GENERATION path.
- Live scores for non-faithfulness RAG maps.
- A successful RAGAS Faithfulness score (latest run is **JUDGE_BLOCKED / NOT SCORED**).
- CI/CD blocking on `GateDecision`.
- Cost, latency, tokens, ROI.
- Dataset-level pass rate or average / combined platform score.
- CLI coverage beyond the P0 MCP demo.

---

## 7. Recommended execution order

1. `pytest tests -q` is the merge gate for deterministic Phase 2 work (392 / 35 deselected / 1 warning today).
2. `pytest -m live --override-ini="addopts=-p no:deepeval"` for live platform tests when configured.
3. Do not treat `python -m pytest` Phase 1 401s as a Phase 2 regression.
4. Leave Phase 1 root tests and Mixtral defaults untouched.

---

## 8. Risks to call out before CI

| Risk | Implication for validation |
|---|---|
| Dual paths (Phase 1 pytest vs runner) | CI must say which command is “platform” vs “legacy live.” |
| Phase 1 401 on OpenRouter | Full `python -m pytest` stays red unless those unmarked tests skip or use a matching Phase 1 host. Do not silently rewrite Phase 1 defaults. |
| RAGAS Faithfulness judge block | A live RAG path can extract successfully and still produce no score. Record `JUDGE_BLOCKED / NOT SCORED`, not FAIL `0.0`. |
| Empty config → gate PASS | A miswired CI job with `evaluations=[]` would pass; CI design (later) must not treat that as a quality bar. |
| No combined platform score | Four path gates must stay independent. |

---

## 9. Scope of this document

Documentation only. Measured evidence lives in `README.md`.

```text
Production code modified: NO
Existing tests modified: NO
Dependencies modified: NO
CI/CD modified: NO
Validation suite: implemented (see tests/ and tests/platform/)
```
