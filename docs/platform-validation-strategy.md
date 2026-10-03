# Platform Validation Test Strategy

**Repository:** `llm-rag-evaluation-ragas`  
**Document type:** planning only (PV-PLAN)  
**Status:** proposed suite — **not implemented**  
**Implementation freeze:** Phase 2 COMPLETE / FROZEN. P3-01, P3-02, P3-03 complete.

This document describes how to **validate** the platform that already exists. It does not change production code, existing tests, dependencies, or CI.

Last measured deterministic baseline (P3-03): `pytest tests -q` → **221 passed**.  
Full `pytest -q` additionally collects four Phase 1 live RAGAS tests that fail when Together Mixtral serverless is unavailable. Those failures are **historical / live-integration**, not Phase 2 architectural defects.

---

## 1. Purpose

Prove, with evidence classes kept distinct, that:

1. Each phase’s contracts still hold.
2. Phase 2 + P3-02/P3-03 compose across component boundaries.
3. The runner pipeline works end-to-end.
4. `DeterministicEvaluator` works without a network.
5. Provider adapters work **when** they are live-validated (they are not, today).
6. Both policy PASS and FAIL, and gate pass/fail, are observable.
7. One trace and many traces in one `EvaluationRun` remain associated correctly.

Do **not** treat stub tests as live provider proof. Do **not** treat Phase 1 live tests as proof that `EvaluationRunner` works.

---

## 2. What exists today (source of truth)

### 2.1 Execution path (actual)

```text
request map + EvaluationConfig
        ↓
EvaluationRunner.run | run_many
        ↓
EvaluationRegistry.get(capability name)
        + injected Evaluator instance
        ↓
Evaluator.evaluate(*args, **kwargs) → EvaluationResult[]
        ↓
normalize_many
        ↓
QualityPolicy.apply → PolicyDecision[]
        ↓
QualityGate.evaluate → GateDecision
        ↓
EvaluationRun on runner.last_run
```

P3-03: `traces[i]` aligns with `trace_evaluations[i]`. Flat `results` / `decisions` are the same objects in trace order. One `QualityGate.evaluate` call uses the **flat** decision list (any `passed is False` fails the run). No average, pass rate, or other run score.

### 2.2 Two evaluation paths

| Path | Location | Enters runner? |
|---|---|---|
| Phase 1 live RAGAS POC | `utils.py`, `conftest.py`, root `test_*.py` | **No** |
| Phase 2/3 core | `src/ai_qe_eval/` | **Yes** |

Phase 1 thresholds live in `utils.DEFAULT_METRIC_THRESHOLDS` / `RAGAS_THRESHOLD_*`. Phase 2/3 thresholds live only on `QualityPolicy`. They are not the same system.

### 2.3 Implemented evaluators

| Class | metric | evaluator string | Live through runner |
|---|---|---|---|
| `DeterministicEvaluator` | `exact_match` | `deterministic` | Not required (pure Python). Smoke exists in `tests/test_end_to_end.py`. |
| `RAGASFaithfulnessEvaluator` | `faithfulness` | `ragas` | **No.** Stubbed in `tests/test_ragas_evaluator.py`. |
| `DeepEvalGEvalCorrectnessEvaluator` | `correctness` | `deepeval` | **No.** Stubbed in `tests/test_deepeval_evaluator.py`. GEval is the mechanism; identity is not `"geval"`. |

Other Phase 1 RAGAS metrics (context precision, context recall, answer relevancy, factual correctness) have **no** Phase 2 adapter.

### 2.4 Current test organization

There is **no** shared `tests/conftest.py` for the core. Fakes live inside each file (`FakeEvaluator`, `RecordingEvaluator`, `StubEvaluator`, `RecordingFaithfulnessMetric`, `RecordingGEvalMetric`).

| Layer | Files |
|---|---|
| Domain | `test_evaluation_trace.py`, `test_trace_events.py`, `test_evaluation_run.py`, `test_evaluation_result.py`, `test_evaluator_contract.py`, `test_evaluation_registry.py`, `test_evaluation_config.py` |
| Adapters | `test_deterministic_evaluator.py`, `test_ragas_evaluator.py`, `test_deepeval_evaluator.py` |
| Normalize / policy / gate | `test_result_normalizer.py`, `test_quality_policy.py`, `test_quality_gate.py` |
| Runner | `test_evaluation_runner.py` |
| P2 E2E doubles | `test_end_to_end.py` |
| P3-02 | `test_evaluation_run_integration.py` |
| P3-03 | `test_multi_trace_run.py` |
| Phase 1 unit | `test_dataset_loading.py`, `test_response_mapping.py`, `test_config_and_api_client.py` |
| Phase 1 live | root `test_context_precision.py`, `test_context_recall.py`, `test_faithfulness.py`, `test_resp_relevancy_factual_correctness.py` |

`pytest.ini`: `testpaths = .`, `pythonpath = src`, `addopts = -p no:deepeval`. Collection includes **both** `tests/` and root live tests.

### 2.5 Coverage that already exists (do not duplicate blindly)

Already proven with doubles or local deterministic code:

- Trace identity into evaluator; config selects capabilities; order preserved.
- Multiple `EvaluationResult`s from one evaluator.
- Normalize before policy; policy owns threshold; gate uses `passed` only.
- Missing capability / missing instance / missing policy → `KeyError`.
- Empty `EvaluationConfig` → no evaluators, `gate.evaluate([])` → PASS.
- `run()` records `last_run` (`GateDecision` return value unchanged).
- `run_many` keeps per-trace groups; empty list → `ValueError`; evaluator exception → no `last_run`.
- Synthetic PASS and FAIL for policy and gate.

**Not proven:** RAGAS Faithfulness or DeepEval GEval **through** `EvaluationRunner` against a real model. Phase 1 live tests do not fill that gap.

---

## 3. Validation principles

1. **Classify evidence:** `UNIT` | `INTEGRATION (doubles)` | `DETERMINISTIC SMOKE` | `LIVE PROVIDER` | `NOT RUN`.
2. **Do not convert** a stub score into a live score claim.
3. **Do not change** Phase 1 live tests, Mixtral default, or gold `"23"` to make a validation suite green.
4. **Do not add** score aggregation, CI product gates, Langfuse, MCP, or new adapters as part of validation.
5. Live tests must **skip** without credentials (`OPENAI_API_KEY`), matching Phase 1 `conftest.py`.
6. Live tests must be **opt-in** (marker, e.g. `pytest -m live`) so `pytest tests -q` stays deterministic. **PROPOSED** for a future implementation increment; not created in PV-PLAN.
7. Preserve evaluator vs policy vs gate: a live score is still only an `EvaluationResult`; PASS/FAIL is still `PolicyDecision` / `GateDecision`.

---

## 4. Proposed validation layers (not yet written)

When a later increment implements this plan, prefer **new** files under something like `tests/platform/` so existing tests stay frozen. Names below are **PROPOSED**.

### Layer A — Phase regression (already the 221)

Command: `pytest tests -q`  
Purpose: no regression of frozen contracts.  
**No new tests required** unless a later code change lands.

### Layer B — Cross-boundary composition

**Goal:** one test module that wires **real** `DeterministicEvaluator` + real `QualityPolicy` + real `QualityGate` + real `normalize_many` + `EvaluationRunner`, without fakes for those layers.

Scenarios:

| ID | Scenario | Expected |
|---|---|---|
| PV-B1 | Single trace, `exact_match` PASS (`output == expected`) | `GateDecision.passed is True`; `last_run.results[0].score == 1.0` |
| PV-B2 | Single trace, `exact_match` FAIL | `passed is False`; score `0.0` |
| PV-B3 | Two traces, same config: one match, one mismatch | `trace_evaluations[0]` PASS, `[1]` FAIL; run gate FAIL (any-fail) |
| PV-B4 | Two capabilities if a second **deterministic** stub is used — **or** only exact_match if no second real deterministic adapter | Do not invent a second live provider here |

This is the highest-value **non-network** platform proof. `test_end_to_end.py` already smokes one matching/mismatching exact_match path; Layer B should add **run_many** with the real deterministic evaluator (gap vs `test_multi_trace_run.py`, which uses stubs).

### Layer C — PASS / FAIL matrix (policy + gate)

Use synthetic `EvaluationResult`s or deterministic scores. Cover:

- Boundary `score == threshold` with `>=` → policy PASS (already in `test_quality_policy.py`).
- Multi-trace: all policies PASS → gate PASS.
- Multi-trace: one policy FAIL → gate FAIL, and the failing decision stays on the correct `TraceEvaluation`.

Do not add severity, blocking, or pass-rate rules.

### Layer D — Negative / failure transparency

Already largely covered. Platform suite should **re-assert** (not redesign):

- Unknown capability, unwired evaluator, missing policy → `KeyError`, `last_run is None`.
- Evaluator `RuntimeError` → propagates; no partial `EvaluationRun`.
- `run_many([])` → `ValueError`.

### Layer E — Live RAGAS Faithfulness through the runner (optional)

**Status today:** NOT RUN.

**Prerequisite:** a Together (or compatible) model that actually serves, injected into `RAGASFaithfulnessEvaluator(llm=...)`. Do **not** silently change `.env.example` Mixtral default.

**Proposed test (skip without key / skip on `model_not_available`):**

1. Build `EvaluationTrace` with `input`, `output`, `retrieval` (`page_content` list), matching Phase 1 mapping semantics.
2. Register capability `faithfulness`; wire `RAGASFaithfulnessEvaluator`; policy e.g. `faithfulness >= 0.0` for “pipeline completed” **or** a documented experimental threshold labeled **EXPERIMENTAL**.
3. `runner.run(trace, config)`.
4. Assert: result `metric=="faithfulness"`, `evaluator=="ragas"`, score is numeric; `last_run.gate_decision` exists.

Do **not** assert production quality of the RAG demo. Do **not** claim this replaces Phase 1’s five metrics.

If Mixtral remains unavailable: keep Layer E skipped; record `LIVE PROVIDER E2E: NOT RUN` with the Together error class.

### Layer F — Live DeepEval GEval (optional)

**Status today:** NOT RUN. Environment constraint: DeepEval 2.7.0 vs frozen RAGAS/OpenAI stack; pytest plugin disabled.

Treat as **DEFERRED** until a isolated extra env exists. Do not `pip install --upgrade deepeval` as part of validation.

### Layer G — Phase 1 live suite (historical)

Keep as-is. Command: `pytest test_context_precision.py ...` when `OPENAI_API_KEY` is set.  
Do not migrate into `EvaluationRun`. Do not use as evidence that P3 multi-trace works.

---

## 5. PASS / FAIL and single / multi-trace matrix

| | Single `run()` | `run_many` (≥2 traces) |
|---|---|---|
| All policies PASS | PV-B1 (deterministic) | PV-B3 variant all-match |
| One policy FAIL | PV-B2 | PV-B3 mixed |
| Empty config | Existing: evaluators not called; gate `[]` PASS | Same per trace if config empty (zero results per trace, then gate `[]` PASS) — **verify in Layer B**; do not change gate |
| Empty traces | N/A | Existing: `ValueError` |

---

## 6. What validation will **not** prove

- Remaining RAGAS metrics as adapters.
- DeepEval live correctness.
- CI/CD blocking on `GateDecision`.
- Cost, latency, tokens, ROI.
- Agent, security, MCP, observability.
- Dataset-level pass rate or average score.
- Event/`turns` evaluation (fields exist; unused by adapters).

---

## 7. Recommended execution order (when implementing)

1. Keep `pytest tests -q` as the merge gate for deterministic work (221 today).
2. Implement Layer B + C + D in a **later** increment (new test module only).
3. Add `pytest.mark.live` for Layer E only after a working judge model is chosen **without** rewriting Phase 1 file defaults as a silent “fix.”
4. Defer Layer F.
5. Leave Phase 1 live tests untouched.

---

## 8. Risks to call out before CI

| Risk | Implication for validation |
|---|---|
| Dual paths (Phase 1 pytest vs runner) | CI must say which command is “platform” vs “legacy live.” |
| Mixtral `model_not_available` | Full `pytest -q` stays red unless live tests skip or a working model is used in **process env**, not a silent default rewrite. |
| `EvaluationResult` has no `trace_id` | Multi-trace assertions must use `trace_evaluations`, not guess from the flat list. |
| Eager `import ragas` via package `__init__` | Collecting `ai_qe_eval` requires RAGAS installed; already true in this repo. |
| Empty config → gate PASS | A miswired CI job with `evaluations=[]` would pass; CI design (later) must not treat that as a quality bar. |

---

## 9. Deliverable of this increment

This file only.

```text
Production code modified: NO
Existing tests modified: NO
Dependencies modified: NO
CI/CD modified: NO
Validation suite implemented: NO
```

Next increment (not PV-PLAN): implement Layers B–D as tests, then optionally Layer E behind a skip marker.
