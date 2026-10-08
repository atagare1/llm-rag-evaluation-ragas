# Architecture

Companion to `README.md`. README is the measured evidence and operator guide. This document explains how the implemented platform is structured and why.

Source of truth: `src/ai_qe_eval` and README. Do not treat this file as a results report or a Phase 1 history.

---

## 1. Problem and architectural goals

AI systems fail in different domains. A RAG answer can be unfaithful after successful retrieval. An agent can take the wrong tool trajectory. An MCP workflow can execute every tool without error and still reach the wrong application state. A judge can fail independently of the system under test.

The platform is one provider-agnostic evaluation engine for four independently gated paths: External RAG, Langfuse OpenAI Agent, Playwright MCP, and Langfuse user-feedback Chatbot.

Goals:

* Score captured evidence, not a single opaque model verdict.
* Keep SUT, evaluator, policy, and gate as separate responsibilities.
* Fail closed **per path**. Never average paths into one platform score.
* Isolate a judge outage from an application defect.

---

## 2. Design principles

* **Evidence in, score out.** Capture helpers extract observations. Evaluators score only the arguments they are given.
* **Thin runner.** `EvaluationRunner` does not rebuild evidence, apply thresholds, or invent a gate rule.
* **Provider-agnostic core.** Domain, normalizer, policy, gate, and runner do not import RAGAS or DeepEval. Adapters do.
* **QE-owned expected behavior.** Expected tool names and chatbot expected answers are supplied by the test, not inferred from telemetry.
* **One executor.** Every platform path goes through `EvaluationRunner`. Capture does not evaluate. Adapters do not gate.
* **Deterministic default.** `pytest tests -q` stays non-live. Live provider tests are opt-in.

---

## 3. High-level architecture

```text
SUT
  → evidence (tools, snapshots, retrieved contexts, conversation turns)
  → Design A request map   {"capability": {"args": [...], "kwargs": {...}}}
  → EvaluationRunner
  → Evaluator.evaluate(...)
  → EvaluationResult[]     (score is not pass/fail)
  → normalize_many         (copy only; scores unchanged)
  → QualityPolicy.apply    (threshold → PolicyDecision.passed)
  → QualityGate.evaluate   (any-fail for that path)
  → GateDecision
```

`EvaluationConfig` names the ordered capabilities. `EvaluationRegistry` is a catalog of names, not instances or thresholds. Evaluators and the policy map are injected.

`EvaluationRun` records the execution. `requests[i]` aligns with `trace_evaluations[i]`. Flat `results` / `decisions` are the same objects in request order — a view, not a second scoring model.

The runner does not rebuild a request map. Capture helpers sit **outside** the core and produce the map.

---

## 4. Four SUT paths

Each path has its own evidence shape and its own gate. They share the runner.

| Path | Evidence | Request map | Evaluated capabilities (implemented) |
|---|---|---|---|
| External RAG | Live HTTP answer + retrieved contexts | `live_rag_demo_request` | Faithfulness through the runner when the judge completes. Other RAG maps exist as adapters. |
| Langfuse OpenAI Agent | `get_many(trace_id)` `TOOL` rows → `ToolInvocation` | `langfuse_tool_correctness_request` | Tool Correctness; Final State. GENERATION G-Eval is unit-mapped only. |
| Playwright MCP | Serialized MCP tool results → `ToolInvocation`; last snapshot for state | `mcp_p0_request` | Tool Correctness; MCP Execution Health; Final State. CLI/web flagship. |
| Langfuse Chatbot | `get_many(session_id)` root `handle-chat-message` SPAN input/output → `ConversationTurn` | turn-relevancy and per-turn G-Eval maps | Turn Relevancy; per-turn G-Eval. Official example app is not in this repo. |

MCP is the interactive demonstration (`python -m ai_qe_eval`, `python -m ai_qe_eval.demo.web`). RAG, Agent, and Chatbot use the same engine through integration and live tests, not CLI scenarios.

---

## 5. Evidence-driven evaluation — no universal `EvaluationTrace`

Evaluators accept `*args, **kwargs` and return `EvaluationResult` lists. Evidence is evaluator-specific:

* RAG metrics need question, answer, and contexts (and a reference when the metric requires one).
* Tool Correctness needs observed and QE-expected invocations.
* Execution Health needs observed `isError` outcomes.
* Final State needs a caller-computed boolean (for example, whether the last snapshot contains `Buy milk`).
* Turn Relevancy and chatbot G-Eval need ordered `ConversationTurn` pairs.

A universal `EvaluationTrace` was removed from the production package. One bag-of-fields object either leaves most fields unused or forces every adapter to pretend the SUT looks the same. Design A request maps keep evidence at the capture boundary and keep the runner generic.

`ToolInvocation` and `ConversationTurn` remain as small, provider-neutral domain types. They are not a substitute trace.

---

## 6. Evaluator categories

Adapters return scores. They do not apply `QualityPolicy` or `QualityGate`.

**Deterministic** — no LLM judge. `exact_match`, `mcp_execution_health` (fail on `isError=true`), `final_state` (caller bool). Used to prove that tool success ≠ outcome success.

**RAG / LLM** — RAGAS Faithfulness; DeepEval faithfulness, answer relevancy, contextual relevancy/precision/recall, hallucination, G-Eval correctness. RAGAS Faithfulness is the only Phase 1 RAGAS metric with a Phase 2 adapter.

**Conversational** — DeepEval Turn Relevancy and per-turn G-Eval on chatbot root-SPAN turns.

**Agent / tool / MCP** — DeepEval Tool Correctness (name/order; expected names are QE-supplied). Combined with Execution Health and Final State on MCP so a healthy tool sequence can still fail the gate.

The default CLI runs only the three MCP P0 evaluators.

---

## 7. Failure-domain isolation and fail-closed behavior

Isolation means a failure is attributed to the layer that actually failed:

| Domain | Example from this platform |
|---|---|
| SUT / application | MCP types `Buy bread`; Final State fails while tools and health pass. |
| Trajectory | Agent or MCP wrong tool order; Tool Correctness fails. |
| Execution | MCP `isError=true`; Execution Health fails. |
| Extraction | Capture could not build a request (missing root input/output, unset session). No score is invented. |
| Judge | RAGAS Faithfulness raised `LLMDidNotFinishException` after a successful RAG HTTP 200. **JUDGE_BLOCKED / NOT SCORED** — not FAIL `0.0`, not an RAG application failure. |

Fail-closed: any `PolicyDecision.passed is False` fails that path’s `QualityGate`. A missing score is not coerced to `0.0`. An empty decision list currently passes (see limitations).

---

## 8. QualityPolicy vs QualityGate

| | `QualityPolicy` | `QualityGate` |
|---|---|---|
| Input | One `EvaluationResult` | The list of `PolicyDecision`s |
| Owns | Operator and threshold (`>`, `>=`, `<`, `<=`, `==`, `!=`) | Any-fail aggregation |
| Output | `PolicyDecision.passed` | `GateDecision.passed` |
| Does not | Execute evaluators, mutate scores, release software | Re-check scores, average paths, talk to CI |

`EvaluationResult.score` is not pass/fail. Policy is not the release decision. The gate is an in-process decision for **one** run, not GitHub Actions, Jenkins, or a combined score across SUTs.

---

## 9. Live validation approach

Evidence classes stay distinct: unit / doubles, deterministic smoke, live provider, and judge-blocked.

* `pytest tests -q` is the Phase 2 regression gate (non-live).
* `@pytest.mark.live` is opt-in. It is the proof for real MCP, Langfuse, and RAG I/O — not the stub adapter tests.
* Positive cases show a path can pass its policies.
* Negative cases are first-class: wrong tool order, wrong expected title, wrong final state, execution error. They prove the gate can fail for the intended reason.
* Phase 1 root RAGAS tests do not enter `EvaluationRunner` and are not platform proof.

Measured outcomes live in README (final live MVP matrix). This document does not restate them as new results.

---

## 10. Key architectural decisions

**Request maps instead of `EvaluationTrace`**

* Problem: four SUTs produce incompatible evidence.
* Options considered: one universal trace; per-evaluator request maps (Design A).
* Decision: Design A. `EvaluationTrace` is gone from production.
* Trade-off: each path needs a capture helper. Gain: the runner never special-cases RAG vs MCP vs chat.

**Provider-agnostic core**

* Problem: RAGAS and DeepEval APIs must not leak into policy or gate.
* Decision: adapters only at the evaluator boundary.
* Trade-off: more mapping code; the core stays testable without a live model.

**Independent per-path gates**

* Problem: MCP can pass while RAG’s judge is blocked.
* Decision: no combined platform score, pass rate, or average.
* Trade-off: no single headline number for a dashboard. Gain: honest attribution.

**QE-supplied expected tools**

* Problem: inferring expected calls from telemetry makes Tool Correctness tautological.
* Decision: expected names come from the test/spec (`evaluation_params=[]` name/order on the default CLI path).
* Trade-off: tests must state intent. Gain: wrong trajectory is detectable.

**MCP as the only interactive flagship**

* Problem: wiring RAG, Agent, and Chatbot into CLI would add process orchestration (browser, Langfuse, Next.js), not evaluation value.
* Decision: CLI/web stay MCP; other paths stay pytest.
* Trade-off: visitors do not click through all four SUTs. Gain: one coherent demo of “tools pass, state fails.”

**Thin runner, injected dependencies**

* Problem: a “smart” orchestrator tends to hide scoring and gating.
* Decision: `run` / `run_many` resolve names, call evaluators, normalize, apply policies, gate once.
* Consequence: a runner exception clears `last_run` for that call. Empty `run_many([])` raises `ValueError`.

---

## 11. Current limitations

Documented in README; not defects to paper over.

* Latest live RAGAS Faithfulness is **JUDGE_BLOCKED / NOT SCORED** (`LLMDidNotFinishException` / `max_tokens`) after SUT HTTP 200 with a valid answer and 4 contexts.
* Agent GENERATION G-Eval and non-faithfulness RAG maps are unit-tested only.
* Remaining Phase 1 RAGAS metrics have no Phase 2 adapter.
* `QualityGate` is not mapped to CI. Empty decision list passes — a miswired empty config would look green.
* No persistence, lineage, or dashboard. `EvaluationRun.to_dict` exists; nothing aggregates separate SUT runs.
* CLI/web do not run RAG or Langfuse.
* Historical Phase 1 root tests remain a second path and currently 401 on OpenRouter. They are not this architecture.

Extend Phase 2 contracts. Do not grow new framework behavior by editing the original RAGAS scripts.
