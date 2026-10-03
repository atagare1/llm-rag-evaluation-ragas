# Spike: ToolCorrectness without EvaluationTrace

Isolated experiment. It is **not** part of `src/ai_qe_eval`.

Question: can settled Langfuse TOOL observations be scored by DeepEval ToolCorrectness without building `EvaluationTrace`, and without Runner / Policy / Gate?

The System Under Test remains `spikes/langfuse_openai_agents`. This folder only maps TOOL rows into evaluator-specific input and calls `ToolCorrectnessMetric.measure`.

## Minimum evaluator input

```text
observed: ordered [(name, arguments?)]
expected: ordered [(name, arguments?)]   # QE spec, never inferred from telemetry
```

Names-only is enough for A→B→C vs A→C→B. Arguments are carried when present; this spike’s deterministic test does not score them (`evaluation_params=[]`).

DeepEval `LLMTestCase` still requires a dummy `input` string. That is not `EvaluationTrace.input`.

## Deterministic test

From the platform root:

```powershell
python -m pytest spikes/langfuse_tool_correctness_direct/test_direct_tool_correctness.py -p no:deepeval -o addopts= -q
```

Recorded observation dicts, no Langfuse/OpenRouter. Observed A→B→C vs expected A→B→C → score 1.0. Same observed vs expected A→C→B → score 0.0.

## Optional live path

Reuses the existing SUT spike and `wait_for_settled_observations`:

```powershell
python spikes/langfuse_tool_correctness_direct/run_live.py
```

Does not construct `EvaluationTrace`.
