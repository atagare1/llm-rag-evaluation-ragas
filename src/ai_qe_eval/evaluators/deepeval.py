"""P2-10 DeepEval GEval correctness evaluator adapter.

Maps EvaluationTrace into DeepEval 2.7.0 LLMTestCase fields and GEval.measure,
then returns one EvaluationResult.

GEval is the DeepEval mechanism; evaluator identity is "deepeval".
Does not apply framework thresholds or PASS/FAIL. Does not auto-register.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.trace import EvaluationTrace

CORRECTNESS_METRIC = "correctness"
DEEPEVAL_EVALUATOR_NAME = "deepeval"
DEFAULT_GEVAL_NAME = "Correctness"
DEFAULT_CRITERIA = (
    "Determine whether the actual output is factually correct "
    "based on the expected output."
)

_LLM_TEST_CASE_CLS: Any = None


def _llm_test_case_cls() -> Any:
    """Load DeepEval 2.7.0 LLMTestCase without importing deepeval package init.

    deepeval/__init__.py pulls optional provider modules (ollama, llama-index,
    anthropic, OpenTelemetry). Those extras would upgrade the frozen RAGAS
    stack. The test-case dataclass itself only needs the stdlib and pydantic.
    """
    global _LLM_TEST_CASE_CLS
    if _LLM_TEST_CASE_CLS is not None:
        return _LLM_TEST_CASE_CLS
    module_path = None
    for entry in sys.path:
        candidate = Path(entry) / "deepeval" / "test_case" / "llm_test_case.py"
        if candidate.is_file():
            module_path = candidate
            break
    if module_path is None:
        raise ImportError("Installed deepeval LLMTestCase module was not found")
    spec = importlib.util.spec_from_file_location(
        "ai_qe_eval._deepeval_llm_test_case",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load deepeval LLMTestCase module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LLM_TEST_CASE_CLS = module.LLMTestCase
    return _LLM_TEST_CASE_CLS


def _trace_to_llm_test_case(trace: EvaluationTrace) -> Any:
    return _llm_test_case_cls()(
        input=trace.input,
        actual_output=trace.output,
        expected_output=trace.expected,
    )


def _default_geval_metric(
    *,
    model: Any | None,
    criteria: str | None,
    evaluation_steps: list[str] | None,
) -> Any:
    from deepeval.metrics.g_eval.g_eval import GEval
    from deepeval.test_case.llm_test_case import LLMTestCaseParams

    kwargs: dict[str, Any] = {
        "name": DEFAULT_GEVAL_NAME,
        "evaluation_params": [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        "model": model,
    }
    if evaluation_steps is not None:
        kwargs["evaluation_steps"] = evaluation_steps
    else:
        kwargs["criteria"] = criteria or DEFAULT_CRITERIA
    return GEval(**kwargs)


class DeepEvalGEvalCorrectnessEvaluator:
    def __init__(
        self,
        *,
        geval_metric: Any | None = None,
        model: Any | None = None,
        criteria: str | None = None,
        evaluation_steps: list[str] | None = None,
    ) -> None:
        self._geval_metric = geval_metric
        self._model = model
        self._criteria = criteria
        self._evaluation_steps = evaluation_steps

    def _metric(self) -> Any:
        if self._geval_metric is not None:
            return self._geval_metric
        return _default_geval_metric(
            model=self._model,
            criteria=self._criteria,
            evaluation_steps=self._evaluation_steps,
        )

    def evaluate(
        self,
        trace: EvaluationTrace,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        test_case = _trace_to_llm_test_case(trace)
        metric = self._metric()
        metric.measure(test_case)
        score = metric.score
        reason = getattr(metric, "reason", None)
        return [
            EvaluationResult(
                metric=CORRECTNESS_METRIC,
                evaluator=DEEPEVAL_EVALUATOR_NAME,
                score=score,
                reason=reason,
                raw_result={
                    "score": score,
                    "reason": reason,
                    "name": getattr(metric, "name", DEFAULT_GEVAL_NAME),
                    "criteria": getattr(metric, "criteria", self._criteria),
                    "evaluation_steps": getattr(
                        metric, "evaluation_steps", self._evaluation_steps
                    ),
                    "input": test_case.input,
                    "actual_output": test_case.actual_output,
                    "expected_output": test_case.expected_output,
                },
            )
        ]
