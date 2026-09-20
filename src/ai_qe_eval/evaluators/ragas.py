"""P2-09 RAGAS Faithfulness evaluator adapter.

Maps EvaluationTrace into RAGAS 0.2.15 Faithfulness (SingleTurnSample +
single_turn_score) and returns one EvaluationResult.

Does not apply thresholds or PASS/FAIL. Does not auto-register capabilities.
Other RAGAS metrics are not implemented here.
"""

from __future__ import annotations

from typing import Any

from ragas import SingleTurnSample
from ragas.metrics import Faithfulness

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.trace import EvaluationTrace

FAITHFULNESS_METRIC = "faithfulness"
RAGAS_EVALUATOR_NAME = "ragas"


def _retrieved_contexts(retrieval: list[Any] | None) -> list[str]:
    if retrieval is None:
        return []
    contexts: list[str] = []
    for item in retrieval:
        if isinstance(item, str) and item:
            contexts.append(item)
        elif isinstance(item, dict):
            page_content = item.get("page_content")
            if isinstance(page_content, str) and page_content:
                contexts.append(page_content)
    return contexts


def _trace_to_single_turn_sample(trace: EvaluationTrace) -> SingleTurnSample:
    return SingleTurnSample(
        user_input=trace.input,
        response=trace.output,
        retrieved_contexts=_retrieved_contexts(trace.retrieval),
    )


def _default_llm_wrapper():
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    from utils import llm_model_name

    load_dotenv()
    llm = ChatOpenAI(model=llm_model_name(), temperature=0)
    return LangchainLLMWrapper(llm)


class RAGASFaithfulnessEvaluator:
    def __init__(
        self,
        *,
        llm: Any | None = None,
        faithfulness_metric: Any | None = None,
    ) -> None:
        self._llm = llm
        self._faithfulness_metric = faithfulness_metric

    def _metric(self) -> Any:
        if self._faithfulness_metric is not None:
            return self._faithfulness_metric
        llm = self._llm if self._llm is not None else _default_llm_wrapper()
        return Faithfulness(llm=llm)

    def evaluate(
        self,
        trace: EvaluationTrace,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        sample = _trace_to_single_turn_sample(trace)
        score = self._metric().single_turn_score(sample)
        return [
            EvaluationResult(
                metric=FAITHFULNESS_METRIC,
                evaluator=RAGAS_EVALUATOR_NAME,
                score=score,
                reason=f"RAGAS faithfulness score={score}.",
                raw_result={
                    "score": score,
                    "user_input": sample.user_input,
                    "response": sample.response,
                    "retrieved_contexts": sample.retrieved_contexts,
                },
            )
        ]
