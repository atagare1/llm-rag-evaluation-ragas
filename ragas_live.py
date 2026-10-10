"""Live RAGAS Llama judge construction (OpenRouter).

Test-only helper for the Layer E Llama tests. Does not change Phase 1
defaults, embeddings, or production evaluators.
"""

from __future__ import annotations

import os

import pytest
from langchain_openai import ChatOpenAI

DEFAULT_RAGAS_LLAMA_JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"
DEFAULT_RAGAS_LLAMA_MAX_TOKENS = 4096


def ragas_llama_judge_model_name() -> str:
    configured = os.getenv("RAGAS_LLM_MODEL")
    if configured is None or configured.strip() == "":
        return DEFAULT_RAGAS_LLAMA_JUDGE_MODEL
    return configured.strip()


def live_ragas_llama_chat() -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        pytest.skip("OPENROUTER_API_KEY or OPENAI_BASE_URL is not set")
    return ChatOpenAI(
        model=ragas_llama_judge_model_name(),
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        max_tokens=DEFAULT_RAGAS_LLAMA_MAX_TOKENS,
    )
