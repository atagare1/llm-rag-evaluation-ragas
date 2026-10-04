"""Unit tests for the live DeepEval OpenRouter judge helper.

Does not call a provider.
"""

from deepeval_live import (
    DEFAULT_DEEPEVAL_JUDGE_MODEL,
    deepeval_judge_model_name,
)


def test_default_deepeval_judge_model_is_openrouter_llama(monkeypatch):
    monkeypatch.delenv("DEEPEVAL_JUDGE_MODEL", raising=False)
    assert DEFAULT_DEEPEVAL_JUDGE_MODEL == "meta-llama/llama-3.3-70b-instruct"
    assert deepeval_judge_model_name() == DEFAULT_DEEPEVAL_JUDGE_MODEL


def test_deepeval_judge_model_env_override(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_JUDGE_MODEL", "  example/model  ")
    assert deepeval_judge_model_name() == "example/model"
