"""Focused checks for the live RAGAS Llama judge helper."""

from ragas_live import (
    DEFAULT_RAGAS_LLAMA_JUDGE_MODEL,
    DEFAULT_RAGAS_LLAMA_MAX_TOKENS,
    live_ragas_llama_chat,
)


def test_live_ragas_llama_chat_sets_output_cap(monkeypatch):
    captured = {}

    class CapturingChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("RAGAS_LLM_MODEL", raising=False)
    monkeypatch.setattr("ragas_live.ChatOpenAI", CapturingChatOpenAI)

    live_ragas_llama_chat()

    assert captured["model"] == DEFAULT_RAGAS_LLAMA_JUDGE_MODEL
    assert captured["max_tokens"] == DEFAULT_RAGAS_LLAMA_MAX_TOKENS
    assert captured["temperature"] == 0
    assert captured["api_key"] == "test-openrouter-key"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
