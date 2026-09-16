# LLM RAG Evaluation using RAGAS + Together AI

This project is a **pytest-based RAG evaluation POC**. It scores answers from an external RAG demo API with [RAGAS](https://github.com/explodinggradients/ragas), using an open-source judge model hosted on [Together AI](https://www.together.ai/) through the OpenAI-compatible client.

This is not an enterprise evaluation framework. Thresholds below are **experimental POC gates** and are **not production-validated**.

---

## What is implemented

* RAGAS metrics: context precision (without reference), context recall, faithfulness, response relevancy, factual correctness
* Together AI via `OPENAI_API_KEY` + `OPENAI_BASE_URL`
* pytest collection of `test_*.py`
* Shared mapping from the live RAG response (`answer`, `retrieved_docs[].page_content`)
* Optional env configuration for model, RAG URL, and thresholds (defaults preserve the original POC)

---

## Installation

```bash
git clone https://github.com/atagare1/llm-rag-evaluation-ragas.git
cd llm-rag-evaluation-ragas
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to `.env` and replace placeholders. Do not commit `.env`.

```env
OPENAI_API_KEY=your_together_ai_key
OPENAI_BASE_URL=https://api.together.xyz/v1
```

Optional (defaults shown):

```env
RAG_API_URL=https://rahulshettyacademy.com/rag-llm/ask
RAGAS_LLM_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
RAGAS_EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
```

Live RAGAS tests skip when `OPENAI_API_KEY` is unset. Unit tests do not call Together or the RAG API.

---

## Running tests

```bash
pytest
```

* Unit tests: dataset loading, response mapping, configuration (no live services).
* Live tests: require Together credentials and the external RAG API.

Experimental default gates: context precision `> 0.8`, context recall `> 0.7`, faithfulness `> 0.8`, answer relevancy `> 0.8`, factual correctness `> 0.8`. Override with `RAGAS_THRESHOLD_*` env vars if needed.

---

## Why Together AI

Together AI is used here as an OpenAI-compatible host for open models such as Mixtral. Model choice is `RAGAS_LLM_MODEL`. This POC does not include multi-model benchmarking.

---

## Future work (not implemented)

* Evaluation reports and dashboards
* Multi-model benchmarking
* Regression / drift detection
* CI/CD quality gates
