# 🤖 LLM RAG Evaluation using RAGAS + Together AI

This project demonstrates an evaluation pipeline for **Retrieval-Augmented Generation (RAG)** using the [RAGAS](https://github.com/explodinggradients/ragas) framework, with inference powered by **open-source LLMs hosted on [Together AI](https://www.together.ai/)**.

---

## 🚀 Key Features

* 🔍 Evaluate RAG pipelines using **RAGAS**
* 💸 Switch from expensive GPT APIs to **Together AI** (e.g., `Mixtral`, `LLaMA`, `Zephyr`)
* 🧪 Simplified `pytest`-based test setup using `conftest.py`
* 🔁 Easy model switching via environment variables

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/atagare1/llm-rag-evaluation-ragas.git
cd llm-rag-evaluation-ragas
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🔐 Secure API Setup

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_together_ai_key
OPENAI_BASE_URL=https://api.together.xyz/v1
```

Ensure `.env` is **ignored** in version control:

```gitignore
.env
```

---

## ⚙️ Test Configuration with Together AI (`conftest.py`)

We use a `pytest` fixture to load the LLM from Together AI. It uses environment variables to remain secure and flexible.

### `conftest.py`

```python
import os
import pytest
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load .env at runtime

@pytest.fixture
def llm_wrapper():
    """
    LLM fixture using Together AI-hosted open models.
    """
    llm = ChatOpenAI(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.7
    )
    return llm
```

---

## ✅ Sample Test

```python
def test_llm_response(llm_wrapper):
    response = llm_wrapper.predict("What is RAG?")
    assert isinstance(response, str)
    assert "retrieval" in response.lower()
```

Run with:

```bash
pytest
```

---

## 🆖 Why Together AI Over OpenAI (ChatGPT)?

| Feature            | Together AI                                | OpenAI (ChatGPT)                    |
| ------------------ | ------------------------------------------ | ----------------------------------- |
| 💸 Cost            | Free & affordable OSS models               | Expensive for GPT-4 & large volumes |
| 🔄 Model Switching | Supports OSS models (Mixtral, LLaMA, etc.) | Proprietary only (GPT-3.5, GPT-4)   |
| 🚀 Performance     | Fast, scalable inference                   | Limited based on pricing tier       |
| 🧠 Transparency    | Open weights & training specs              | Black-box models                    |

---

## 📊 Future Enhancements

* Integrate RAGAS metrics into test reports
* Utilise RAGAS dashboard
* Add benchmarking across multiple models (e.g., `Mixtral` vs `LLaMA`)
* Plug into CI/CD with GitHub Actions



