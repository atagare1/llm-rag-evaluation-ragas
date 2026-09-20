import os

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from utils import build_embeddings_wrapper, build_single_turn_sample, get_api_response, llm_model_name

load_dotenv()


def together_api_key_configured():
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def llm_wrapper():
    if not together_api_key_configured():
        pytest.skip("OPENAI_API_KEY is not set; skipping live RAGAS evaluation")
    llm = ChatOpenAI(model=llm_model_name(), temperature=0)
    return LangchainLLMWrapper(llm)


@pytest.fixture
def embeddings_wrapper():
    if not together_api_key_configured():
        pytest.skip("OPENAI_API_KEY is not set; skipping live RAGAS evaluation")
    return build_embeddings_wrapper()


@pytest.fixture
def get_test_data(request):
    if not together_api_key_configured():
        pytest.skip("OPENAI_API_KEY is not set; skipping live RAG evaluation")
    passed_data = request.param
    response_data = get_api_response(passed_data)
    sample = build_single_turn_sample(passed_data, response_data)
    print("retrieved_context_count", len(sample.retrieved_contexts or []))
    return sample
