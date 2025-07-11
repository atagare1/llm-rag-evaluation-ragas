import os

import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper


@pytest.fixture
def llm_wrapper():
    os.environ["OPENAI_API_KEY"] = "a16b6a698280c4de215a803662f56ff170ddae3e662c3a303ba3a6ef79fc8ce9"
    os.environ["OPENAI_BASE_URL"] = "https://api.together.xyz/v1"

    # Use a free open-source model hosted on Together
    llm = ChatOpenAI(model="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    return langchain_llm