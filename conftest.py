import os

import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from dotenv import load_dotenv
load_dotenv()
@pytest.fixture
def llm_wrapper():
    # Use a free open-source model hosted on Together
    llm = ChatOpenAI(model="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    return langchain_llm