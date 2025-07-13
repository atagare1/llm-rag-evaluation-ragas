import os
import pytest
import requests
from langchain.chat_models import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference

from utils import read_test_data, get_api_response

@pytest.mark.asyncio
@pytest.mark.parametrize("get_test_data",[read_test_data("rag_test_data.json")],indirect=True)
async def test_context_precision(llm_wrapper,get_test_data):
    # Use Together AI endpoint and key
    context_precision = LLMContextPrecisionWithoutReference(llm=llm_wrapper)
    # Generate score
    score = await context_precision.single_turn_ascore(get_test_data)
    print(score)
    assert score >0.8

@pytest.fixture
def get_test_data(request):
    passed_data = request.param
    responseData = get_api_response(passed_data)
    print(responseData)

    sample= SingleTurnSample(
        user_input=passed_data["question"],
        response=responseData["answer"],
        retrieved_contexts=[ doc["page_content"] for doc in responseData.get("retrieved_docs")]
        #retrieved_contexts=[responseData["retrieved_docs"][0]["page_content"],
           #                 responseData["retrieved_docs"][1]["page_content"],]

    )
    return sample