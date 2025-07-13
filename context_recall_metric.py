import pytest
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference, LLMContextRecall
from utils import get_api_response, read_test_data

@pytest.mark.asyncio
@pytest.mark.parametrize("get_test_data",[read_test_data("rag_test_data.json")],indirect=True)
async def test_context_recall(llm_wrapper,get_test_data):
    context_recall = LLMContextRecall(llm=llm_wrapper)
    # Generate score
    score = await context_recall.single_turn_ascore(get_test_data)
    print(score)
    assert score >0.7

@pytest.fixture
def get_test_data(request):
    passed_data=request.param
    responseData=get_api_response(passed_data)
    print(responseData)

    sample= SingleTurnSample(
        user_input=passed_data["question"],
        retrieved_contexts=[responseData["retrieved_docs"][0]["page_content"],
                            responseData["retrieved_docs"][1]["page_content"],],
        reference=passed_data["reference"]

    )
    return sample