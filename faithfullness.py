import pytest
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference, LLMContextRecall, Faithfulness
from utils import get_api_response, read_test_data

@pytest.mark.asyncio
@pytest.mark.parametrize("get_test_data",[read_test_data('rag_test_data_faithfulness.json')],indirect=True)
async def test_failthfulness(llm_wrapper,get_test_data):
    failthfulness = Faithfulness(llm=llm_wrapper)
    # Generate score
    score = await failthfulness.single_turn_ascore(get_test_data)
    print(score)
    assert score >0.8

@pytest.fixture
def get_test_data(request):
    passed_data=request.param
    responseData=get_api_response(passed_data)
    print(responseData)

    sample= SingleTurnSample(
        user_input=passed_data["question"],
        response=responseData["Answer"],
        retrieved_contexts=[responseData["retrieved_docs"][0]["page_content"],
                            ]
    )
    return sample