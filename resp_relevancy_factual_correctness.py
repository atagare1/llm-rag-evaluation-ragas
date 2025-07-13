import pytest
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import LLMContextPrecisionWithoutReference, LLMContextRecall, Faithfulness, ResponseRelevancy, \
    FactualCorrectness
from utils import get_api_response, read_test_data

@pytest.mark.asyncio
@pytest.mark.parametrize("get_test_data",[read_test_data('rag_test_data_faithfulness.json')],indirect=True)
async def test_resp_relevancy_and_factual_correctness(llm_wrapper,get_test_data):
    metrics = [ResponseRelevancy(llm=llm_wrapper), FactualCorrectness(llm=llm_wrapper)]
    # Generate score
    eval_dataset=EvaluationDataset([get_test_data])
    results=evaluate(dataset=eval_dataset,metrics=metrics)

@pytest.fixture
def get_test_data(request):
    passed_data=request.param
    responseData=get_api_response(passed_data)
    print(responseData)

    sample= SingleTurnSample(
        user_input=passed_data["question"],
        response=responseData["answer"],
        retrieved_contexts=[responseData["retrieved_docs"][0]["page_content"],
                            ],
        reference=passed_data["reference"]
    )
    return sample