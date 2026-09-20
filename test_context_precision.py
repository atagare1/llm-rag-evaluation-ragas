import pytest
from ragas.metrics import LLMContextPrecisionWithoutReference

from utils import metric_threshold, read_test_data


@pytest.mark.asyncio
@pytest.mark.parametrize("get_test_data", [read_test_data("rag_test_data.json")], indirect=True)
async def test_context_precision(llm_wrapper, get_test_data):
    context_precision = LLMContextPrecisionWithoutReference(llm=llm_wrapper)
    score = await context_precision.single_turn_ascore(get_test_data)
    print(score)
    # Experimental POC threshold. Not production-validated.
    assert score > metric_threshold("context_precision")
