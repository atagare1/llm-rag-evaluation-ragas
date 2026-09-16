import pytest
from ragas.metrics import Faithfulness

from utils import metric_threshold, read_test_data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_test_data", [read_test_data("rag_test_data_faithfulness.json")], indirect=True
)
async def test_faithfulness(llm_wrapper, get_test_data):
    faithfulness = Faithfulness(llm=llm_wrapper)
    score = await faithfulness.single_turn_ascore(get_test_data)
    print(score)
    # Experimental POC threshold. Not production-validated.
    assert score > metric_threshold("faithfulness")
