import pytest

from utils import (
    extract_answer,
    extract_retrieved_contexts,
    map_rag_response,
)


QUESTION = {"question": "How many articles?", "reference": "23"}


def test_extract_answer_uses_lowercase_answer_field():
    assert extract_answer({"answer": "There are 23 articles."}) == "There are 23 articles."


def test_extract_answer_rejects_capitalized_answer_field():
    with pytest.raises(ValueError, match="missing required field 'answer'"):
        extract_answer({"Answer": "There are 23 articles."})


def test_extract_answer_rejects_non_object_response():
    with pytest.raises(ValueError, match="JSON object"):
        extract_answer(["not", "an", "object"])


def test_extract_answer_rejects_non_string_answer():
    with pytest.raises(ValueError, match="must be a string"):
        extract_answer({"answer": 23})


def test_extract_retrieved_contexts_uses_all_page_contents():
    response = {
        "retrieved_docs": [
            {"file_name": "a.docx", "page_content": "first"},
            {"file_name": "b.docx", "page_content": "second"},
            {"file_name": "c.docx", "page_content": "third"},
        ]
    }
    assert extract_retrieved_contexts(response) == ["first", "second", "third"]


def test_extract_retrieved_contexts_missing_docs_returns_empty_list():
    assert extract_retrieved_contexts({"answer": "ok"}) == []


def test_extract_retrieved_contexts_skips_invalid_or_empty_docs():
    response = {
        "retrieved_docs": [
            {"file_name": "a.docx", "page_content": "keep"},
            {"file_name": "b.docx", "page_content": ""},
            {"file_name": "c.docx"},
            "not-a-dict",
            {"file_name": "d.docx", "page_content": "also keep"},
        ]
    }
    assert extract_retrieved_contexts(response) == ["keep", "also keep"]


def test_extract_retrieved_contexts_rejects_non_list_docs():
    with pytest.raises(ValueError, match="must be a list"):
        extract_retrieved_contexts({"retrieved_docs": {"page_content": "x"}})


def test_map_rag_response_builds_single_turn_fields():
    response = {
        "answer": "There are 23 articles.",
        "retrieved_docs": [{"page_content": "ctx-1"}, {"page_content": "ctx-2"}],
    }
    mapped = map_rag_response(QUESTION, response)
    assert mapped == {
        "user_input": "How many articles?",
        "response": "There are 23 articles.",
        "retrieved_contexts": ["ctx-1", "ctx-2"],
        "reference": "23",
    }


def test_map_rag_response_does_not_assume_two_documents():
    response = {
        "answer": "There are 23 articles.",
        "retrieved_docs": [{"page_content": "only-one"}],
    }
    mapped = map_rag_response(QUESTION, response)
    assert mapped["retrieved_contexts"] == ["only-one"]
