import json
import requests

def read_test_data():
    path="test-data/rag_test_data.json"
    with open(path) as f:
        return json.load(f)


def get_api_response(passed_data):
    responseData= requests.post(url="https://rahulshettyacademy.com/rag-llm/ask",
                                json={
    "question":passed_data["question"],
    "chat_history":[]
    }).json()
    return responseData