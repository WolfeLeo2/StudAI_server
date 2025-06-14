import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_summarize_endpoint():
    test_text = "This is a long text that needs to be summarized. It contains multiple sentences and should be shortened while maintaining the main ideas."
    response = client.post(
        "/summarize",
        json={
            "text": test_text,
            "max_length": 50,
            "min_length": 10
        }
    )
    assert response.status_code == 200
    assert "summary" in response.json()
    assert len(response.json()["summary"]) > 0

def test_question_answer_endpoint():
    response = client.post(
        "/question-answer",
        json={
            "question": "What is the capital of France?",
            "context": "Paris is the capital and largest city of France."
        }
    )
    assert response.status_code == 200
    assert "answer" in response.json()
    assert len(response.json()["answer"]) > 0

def test_invalid_input():
    response = client.post(
        "/summarize",
        json={
            "text": "",
            "max_length": -1
        }
    )
    assert response.status_code == 500