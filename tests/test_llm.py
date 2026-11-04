import pytest
from backend.app.generation import llm


def test_rate_limit_error(monkeypatch):
    def mock_create(**kwargs):
        raise Exception("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(
        llm.client.interactions,
        "create",
        mock_create
    )

    with pytest.raises(RuntimeError, match="Gemini API rate limit reached"):
        llm.generate_answer(
            "What is RAG?",
            "RAG combines retrieval and generation."
        )


def test_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        llm.generate_answer("", "Some context")


def test_empty_context():
    with pytest.raises(ValueError, match="Document context cannot be empty"):
        llm.generate_answer("What is RAG?", "")