
import pytest

from backend.app.rag.pipeline import (
    build_vector_store,
    retrieve_chunks,
    answer_question
)

def test_build_vector_store():
    store = build_vector_store("data/sample.pdf")

    assert store.index.ntotal > 0
    assert len(store.chunks) == store.index.ntotal

    assert store.chunks[0]["page"] == 1
    assert store.chunks[0]["text"]


def test_empty_pdf_raises_error(tmp_path):
    from backend.app.ingestion.pdf_loader import extract_text_from_pdf
    import fitz

    empty_pdf = tmp_path / "empty.pdf"

    doc = fitz.open()
    doc.new_page()
    doc.save(empty_pdf)
    doc.close()

    with pytest.raises(ValueError, match="no extractable text"):
        build_vector_store(str(empty_pdf))


def test_retrieve_relevant_chunks():
    store = build_vector_store("data/sample.pdf")

    results = retrieve_chunks(
        store,
        "What is retrieval augmented generation?",
        top_k=1
    )

    assert len(results) == 1
    assert results[0]["text"]
    assert results[0]["page"] == 2
    assert "similarity" in results[0]

def test_answer_question(monkeypatch):
    from backend.app.rag import pipeline

    store = build_vector_store("data/sample.pdf")

    def mock_generate_answer(question, context):
        assert question == "What does RAG combine?"
        assert "RAG combines document retrieval" in context
        return "RAG combines document retrieval with language model generation."

    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        mock_generate_answer
    )

    result = answer_question(store, "What does RAG combine?")

    assert result["answer"]
    assert len(result["sources"]) > 0
    assert result["sources"][0]["page"] == 2


def test_answer_question_returns_sources(monkeypatch):
    store = build_vector_store("data/sample.pdf")

    from backend.app.rag import pipeline

    def mock_generate_answer(question, context):
        return "Test answer"

    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        mock_generate_answer
    )
    result = answer_question(store, "What does RAG combine?")

    assert result["answer"] == "Test answer"
    assert all("page" in source for source in result["sources"])
    assert all("similarity" in source for source in result["sources"])


def test_answer_question_without_relevant_chunks(monkeypatch):
    from backend.app.rag import pipeline

    store = build_vector_store("data/sample.pdf")

    def mock_retrieve_chunks(vector_store, question, top_k):
        return []

    def mock_generate_answer(question, context):
        raise AssertionError("Gemini should not be called")

    monkeypatch.setattr(
        pipeline,
        "retrieve_chunks",
        mock_retrieve_chunks
    )

    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        mock_generate_answer
    )

    result = answer_question(store, "Some unrelated question")

    assert result["sources"] == []
    assert "couldn't find relevant information" in result["answer"]