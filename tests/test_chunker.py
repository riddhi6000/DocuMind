
from backend.app.ingestion.chunker import chunk_text, chunk_pages


def test_chunk_text():
    text = "This is a sample document about retrieval augmented generation."

    chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk) <= 30 for chunk in chunks)


def test_chunk_pages():
    pages = [
        {"page": 1, "text": "DocuMind is an AI-powered PDF assistant."},
        {"page": 2, "text": "RAG combines retrieval with generation."}
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) == 2
    assert chunks[0]["page"] == 1
    assert chunks[1]["page"] == 2
    assert chunks[0]["chunk_id"] == 0
    assert chunks[1]["chunk_id"] == 1
