
import numpy as np
import pytest

from backend.app.retrieval.vector_store import VectorStore


def test_add_and_search_chunks():
    store = VectorStore(dimension=3)

    chunks = [
        {"chunk_id": 0, "page": 1, "text": "Machine learning"},
        {"chunk_id": 1, "page": 2, "text": "Cooking recipes"},
        {"chunk_id": 2, "page": 3, "text": "Artificial intelligence"}
    ]

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.2, 0.0]
    ], dtype=np.float32)

    store.add_chunks(chunks, embeddings)

    query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    results = store.search(query, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == 0
    assert results[1]["chunk_id"] == 2
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_empty_index():
    store = VectorStore(dimension=3)

    results = store.search(np.array([1.0, 0.0, 0.0]), top_k=3)

    assert results == []


def test_mismatched_chunk_count():
    store = VectorStore(dimension=3)

    chunks = [{"chunk_id": 0, "page": 1, "text": "Example"}]
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    with pytest.raises(ValueError):
        store.add_chunks(chunks, embeddings)
