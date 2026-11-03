
from backend.app.embeddings.embedder import generate_embeddings


def test_generate_embeddings():
    texts = [
        "I love programming.",
        "Python is a programming language.",
        "The weather is sunny today."
    ]

    embeddings = generate_embeddings(texts)

    assert embeddings.shape == (3, 384)
    assert embeddings.dtype.name == "float32"
