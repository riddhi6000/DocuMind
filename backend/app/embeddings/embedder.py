
from functools import lru_cache

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts: list[str]):
    """
    Convert a list of text strings into numerical embeddings.

    Args:
        texts: List of text chunks.

    Returns:
        A NumPy array containing the embeddings.
    """

    model = get_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings
