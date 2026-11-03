
from sentence_transformers import SentenceTransformer


# Load the embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts: list[str]):
    """
    Convert a list of text strings into numerical embeddings.

    Args:
        texts: List of text chunks.

    Returns:
        A NumPy array containing the embeddings.
    """

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings
