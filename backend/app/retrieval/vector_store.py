
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int = 384):
        """
        Initialize a FAISS index for normalized embeddings.

        Args:
            dimension: Number of dimensions in each embedding.
        """
        self.dimension = dimension

        # Inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)

        # Store original chunk information separately
        self.chunks = []

    def add_chunks(self, chunks: list[dict], embeddings: np.ndarray):
        """
        Add embeddings and their corresponding metadata to FAISS.
        """

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimensions do not match the index")

        if not np.issubdtype(embeddings.dtype, np.number):
            raise ValueError("Embeddings must be numeric")

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Retrieve the top-k most similar chunks.
        """

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        if query_embedding.shape == (self.dimension,):
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape != (1, self.dimension):
            raise ValueError("Query embedding has an invalid shape")

        if not np.all(np.isfinite(query_embedding)):
            raise ValueError("Query embedding must contain finite values")

        query_embedding = np.ascontiguousarray(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            results.append({
                **self.chunks[index],
                "similarity": float(score)
            })

        return results
