
import numpy as np

from backend.app.ingestion.pdf_loader import extract_text_from_pdf
from backend.app.ingestion.chunker import chunk_pages
from backend.app.embeddings.embedder import generate_embeddings
from backend.app.retrieval.vector_store import VectorStore


def build_vector_store(file_path: str) -> VectorStore:
    """
    Process a PDF and build a searchable FAISS vector store.

    Pipeline:
    PDF -> Text Extraction -> Chunking -> Embeddings -> FAISS
    """

    # Step 1: Extract text from PDF
    pages = extract_text_from_pdf(file_path)

    if not pages:
        raise ValueError("The PDF contains no extractable text")

    # Step 2: Split extracted text into chunks
    chunks = chunk_pages(
        pages,
        chunk_size=500,
        chunk_overlap=100
    )

    if not chunks:
        raise ValueError("No text chunks could be generated")

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(
        [chunk["text"] for chunk in chunks]
    )

    # Step 4: Initialize FAISS using the embedding dimension
    dimension = embeddings.shape[1]
    vector_store = VectorStore(dimension=dimension)

    # Step 5: Add embeddings and metadata
    vector_store.add_chunks(chunks, embeddings)

    return vector_store
