
import numpy as np

from backend.app.ingestion.pdf_loader import extract_text_from_pdf
from backend.app.ingestion.chunker import chunk_pages
from backend.app.embeddings.embedder import generate_embeddings
from backend.app.retrieval.vector_store import VectorStore
from backend.app.generation.llm import generate_answer

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

def retrieve_chunks(
    vector_store: VectorStore,
    question: str,
    top_k: int = 3
) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a question.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty")

    # Convert the user's question into an embedding
    query_embedding = generate_embeddings([question])

    # Search FAISS for the most similar document chunks
    results = vector_store.search(
        query_embedding,
        top_k=top_k
    )

    return results

def answer_question(vector_store, question: str, top_k: int = 3) -> dict:
    """
    Answer a question using retrieved PDF chunks and Gemini.
    """

    # Step 1: Retrieve relevant chunks from FAISS
    retrieved_chunks = retrieve_chunks(
        vector_store,
        question,
        top_k
    )

    if not retrieved_chunks:
        return {
            "answer": "I couldn't find relevant information in the uploaded document.",
            "sources": []
        }

    # Step 2: Build context with page references
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    # Step 3: Generate an answer using Gemini
    answer = generate_answer(question, context)

    # Step 4: Return answer and source metadata
    sources = [
        {
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "similarity": chunk["similarity"],
            "text": chunk["text"]
        }
        for chunk in retrieved_chunks
    ]

    return {
        "answer": answer,
        "sources": sources
    }