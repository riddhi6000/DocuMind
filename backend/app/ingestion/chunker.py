
def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text.
        chunk_size: Maximum approximate characters per chunk.
        chunk_overlap: Approximate characters repeated between chunks.

    Returns:
        A list of text chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer ending at a word boundary.
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        # Start the next chunk with some overlapping text.
        next_start = end - chunk_overlap

        # Avoid starting in the middle of a word.
        while next_start < end and not text[next_start].isspace():
            next_start += 1

        start = next_start + 1

    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[dict]:
    """
    Chunk extracted PDF pages while preserving page metadata.
    """

    all_chunks = []

    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            chunk_size,
            chunk_overlap
        )

        for chunk in page_chunks:
            all_chunks.append({
                "chunk_id": len(all_chunks),
                "page": page["page"],
                "text": chunk
            })

    return all_chunks
