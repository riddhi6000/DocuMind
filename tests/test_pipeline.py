
import pytest

from backend.app.rag.pipeline import build_vector_store


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
