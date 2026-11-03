
import fitz


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF, preserving page numbers.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of dictionaries containing page numbers and extracted text.
    """
    pages = []

    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages
