# DocuMind — AI-Powered PDF Assistant

DocuMind is a Retrieval-Augmented Generation (RAG) application designed to help users interact with PDF documents through natural language questions.

Instead of relying solely on an LLM's pretrained knowledge, DocuMind aims to retrieve relevant information from uploaded documents and use it to generate context-aware, grounded responses with source references.

## Features

### Implemented
- PDF text extraction using PyMuPDF.
- Page-level metadata preservation.
- Text chunking with configurable chunk size and overlap.
- Semantic embedding generation using Sentence Transformers.
- Automated tests using pytest.

### Planned
- Vector storage and similarity search using FAISS.
- LLM-powered answer generation using Gemini.
- Context-aware question answering.
- Source attribution with document and page references.
- PDF upload interface using Streamlit.
- FastAPI backend integration.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI |
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| LLM | Google Gemini |
| Testing | pytest |
| Version Control | Git & GitHub |

## Architecture

The planned RAG pipeline:

```text
PDF Upload
    |
    v
PDF Text Extraction
    |
    v
Text Chunking
    |
    v
Embedding Generation
    |
    v
FAISS Vector Store
    |
    v
Semantic Retrieval
    |
    v
LLM Context Generation
    |
    v
Grounded Answer + Sources
```

## Project Structure

```text
DocuMind/
│
├── backend/
│   ├── app/
│   │   ├── ingestion/
│   │   │   ├── pdf_loader.py
│   │   │   └── chunker.py
│   │   │
│   │   ├── embeddings/
│   │   │   └── embedder.py
│   │   │
│   │   ├── retrieval/
│   │   ├── generation/
│   │   └── rag/
│   │
│   └── requirements.txt
│
├── frontend/
│   └── streamlit_app.py
│
├── tests/
│   ├── test_chunker.py
│   └── test_embedder.py
│
├── data/
├── .env.example
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/riddhi6000/DocuMind.git
cd DocuMind
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and configure your Gemini API key when the generation module is implemented.

### 5. Run tests

```bash
python -m pytest -v
```

## Development Status

DocuMind is actively under development. The initial document ingestion, chunking, and embedding modules have been implemented and tested.

Upcoming development will focus on vector retrieval, LLM integration, and the complete question-answering pipeline.

## Author

**Riddhi Jain**

B.Tech Information Technology

[GitHub](https://github.com/riddhi6000)
