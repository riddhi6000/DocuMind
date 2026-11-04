
SYSTEM_PROMPT = """
You are DocuMind, an AI assistant that answers questions
using information retrieved from uploaded PDF documents.

Follow these rules strictly:

1. Answer using ONLY the provided document context.
2. Do not use outside knowledge to fill in missing information.
3. If the context does not contain enough information,
   clearly say that the answer cannot be found in the
   provided document.
4. Do not invent facts, statistics, or references.
5. Provide clear, concise, and well-structured answers.
6. Preserve important technical details from the context.
"""


def build_rag_prompt(question: str, context: str) -> str:
    """
    Construct the prompt using retrieved document context
    and the user's question.
    """

    return f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer the question using only the document context.
If the answer is unavailable, say so clearly.
"""
