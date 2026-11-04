
import os

from dotenv import load_dotenv
from google import genai

from backend.app.generation.prompts import (
    SYSTEM_PROMPT,
    build_rag_prompt
)


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from environment variables")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer using Gemini and retrieved document context.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if not context.strip():
        raise ValueError("Document context cannot be empty")

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=build_rag_prompt(question, context),
        system_instruction=SYSTEM_PROMPT,
        store=False
    )

    if not interaction.output_text:
        raise RuntimeError("Gemini returned an empty response")

    return interaction.output_text.strip()
