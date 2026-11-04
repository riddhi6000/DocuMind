
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.app.generation.prompts import (
    SYSTEM_PROMPT,
    build_rag_prompt
)


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from environment variables")

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=1
        )
    )
)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer using Gemini and retrieved document context.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if not context.strip():
        raise ValueError("Document context cannot be empty")

    try:
        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=build_rag_prompt(question, context),
            system_instruction=SYSTEM_PROMPT,
            store=False,
            timeout=30000
        )

    except Exception as e:
        error_message = str(e)
        status_code = getattr(e, "status_code", None)

        if status_code == 429 or "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
            raise RuntimeError(
                "Gemini API rate limit reached. Please try again after your quota resets."
            ) from e

        elif status_code == 503 or "503" in error_message or "UNAVAILABLE" in error_message:
            raise RuntimeError(
                "Gemini is temporarily unavailable. Please try again later."
            ) from e

        elif "timeout" in error_message.lower():
            raise RuntimeError(
                "Gemini request timed out. Please try again."
            ) from e

        raise

    if not interaction.output_text:
        raise RuntimeError("Gemini returned an empty response")

    return interaction.output_text.strip()
