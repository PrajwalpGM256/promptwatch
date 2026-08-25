import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai

load_dotenv()

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def api_key() -> str:
    """Read GEMINI_API_KEY from the environment.

    Raises:
        RuntimeError: if it is not set.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return key


@lru_cache(maxsize=1)
def _cached_client(key: str) -> genai.Client:
    return genai.Client(api_key=key)


def client() -> genai.Client:
    """Return a cached Gemini client so connections are reused across calls."""
    return _cached_client(api_key())
