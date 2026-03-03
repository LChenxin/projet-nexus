# backend/llm.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
from openai import OpenAI
from backend import config


# =========================
# 1) Client initialization
# =========================
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


# =========================
# 2) Chat API wrapper
# =========================
def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "text": str,
        "raw": response_object
      }
    """
    client = get_client()

    resp = client.chat.completions.create(
        model=model or config.LLM_MODEL,
        messages=messages,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        max_tokens=max_output_tokens or config.LLM_MAX_OUTPUT_TOKENS,
    )

    # OpenAI responses: resp.choices[0].message.content
    text = resp.choices[0].message.content or ""
    return {"text": text, "raw": resp}


# =========================
# 3) Embeddings wrapper
# =========================
def embed(
    texts: List[str],
    model: Optional[str] = None,
) -> List[List[float]]:
    client = get_client()

    # OpenAI embeddings: response.data[i].embedding
    resp = client.embeddings.create(
        model=model or config.EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in resp.data]