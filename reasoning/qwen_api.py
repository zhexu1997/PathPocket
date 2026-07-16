"""Simple OpenAI-compatible API wrapper for PathPocket reasoning."""

from __future__ import annotations

import asyncio
import base64
import os
from typing import Any, Dict, List, Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

_client: Optional[AsyncOpenAI] = None

def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL_NAME", "").strip() or "qwen3.7-plus"

def get_client() -> AsyncOpenAI:
    global _client
    if _client is not None:
        return _client
    
    if AsyncOpenAI is None:
        raise ImportError("openai package is required. pip install openai")
        
    base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
        
    _client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return _client

async def chat_multimodal(
    messages: List[Dict[str, Any]], 
    model: str, 
    **kwargs: Any
) -> str:
    """Send a chat request (with optional base64 images in messages)."""
    client = get_client()
    
    # Base Qwen/OpenAI arguments
    api_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        "top_p": float(os.getenv("OPENAI_TOP_P", "0.8")),
    }
    
    max_tokens = os.getenv("OPENAI_MAX_TOKENS", "4096")
    if max_tokens.isdigit():
        api_kwargs["max_tokens"] = int(max_tokens)
        
    api_kwargs.update(kwargs)
    
    # Retries built-in to the openai client, but we can do a simple loop
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(**api_kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 ** attempt)
    return ""
