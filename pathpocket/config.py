"""
Configuration for PathPocket
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathPocketConfig:
    """Configuration for PathPocket system"""

    working_dir: str = "./pathpocket_storage"
    """Working directory for storage"""

    # Multimodal processing flags
    enable_image_processing: bool = True
    """Enable image content processing"""

    enable_table_processing: bool = True
    """Enable table content processing"""

    # Context extraction configuration
    context_window: int = 1
    """Window size for context extraction"""

    context_mode: str = "page"
    """Context extraction mode: 'page', 'chunk', or 'token'"""

    max_context_tokens: int = 2000
    """Maximum context tokens"""

    include_headers: bool = True
    """Whether to include headers/titles in context"""

    include_captions: bool = True
    """Whether to include captions in context"""

    context_filter_content_types: List[str] = field(default_factory=lambda: ["text"])
    """Content types to include in context"""

    # RAG core configuration
    rag_kwargs: dict = field(default_factory=dict)
    """Additional keyword arguments for RAG core initialization"""
