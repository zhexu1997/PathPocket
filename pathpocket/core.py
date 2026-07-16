"""
Core utilities for PathPocket
Replaces LightRAG dependencies with self-contained implementations
"""

import logging
import hashlib
import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from functools import partial
from dataclasses import dataclass

# Constants
GRAPH_FIELD_SEP = "<SEP>"  # Match LightRAG's constant
DEFAULT_ENTITY_NAME_MAX_LENGTH = 200
DEFAULT_TUPLE_DELIMITER = "<|#|>"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
SOURCE_IDS_LIMIT_METHOD_FIFO = "FIFO"
SOURCE_IDS_LIMIT_METHOD_KEEP = "KEEP"
DEFAULT_SOURCE_IDS_LIMIT_METHOD = SOURCE_IDS_LIMIT_METHOD_KEEP
DEFAULT_SUMMARY_LANGUAGE = "English"
DEFAULT_MAX_FILE_PATHS = 10
DEFAULT_FILE_PATH_MORE_PLACEHOLDER = "more"

# Initialize logger
logger = logging.getLogger("pathpocket")
logger.propagate = False
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def compute_args_hash(content: str) -> str:
    """Compute MD5 hash of content"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute a unique ID for a given content string.
    The ID is a combination of the given prefix and the MD5 hash of the content string.
    """
    return prefix + compute_args_hash(content)


def split_string_by_multi_markers(text: str, markers: List[str]) -> List[str]:
    """Split string by multiple markers"""
    if not markers:
        return [text]
    
    # Use first marker as primary splitter
    parts = text.split(markers[0])
    
    # Further split by other markers if needed
    result = []
    for part in parts:
        if len(markers) > 1:
            result.extend(split_string_by_multi_markers(part, markers[1:]))
        else:
            result.append(part)
    
    return result


def fix_tuple_delimiter_corruption(record: str, delimiter_core: str, tuple_delimiter: str) -> str:
    """Fix corrupted tuple delimiters in LLM output"""
    # Replace variations of delimiter with correct one
    variations = [
        f"<|{delimiter_core}|>",
        f"<| {delimiter_core} |>",
        f"<|{delimiter_core.upper()}|>",
        f"<| {delimiter_core.upper()} |>",
    ]
    for variation in variations:
        if variation != tuple_delimiter:
            record = record.replace(variation, tuple_delimiter)
    return record


def sanitize_and_normalize_extracted_text(text: str, remove_inner_quotes: bool = False) -> str:
    """Sanitize and normalize extracted text"""
    if not text:
        return ""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove quotes if requested
    if remove_inner_quotes:
        text = text.strip('"\'')
    
    return text


def _truncate_entity_identifier(identifier: str, max_length: int, chunk_key: str, identifier_type: str) -> str:
    """Truncate entity identifier if too long"""
    if len(identifier) <= max_length:
        return identifier
    
    truncated = identifier[:max_length]
    logger.warning(
        f"{chunk_key}: {identifier_type} `{identifier}` exceeds maximum length ({max_length}), "
        f"truncated to `{truncated}`"
    )
    return truncated


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from the text"""
    return re.sub(
        r"^(<think>.*?</think>|.*</think>)", "", text, flags=re.DOTALL
    ).strip()


def sanitize_text_for_encoding(text: str, replacement_char: str = "") -> str:
    """Sanitize text to ensure safe UTF-8 encoding"""
    if not text:
        return text
    
    try:
        text = text.strip()
        text.encode("utf-8")
        
        # Remove surrogate characters
        sanitized = ""
        for char in text:
            code_point = ord(char)
            if 0xD800 <= code_point <= 0xDFFF or code_point in [0xFFFE, 0xFFFF]:
                sanitized += replacement_char
            else:
                sanitized += char
        
        return sanitized
    except Exception:
        return text


def pack_user_ass_to_openai_messages(*args: str) -> List[Dict[str, str]]:
    """Pack user and assistant messages for OpenAI format"""
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def is_float_regex(value: str) -> bool:
    """Check if string is a float"""
    import re
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def merge_source_ids(existing_ids: List[str] | None, new_ids: List[str] | None) -> List[str]:
    """Merge two lists of source IDs while preserving order and removing duplicates"""
    merged: List[str] = []
    seen: set[str] = set()
    
    for sequence in (existing_ids, new_ids):
        if not sequence:
            continue
        for source_id in sequence:
            if not source_id:
                continue
            if source_id not in seen:
                seen.add(source_id)
                merged.append(source_id)
    
    return merged


def apply_source_ids_limit(
    source_ids: List[str],
    limit: int,
    method: str,
    identifier: str | None = None,
) -> List[str]:
    """Apply a limit strategy to a sequence of source IDs"""
    SOURCE_IDS_LIMIT_METHOD_FIFO = "FIFO"
    SOURCE_IDS_LIMIT_METHOD_KEEP = "KEEP"
    
    if limit <= 0:
        return []
    
    source_ids_list = list(source_ids)
    if len(source_ids_list) <= limit:
        return source_ids_list
    
    normalized_method = method.upper() if method else "KEEP"
    
    if normalized_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
        truncated = source_ids_list[-limit:]
    else:  # KEEP
        truncated = source_ids_list[:limit]
    
    if identifier and len(truncated) < len(source_ids_list):
        logger.debug(
            f"Source_id truncated: {identifier} | {normalized_method} keeping {len(truncated)} of {len(source_ids_list)} entries"
        )
    
    return truncated
