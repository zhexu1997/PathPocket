"""
Query functionality for PathPocket
Copied from raganything.query and adapted for PathPocket
"""

import json
import hashlib
import os
import re
import asyncio
from typing import Dict, List, Any, Literal, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

from pathpocket.core import logger, compute_mdhash_id
from pathpocket.prompt import PROMPTS
from pathpocket.utils import (
    get_processor_for_type,
    encode_image_to_base64,
    validate_image_file,
)
from pathpocket.lightrag_utils import (
    dedupe_similar_image_content_caption,
    strip_chunk_local_path_hints_for_llm,
)
from pathpocket.evidence_level_catalog import enrich_evidence_items_from_catalog

_PLACEHOLDER_CAPTIONS = frozenset({"Image:", "Image", "图像:", "图像", ""})


def _is_usable_similar_image_caption(text: Any) -> bool:
    s = str(text or "").strip()
    return bool(s) and s not in _PLACEHOLDER_CAPTIONS


def _clean_similar_image_caption_text(raw_content: str) -> str:
    """Strip image path / entity boilerplate from caption text."""
    if not raw_content:
        return ""
    caption = str(raw_content)
    caption = re.sub(
        r"Entity:\s*Image_\w+\s*\(image\)\s*\n?",
        "",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(
        r"Image\s+Content\s*\(Path:\s*[^)]+\)\s*[：:]\s*\n?",
        "",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(
        r"Table\s+Content\s*\(Path:\s*[^)]+\)\s*[：:]\s*\n?",
        "",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(r"\(Path:\s*[^)]+\)", "", caption, flags=re.IGNORECASE)
    caption = re.sub(
        r"/\S+\.(jpg|jpeg|png|bmp|tiff|gif|webp)",
        "",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(r"图像路径[：:]\s*[^\n]+", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r"Image\s+Path[：:]\s*[^\n]+", "", caption, flags=re.IGNORECASE)
    caption = re.sub(
        r'["\']image_path["\']\s*:\s*["\'][^"\']+["\']',
        "",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(
        r'["\']img_path["\']\s*:\s*["\'][^"\']+["\']',
        "",
        caption,
        flags=re.IGNORECASE,
    )
    caption = re.sub(
        r"Entity:\s*[^\n]+\(image\)\s*\n\s*\n",
        "",
        caption,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    caption = re.sub(r"\n\s*\n\s*\n+", "\n\n", caption)
    return caption.strip()


def _inline_caption_from_similar_image_row(img_info: Dict[str, Any]) -> str:
    meta = img_info.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    raw = (
        img_info.get("caption")
        or img_info.get("content")
        or meta.get("content")
        or meta.get("caption")
        or ""
    )
    return _clean_similar_image_caption_text(str(raw)) if raw else ""


@dataclass
class QueryParam:
    """Configuration parameters for query execution"""
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    only_need_context: bool = False
    only_need_prompt: bool = False
    model_func: Optional[Callable] = None  # Optional model function override
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    top_k: int = 40
    chunk_top_k: int = 40
    max_entity_tokens: int = 2000
    max_relation_tokens: int = 2000
    max_total_tokens: int = 12000
    conversation_history: List[Dict[str, str]] = field(default_factory=list)  # Conversation history
    history_turns: int = 3
    user_prompt: Optional[str] = None
    enable_rerank: bool = True
    include_references: bool = False


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.
    """
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop
    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


class QueryMixin:
    """QueryMixin class containing query functionality for PathPocket"""

    def _generate_multimodal_cache_key(
        self, query: str, multimodal_content: List[Dict[str, Any]], mode: str, **kwargs
    ) -> str:
        """
        Generate cache key for multimodal query

        Args:
            query: Base query text
            multimodal_content: List of multimodal content
            mode: Query mode
            **kwargs: Additional parameters

        Returns:
            str: Cache key hash
        """
        # Create a normalized representation of the query parameters
        cache_data = {
            "query": query.strip(),
            "mode": mode,
        }

        # Normalize multimodal content for stable caching
        normalized_content = []
        if multimodal_content:
            import hashlib
            for item in multimodal_content:
                if isinstance(item, dict):
                    normalized_item = {}
                    for key, value in item.items():
                        # For image paths, use image content hash for stable caching
                        # This ensures same images get same cache key regardless of path
                        if key in [
                            "img_path",
                            "image_path",
                        ] and isinstance(value, str):
                            try:
                                # Try to compute hash of image content
                                if Path(value).exists():
                                    with open(value, "rb") as f:
                                        image_bytes = f.read()
                                        image_hash = hashlib.md5(image_bytes).hexdigest()
                                        normalized_item[f"{key}_hash"] = image_hash
                                        # Also keep basename for reference
                                        normalized_item[f"{key}_name"] = Path(value).name
                                    continue
                            except Exception:
                                # If can't read image, fallback to basename
                                pass
                            # Fallback: use basename if can't hash
                            normalized_item[key] = Path(value).name
                        # For base64-encoded images, use the base64 hash
                        elif key == "image_base64" and isinstance(value, str):
                            # Hash the base64 string to create stable cache key
                            normalized_item[f"{key}_hash"] = hashlib.md5(
                                value.encode()
                            ).hexdigest()
                        # For file paths (non-image), use basename
                        elif key == "file_path" and isinstance(value, str):
                            normalized_item[key] = Path(value).name
                        # For large content, create a hash instead of storing directly
                        elif (
                            key in ["table_data", "table_body"]
                            and isinstance(value, str)
                            and len(value) > 200
                        ):
                            normalized_item[f"{key}_hash"] = hashlib.md5(
                                value.encode()
                            ).hexdigest()
                        else:
                            normalized_item[key] = value
                    normalized_content.append(normalized_item)
                else:
                    normalized_content.append(item)

        cache_data["multimodal_content"] = normalized_content

        # Add relevant kwargs to cache data
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "stream",
                "response_type",
                "top_k",
                "max_tokens",
                "temperature",
            ]
        }
        cache_data.update(relevant_kwargs)

        # Generate hash from the cache data
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()

        return f"multimodal_query:{cache_hash}"

    async def aquery(
        self, query: str, mode: str = "mix", system_prompt: str | None = None, **kwargs
    ) -> str:
        """
        Pure text query - directly calls RAG engine's query functionality

        Args:
            query: Query text
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            system_prompt: Optional system prompt to include.
            **kwargs: Other query parameters, will be passed to QueryParam
                - vlm_enhanced: bool, default True when vision_model_func is available.

        Returns:
            str: Query result
        """
        if self.rag_engine is None:
            raise ValueError(
                "No RAG engine instance available. Please process documents first or provide a pre-initialized RAG engine instance."
            )

        # Check if VLM enhanced query should be used
        vlm_enhanced = kwargs.pop("vlm_enhanced", None)

        # Auto-determine VLM enhanced based on availability
        if vlm_enhanced is None:
            vlm_enhanced = (
                hasattr(self, "vision_model_func")
                and self.vision_model_func is not None
            )

        # Use VLM enhanced query if enabled and available
        if (
            vlm_enhanced
            and hasattr(self, "vision_model_func")
            and self.vision_model_func
        ):
            return await self.aquery_vlm_enhanced(
                query, mode=mode, system_prompt=system_prompt, **kwargs
            )
        elif vlm_enhanced and (
            not hasattr(self, "vision_model_func") or not self.vision_model_func
        ):
            self.logger.warning(
                "VLM enhanced query requested but vision_model_func is not available, falling back to normal query"
            )

        # Create query parameters
        query_param = QueryParam(mode=mode, **kwargs)

        self.logger.info(f"Executing text query: {query[:100]}...")
        self.logger.info(f"Query mode: {mode}")

        # Call RAG engine's query method
        result = await self.rag_engine.aquery(
            query, param=query_param, system_prompt=system_prompt
        )

        self.logger.info("Text query completed")
        return result

    async def aquery_with_multimodal(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]] = None,
        mode: str = "mix",
        **kwargs,
    ) -> str:
        """
        Multimodal query - combines text and multimodal content for querying

        Args:
            query: Base query text
            multimodal_content: List of multimodal content
            mode: Query mode
            **kwargs: Other query parameters

        Returns:
            str: Query result
        """
        # Ensure RAG engine is initialized
        await self._ensure_rag_engine_initialized()

        self.logger.info(f"Executing multimodal query: {query[:100]}...")
        self.logger.info(f"Query mode: {mode}")

        # If no multimodal content, fallback to pure text query
        if not multimodal_content:
            self.logger.info("No multimodal content provided, executing text query")
            return await self.aquery(query, mode=mode, **kwargs)

        # Check if we have images and vision_model_func is available
        has_images = any(c.get("type") == "image" for c in multimodal_content)
        can_use_vision = hasattr(self, "vision_model_func") and self.vision_model_func
        
        self.logger.debug(f"Multimodal query check: has_images={has_images}, can_use_vision={can_use_vision}")
        
        # If we have images and vision model is available, use it directly
        if has_images and can_use_vision:
            self.logger.info("Using vision model with knowledge graph context for image query")
            
            # For image queries, retrieve KG context WITHOUT using cache
            # This ensures fresh context based on current query, not cached text query results
            # Pass disable_cache=True to ensure we get fresh results
            try:
                # Temporarily disable cache for KG context retrieval in image queries
                # to avoid using cached text query results
                original_cache_setting = None
                if (
                    hasattr(self, "rag_engine")
                    and self.rag_engine
                    and hasattr(self.rag_engine, "llm_response_cache")
                    and self.rag_engine.llm_response_cache
                ):
                    original_cache_setting = self.rag_engine.llm_response_cache.global_config.get(
                        "enable_llm_cache", True
                    )
                    # Temporarily disable cache for this KG context retrieval
                    self.rag_engine.llm_response_cache.global_config["enable_llm_cache"] = False
                
                try:
                    kg_result = await self.aquery(query, mode=mode, **kwargs)
                    kg_context = kg_result if kg_result else ""
                    self.logger.debug(f"Retrieved KG context length: {len(kg_context)}")
                finally:
                    # Restore original cache setting
                    if original_cache_setting is not None:
                        self.rag_engine.llm_response_cache.global_config["enable_llm_cache"] = original_cache_setting
            except Exception as e:
                self.logger.warning(f"Failed to retrieve KG context: {e}")
                kg_context = ""
            
            # Get all image paths
            image_paths = [c.get("img_path") for c in multimodal_content if c.get("type") == "image" and c.get("img_path")]
            
            if image_paths:
                self.logger.info(f"Processing {len(image_paths)} images with vision model (parallel processing enabled)")
                try:
                    import base64
                    from pathlib import Path
                    
                    # Retrieve similar images with structured data (parallel processing)
                    image_retrieval_data = await self._retrieve_similar_images_from_multimodal_content(
                        multimodal_content,
                        top_k_per_image=3,
                        return_structured=True,
                        rerank_query=query,
                        enable_rerank=kwargs.get("enable_rerank", True),
                    )
                    
                    similar_entity_names = image_retrieval_data.get("entity_names", [])
                    similar_chunk_ids = image_retrieval_data.get("chunk_ids", [])
                    image_retrieval_context = image_retrieval_data.get("text_context", "")
                    
                    # Build enhanced KG context with similar images' entities
                    enhanced_kg_context = kg_context
                    if similar_entity_names:
                        # Query entities by name to get more context
                        if hasattr(self, "rag_engine") and self.rag_engine and hasattr(self.rag_engine, "entities_vdb"):
                            try:
                                entities_vdb = self.rag_engine.entities_vdb
                                if entities_vdb:
                                    for entity_name in similar_entity_names[:5]:  # Limit to 5 entities
                                        try:
                                            entity_results = await entities_vdb.query(entity_name, top_k=2)
                                            if entity_results:
                                                for entity in entity_results:
                                                    entity_content = entity.get("content", entity.get("entity_name", ""))
                                                    if entity_content:
                                                        enhanced_kg_context += f"\n\nRelated entity '{entity_name}': {entity_content[:200]}"
                                        except Exception as e:
                                            self.logger.debug(f"Error querying entity {entity_name}: {e}")
                            except Exception as e:
                                self.logger.warning(f"Error accessing entities_vdb: {e}")
                    
                    # Get chunks from similar images
                    similar_chunks_text = ""
                    if similar_chunk_ids and hasattr(self, "rag_engine") and self.rag_engine:
                        try:
                            text_chunks_db = self.rag_engine.text_chunks
                            if text_chunks_db:
                                for chunk_id in similar_chunk_ids[:5]:  # Limit to 5 chunks
                                    try:
                                        chunk_data = await text_chunks_db.get_by_id(chunk_id)
                                        if chunk_data:
                                            if isinstance(chunk_data, dict):
                                                chunk_content = chunk_data.get("content", chunk_data.get("chunk", ""))
                                            else:
                                                chunk_content = str(chunk_data)
                                            if chunk_content:
                                                _cc = strip_chunk_local_path_hints_for_llm(chunk_content)
                                                similar_chunks_text += f"\n\nRelated chunk: {_cc[:300]}"
                                    except Exception as e:
                                        self.logger.debug(f"Error getting chunk {chunk_id}: {e}")
                        except Exception as e:
                            self.logger.warning(f"Error retrieving chunks: {e}")
                    
                    # Process all images in parallel (encode to base64)
                    async def encode_image(image_path: str):
                        """Encode image to base64"""
                        if not Path(image_path).exists():
                            raise FileNotFoundError(f"Image file not found: {image_path}")
                        with open(image_path, "rb") as f:
                            image_bytes = f.read()
                            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                            return image_base64.strip().replace("\n", "").replace("\r", "").replace(" ", "")
                    
                    # Parallel encode all images
                    encode_tasks = [encode_image(img_path) for img_path in image_paths]
                    image_base64_list = await asyncio.gather(*encode_tasks, return_exceptions=True)
                    
                    # Filter out exceptions and use valid images
                    valid_images = [img for img in image_base64_list if not isinstance(img, Exception)]
                    if not valid_images:
                        raise ValueError("No valid images could be encoded")
                    
                    # Use first image for vision model (can be extended to support multiple images)
                    image_base64 = valid_images[0]
                    
                    # Build enhanced prompt with KG context, similar images info, and related chunks
                    enhanced_prompt_parts = [f"User question: {query}"]
                    
                    if enhanced_kg_context:
                        enhanced_prompt_parts.append(f"\n\nRelevant knowledge from knowledge graph:\n{enhanced_kg_context}")
                    
                    if image_retrieval_context:
                        enhanced_prompt_parts.append(f"\n\n{image_retrieval_context}")
                    
                    if similar_chunks_text:
                        enhanced_prompt_parts.append(f"\n\n=== Related Text Chunks from Similar Images ==={similar_chunks_text}")
                    
                    enhanced_prompt_parts.append("\n\nPlease analyze the image and provide a comprehensive answer based on:")
                    enhanced_prompt_parts.append("1. The image content you see")
                    if enhanced_kg_context:
                        enhanced_prompt_parts.append("2. The relevant knowledge from the knowledge graph above")
                    if image_retrieval_context:
                        enhanced_prompt_parts.append("3. The similar images found in the knowledge base")
                    if similar_chunks_text:
                        enhanced_prompt_parts.append("4. The related text chunks from similar images")
                    enhanced_prompt_parts.append("\nCombine all sources of information to give a detailed and accurate answer.")
                    
                    enhanced_prompt = "".join(enhanced_prompt_parts)
                    
                    # Call vision model with enhanced prompt
                    vision_answer = await self.vision_model_func(
                        enhanced_prompt,
                        image_data=image_base64,
                        system_prompt="You are a professional medical image analyst. Analyze the image carefully and provide detailed information based on both the image content and the relevant medical knowledge provided."
                    )
                    
                    self.logger.info("Vision model query completed with enhanced context from similar images")
                    return vision_answer
                except Exception as e:
                    self.logger.warning(f"Failed to use vision model: {e}, falling back to enhanced query")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    # Fall through to enhanced query method

        # For image queries, disable cache to ensure fresh results based on actual image content
        # Image queries should not use cache because:
        # 1. Same query text with different images should produce different results
        # 2. Image content hash in cache key may not be reliable for temporary uploaded files
        # 3. Similar images retrieved from VDB may vary, affecting the final answer
        has_images = any(c.get("type") == "image" for c in multimodal_content) if multimodal_content else False
        
        # Only check cache for non-image multimodal queries (e.g., tables, latex)
        cached_result = None
        if not has_images:
            # Generate cache key for multimodal query (non-image)
            cache_key = self._generate_multimodal_cache_key(
                query, multimodal_content, mode, **kwargs
            )
            
            # Check cache if available and enabled
            if (
                hasattr(self, "rag_engine")
                and self.rag_engine
                and hasattr(self.rag_engine, "llm_response_cache")
                and self.rag_engine.llm_response_cache
            ):
                if self.rag_engine.llm_response_cache.global_config.get(
                    "enable_llm_cache", True
                ):
                    try:
                        cached_result = await self.rag_engine.llm_response_cache.get_by_id(
                            cache_key
                        )
                        if cached_result and isinstance(cached_result, dict):
                            result_content = cached_result.get("return")
                            if result_content:
                                self.logger.info(
                                    f"Multimodal query cache hit: {cache_key[:16]}..."
                                )
                                return result_content
                    except Exception as e:
                        self.logger.debug(f"Error accessing multimodal query cache: {e}")
        else:
            # For image queries, skip cache to ensure fresh results
            self.logger.info("Image query detected - skipping cache to ensure fresh results based on actual image content")

        # Process multimodal content to generate enhanced query text
        enhanced_query = await self._process_multimodal_query_content(
            query, multimodal_content
        )

        self.logger.info(
            f"Generated enhanced query length: {len(enhanced_query)} characters"
        )
        
        # Retrieve similar images with structured data (parallel processing enabled)
        image_retrieval_data = await self._retrieve_similar_images_from_multimodal_content(
            multimodal_content,
            top_k_per_image=3,
            return_structured=True,
            rerank_query=query,
            enable_rerank=kwargs.get("enable_rerank", True),
        )
        
        # Use similar images' entities for extended retrieval
        similar_entity_names = image_retrieval_data.get("entity_names", [])
        similar_chunk_ids = image_retrieval_data.get("chunk_ids", [])
        image_retrieval_context = image_retrieval_data.get("text_context", "")
        
        # Build enhanced query with entity names from similar images
        if similar_entity_names:
            entity_context = f"\n\nRelated entities from similar images: {', '.join(similar_entity_names)}"
            enhanced_query += entity_context
            self.logger.info(f"Added {len(similar_entity_names)} entity names from similar images to query")
        
        # Add image retrieval text context
        if image_retrieval_context:
            enhanced_query += "\n\n" + image_retrieval_context
            self.logger.info("Added image retrieval context to query")
        
        # Get chunks from similar images and add to context
        similar_chunks_context = ""
        if similar_chunk_ids and hasattr(self, "rag_engine") and self.rag_engine:
            try:
                text_chunks_db = self.rag_engine.text_chunks
                if text_chunks_db:
                    similar_chunks = []
                    for chunk_id in similar_chunk_ids[:10]:  # Limit to 10 chunks to avoid too much context
                        try:
                            chunk_data = await text_chunks_db.get_by_id(chunk_id)
                            if chunk_data:
                                # Handle different storage formats
                                if isinstance(chunk_data, dict):
                                    chunk_content = chunk_data.get("content", chunk_data.get("chunk", ""))
                                else:
                                    chunk_content = str(chunk_data)
                                if chunk_content:
                                    _cc = strip_chunk_local_path_hints_for_llm(chunk_content)
                                    similar_chunks.append(f"Chunk {chunk_id}: {_cc[:200]}...")
                        except Exception as e:
                            self.logger.debug(f"Error getting chunk {chunk_id}: {e}")
                            continue
                    
                    if similar_chunks:
                        similar_chunks_context = "\n\n=== Related Text Chunks from Similar Images ===\n"
                        similar_chunks_context += "\n".join(similar_chunks[:5])  # Limit to 5 chunks in context
                        enhanced_query += similar_chunks_context
                        self.logger.info(f"Added {len(similar_chunks)} text chunks from similar images to context")
            except Exception as e:
                self.logger.warning(f"Error retrieving chunks from similar images: {e}")
        
        # Execute enhanced query
        result = await self.aquery(enhanced_query, mode=mode, **kwargs)

        # Save to cache if available and enabled
        if (
            hasattr(self, "rag_engine")
            and self.rag_engine
            and hasattr(self.rag_engine, "llm_response_cache")
            and self.rag_engine.llm_response_cache
        ):
            if self.rag_engine.llm_response_cache.global_config.get(
                "enable_llm_cache", True
            ):
                try:
                    # Create cache entry for multimodal query
                    cache_entry = {
                        "return": result,
                        "cache_type": "multimodal_query",
                        "original_query": query,
                        "multimodal_content_count": len(multimodal_content),
                        "mode": mode,
                    }

                    await self.rag_engine.llm_response_cache.upsert(
                        {cache_key: cache_entry}
                    )
                    self.logger.info(
                        f"Saved multimodal query result to cache: {cache_key[:16]}..."
                    )
                except Exception as e:
                    self.logger.debug(f"Error saving multimodal query to cache: {e}")

        # Ensure cache is persisted to disk
        if (
            hasattr(self, "rag_engine")
            and self.rag_engine
            and hasattr(self.rag_engine, "llm_response_cache")
            and self.rag_engine.llm_response_cache
        ):
            try:
                await self.rag_engine.llm_response_cache.index_done_callback()
            except Exception as e:
                self.logger.debug(f"Error persisting multimodal query cache: {e}")

        self.logger.info("Multimodal query completed")
        return result

    async def aquery_vlm_enhanced(
        self, query: str, mode: str = "mix", system_prompt: str | None = None, 
        multimodal_content: List[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """
        VLM enhanced query - uses VLM to process images and text
        
        If multimodal_content is provided, uses those images directly.
        Otherwise, extracts image paths from retrieved context and processes them.

        Args:
            query: User query
            mode: Underlying RAG engine query mode
            system_prompt: Optional system prompt to include
            multimodal_content: Optional list of multimodal content (e.g., images)
                Each element should have:
                - type: "image"
                - img_path: Path to image file
            **kwargs: Other query parameters

        Returns:
            str: VLM query result
        """
        # Ensure VLM is available
        if not hasattr(self, "vision_model_func") or not self.vision_model_func:
            raise ValueError(
                "VLM enhanced query requires vision_model_func. "
                "Please provide a vision model function when initializing PathPocket."
            )

        # Ensure RAG engine is initialized
        await self._ensure_rag_engine_initialized()

        self.logger.info(f"Executing VLM enhanced query: {query[:100]}...")

        # Clear previous image cache
        if hasattr(self, "_current_images_base64"):
            delattr(self, "_current_images_base64")

        # Initialize image cache
        self._current_images_base64 = []
        images_found = 0
        enhanced_prompt = ""

        # If multimodal_content is provided, use those images directly
        if multimodal_content:
            self.logger.info(f"Processing {len(multimodal_content)} multimodal content items")
            
            # First, retrieve relevant context from knowledge graph
            try:
                query_param = QueryParam(mode=mode, only_need_prompt=True, **kwargs)
                kg_context = await self.rag_engine.aquery(query, param=query_param)
                if not isinstance(kg_context, str):
                    kg_context = str(kg_context) if kg_context is not None else ""
                enhanced_prompt = kg_context
            except Exception as e:
                self.logger.warning(f"Failed to retrieve KG context: {e}")
                enhanced_prompt = ""

            # Process images from multimodal_content
            import base64
            from pathlib import Path
            
            for i, content in enumerate(multimodal_content):
                if content.get("type") == "image":
                    image_path = content.get("img_path")
                    if image_path and Path(image_path).exists():
                        try:
                            with open(image_path, "rb") as f:
                                image_bytes = f.read()
                                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                                # Clean base64 string
                                image_base64 = image_base64.strip().replace("\n", "").replace("\r", "").replace(" ", "")
                                self._current_images_base64.append(image_base64)
                                images_found += 1
                                # Add image marker to prompt
                                enhanced_prompt += f"\n\nImage Path: {image_path}\n[VLM_IMAGE_{images_found}]"
                                self.logger.info(f"Added image {images_found}: {image_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to process image {image_path}: {e}")
            
            if not images_found:
                self.logger.warning("No valid images found in multimodal_content, falling back to normal query")
                query_param = QueryParam(mode=mode, **kwargs)
                return await self.rag_engine.aquery(
                    query, param=query_param, system_prompt=system_prompt
                )
        else:
            # Original behavior: extract images from retrieved context
            # 1. Get original retrieval prompt (without generating final answer)
            query_param = QueryParam(mode=mode, only_need_prompt=True, **kwargs)
            raw_prompt = await self.rag_engine.aquery(query, param=query_param)

            self.logger.debug("Retrieved raw prompt from RAG engine")

            # Ensure raw_prompt is a string
            if not isinstance(raw_prompt, str):
                raw_prompt = str(raw_prompt) if raw_prompt is not None else ""

            # 2. Extract and process image paths
            enhanced_prompt, images_found = await self._process_image_paths_for_vlm(
                raw_prompt
            )

            if not images_found:
                self.logger.info("No valid images found, falling back to normal query")
                # Fallback to normal query
                query_param = QueryParam(mode=mode, **kwargs)
                return await self.rag_engine.aquery(
                    query, param=query_param, system_prompt=system_prompt
                )

        self.logger.info(f"Processed {images_found} images for VLM")

        # 3. Build VLM message format
        messages = self._build_vlm_messages_with_images(
            enhanced_prompt, query, system_prompt
        )

        # 4. Call VLM for question answering
        result = await self._call_vlm_with_multimodal_content(messages)

        self.logger.info("VLM enhanced query completed")
        return result

    async def _process_multimodal_query_content(
        self, base_query: str, multimodal_content: List[Dict[str, Any]]
    ) -> str:
        """
        Process multimodal query content to generate enhanced query text

        Args:
            base_query: Base query text
            multimodal_content: List of multimodal content

        Returns:
            str: Enhanced query text
        """
        self.logger.info("Starting multimodal query content processing...")

        enhanced_parts = [f"User query: {base_query}"]

        for i, content in enumerate(multimodal_content):
            content_type = content.get("type", "unknown")
            self.logger.info(
                f"Processing {i+1}/{len(multimodal_content)} multimodal content: {content_type}"
            )

            try:
                # Get appropriate processor
                processor = get_processor_for_type(self.modal_processors, content_type)

                if processor:
                    # Generate content description
                    description = await self._generate_query_content_description(
                        processor, content, content_type
                    )
                    enhanced_parts.append(
                        f"\nRelated {content_type} content: {description}"
                    )
                else:
                    # If no appropriate processor, use basic description
                    basic_desc = str(content)[:200]
                    enhanced_parts.append(
                        f"\nRelated {content_type} content: {basic_desc}"
                    )

            except Exception as e:
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                # Continue processing other content
                continue

        enhanced_query = "\n".join(enhanced_parts)
        enhanced_query += PROMPTS.get("QUERY_ENHANCEMENT_SUFFIX", "\n\nPlease answer the user's question based on the provided context and multimodal content.")

        self.logger.info("Multimodal query content processing completed")
        return enhanced_query

    async def _generate_query_content_description(
        self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """
        Generate content description for query

        Args:
            processor: Multimodal processor
            content: Content data
            content_type: Content type

        Returns:
            str: Content description
        """
        try:
            if content_type == "image":
                return await self._describe_image_for_query(processor, content)
            elif content_type == "table":
                return await self._describe_table_for_query(processor, content)
            else:
                return await self._describe_generic_for_query(
                    processor, content, content_type
                )

        except Exception as e:
            self.logger.error(f"Error generating {content_type} description: {str(e)}")
            return f"{content_type} content: {str(content)[:100]}"

    async def _describe_image_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate image description for query"""
        image_path = content.get("img_path")
        captions = content.get("image_caption", content.get("img_caption", []))
        footnotes = content.get("image_footnote", content.get("img_footnote", []))

        if image_path and Path(image_path).exists():
            # If image exists, use vision model to generate description
            image_base64 = encode_image_to_base64(image_path)
            if image_base64:
                prompt = PROMPTS.get("QUERY_IMAGE_DESCRIPTION", "Describe this image in detail.")
                description = await processor.modal_caption_func(
                    prompt,
                    image_data=image_base64,
                    system_prompt=PROMPTS.get("QUERY_IMAGE_ANALYST_SYSTEM", "You are an expert at analyzing images."),
                )
                return description
        return ""

    async def _batch_fetch_chunk_contents_by_ids(
        self, chunk_ids: List[str]
    ) -> Dict[str, str]:
        """Batch-load chunk bodies for similar-image caption hydration."""
        ids = list(dict.fromkeys(c for c in chunk_ids if c))
        if not ids:
            return {}
        out: Dict[str, str] = {}
        rag_engine = getattr(self, "rag_engine", None)
        if not rag_engine:
            return out
        try:
            text_chunks = getattr(rag_engine, "text_chunks", None)
            if text_chunks and hasattr(text_chunks, "get_by_ids"):
                rows = await text_chunks.get_by_ids(ids)
                for item in rows or []:
                    if isinstance(item, dict) and item.get("id"):
                        content = item.get("content", "")
                        if content:
                            out[str(item["id"])] = str(content)
                missing = [i for i in ids if i not in out]
            else:
                missing = ids

            if missing and getattr(rag_engine, "chunks_vdb", None):
                chunks_vdb = rag_engine.chunks_vdb
                db = getattr(chunks_vdb, "db", None)
                if db:
                    chunk_sql = (
                        "SELECT id, content FROM VDB_CHUNKS "
                        "WHERE workspace = $1 AND id = ANY($2::varchar[])"
                    )
                    chunk_results = await db.query(
                        chunk_sql,
                        [chunks_vdb.workspace, missing],
                        multirows=True,
                    )
                    if isinstance(chunk_results, dict):
                        chunk_results = [chunk_results]
                    for item in chunk_results or []:
                        if isinstance(item, dict) and item.get("id"):
                            content = item.get("content", "")
                            if content:
                                out[str(item["id"])] = str(content)
        except Exception as ex:
            self.logger.debug("Batch chunk fetch for image captions failed: %s", ex)
        return out

    async def _hydrate_similar_image_row_captions(
        self, rows: List[Dict[str, Any]]
    ) -> None:
        """Fill ``_caption`` on each row: inline VDB content first, then batch chunk lookup."""
        if not rows:
            return
        for row in rows:
            inline = _inline_caption_from_similar_image_row(row)
            if _is_usable_similar_image_caption(inline):
                row["_caption"] = inline

        missing = [
            r
            for r in rows
            if not r.get("_caption") and str(r.get("chunk_id") or "").strip()
        ]
        if missing:
            chunk_map = await self._batch_fetch_chunk_contents_by_ids(
                [str(r["chunk_id"]) for r in missing]
            )
            for row in missing:
                cid = str(row.get("chunk_id") or "").strip()
                raw = chunk_map.get(cid, "")
                cap = _clean_similar_image_caption_text(raw) if raw else ""
                if _is_usable_similar_image_caption(cap):
                    row["_caption"] = cap

        for row in rows:
            if row.get("_caption"):
                continue
            cap = (await self._resolve_similar_image_caption_text(row)).strip()
            if cap:
                row["_caption"] = cap

    async def _resolve_similar_image_caption_text(
        self, img_info: Dict[str, Any]
    ) -> str:
        """
        从行内 content / chunk / 图实体补全并清洗一条相似图像的 caption。
        优先使用 VDB_IMAGES 已返回的 content，避免重复 DB 查询。
        """
        inline = _inline_caption_from_similar_image_row(img_info)
        if _is_usable_similar_image_caption(inline):
            return inline
        if img_info.get("_caption"):
            return str(img_info["_caption"]).strip()

        entity_name = img_info.get("entity_name", "")
        chunk_id = img_info.get("chunk_id", "")
        raw_content = ""
        rag_engine = getattr(self, "rag_engine", None)

        if chunk_id and rag_engine:
            try:
                if hasattr(rag_engine, "text_chunks") and rag_engine.text_chunks:
                    chunk_data = await rag_engine.text_chunks.get_by_id(chunk_id)
                    if chunk_data and isinstance(chunk_data, dict):
                        raw_content = chunk_data.get("content", "")
                if (
                    not raw_content
                    and hasattr(rag_engine, "chunks_vdb")
                    and rag_engine.chunks_vdb
                ):
                    if hasattr(rag_engine.chunks_vdb, "db") and rag_engine.chunks_vdb.db:
                        chunk_sql = (
                            "SELECT content FROM VDB_CHUNKS WHERE workspace = $1 AND id = $2"
                        )
                        chunk_results = await rag_engine.chunks_vdb.db.query(
                            chunk_sql,
                            [rag_engine.chunks_vdb.workspace, chunk_id],
                        )
                        if chunk_results:
                            if isinstance(chunk_results, dict):
                                chunk_results = [chunk_results]
                            raw_content = chunk_results[0].get("content", "")
            except Exception:
                pass

        final_content = _clean_similar_image_caption_text(raw_content) if raw_content else ""

        if not _is_usable_similar_image_caption(final_content) and entity_name and rag_engine:
            try:
                if hasattr(rag_engine, "chunk_entity_relation_graph"):
                    node_data = await rag_engine.chunk_entity_relation_graph.get_node(
                        entity_name
                    )
                    if node_data and isinstance(node_data, dict):
                        desc = _clean_similar_image_caption_text(
                            node_data.get("description", "")
                        )
                        if _is_usable_similar_image_caption(desc):
                            final_content = desc
            except Exception:
                pass

            if not _is_usable_similar_image_caption(final_content):
                try:
                    if hasattr(rag_engine, "entities_vdb"):
                        from pathpocket.lightrag_utils import compute_mdhash_id

                        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
                        vdb_data = await rag_engine.entities_vdb.get_by_id(entity_id)
                        if vdb_data:
                            vdb_content = vdb_data.get("content", "")
                            if vdb_content:
                                desc = (
                                    vdb_content.split("\n", 1)[-1]
                                    if "\n" in vdb_content
                                    else vdb_content
                                )
                                desc = _clean_similar_image_caption_text(desc)
                                if _is_usable_similar_image_caption(desc):
                                    final_content = desc
                except Exception:
                    pass

        return final_content.strip()

    def _vector_candidate_top_k_for_similar_images(self, top_k_per_image: int) -> int:
        """向量库多取候选，便于相似度阈值过滤后再 rerank、按分数取 top_k。"""
        k = int(top_k_per_image) if top_k_per_image else 1
        return min(300, max(50, k * 50))

    async def _pick_top_similar_images_by_rerank(
        self,
        rows: List[Dict[str, Any]],
        rerank_query: str,
        enable_rerank: bool,
        structured_retrieval: Optional[Dict[str, Any]],
        top_k: int,
    ) -> tuple[List[Dict[str, Any]], bool, bool]:
        """
        对单张查询图对应的候选行：rerank 后按 rerank_score 降序取前 top_k 条。

        Returns:
            (picked_rows, caption_rerank_applied, per_option_style)
        """
        rq = (rerank_query or "").strip()
        _enable_rr = True if enable_rerank is None else bool(enable_rerank)
        if not rows or top_k <= 0:
            return [], False, False
        if not rq or not _enable_rr or not self.rag_engine:
            rows_sorted = sorted(
                rows,
                key=lambda x: float(x.get("similarity", 0.0) or 0.0),
                reverse=True,
            )
            return rows_sorted[:top_k], False, False

        from pathpocket.operate import (
            _extract_stem_from_mcq_query,
            _max_rerank_scores_across_queries,
        )

        gc = getattr(self.rag_engine, "__dict__", None) or {}
        if not gc.get("rerank_model_func"):
            rows_sorted = sorted(
                rows,
                key=lambda x: float(x.get("similarity", 0.0) or 0.0),
                reverse=True,
            )
            return rows_sorted[:top_k], False, False

        sr = structured_retrieval if isinstance(structured_retrieval, dict) else {}
        mix_active = bool(sr.get("mix_structured_retrieval_active"))
        cand_answers = [
            c.strip()
            for c in (sr.get("rerank_query_strings_per_option") or [])
            if isinstance(c, str) and c.strip()
        ]
        if mix_active and cand_answers:
            queries = cand_answers
            per_option_style = True
        elif mix_active:
            stem = _extract_stem_from_mcq_query(rq).strip() or rq
            queries = [stem] if stem.strip() else []
            per_option_style = False
        else:
            queries = [rq]
            per_option_style = False

        if not queries:
            rows_sorted = sorted(
                rows,
                key=lambda x: float(x.get("similarity", 0.0) or 0.0),
                reverse=True,
            )
            return rows_sorted[:top_k], False, False

        await self._hydrate_similar_image_row_captions(rows)
        docs_for_rr: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            d.pop("_caption", None)
            cap = str(row.get("_caption") or "").strip()
            if not cap:
                en = (d.get("entity_name") or "").strip()
                cap = f"[Image entity] {en}" if en else (
                    Path(d.get("image_path") or "").name or "image"
                )
            d["content"] = cap[:12000]
            docs_for_rr.append(d)
        doc_tx = [d["content"] for d in docs_for_rr]

        try:
            max_s, best_q, rq_used = await _max_rerank_scores_across_queries(
                doc_tx, queries, gc, True
            )
            scored: List[Dict[str, Any]] = []
            for i, row in enumerate(docs_for_rr):
                sc = float(max_s.get(i, 0.0))
                rc = dict(row)
                rc["rerank_score"] = sc
                rc["rerank_query"] = best_q.get(i, "")
                rc["rerank_queries"] = list(rq_used)
                scored.append(rc)
            scored.sort(
                key=lambda x: float(x.get("rerank_score", 0.0) or 0.0),
                reverse=True,
            )
            min_rr_img = float(
                getattr(
                    self.rag_engine,
                    "min_rerank_score_similar_images",
                    getattr(self.rag_engine, "min_rerank_score", 0.5),
                )
                or 0.0
            )
            if min_rr_img > 0.0:
                scored = [
                    x
                    for x in scored
                    if float(x.get("rerank_score", 0.0) or 0.0) >= min_rr_img
                ]
            return scored[:top_k], True, per_option_style
        except Exception as ex:
            self.logger.warning(
                "Per-query-image similar image rerank failed, using vector order: %s",
                ex,
            )
            rows_sorted = sorted(
                rows,
                key=lambda x: float(x.get("similarity", 0.0) or 0.0),
                reverse=True,
            )
            return rows_sorted[:top_k], False, False

    async def _retrieve_similar_images_from_multimodal_content(
        self,
        multimodal_content: List[Dict[str, Any]],
        top_k_per_image: int = 3,
        return_structured: bool = False,
        rerank_query: Optional[str] = None,
        enable_rerank: Optional[bool] = None,
        rerank_top_n: Optional[int] = None,
        structured_retrieval: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        从多模态内容中检索相似图像（支持并行处理）
        
        Args:
            multimodal_content: 多模态内容列表
            top_k_per_image: 每张查询图在「向量相似度 ≥ 引擎阈值」的候选上，按 rerank 分数取前该条数；
                向量侧会先多取候选再过滤（见 ``_vector_candidate_top_k_for_similar_images``）。
            return_structured: 如果为True，返回结构化数据（实体列表、文档块ID列表等）；否则返回格式化的文本
            rerank_query: 用户查询文本；与 operate 中关系/chunk 一致：mix 结构化且 metadata 含选项时用
                per-option rerank，否则 mix 结构化用题干，否则用全文。
            enable_rerank: 是否重排；None 时默认 True
            rerank_top_n: 已弃用；保留兼容。最终条数由 ``top_k_per_image`` 决定。
            structured_retrieval: ``aquery_data`` 返回的 ``metadata.structured_retrieval``（含
                ``mix_structured_retrieval_active``、``rerank_query_strings_per_option``、阈值等）
        
        Returns:
            如果 return_structured=True: Dict包含:
                - similar_images: List[Dict] - 所有相似图像信息
                - entity_names: List[str] - 所有相似图像关联的实体名称
                - chunk_ids: List[str] - 所有相似图像关联的文档块ID
                - text_context: str - 格式化的文本上下文
            否则: str - 格式化的相似图像检索结果文本
        """
        if not multimodal_content:
            return {} if return_structured else ""
        
        # 检查pathology_images_vdb是否可用
        if not hasattr(self, "rag_engine") or not self.rag_engine:
            return {} if return_structured else ""
        
        if not hasattr(self.rag_engine, "pathology_images_vdb") or not self.rag_engine.pathology_images_vdb:
            self.logger.debug("pathology_images_vdb not available, skipping image retrieval")
            return {} if return_structured else ""
        
        pathology_images_vdb = self.rag_engine.pathology_images_vdb
        
        # 确保pathology_images_vdb已初始化
        # 支持多种存储类型：SimpleVectorStorage, PGVectorStorage, NanoVectorDBStorage
        try:
            # 首先调用initialize
            if hasattr(pathology_images_vdb, 'initialize'):
                if not getattr(pathology_images_vdb, '_initialized', False):
                    self.logger.info(f"Initializing pathology_images_vdb (type: {type(pathology_images_vdb)})")
                    await pathology_images_vdb.initialize()
                    setattr(pathology_images_vdb, '_initialized', True)
                    self.logger.debug("pathology_images_vdb initialized")
            
            # 对于PGVectorStorage，检查db是否已初始化
            if hasattr(pathology_images_vdb, 'db'):
                if pathology_images_vdb.db is None:
                    self.logger.warning("pathology_images_vdb.db is None, re-initializing...")
                    if hasattr(pathology_images_vdb, 'initialize'):
                        await pathology_images_vdb.initialize()
                        setattr(pathology_images_vdb, '_initialized', True)
                    if pathology_images_vdb.db is None:
                        self.logger.error("pathology_images_vdb.db is still None after re-initialization")
                        return {} if return_structured else ""
                else:
                    self.logger.debug("pathology_images_vdb.db is initialized (PGVectorStorage)")
            
            # 对于SimpleVectorStorage，检查_client是否已初始化
            elif hasattr(pathology_images_vdb, '_client'):
                if pathology_images_vdb._client is None:
                    self.logger.warning("pathology_images_vdb._client is None, re-initializing...")
                    if hasattr(pathology_images_vdb, 'initialize'):
                        await pathology_images_vdb.initialize()
                        setattr(pathology_images_vdb, '_initialized', True)
                    if pathology_images_vdb._client is None:
                        self.logger.error("pathology_images_vdb._client is still None after re-initialization")
                        return {} if return_structured else ""
                else:
                    self.logger.debug("pathology_images_vdb._client is initialized (SimpleVectorStorage)")
            
            # 对于NanoVectorDBStorage，确保客户端可用
            elif hasattr(pathology_images_vdb, '_get_client'):
                try:
                    client = await pathology_images_vdb._get_client()
                    if client is None:
                        self.logger.warning("pathology_images_vdb._get_client() returned None")
                    else:
                        self.logger.debug("pathology_images_vdb client retrieved successfully (NanoVectorDBStorage)")
                except Exception as e:
                    self.logger.error(f"Failed to get pathology_images_vdb client: {e}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    return {} if return_structured else ""
        except Exception as e:
            self.logger.error(f"Failed to initialize pathology_images_vdb: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {} if return_structured else ""
        
        # 提取所有图像路径
        image_paths = []
        for content in multimodal_content:
            content_type = content.get("type", "").lower()
            if content_type == "image":
                image_path = content.get("img_path") or content.get("image_path")
                if image_path and Path(image_path).exists():
                    image_paths.append(image_path)
        
        if not image_paths:
            self.logger.debug("No valid image paths found in multimodal content")
            return {} if return_structured else ""
        
        self.logger.info(
            "Retrieving similar images for %d query images: final top_k=%d per query image "
            "(after vector sim ≥ pathology_images_cosine_better_than_threshold + rerank); "
            "vector stage uses larger candidate pool",
            len(image_paths),
            top_k_per_image,
        )

        # 结构化数据收集（相似图列表在检索后按 query_image 组装）
        all_similar_images: List[Dict[str, Any]] = []
        all_entity_names = set()
        all_chunk_ids = set()
        retrieval_results = []
        
        try:
            # 获取Virchow2特征提取器
            if not hasattr(pathology_images_vdb, "embedding_func") or not pathology_images_vdb.embedding_func:
                self.logger.warning("pathology_images_vdb embedding_func not available")
                return {} if return_structured else ""
            
            virchow2_extractor = pathology_images_vdb.embedding_func.virchow2_extractor
            candidate_k = int(os.getenv("SIMILAR_IMAGES_CANDIDATE_K", "300"))

            sim_thr = float(
                getattr(
                    self.rag_engine,
                    "pathology_images_cosine_better_than_threshold",
                    0.0,
                )
                or 0.0
            )

            # 并行处理：为每个查询图像创建检索任务
            async def retrieve_for_one_image(query_image_path: str):
                """为单个图像检索相似图像"""
                try:
                    # 在查询前再次确保客户端已初始化（防止并发问题）
                    # 这是必要的，因为并行任务可能在初始化完成前就开始执行
                    # 首先确保 initialize() 已被调用
                    if hasattr(pathology_images_vdb, 'initialize'):
                        if not getattr(pathology_images_vdb, '_initialized', False):
                            self.logger.debug(f"Initializing pathology_images_vdb in retrieve_for_one_image for {Path(query_image_path).name}")
                            try:
                                await pathology_images_vdb.initialize()
                                setattr(pathology_images_vdb, '_initialized', True)
                                self.logger.debug("pathology_images_vdb initialized in retrieve_for_one_image")
                            except Exception as init_e:
                                self.logger.error(f"Failed to initialize pathology_images_vdb in retrieve_for_one_image: {init_e}")
                                import traceback
                                self.logger.debug(traceback.format_exc())
                                # 如果初始化失败，返回空结果
                                return {
                                    "query_image": query_image_path,
                                    "similar_images": []
                                }
                    
                    # 对于 SimpleVectorStorage，检查 _client 是否已初始化
                    if hasattr(pathology_images_vdb, '_client'):
                        if pathology_images_vdb._client is None:
                            # 如果 _client 是 None，尝试重新初始化
                            self.logger.warning(f"pathology_images_vdb._client is None, re-initializing for {Path(query_image_path).name}")
                            self.logger.info(f"pathology_images_vdb state: _initialized={getattr(pathology_images_vdb, '_initialized', 'N/A')}, type={type(pathology_images_vdb)}")
                            if hasattr(pathology_images_vdb, 'initialize'):
                                try:
                                    self.logger.info(f"Calling pathology_images_vdb.initialize() for {Path(query_image_path).name}")
                                    await pathology_images_vdb.initialize()
                                    setattr(pathology_images_vdb, '_initialized', True)
                                    self.logger.info(f"After initialize(): _initialized={getattr(pathology_images_vdb, '_initialized', False)}, _client={pathology_images_vdb._client is not None if hasattr(pathology_images_vdb, '_client') else 'N/A'}")
                                    # 再次检查 _client
                                    if pathology_images_vdb._client is None:
                                        self.logger.error(f"pathology_images_vdb._client is still None after re-initialization for {Path(query_image_path).name}")
                                        self.logger.error(f"pathology_images_vdb type: {type(pathology_images_vdb)}")
                                        self.logger.error(f"pathology_images_vdb has _lock: {hasattr(pathology_images_vdb, '_lock')}")
                                        self.logger.error(f"pathology_images_vdb embedding_func: {hasattr(pathology_images_vdb, 'embedding_func') and pathology_images_vdb.embedding_func is not None}")
                                        # Try to force initialization by calling query which will initialize
                                        self.logger.info("Attempting to force initialization by accessing _client through query method...")
                                        return {
                                            "query_image": query_image_path,
                                            "similar_images": []
                                        }
                                except Exception as init_e:
                                    self.logger.error(f"Failed to re-initialize pathology_images_vdb in retrieve_for_one_image: {init_e}")
                                    import traceback
                                    self.logger.error(traceback.format_exc())
                                    return {
                                        "query_image": query_image_path,
                                        "similar_images": []
                                    }
                    
                    # 对于使用 _get_client 的存储（如 NanoVectorDBStorage），确保客户端可用
                    if hasattr(pathology_images_vdb, '_get_client'):
                        try:
                            client = await pathology_images_vdb._get_client()
                            if client is None:
                                self.logger.error(f"pathology_images_vdb._get_client() returned None for {Path(query_image_path).name}")
                                return {
                                    "query_image": query_image_path,
                                    "similar_images": []
                                }
                        except Exception as client_e:
                            self.logger.error(f"Failed to get pathology_images_vdb client in retrieve_for_one_image: {client_e}")
                            import traceback
                            self.logger.debug(traceback.format_exc())
                            return {
                                "query_image": query_image_path,
                                "similar_images": []
                            }
                    
                    # 提取查询图像的特征向量
                    query_features = await virchow2_extractor.extract_features(query_image_path)
                    
                    # 确保是1D数组
                    if isinstance(query_features, np.ndarray):
                        if query_features.ndim > 1:
                            query_features = query_features.flatten()
                        query_embedding = query_features.tolist()
                    elif isinstance(query_features, list):
                        if query_features and isinstance(query_features[0], np.ndarray):
                            query_embedding = query_features[0].flatten().tolist()
                        else:
                            query_embedding = query_features
                    else:
                        query_embedding = list(query_features)
                    
                    # 在pathology_images_vdb中搜索相似图像
                    # query方法内部会调用_get_client()，所以需要确保initialize()已被调用
                    self.logger.info(
                        "Querying pathology_images_vdb for %s: vector_candidate_top_k=%d "
                        "(final per-image k after sim≥threshold + rerank=%d), embedding_dim=%d",
                        Path(query_image_path).name,
                        candidate_k,
                        top_k_per_image,
                        len(query_embedding) if query_embedding else 0,
                    )
                    similar_images = await pathology_images_vdb.query(
                        query="",
                        top_k=candidate_k,
                        query_embedding=query_embedding,
                    )
                    
                    self.logger.info(f"Query returned {len(similar_images) if similar_images else 0} similar images for {Path(query_image_path).name}")
                    if similar_images:
                        first_result = similar_images[0] if len(similar_images) > 0 else None
                        if first_result:
                            self.logger.debug(f"First result keys: {list(first_result.keys()) if isinstance(first_result, dict) else 'N/A'}")
                            self.logger.debug(f"First result distance: {first_result.get('distance', 'N/A') if isinstance(first_result, dict) else 'N/A'}")
                            self.logger.debug(f"First result sample: {str(first_result)[:200] if isinstance(first_result, dict) else str(first_result)[:200]}")
                    else:
                        self.logger.warning(f"No similar images returned from query for {Path(query_image_path).name}")
                        # Check if database is empty
                        try:
                            if hasattr(pathology_images_vdb, '_client') and pathology_images_vdb._client:
                                if hasattr(pathology_images_vdb._client, '__len__'):
                                    db_size = len(pathology_images_vdb._client)
                                    self.logger.info(f"pathology_images_vdb contains {db_size} items")
                                else:
                                    self.logger.warning(f"Cannot determine pathology_images_vdb size (client type: {type(pathology_images_vdb._client)})")
                            else:
                                self.logger.warning(f"pathology_images_vdb._client is None or not available")
                        except Exception as check_e:
                            self.logger.debug(f"Error checking pathology_images_vdb size: {check_e}")
                    
                    return {
                        "query_image": query_image_path,
                        "similar_images": similar_images or []
                    }
                except Exception as e:
                    self.logger.error(f"Error retrieving similar images for {query_image_path}: {e}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    return {
                        "query_image": query_image_path,
                        "similar_images": []
                    }
            
            # 并行执行所有图像的检索
            tasks = [retrieve_for_one_image(img_path) for img_path in image_paths]
            results = await asyncio.gather(*tasks)
            
            # 每张查询图：向量多候选 → 相似度阈值 → caption rerank → 按 rerank_score 取前 top_k
            rq = (rerank_query or "").strip()
            _enable_rr = True if enable_rerank is None else bool(enable_rerank)
            caption_rerank_applied = False
            per_option_rerank = False
            all_similar_images.clear()

            for result in results:
                query_image_path = result["query_image"]
                similar_images = result.get("similar_images") or []
                if not similar_images:
                    self.logger.debug(
                        "No similar images found for %s", Path(query_image_path).name
                    )
                    continue
                rows: List[Dict[str, Any]] = []
                for i, img_result in enumerate(similar_images, 1):
                    if not isinstance(img_result, dict):
                        continue
                    metadata = img_result.get("metadata", {})
                    image_path = metadata.get("image_path") or img_result.get(
                        "image_path", ""
                    )
                    chunk_id = metadata.get("chunk_id") or img_result.get(
                        "chunk_id", ""
                    )
                    file_path = metadata.get("file_path") or img_result.get(
                        "file_path", ""
                    )
                    entity_name = metadata.get("entity_name") or img_result.get(
                        "entity_name", ""
                    )
                    distance = img_result.get("distance")
                    if distance is None:
                        distance = img_result.get("__metrics__", float("inf"))
                    if distance == float("inf") or distance is None:
                        similarity = 0.0
                    elif distance <= 1:
                        similarity = max(0.0, 1.0 - distance)
                    else:
                        similarity = max(0.0, 1.0 - (distance / 2.0))
                    self.logger.debug(
                        "Image result %d: distance=%s similarity=%.4f path=%s",
                        i,
                        distance,
                        similarity,
                        (image_path[:50] if image_path else "N/A"),
                    )
                    content = metadata.get("content") or img_result.get("content", "")
                    rows.append(
                        {
                            "image_path": image_path,
                            "chunk_id": chunk_id,
                            "file_path": file_path,
                            "entity_name": entity_name,
                            "similarity": similarity,
                            "query_image": query_image_path,
                            "content": content,
                            "metadata": metadata,
                        }
                    )
                if sim_thr > 0.0:
                    before_ct = len(rows)
                    rows = [
                        r
                        for r in rows
                        if float(r.get("similarity", 0.0) or 0.0) >= sim_thr
                    ]
                    self.logger.info(
                        "Similar images for %s: vector candidates=%d sim>=%.4f -> %d",
                        Path(query_image_path).name,
                        before_ct,
                        sim_thr,
                        len(rows),
                    )
                picked, cap_ap, po = await self._pick_top_similar_images_by_rerank(
                    rows,
                    rq,
                    _enable_rr,
                    structured_retrieval,
                    top_k_per_image,
                )
                if cap_ap:
                    caption_rerank_applied = True
                if po:
                    per_option_rerank = True
                all_similar_images.extend(picked)
                self.logger.info(
                    "Selected %d similar images for %s (vector_hits=%d, final_k<=%d)",
                    len(picked),
                    Path(query_image_path).name,
                    len(similar_images),
                    top_k_per_image,
                )

            _el_gc = {
                "evidence_level_title_category_json": getattr(
                    self.rag_engine,
                    "evidence_level_title_category_json",
                    None,
                ),
            }
            _tc = getattr(self.rag_engine, "text_chunks", None)
            if _tc and all_similar_images:
                for _sim in all_similar_images:
                    if not isinstance(_sim, dict):
                        continue
                    _fp = str(_sim.get("file_path") or "").strip()
                    if _fp and _fp.lower() not in ("unknown", "unknown_source"):
                        continue
                    _cid = _sim.get("chunk_id")
                    if not _cid:
                        continue
                    try:
                        _chunk_data = await _tc.get_by_id(_cid)
                    except Exception:
                        _chunk_data = None
                    if isinstance(_chunk_data, dict):
                        _cf = str(_chunk_data.get("file_path") or "").strip()
                        if _cf:
                            _sim["file_path"] = _cf
            enrich_evidence_items_from_catalog(all_similar_images, _el_gc)

            for it in all_similar_images:
                en = it.get("entity_name")
                if en:
                    all_entity_names.add(en)
                cid = it.get("chunk_id")
                if cid:
                    all_chunk_ids.add(cid)

            from collections import OrderedDict

            retrieval_results = []
            if all_similar_images:
                by_qimg: "OrderedDict[str, List[dict]]" = OrderedDict()
                for r in all_similar_images:
                    qi = r.get("query_image") or ""
                    by_qimg.setdefault(qi, []).append(r)
                for qi, rows in by_qimg.items():
                    hdr = f"\nSimilar images for query image: {Path(qi).name}"
                    if rq and _enable_rr and caption_rerank_applied:
                        hdr += (
                            " (per-option caption rerank, same as relations)"
                            if per_option_rerank
                            else " (caption rerank applied)"
                        )
                    rp = [hdr]
                    for j, r in enumerate(rows, 1):
                        image_path = r.get("image_path") or ""
                        fp = r.get("file_path") or ""
                        ent = r.get("entity_name") or ""
                        sim = float(r.get("similarity", 0.0) or 0.0)
                        rs = r.get("rerank_score")
                        rs_s = f", rerank_score={float(rs):.3f}" if rs is not None else ""
                        ev_s = ""
                        if r.get("evidence_level") is not None:
                            try:
                                ev_s = f", evidence_level={int(r['evidence_level'])}"
                            except (TypeError, ValueError):
                                ev_s = f", evidence_level={r.get('evidence_level')}"
                            elab = r.get("evidence_level_label")
                            if elab:
                                ev_s += f" ({elab})"
                        rp.append(
                            f"  {j}. Image: {Path(image_path).name if image_path else 'N/A'} "
                            f"(vector_similarity={sim:.3f}{rs_s}, file: {Path(fp).name if fp else 'N/A'}, "
                            f"entity: {ent or 'N/A'}{ev_s})"
                        )
                    retrieval_results.append("\n".join(rp))

        except Exception as e:
            self.logger.error(f"Error in image retrieval: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {} if return_structured else ""
        
        # 构建文本上下文
        if retrieval_results:
            text_context = "\n\n=== Similar Images Retrieved from Knowledge Base ===\n"
            text_context += "\n".join(retrieval_results)
            text_context += "\n\nThese similar images may provide relevant context for answering the query."
        else:
            text_context = ""
        
        # 根据 return_structured 返回相应格式
        if return_structured:
            for _sim in all_similar_images:
                if isinstance(_sim, dict):
                    dedupe_similar_image_content_caption(_sim)
            return {
                "similar_images": all_similar_images,
                "entity_names": list(all_entity_names),
                "chunk_ids": list(all_chunk_ids),
                "text_context": text_context
            }
        else:
            return text_context

        # If image doesn't exist or processing failed, use existing information
        parts = []
        if image_path:
            parts.append(f"Image path: {image_path}")
        if captions:
            parts.append(f"Image captions: {', '.join(captions)}")
        if footnotes:
            parts.append(f"Image footnotes: {', '.join(footnotes)}")

        return "; ".join(parts) if parts else "Image content information incomplete"

    async def _describe_table_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate table description for query"""
        table_data = content.get("table_data", "")
        table_caption = content.get("table_caption", "")

        prompt = PROMPTS.get("QUERY_TABLE_ANALYSIS", "Analyze this table: {table_data}").format(
            table_data=table_data, table_caption=table_caption
        )

        description = await processor.modal_caption_func(
            prompt, system_prompt=PROMPTS.get("QUERY_TABLE_ANALYST_SYSTEM", "You are an expert at analyzing tables.")
        )

        return description

    async def _describe_generic_for_query(
        self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """Generate generic content description for query"""
        content_str = str(content)

        prompt = PROMPTS.get("QUERY_GENERIC_ANALYSIS", "Analyze this {content_type} content: {content_str}").format(
            content_type=content_type, content_str=content_str
        )

        description = await processor.modal_caption_func(
            prompt,
            system_prompt=PROMPTS.get("QUERY_GENERIC_ANALYST_SYSTEM", "You are an expert at analyzing content.").format(
                content_type=content_type
            ),
        )

        return description

    async def _process_image_paths_for_vlm(self, prompt: str) -> tuple[str, int]:
        """
        Process image paths in prompt, keeping original paths and adding VLM markers

        Args:
            prompt: Original prompt

        Returns:
            tuple: (processed prompt, image count)
        """
        enhanced_prompt = prompt
        images_processed = 0

        # Initialize image cache
        self._current_images_base64 = []

        # Enhanced regex pattern for matching image paths
        image_path_pattern = (
            r"Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))"
        )

        # First, let's see what matches we find
        matches = re.findall(image_path_pattern, prompt)
        self.logger.info(f"Found {len(matches)} image path matches in prompt")

        def replace_image_path(match):
            nonlocal images_processed

            image_path = match.group(1).strip()
            self.logger.debug(f"Processing image path: '{image_path}'")

            # Validate path format (basic check)
            if not image_path or len(image_path) < 3:
                self.logger.warning(f"Invalid image path format: {image_path}")
                return match.group(0)  # Keep original

            # Use utility function to validate image file
            self.logger.debug(f"Calling validate_image_file for: {image_path}")
            is_valid = validate_image_file(image_path)
            self.logger.debug(f"Validation result for {image_path}: {is_valid}")

            if not is_valid:
                self.logger.warning(f"Image validation failed for: {image_path}")
                return match.group(0)  # Keep original if validation fails

            try:
                # Encode image to base64 using utility function
                self.logger.debug(f"Attempting to encode image: {image_path}")
                image_base64 = encode_image_to_base64(image_path)
                if image_base64:
                    images_processed += 1
                    # Save base64 to instance variable for later use
                    self._current_images_base64.append(image_base64)

                    # Keep original path info and add VLM marker
                    result = f"Image Path: {image_path}\n[VLM_IMAGE_{images_processed}]"
                    self.logger.debug(
                        f"Successfully processed image {images_processed}: {image_path}"
                    )
                    return result
                else:
                    self.logger.error(f"Failed to encode image: {image_path}")
                    return match.group(0)  # Keep original if encoding failed

            except Exception as e:
                self.logger.error(f"Failed to process image {image_path}: {e}")
                return match.group(0)  # Keep original

        # Execute replacement
        enhanced_prompt = re.sub(
            image_path_pattern, replace_image_path, enhanced_prompt
        )

        return enhanced_prompt, images_processed

    def _build_vlm_messages_with_images(
        self, enhanced_prompt: str, user_query: str, system_prompt: str
    ) -> List[Dict]:
        """
        Build VLM message format, using markers to correspond images with text positions

        Args:
            enhanced_prompt: Enhanced prompt with image markers
            user_query: User query

        Returns:
            List[Dict]: VLM message format
        """
        images_base64 = getattr(self, "_current_images_base64", [])

        if not images_base64:
            # Pure text mode
            return [
                {
                    "role": "user",
                    "content": f"Context:\n{enhanced_prompt}\n\nUser Question: {user_query}",
                }
            ]

        # Build multimodal content
        content_parts = []

        # Split text at image markers and insert images
        text_parts = enhanced_prompt.split("[VLM_IMAGE_")

        for i, text_part in enumerate(text_parts):
            if i == 0:
                # First text part
                if text_part.strip():
                    content_parts.append({"type": "text", "text": text_part})
            else:
                # Find marker number and insert corresponding image
                marker_match = re.match(r"(\d+)\](.*)", text_part, re.DOTALL)
                if marker_match:
                    image_num = (
                        int(marker_match.group(1)) - 1
                    )  # Convert to 0-based index
                    remaining_text = marker_match.group(2)

                    # Insert corresponding image
                    if 0 <= image_num < len(images_base64):
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{images_base64[image_num]}"
                                },
                            }
                        )

                    # Insert remaining text
                    if remaining_text.strip():
                        content_parts.append({"type": "text", "text": remaining_text})

        # Add user question
        content_parts.append(
            {
                "type": "text",
                "text": f"\n\nUser Question: {user_query}\n\nPlease answer based on the context and images provided.",
            }
        )
        base_system_prompt = "You are a helpful assistant that can analyze both text and image content to provide comprehensive answers."

        if system_prompt:
            full_system_prompt = base_system_prompt + " " + system_prompt
        else:
            full_system_prompt = base_system_prompt

        return [
            {
                "role": "system",
                "content": full_system_prompt,
            },
            {
                "role": "user",
                "content": content_parts,
            },
        ]

    async def _call_vlm_with_multimodal_content(self, messages: List[Dict]) -> str:
        """
        Call VLM to process multimodal content

        Args:
            messages: VLM message format

        Returns:
            str: VLM response result
        """
        try:
            user_message = messages[1]
            content = user_message["content"]
            system_prompt = messages[0]["content"]

            if isinstance(content, str):
                # Pure text mode
                result = await self.vision_model_func(
                    content, system_prompt=system_prompt
                )
            else:
                # Multimodal mode - pass complete messages directly to VLM
                result = await self.vision_model_func(
                    "",  # Empty prompt since we're using messages format
                    messages=messages,
                )

            return result

        except Exception as e:
            self.logger.error(f"VLM call failed: {e}")
            raise

    # Synchronous versions of query methods
    def query(self, query: str, mode: str = "mix", **kwargs) -> str:
        """
        Synchronous version of pure text query

        Args:
            query: Query text
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters

        Returns:
            str: Query result
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, mode=mode, **kwargs))

    def query_with_multimodal(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]] = None,
        mode: str = "mix",
        **kwargs,
    ) -> str:
        """
        Synchronous version of multimodal query

        Args:
            query: Base query text
            multimodal_content: List of multimodal content
            mode: Query mode
            **kwargs: Other query parameters

        Returns:
            str: Query result
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_multimodal(query, multimodal_content, mode=mode, **kwargs)
        )
