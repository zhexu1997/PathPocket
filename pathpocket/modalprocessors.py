"""
PathPocket Modal Processors
Based on caption and text information only (no vision model)
"""

import json
import re
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

from pathpocket.core import logger, compute_mdhash_id
from dataclasses import dataclass


@dataclass
class ContextConfig:
    """Configuration for context extraction"""

    context_window: int = 1
    context_mode: str = "page"
    max_context_tokens: int = 2000
    include_headers: bool = True
    include_captions: bool = True
    filter_content_types: List[str] = None

    def __post_init__(self):
        if self.filter_content_types is None:
            self.filter_content_types = ["text"]


class ContextExtractor:
    """Universal context extractor supporting multiple content source formats"""

    def __init__(self, config: ContextConfig = None, tokenizer=None):
        self.config = config or ContextConfig()
        self.tokenizer = tokenizer

    def extract_context(
        self,
        content_source: Any,
        current_item_info: Dict[str, Any],
        content_format: str = "auto",
    ) -> str:
        """Extract context for current item from content source"""
        if not content_source and not self.config.context_window:
            return ""

        try:
            if content_format == "minerU" and isinstance(content_source, list):
                return self._extract_from_content_list(content_source, current_item_info)
            elif isinstance(content_source, list):
                return self._extract_from_content_list(content_source, current_item_info)
            elif isinstance(content_source, str):
                return self._extract_from_text_source(content_source, current_item_info)
            else:
                logger.warning(f"Unsupported content source type: {type(content_source)}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return ""

    def _extract_from_content_list(
        self, content_list: List[Dict], current_item_info: Dict
    ) -> str:
        """Extract context from MinerU-style content list"""
        if self.config.context_mode == "page":
            return self._extract_page_context(content_list, current_item_info)
        else:
            return self._extract_chunk_context(content_list, current_item_info)

    def _extract_page_context(
        self, content_list: List[Dict], current_item_info: Dict
    ) -> str:
        """Extract context based on page boundaries"""
        current_page = current_item_info.get("page_idx", 0)
        window_size = self.config.context_window

        start_page = max(0, current_page - window_size)
        end_page = current_page + window_size + 1

        context_texts = []

        for item in content_list:
            item_page = item.get("page_idx", 0)
            item_type = item.get("type", "")

            if (
                start_page <= item_page < end_page
                and item_type in self.config.filter_content_types
            ):
                text_content = self._extract_text_from_item(item)
                if text_content and text_content.strip():
                    if item_page != current_page:
                        context_texts.append(f"[Page {item_page}] {text_content}")
                    else:
                        context_texts.append(text_content)

        context = "\n".join(context_texts)
        return self._truncate_context(context)

    def _extract_chunk_context(
        self, content_list: List[Dict], current_item_info: Dict
    ) -> str:
        """Extract context based on content chunks"""
        current_index = current_item_info.get("index", 0)
        window_size = self.config.context_window

        start_idx = max(0, current_index - window_size)
        end_idx = min(len(content_list), current_index + window_size + 1)

        context_texts = []

        for i in range(start_idx, end_idx):
            if i != current_index:
                item = content_list[i]
                item_type = item.get("type", "")

                if item_type in self.config.filter_content_types:
                    text_content = self._extract_text_from_item(item)
                    if text_content and text_content.strip():
                        context_texts.append(text_content)

        context = "\n".join(context_texts)
        return self._truncate_context(context)

    def _extract_text_from_item(self, item: Dict) -> str:
        """Extract text content from a content item"""
        item_type = item.get("type", "")

        if item_type == "text":
            text = item.get("text", "")
            text_level = item.get("text_level", 0)

            if self.config.include_headers and text_level > 0:
                return f"{'#' * text_level} {text}"
            return text

        elif item_type == "image" and self.config.include_captions:
            captions = item.get("image_caption", item.get("img_caption", []))
            if captions:
                return f"[Image: {', '.join(captions)}]"

        elif item_type == "table" and self.config.include_captions:
            captions = item.get("table_caption", [])
            if captions:
                return f"[Table: {', '.join(captions)}]"

        return ""

    def _extract_from_text_source(
        self, text_source: str, current_item_info: Dict
    ) -> str:
        """Extract context from plain text source"""
        return self._truncate_context(text_source)

    def _truncate_context(self, context: str) -> str:
        """Truncate context to maximum token limit"""
        if not context:
            return ""

        if self.tokenizer:
            tokens = self.tokenizer.encode(context)
            if len(tokens) <= self.config.max_context_tokens:
                return context

            truncated_tokens = tokens[: self.config.max_context_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens)

            last_period = truncated_text.rfind(".")
            last_newline = truncated_text.rfind("\n")

            if last_period > len(truncated_text) * 0.8:
                return truncated_text[: last_period + 1]
            elif last_newline > len(truncated_text) * 0.8:
                return truncated_text[:last_newline]
            else:
                return truncated_text + "..."
        else:
            if len(context) <= self.config.max_context_tokens:
                return context

            truncated = context[: self.config.max_context_tokens]
            last_period = truncated.rfind(".")
            last_newline = truncated.rfind("\n")

            if last_period > len(truncated) * 0.8:
                return truncated[: last_period + 1]
            elif last_newline > len(truncated) * 0.8:
                return truncated[:last_newline]
            else:
                return truncated + "..."


class BaseModalProcessor:
    """Base class for modal processors"""

    def __init__(
        self,
        rag_engine,
        modal_caption_func,
        context_extractor: ContextExtractor = None,
    ):
        """Initialize base processor

        Args:
            rag_engine: RAG engine instance
            modal_caption_func: LLM function for generating descriptions (caption-based, no vision model)
            context_extractor: Context extractor instance
        """
        from dataclasses import asdict
        
        self.rag_engine = rag_engine
        self.modal_caption_func = modal_caption_func

        # Use RAG engine's storage instances
        self.text_chunks_db = rag_engine.text_chunks
        self.chunks_vdb = rag_engine.chunks_vdb
        self.entities_vdb = rag_engine.entities_vdb
        self.relationships_vdb = rag_engine.relationships_vdb
        self.knowledge_graph_inst = rag_engine.chunk_entity_relation_graph

        # Use RAG engine's configuration and functions
        self.embedding_func = rag_engine.embedding_func
        self.llm_model_func = rag_engine.llm_model_func
        self.global_config = asdict(rag_engine)
        self.hashing_kv = rag_engine.llm_response_cache
        self.tokenizer = rag_engine.tokenizer

        # Initialize context extractor with tokenizer if not provided
        if context_extractor is None:
            self.context_extractor = ContextExtractor(tokenizer=self.tokenizer)
        else:
            self.context_extractor = context_extractor
            # Update tokenizer if context_extractor doesn't have one
            if self.context_extractor.tokenizer is None:
                self.context_extractor.tokenizer = self.tokenizer

        self.content_source = None
        self.content_format = "auto"

    def set_content_source(self, content_source: Any, content_format: str = "auto"):
        """Set content source for context extraction"""
        self.content_source = content_source
        self.content_format = content_format

    def _get_context_for_item(self, item_info: Dict[str, Any]) -> str:
        """Get context for current processing item"""
        if not self.content_source:
            return ""

        try:
            context = self.context_extractor.extract_context(
                self.content_source, item_info, self.content_format
            )
            return context
        except Exception as e:
            logger.error(f"Error getting context for item {item_info}: {e}")
            return ""

    def _parse_response(self, response: str, entity_name: str = None) -> Tuple[str, Dict[str, Any]]:
        """Parse LLM response to extract description and entity info"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)

                detailed_description = data.get("detailed_description", response)
                entity_info = data.get("entity_info", {})

                if not entity_info:
                    entity_info = {
                        "entity_name": entity_name or f"modal_entity_{compute_mdhash_id(response)}",
                        "entity_type": "generic",
                        "summary": detailed_description[:100] if detailed_description else "",
                    }

                return detailed_description, entity_info
            else:
                # Fallback: use entire response as description
                fallback_entity = {
                    "entity_name": entity_name or f"modal_entity_{compute_mdhash_id(response)}",
                    "entity_type": "generic",
                    "summary": response[:100] if response else "",
                }
                return response, fallback_entity
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            fallback_entity = {
                "entity_name": entity_name or f"modal_entity_{compute_mdhash_id(response)}",
                "entity_type": "generic",
                "summary": response[:100] if response else "",
            }
            return response, fallback_entity

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate description and entity info only (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement this method")


class ImageModalProcessor(BaseModalProcessor):
    """Processor specialized for image content (caption-based, no vision model, no LLM description generation)
    Supports Virchow2 feature extraction for pathology images
    """

    def __init__(
        self,
        rag_engine,
        modal_caption_func,
        context_extractor: ContextExtractor = None,
        conch_feature_func=None,
        is_pathology_image_func=None,
    ):
        """
        Initialize ImageModalProcessor
        
        Args:
            rag_engine: RAG engine instance
            modal_caption_func: LLM function for generating descriptions
            context_extractor: Context extractor instance
            conch_feature_func: Virchow2 feature extraction function (optional)
            is_pathology_image_func: Function to check if image is pathology image (optional)
        """
        super().__init__(rag_engine, modal_caption_func, context_extractor)
        self.conch_feature_func = conch_feature_func
        self.is_pathology_image_func = is_pathology_image_func or self._default_is_pathology_image
        
    def _default_is_pathology_image(self, image_path: str, content_data: Dict[str, Any] = None) -> bool:
        """
        Default function to check if image is pathology image
        
        Args:
            image_path: Path to image file
            content_data: Image content data dictionary
            
        Returns:
            True if image is considered pathology image, False otherwise
        """
        # Default: check if image path contains pathology-related keywords
        if not image_path:
            return False
            
        image_path_lower = image_path.lower()
        pathology_keywords = ["pathology", "histology", "histopathology", "biopsy", "tissue"]
        
        # Check path
        if any(keyword in image_path_lower for keyword in pathology_keywords):
            return True
            
        # Check captions if available
        if content_data:
            captions = content_data.get("image_caption", content_data.get("img_caption", []))
            caption_text = " ".join(captions).lower() if captions else ""
            if any(keyword in caption_text for keyword in pathology_keywords):
                return True
                
        return False
    
    async def extract_and_store_conch_features(
        self,
        image_path: str,
        chunk_id: str,
        entity_name: str,
        file_path: str,
    ) -> bool:
        """
        Extract Virchow2 features and store in vector database
        
        Args:
            image_path: Path to image file
            chunk_id: Associated chunk ID (for linking to text vector)
            entity_name: Entity name for the image
            file_path: Source file path
            
        Returns:
            True if features were extracted and stored successfully, False otherwise
        """
        if not self.conch_feature_func:
            return False
            
        if not self.rag_engine.pathology_images_vdb:
            logger.debug("Image feature storage (pathology_images_vdb) not available")
            return False
            
        try:
            # Extract features using Virchow2
            features = await self.conch_feature_func.func([image_path])
            logger.debug(f"extract_and_store_conch_features: received {len(features) if features else 0} features")
            if not features or len(features) == 0:
                logger.warning(f"Failed to extract Virchow2 features for {image_path}")
                return False
            
            feature_vector = features[0]
            logger.debug(f"extract_and_store_conch_features: feature_vector type={type(feature_vector)}, shape={feature_vector.shape if isinstance(feature_vector, np.ndarray) else 'N/A'}")
            
            # Convert to numpy array if needed and ensure it's 1D
            if not isinstance(feature_vector, np.ndarray):
                feature_vector = np.array(feature_vector)
            
            # Ensure vector is 1D (flatten if needed)
            if feature_vector.ndim > 1:
                feature_vector = feature_vector.flatten()
            elif feature_vector.ndim == 0:
                # Scalar, convert to 1D array
                feature_vector = np.array([feature_vector])
            
            # Verify vector shape
            expected_dim = getattr(self.conch_feature_func, 'embedding_dim', None)
            if expected_dim and feature_vector.shape[0] != expected_dim:
                logger.warning(f"Feature vector dimension mismatch: expected {expected_dim}, got {feature_vector.shape[0]}")
            
            logger.debug(f"Virchow2 feature vector shape: {feature_vector.shape}, dtype: {feature_vector.dtype}")
            
            # Prepare data for storage
            # Use chunk_id-based ID to link with text vector in chunks_vdb
            feature_id = f"img_virchow2_{chunk_id}"
            feature_data = {
                feature_id: {
                    "content": f"Image: {image_path}",  # Placeholder content for compatibility
                    "image_path": image_path,
                    "chunk_id": chunk_id,  # Link to text vector chunk
                    "file_path": file_path,
                    "entity_name": entity_name,
                    "__vector__": feature_vector,  # Pre-computed Virchow2 feature vector
                }
            }
            
            # Store features in vector database
            # Note: This stores Virchow2 visual features
            # Text features are already stored in chunks_vdb with the same chunk_id
            await self.rag_engine.pathology_images_vdb.upsert(feature_data)
            
            logger.debug(f"Stored Virchow2 features for image: {image_path} (chunk_id: {chunk_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting/storing Virchow2 features for {image_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate image description directly from caption (no LLM call, no context)

        Args:
            modal_content: Image content to process
            content_type: Type of modal content ("image")
            item_info: Item information (not used, kept for compatibility)
            entity_name: Optional predefined entity name

        Returns:
            Tuple of (caption_text, entity_info)
        """
        try:
            # Parse image content
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"description": modal_content}
            else:
                content_data = modal_content

            image_path = content_data.get("img_path", "")
            captions = content_data.get(
                "image_caption", content_data.get("img_caption", [])
            )
            footnotes = content_data.get(
                "image_footnote", content_data.get("img_footnote", [])
            )

            # Build description directly from caption and footnotes (no LLM call)
            caption_parts = []
            if captions:
                caption_parts.extend(captions)
            if footnotes:
                caption_parts.extend(footnotes)
            
            description = " ".join(caption_parts) if caption_parts else f"Image: {image_path}"

            # Generate entity name using hash code (like RAGAnything)
            # Use hash of modal content to create unique entity name
            if not entity_name:
                # Generate hash-based entity name with (image) suffix
                content_hash = compute_mdhash_id(str(modal_content))[:16]
                entity_name = f"Image_{content_hash} (image)"

            entity_info = {
                "entity_name": entity_name,
                "entity_type": "image",
                "summary": description[:200] if description else "",  # Use first 200 characters
            }

            return description, entity_info

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"Image_{compute_mdhash_id(str(modal_content))[:16]} (image)",
                "entity_type": "image",
                "summary": f"Image content: {str(modal_content)[:200]}",
            }
            return str(modal_content), fallback_entity


class TableModalProcessor(BaseModalProcessor):
    """Processor specialized for table content (caption-based, no LLM description generation)"""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate table description directly from caption and body (no LLM call, no context)

        Args:
            modal_content: Table content to process
            content_type: Type of modal content ("table")
            item_info: Item information (not used, kept for compatibility)
            entity_name: Optional predefined entity name

        Returns:
            Tuple of (table_text, entity_info)
        """
        try:
            # Parse table content
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"description": modal_content}
            else:
                content_data = modal_content

            table_img_path = content_data.get("img_path", "")
            table_caption = content_data.get("table_caption", [])
            table_body = content_data.get("table_body", "")
            table_footnote = content_data.get("table_footnote", [])

            # Build description directly from caption, body, and footnotes (no LLM call)
            description_parts = []
            if table_caption:
                description_parts.append("Caption: " + ", ".join(table_caption))
            if table_body:
                description_parts.append("Table body: " + str(table_body))
            if table_footnote:
                description_parts.append("Footnotes: " + ", ".join(table_footnote))
            
            description = "\n".join(description_parts) if description_parts else f"Table: {table_img_path}"

            # Generate entity name using hash code (like RAGAnything)
            # Use hash of modal content to create unique entity name
            if not entity_name:
                # Generate hash-based entity name with (table) suffix
                content_hash = compute_mdhash_id(str(modal_content))[:16]
                entity_name = f"Table_{content_hash} (table)"

            entity_info = {
                "entity_name": entity_name,
                "entity_type": "table",
                "summary": description[:200] if description else "",  # Use first 200 characters
            }

            return description, entity_info

        except Exception as e:
            logger.error(f"Error processing table: {e}")
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"Table_{compute_mdhash_id(str(modal_content))[:16]} (table)",
                "entity_type": "table",
                "summary": f"Table content: {str(modal_content)[:200]}",
            }
            return str(modal_content), fallback_entity


class GenericModalProcessor(BaseModalProcessor):
    """Processor for generic modal content (no LLM description generation)"""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate generic content description directly from content (no LLM call, no context)"""
        try:
            content = str(modal_content)

            # Use content directly as description (no LLM call)
            description = content

            # Generate entity name from content or use provided
            if not entity_name:
                entity_name = f"{content_type.title()}_{compute_mdhash_id(str(modal_content))[:8]}"

            entity_info = {
                "entity_name": entity_name,
                "entity_type": content_type,
                "summary": description[:100] if description else "",
            }

            return description, entity_info

        except Exception as e:
            logger.error(f"Error processing generic content: {e}")
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"generic_{compute_mdhash_id(str(modal_content))}",
                "entity_type": content_type,
                "summary": f"Content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity
