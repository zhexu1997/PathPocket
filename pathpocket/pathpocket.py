"""
PathPocket Main Module
Caption-based multimodal RAG system (no vision model)
Uses RAG engine for storage and query, only customizes file parsing and multimodal instance building
"""

import os
import asyncio
import atexit
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from dataclasses import dataclass, field

from pathpocket.rag_engine import PathPocketEngine
from pathpocket.core import logger, compute_mdhash_id
from pathpocket.shared_storage import (
    pathpocket_init_pipeline_status,
    pathpocket_init_share_data,
    pathpocket_set_default_workspace,
)

from pathpocket.query_mixin import QueryMixin
from pathpocket.utils import insert_text_content_with_multimodal_content as pathpocket_insert_text_with_multimodal
from pathpocket.processor import ProcessorMixin

from pathpocket.config import PathPocketConfig
from pathpocket.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    GenericModalProcessor,
    ContextExtractor,
    ContextConfig,
)


@dataclass
class PathPocket(QueryMixin, ProcessorMixin):
    """PathPocket: Caption-based multimodal RAG system
    Uses RAG engine for storage and query, only customizes file parsing and multimodal instance building
    """

    # Core Components
    rag_engine: Optional[PathPocketEngine] = field(default=None)
    """Optional pre-initialized PathPocket engine instance."""

    llm_model_func: Optional[Callable] = field(default=None)
    """LLM model function for text analysis (used for caption-based extraction)."""

    vision_model_func: Optional[Callable] = field(default=None)
    """Vision model function for image analysis during queries (optional)."""

    embedding_func: Optional[Callable] = field(default=None)
    """Embedding function for text vectorization."""
    
    conch_feature_func: Optional[Callable] = field(default=None)
    """Virchow2 feature extraction function for pathology images (optional).
    Should be a wrapper that matches the embedding function interface:
    async def func(image_paths: List[str]) -> List[np.ndarray]
    """

    config: Optional[PathPocketConfig] = field(default=None)
    """Configuration object."""

    # RAG Engine Configuration
    rag_engine_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments for RAG engine initialization when rag_engine is not provided.
    This allows passing all RAG engine configuration parameters like:
    - kv_storage, vector_storage, graph_storage, doc_status_storage
    - top_k, chunk_top_k, max_entity_tokens, max_relation_tokens, max_total_tokens
    - cosine_threshold, related_chunk_number
    - chunk_token_size, chunk_overlap_token_size, tokenizer, tiktoken_model_name
    - embedding_batch_num, embedding_func_max_async, embedding_cache_config
    - llm_model_name, llm_model_max_token_size, llm_model_max_async, llm_model_kwargs
    - rerank_model_func, vector_db_storage_cls_kwargs, enable_llm_cache
    - max_parallel_insert, max_graph_nodes, addon_params, etc.
    """

    # Internal State
    modal_processors: Dict[str, Any] = field(default_factory=dict, init=False)
    """Dictionary of multimodal processors."""

    context_extractor: Optional[ContextExtractor] = field(default=None, init=False)
    """Context extractor for providing surrounding content to modal processors."""

    parse_cache: Optional[Any] = field(default=None, init=False)
    """Parse result cache storage using RAG engine KV storage."""

    def __post_init__(self):
        """Post-initialization setup"""
        # Initialize configuration if not provided
        if self.config is None:
            self.config = PathPocketConfig()

        # Set working directory
        self.working_dir = self.config.working_dir

        # Set up logger (use existing logger, don't configure it)
        self.logger = logger

        # Register close method for cleanup
        atexit.register(self.close)

        # Create working directory if needed
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self.logger.info(f"Created working directory: {self.working_dir}")

        # Log configuration info
        self.logger.info("PathPocket initialized with config:")
        self.logger.info(f"  Working directory: {self.config.working_dir}")
        self.logger.info(
            f"  Multimodal processing - Image: {self.config.enable_image_processing}, "
            f"Table: {self.config.enable_table_processing}"
        )

    def close(self):
        """Cleanup resources when object is destroyed"""
        try:
            import asyncio

            if asyncio.get_event_loop().is_running():
                # If we're in an async context, schedule cleanup
                asyncio.create_task(self.finalize_storages())
            else:
                # Run cleanup synchronously
                asyncio.run(self.finalize_storages())
        except Exception as e:
            # Use print instead of logger since logger might be cleaned up already
            print(f"Warning: Failed to finalize PathPocket storages: {e}")

    @property
    def rag_core(self):
        """Alias for rag_engine to maintain compatibility with query mixin"""
        if self.rag_engine is None:
            raise ValueError(
                "No RAG engine instance available. Please process documents first or provide a pre-initialized RAG engine instance."
            )
        return self.rag_engine

    async def _ensure_rag_core_initialized(self):
        """Ensure RAG core (engine) is initialized"""
        result = await self._ensure_rag_engine_initialized()
        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            raise ValueError(
                f"No RAG engine instance available. {error}"
            )

    def _create_context_config(self) -> ContextConfig:
        """Create context configuration from PathPocket config"""
        return ContextConfig(
            context_window=self.config.context_window,
            context_mode=self.config.context_mode,
            max_context_tokens=self.config.max_context_tokens,
            include_headers=self.config.include_headers,
            include_captions=self.config.include_captions,
            filter_content_types=self.config.context_filter_content_types,
        )

    def _create_context_extractor(self) -> ContextExtractor:
        """Create context extractor with tokenizer from RAG engine"""
        if self.rag_engine is None:
            raise ValueError(
                "RAG engine must be initialized before creating context extractor"
            )

        context_config = self._create_context_config()
        return ContextExtractor(
            config=context_config, tokenizer=self.rag_engine.tokenizer
        )

    def _initialize_processors(self):
        """Initialize multimodal processors (caption-based, no vision model, no LLM description generation)"""
        if self.rag_engine is None:
            raise ValueError(
                "RAG engine instance must be initialized before creating processors"
            )

        # Create context extractor (not used, but kept for compatibility)
        self.context_extractor = self._create_context_extractor()

        # Create different multimodal processors based on configuration
        # Note: modal_caption_func is not actually used since we don't generate descriptions
        # But we still need to pass it for compatibility
        self.modal_processors = {}

        if self.config.enable_image_processing:
            self.modal_processors["image"] = ImageModalProcessor(
                rag_engine=self.rag_engine,
                modal_caption_func=self.llm_model_func,  # Not used, but kept for compatibility
                context_extractor=None,  # Not used - no context extraction
                conch_feature_func=self.conch_feature_func,  # Virchow2 feature extraction
                is_pathology_image_func=None,  # Use default pathology image detection
            )

        if self.config.enable_table_processing:
            self.modal_processors["table"] = TableModalProcessor(
                rag_engine=self.rag_engine,
                modal_caption_func=self.llm_model_func,  # Not used, but kept for compatibility
                context_extractor=None,  # Not used - no context extraction
            )

        # Always include generic processor as fallback
        self.modal_processors["generic"] = GenericModalProcessor(
            rag_engine=self.rag_engine,
            modal_caption_func=self.llm_model_func,  # Not used, but kept for compatibility
            context_extractor=None,  # Not used - no context extraction
        )

        self.logger.info("Multimodal processors initialized (caption-based, no LLM description generation)")
        self.logger.info(f"Available processors: {list(self.modal_processors.keys())}")

    async def _ensure_rag_engine_initialized(self):
        """Ensure RAG engine instance is initialized, create if necessary"""
        try:
            if self.rag_engine is not None:
                # RAG engine was pre-provided, but we need to ensure it's properly initialized
                try:
                    # Ensure RAG engine storages are initialized
                    if (
                        not hasattr(self.rag_engine, "_storages_status")
                        or self.rag_engine._storages_status.name != "INITIALIZED"
                    ):
                        self.logger.info(
                            "Initializing storages for pre-provided RAG engine instance"
                        )
                        await self.rag_engine.initialize_storages()
                        await pathpocket_init_pipeline_status()

                    # Initialize parse cache if not already done
                    if self.parse_cache is None:
                        self.logger.info(
                            "Initializing parse cache for pre-provided RAG engine instance"
                        )
                        self.parse_cache = (
                            self.rag_engine.key_string_value_json_storage_cls(
                                namespace="parse_cache",
                                workspace=self.rag_engine.workspace,
                                global_config=self.rag_engine.__dict__,
                                embedding_func=self.embedding_func,
                            )
                        )
                        await self.parse_cache.initialize()

                    # Initialize processors if not already done
                    if not self.modal_processors:
                        self._initialize_processors()

                    return {"success": True}

                except Exception as e:
                    error_msg = (
                        f"Failed to initialize pre-provided RAG engine instance: {str(e)}"
                    )
                    self.logger.error(error_msg, exc_info=True)
                    return {"success": False, "error": error_msg}

            # Validate required functions for creating new RAG engine instance
            if self.llm_model_func is None:
                error_msg = "llm_model_func must be provided when RAG engine is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            if self.embedding_func is None:
                error_msg = "embedding_func must be provided when RAG engine is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            # Prepare RAG engine initialization parameters
            rag_params = {
                "working_dir": self.working_dir,
                "llm_model_func": self.llm_model_func,
                "embedding_func": self.embedding_func,
            }

            # Extract addon_params from rag_engine_kwargs before merging (to merge instead of replace)
            user_addon_params = {}
            if "addon_params" in self.rag_engine_kwargs:
                user_addon_params = self.rag_engine_kwargs.pop("addon_params", {})
                if not isinstance(user_addon_params, dict):
                    user_addon_params = {}
            
            # Merge user-provided rag_engine_kwargs (without addon_params)
            rag_params.update(self.rag_engine_kwargs)
            
            # Initialize addon_params in rag_params
            if "addon_params" not in rag_params:
                rag_params["addon_params"] = {}
            elif not isinstance(rag_params["addon_params"], dict):
                rag_params["addon_params"] = {}
            
            # Merge user-provided addon_params first
            rag_params["addon_params"].update(user_addon_params)
            
            # Add conch_feature_func (Virchow2) to addon_params if provided
            # Note: conch_feature_func should NOT be passed directly to LightRAG.__init__()
            # It should only be in addon_params, which is used by rag_engine.py to initialize pathology_images_vdb
            if self.conch_feature_func:
                rag_params["addon_params"]["conch_feature_func"] = self.conch_feature_func
                self.logger.info("Added conch_feature_func to addon_params for pathology_images_vdb initialization")
                self.logger.debug(f"addon_params keys after adding conch_feature_func: {list(rag_params['addon_params'].keys())}")
                self.logger.debug(f"conch_feature_func in addon_params: {'conch_feature_func' in rag_params['addon_params']}")
            else:
                self.logger.warning("conch_feature_func is None, pathology_images_vdb will not be initialized")
                self.logger.debug(f"self.conch_feature_func: {self.conch_feature_func}")

            # Log the parameters being used for initialization (excluding sensitive data)
            log_params = {
                k: v
                for k, v in rag_params.items()
                if not callable(v)
                and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            self.logger.info(f"Initializing RAG engine with parameters: {log_params}")

            try:
                # Initialize shared data before creating RAG engine
                pathpocket_init_share_data(workers=8)
                
                # Create RAG engine instance with merged parameters
                # Debug: Check addon_params before creating engine
                if "addon_params" in rag_params:
                    self.logger.debug(f"Before creating engine - addon_params keys: {list(rag_params['addon_params'].keys())}")
                    self.logger.debug(f"Before creating engine - conch_feature_func in addon_params: {'conch_feature_func' in rag_params['addon_params']}")
                
                self.rag_engine = PathPocketEngine(**rag_params)
                
                # Debug: Check addon_params after creating engine
                if hasattr(self.rag_engine, "addon_params"):
                    self.logger.debug(f"After creating engine - addon_params type: {type(self.rag_engine.addon_params)}")
                    if isinstance(self.rag_engine.addon_params, dict):
                        self.logger.debug(f"After creating engine - addon_params keys: {list(self.rag_engine.addon_params.keys())}")
                        self.logger.debug(f"After creating engine - conch_feature_func in addon_params: {'conch_feature_func' in self.rag_engine.addon_params}")
                
                # Set default workspace before initializing pipeline status
                pathpocket_set_default_workspace(self.rag_engine.workspace)
                
                await self.rag_engine.initialize_storages()
                await pathpocket_init_pipeline_status(workspace=self.rag_engine.workspace)

                # Initialize parse cache storage using RAG engine's KV storage
                self.parse_cache = self.rag_engine.key_string_value_json_storage_cls(
                    namespace="parse_cache",
                    workspace=self.rag_engine.workspace,
                    global_config=self.rag_engine.__dict__,
                    embedding_func=self.embedding_func,
                )
                await self.parse_cache.initialize()

                # Initialize processors after RAG engine is ready
                self._initialize_processors()

                self.logger.info(
                    "RAG engine, parse cache, and multimodal processors initialized"
                )
                return {"success": True}

            except Exception as e:
                error_msg = f"Failed to initialize RAG engine instance: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"Unexpected error during RAG engine initialization: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg}

    async def finalize_storages(self):
        """Finalize all storages including parse cache and RAG engine storages"""
        try:
            tasks = []
            
            # Finalize parse cache if it exists
            if self.parse_cache:
                tasks.append(self.parse_cache.finalize())
            
            # Finalize RAG engine storages if it exists
            if self.rag_engine:
                tasks.append(self.rag_engine.finalize_storages())
            
            # Run all finalization tasks concurrently
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                self.logger.info("Successfully finalized all PathPocket storages")
        except Exception as e:
            self.logger.error(f"Error during storage finalization: {e}")

    def set_content_source_for_context(
        self, content_source, content_format: str = "auto"
    ):
        """Set content source for context extraction in all modal processors"""
        if not self.modal_processors:
            self.logger.warning(
                "Modal processors not initialized. Content source will be set when processors are created."
            )
            return

        for processor_name, processor in self.modal_processors.items():
            try:
                processor.set_content_source(content_source, content_format)
                self.logger.debug(f"Set content source for {processor_name} processor")
            except Exception as e:
                self.logger.error(
                    f"Failed to set content source for {processor_name}: {e}"
                )

        self.logger.info(
            f"Content source set for context extraction (format: {content_format})"
        )

    async def insert_content_list(
        self,
        content_list: List[Dict[str, Any]],
        file_path: str = "unknown",
        split_by_character: str = "\n\n",
        split_by_character_only: bool = False,
        doc_id: str = None,
        display_stats: bool = True,
    ):
        """
        Insert content list into knowledge graph
        Uses RAG engine's text insertion for text content
        and custom multimodal processing for multimodal content

        Args:
            content_list: List of content items (from MinerU or similar parser)
            file_path: File path for citation
            split_by_character: Character(s) to split text by
            split_by_character_only: Whether to only split by character (no token-based splitting)
            doc_id: Optional document ID
            display_stats: Whether to display statistics
        """
        # Ensure RAG engine is initialized
        init_result = await self._ensure_rag_engine_initialized()
        if not init_result.get("success"):
            raise RuntimeError(f"Failed to initialize RAG engine: {init_result.get('error')}")

        # Set content source for context extraction
        self.set_content_source_for_context(content_list, content_format="minerU")

        # Separate text and multimodal content using utils function
        from pathpocket.utils import separate_content
        text_content, multimodal_items = separate_content(content_list)
        
        # Count text items for statistics
        text_items_count = sum(1 for item in content_list if item.get("type", "text") == "text")

        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = compute_mdhash_id(
                "\n".join([str(item) for item in content_list]), prefix="doc-"
            )

        # Insert text content using RAG engine's text insertion method
        if text_content.strip():
            # Use RAG engine's text insertion method directly with the merged text content
            await pathpocket_insert_text_with_multimodal(
                rag_engine=self.rag_engine,
                input=text_content,
                file_paths=os.path.basename(file_path),
                ids=doc_id,
                split_by_character=split_by_character,
                split_by_character_only=split_by_character_only,
            )
            
            # Wait for processing to complete and get actual chunk count
            if display_stats:
                try:
                    # Get document status to see actual chunk count
                    doc_status_data = await self.rag_engine.doc_status.get_by_id(doc_id)
                    if doc_status_data:
                        actual_chunks_count = doc_status_data.get("chunks_count", 0)
                        self.logger.info(
                            f"Text insertion stats: {len(text_content)} characters -> "
                            f"{actual_chunks_count} actual chunks after RAG engine processing"
                        )
                except Exception as e:
                    self.logger.debug(f"Could not get chunk count from doc_status: {e}")
        else:
            # No text content found
            self.logger.info(f"No text content found in content_list for doc {doc_id}")

        # Process multimodal content using caption-based extraction
        if multimodal_items:
            # Check if multimodal content is already processed
            try:
                existing_doc_status = await self.rag_engine.doc_status.get_by_id(doc_id)
                if existing_doc_status:
                    multimodal_processed = existing_doc_status.get(
                        "multimodal_processed", False
                    )
                    
                    if multimodal_processed:
                        self.logger.info(
                            f"Document {doc_id} multimodal content is already processed, skipping"
                        )
                        if display_stats:
                            self.logger.info(
                                f"Skipped multimodal processing: {len(multimodal_items)} items already processed"
                            )
                        return
            except Exception as e:
                self.logger.debug(f"Error checking document status for {doc_id}: {e}")
                # Continue with processing if check fails

            # Process multimodal content using ProcessorMixin's method
            await self._process_multimodal_content(
                multimodal_items=multimodal_items,
                file_path=file_path,
                doc_id=doc_id,
            )
        else:
            # If no multimodal content, mark multimodal processing as complete
            # This ensures the document status properly reflects completion of all processing
            await self._mark_multimodal_processing_complete(doc_id)
            self.logger.debug(
                f"No multimodal content found in document {doc_id}, marked multimodal processing as complete"
            )

        if display_stats:
            self.logger.info(
                f"Inserted content: {text_items_count} text items, {len(multimodal_items)} multimodal items"
            )
