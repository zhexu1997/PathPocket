"""
Internal RAG Core for PathPocket
Replaces LightRAG with self-contained implementation
"""

import os
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

from pathpocket.core import logger
from pathpocket.storage import (
    BaseKVStorage,
    BaseVectorStorage,
    BaseGraphStorage,
    JsonKVStorage,
    SimpleVectorStorage,
    NetworkXGraphStorage,
)


class StorageStatus(Enum):
    """Storage initialization status"""
    NOT_INITIALIZED = "NOT_INITIALIZED"
    INITIALIZING = "INITIALIZING"
    INITIALIZED = "INITIALIZED"


@dataclass
class EmbeddingFunc:
    """Embedding function wrapper"""
    embedding_dim: int
    max_token_size: int
    func: Callable
    
    async def __call__(self, texts: List[str]) -> List[List[float]]:
        """Call embedding function"""
        return await self.func(texts)


@dataclass
class DocStatus:
    """Document processing status"""
    doc_id: str
    status: str = "processing"
    chunks_count: int = 0
    create_time: int = field(default_factory=lambda: int(time.time()))
    update_time: int = field(default_factory=lambda: int(time.time()))


class RAGCore:
    """Internal RAG core implementation"""
    
    def __init__(
        self,
        working_dir: str = "./rag_storage",
        llm_model_func: Optional[Callable] = None,
        embedding_func: Optional[EmbeddingFunc] = None,
        tokenizer: Optional[Any] = None,
        **kwargs
    ):
        self.working_dir = working_dir
        self.llm_model_func = llm_model_func
        self.embedding_func = embedding_func
        self.tokenizer = tokenizer
        
        # Storage instances
        self.text_chunks: Optional[BaseKVStorage] = None
        self.chunks_vdb: Optional[BaseVectorStorage] = None
        self.entities_vdb: Optional[BaseVectorStorage] = None
        self.relationships_vdb: Optional[BaseVectorStorage] = None
        self.chunk_entity_relation_graph: Optional[BaseGraphStorage] = None
        self.full_docs: Optional[BaseKVStorage] = None
        self.full_entities: Optional[BaseKVStorage] = None
        self.full_relations: Optional[BaseKVStorage] = None
        self.entity_chunks: Optional[BaseKVStorage] = None
        self.relation_chunks: Optional[BaseKVStorage] = None
        self.doc_status: Optional[BaseKVStorage] = None
        self.llm_response_cache: Optional[BaseKVStorage] = None
        
        # Pathology image feature storage (CONCH1.5 features)
        self.pathology_images_vdb: Optional[BaseVectorStorage] = None
        
        # Status
        self._storages_status = StorageStatus.NOT_INITIALIZED
        
        # Configuration
        self.workspace = kwargs.get("workspace", "")
        self.llm_model_max_async = kwargs.get("llm_model_max_async", 4)
        self.max_parallel_insert = kwargs.get("max_parallel_insert", 2)
        self.chunk_token_size = kwargs.get("chunk_token_size", 1200)
        self.chunk_overlap_token_size = kwargs.get("chunk_overlap_token_size", 100)
        
        # Addon params
        self.addon_params = kwargs.get("addon_params", {})
        
        # Create working directory
        os.makedirs(working_dir, exist_ok=True)
    
    async def initialize_storages(self):
        """Initialize all storage backends"""
        if self._storages_status == StorageStatus.INITIALIZED:
            return
        
        self._storages_status = StorageStatus.INITIALIZING
        
        global_config = {
            "working_dir": self.working_dir,
            "workspace": self.workspace,
            "embedding_batch_num": getattr(self, 'embedding_batch_num', 10),
        }
        
        # Initialize KV storages
        self.text_chunks = JsonKVStorage(
            namespace="text_chunks",
            workspace=self.workspace,
            global_config=global_config
        )
        await self.text_chunks.initialize()
        
        self.full_docs = JsonKVStorage(
            namespace="full_docs",
            workspace=self.workspace,
            global_config=global_config
        )
        await self.full_docs.initialize()
        
        self.full_entities = JsonKVStorage(
            namespace="full_entities",
            workspace=self.workspace,
            global_config=global_config
        )
        await self.full_entities.initialize()
        
        self.full_relations = JsonKVStorage(
            namespace="full_relations",
            workspace=self.workspace,
            global_config=global_config
        )
        await self.full_relations.initialize()
        
        self.entity_chunks = JsonKVStorage(
            namespace="entity_chunks",
            workspace=self.workspace,
            global_config=global_config
        )
        await self.entity_chunks.initialize()
        
        self.relation_chunks = JsonKVStorage(
            namespace="relation_chunks",
            workspace=self.workspace,
            global_config=global_config
        )
        await self.relation_chunks.initialize()
        
        self.doc_status = JsonKVStorage(
            namespace="doc_status",
            workspace=self.workspace,
            global_config=global_config
        )
        await self.doc_status.initialize()
        
        self.llm_response_cache = JsonKVStorage(
            namespace="llm_response_cache",
            workspace=self.workspace,
            global_config=global_config
        )
        await self.llm_response_cache.initialize()
        
        # Initialize vector storages (use same namespaces and meta_fields as LightRAG)
        self.chunks_vdb = SimpleVectorStorage(
            namespace="chunks",
            workspace=self.workspace,
            global_config=global_config,
            embedding_func=self.embedding_func,
            meta_fields={"full_doc_id", "content", "file_path"}
        )
        await self.chunks_vdb.initialize()
        
        self.entities_vdb = SimpleVectorStorage(
            namespace="entities",
            workspace=self.workspace,
            global_config=global_config,
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "source_id", "content", "file_path"}
        )
        await self.entities_vdb.initialize()
        
        self.relationships_vdb = SimpleVectorStorage(
            namespace="relationships",
            workspace=self.workspace,
            global_config=global_config,
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "source_id", "content", "file_path"}
        )
        await self.relationships_vdb.initialize()
        
        # Initialize graph storage (use same namespace as LightRAG)
        self.chunk_entity_relation_graph = NetworkXGraphStorage(
            namespace="chunk_entity_relation",
            workspace=self.workspace,
            global_config=global_config
        )
        await self.chunk_entity_relation_graph.initialize()
        
        # Initialize pathology image feature storage (Virchow2 features)
        # Only initialize if conch_feature_func is provided AND instance doesn't already exist
        # (instance may have been created in rag_engine.py __post_init__)
        conch_feature_func = kwargs.get("conch_feature_func")
        if conch_feature_func and not self.pathology_images_vdb:
            # Use the same vector storage type as the main vector storage (PGVectorStorage or SimpleVectorStorage)
            vector_storage_type = global_config.get("vector_storage", "SimpleVectorStorage")
            logger.info(f"Creating pathology_images_vdb in RAGCore.initialize_storages using {vector_storage_type}")
            
            if vector_storage_type == "PGVectorStorage":
                # Use PGVectorStorage to access VDB_IMAGES table in PostgreSQL
                from pathpocket.lightrag_kg.postgres_impl import PGVectorStorage
                self.pathology_images_vdb = PGVectorStorage(
                    namespace="pathology_images",
                    workspace=self.workspace,
                    global_config=global_config,
                    embedding_func=conch_feature_func,
                    meta_fields={"image_path", "chunk_id", "file_path", "entity_name", "content"}
                )
                await self.pathology_images_vdb.initialize()
                logger.info("Pathology image feature storage initialized (Virchow2) as PGVectorStorage (VDB_IMAGES table)")
            else:
                # Use SimpleVectorStorage (NanoVectorDB) for local storage
                from pathpocket.lightrag_namespace import resolve_vector_store_cosine_threshold
                from pathpocket.storage import SimpleVectorStorage

                _img_vdb_cos = resolve_vector_store_cosine_threshold(
                    "pathology_images", global_config
                )
                self.pathology_images_vdb = SimpleVectorStorage(
                    namespace="pathology_images",
                    workspace=self.workspace,
                    global_config=global_config,
                    embedding_func=conch_feature_func,
                    meta_fields={"image_path", "chunk_id", "file_path", "entity_name"},
                    cosine_better_than_threshold=_img_vdb_cos,
                )
                await self.pathology_images_vdb.initialize()
                logger.info("Pathology image feature storage initialized (Virchow2) as SimpleVectorStorage")
        elif self.pathology_images_vdb:
            logger.info("pathology_images_vdb already exists, skipping creation in RAGCore.initialize_storages")
            # Ensure it's initialized
            if hasattr(self.pathology_images_vdb, 'initialize'):
                await self.pathology_images_vdb.initialize()
        elif conch_feature_func:
            logger.warning("conch_feature_func provided but pathology_images_vdb creation failed")
        
        self._storages_status = StorageStatus.INITIALIZED
        logger.info("RAGCore storages initialized")
    
    async def finalize_storages(self):
        """Finalize all storages"""
        storages = [
            self.text_chunks,
            self.chunks_vdb,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
            self.full_docs,
            self.full_entities,
            self.full_relations,
            self.entity_chunks,
            self.relation_chunks,
            self.doc_status,
            self.llm_response_cache,
            self.pathology_images_vdb,
        ]
        
        for storage in storages:
            if storage:
                try:
                    await storage.finalize()
                except Exception as e:
                    logger.warning(f"Error finalizing storage: {e}")
        
        logger.info("RAGCore storages finalized")
    
    def __dict__(self):
        """Return configuration dict for compatibility"""
        return {
            "working_dir": self.working_dir,
            "workspace": self.workspace,
            "llm_model_func": self.llm_model_func,
            "embedding_func": self.embedding_func,
            "tokenizer": self.tokenizer,
            "llm_model_max_async": self.llm_model_max_async,
            "max_parallel_insert": self.max_parallel_insert,
            "addon_params": self.addon_params,
        }
