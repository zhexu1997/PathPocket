"""
Storage interfaces and implementations for PathPocket
Self-contained storage system without LightRAG dependencies
"""

import os
import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import defaultdict

from pathpocket.core import logger


@dataclass
class BaseKVStorage(ABC):
    """Base class for key-value storage"""
    namespace: str
    workspace: str = ""
    global_config: Dict[str, Any] = field(default_factory=dict)
    
    @abstractmethod
    async def initialize(self):
        """Initialize the storage"""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get data by ID"""
        pass
    
    @abstractmethod
    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Insert or update data"""
        pass
    
    @abstractmethod
    async def index_done_callback(self) -> None:
        """Commit storage operations"""
        pass
    
    @abstractmethod
    async def finalize(self) -> None:
        """Finalize storage"""
        pass


@dataclass
class BaseVectorStorage(ABC):
    """Base class for vector storage"""
    namespace: str
    workspace: str = ""
    global_config: Dict[str, Any] = field(default_factory=dict)
    embedding_func: Any = None
    cosine_better_than_threshold: float = 0.2
    
    @abstractmethod
    async def initialize(self):
        """Initialize the storage"""
        pass
    
    @abstractmethod
    async def query(self, query: str, top_k: int, query_embedding: List[float] = None) -> List[Dict[str, Any]]:
        """Query vectors"""
        pass
    
    @abstractmethod
    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Insert or update vectors"""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get vector by ID"""
        pass
    
    @abstractmethod
    async def index_done_callback(self) -> None:
        """Commit storage operations"""
        pass
    
    @abstractmethod
    async def finalize(self) -> None:
        """Finalize storage"""
        pass


@dataclass
class BaseGraphStorage(ABC):
    """Base class for graph storage"""
    namespace: str
    workspace: str = ""
    global_config: Dict[str, Any] = field(default_factory=dict)
    
    @abstractmethod
    async def initialize(self):
        """Initialize the storage"""
        pass
    
    @abstractmethod
    async def get_node(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get node by entity name"""
        pass
    
    @abstractmethod
    async def upsert_node(self, entity_name: str, node_data: Dict[str, Any]) -> None:
        """Insert or update node"""
        pass
    
    @abstractmethod
    async def get_edge(self, src_id: str, tgt_id: str) -> Optional[Dict[str, Any]]:
        """Get edge between two nodes"""
        pass
    
    @abstractmethod
    async def upsert_edge(self, src_id: str, tgt_id: str, edge_data: Dict[str, Any]) -> None:
        """Insert or update edge"""
        pass
    
    @abstractmethod
    async def has_edge(self, src_id: str, tgt_id: str) -> bool:
        """Check if edge exists"""
        pass
    
    @abstractmethod
    async def get_node_edges(self, entity_name: str) -> List[tuple]:
        """Get all edges for a node"""
        pass
    
    @abstractmethod
    async def finalize(self) -> None:
        """Finalize storage"""
        pass


class JsonKVStorage(BaseKVStorage):
    """JSON-based key-value storage implementation"""
    
    def __init__(self, namespace: str, workspace: str = "", global_config: Dict[str, Any] = None):
        """Initialize JsonKVStorage"""
        super().__init__(namespace=namespace, workspace=workspace, global_config=global_config or {})
        working_dir = self.global_config.get("working_dir", "./rag_storage")
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
        
        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"kv_store_{self.namespace}.json")
        self._data: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize storage from file"""
        async with self._lock:
            # Don't reinitialize if already initialized and has data
            if self._initialized and len(self._data) > 0:
                logger.debug(f"Storage {self.namespace} already initialized with {len(self._data)} records, skipping")
                return
            
            if os.path.exists(self._file_name):
                try:
                    with open(self._file_name, 'r', encoding='utf-8') as f:
                        loaded_data = json.load(f)
                        # Only update if we don't have data in memory
                        if len(self._data) == 0:
                            self._data = loaded_data
                            logger.info(f"Loaded {len(self._data)} records from {self.namespace}")
                        else:
                            # Merge with existing data
                            self._data.update(loaded_data)
                            logger.info(f"Merged {len(loaded_data)} records from file with {len(self._data)} in-memory records for {self.namespace}")
                except Exception as e:
                    logger.warning(f"Error loading {self.namespace}: {e}, starting fresh")
                    if len(self._data) == 0:
                        self._data = {}
            else:
                if len(self._data) == 0:
                    self._data = {}
            self._initialized = True
    
    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get data by ID"""
        async with self._lock:
            return self._data.get(id)
    
    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Insert or update data"""
        # Initialize if not already done (outside lock to avoid deadlock)
        if not self._initialized:
            logger.warning(f"Storage {self.namespace} not initialized, initializing now")
            await self.initialize()
        
        async with self._lock:
            self._data.update(data)
            logger.debug(f"Upserted {len(data)} records to {self.namespace}, total: {len(self._data)}")
    
    async def index_done_callback(self) -> None:
        """Write data to file"""
        async with self._lock:
            try:
                # Ensure directory exists
                file_dir = os.path.dirname(self._file_name)
                if file_dir:
                    os.makedirs(file_dir, exist_ok=True)
                
                # Log data before saving
                data_count = len(self._data)
                if data_count == 0:
                    logger.warning(f"Warning: {self.namespace} storage is empty when saving to {self._file_name}")
                else:
                    logger.info(f"Saving {data_count} records to {self._file_name}")
                
                with open(self._file_name, 'w', encoding='utf-8') as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
                logger.debug(f"Saved {data_count} records to {self._file_name}")
            except Exception as e:
                logger.error(f"Error writing {self.namespace} to {self._file_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    async def finalize(self) -> None:
        """Finalize storage"""
        await self.index_done_callback()


class SimpleVectorStorage(BaseVectorStorage):
    """Vector storage implementation using NanoVectorDB (matching LightRAG/raganything behavior)"""
    
    def __init__(self, namespace: str, workspace: str = "", global_config: Dict[str, Any] = None, 
                 embedding_func: Any = None, cosine_better_than_threshold: float = 0.2, meta_fields: set = None):
        """Initialize SimpleVectorStorage"""
        super().__init__(namespace=namespace, workspace=workspace, global_config=global_config or {},
                        embedding_func=embedding_func, cosine_better_than_threshold=cosine_better_than_threshold)
        
        # Set meta_fields based on namespace (matching LightRAG)
        if meta_fields is None:
            if namespace == "entities":
                self.meta_fields = {"entity_name", "source_id", "content", "file_path"}
            elif namespace == "relationships":
                self.meta_fields = {"src_id", "tgt_id", "source_id", "content", "file_path"}
            elif namespace == "chunks":
                self.meta_fields = {"full_doc_id", "content", "file_path"}
            else:
                self.meta_fields = {"content", "source_id", "file_path"}
        else:
            self.meta_fields = meta_fields
        
        working_dir = self.global_config.get("working_dir", "./rag_storage")
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
        
        os.makedirs(workspace_dir, exist_ok=True)
        self._vdb_file = os.path.join(workspace_dir, f"vdb_{self.namespace}.json")
        
        # Get embedding batch size from config
        self._max_batch_size = self.global_config.get("embedding_batch_num", 10)
        
        # Require NanoVectorDB (matching LightRAG/raganything behavior - no fallback)
        try:
            from nano_vectordb import NanoVectorDB
            self.nano_vectordb = NanoVectorDB
        except ImportError:
            raise ImportError(
                "NanoVectorDB is required but not installed. "
                "Please install it with: pip install nano-vectordb"
            )
        
        # Create client in __init__ if embedding_func is available (matching LightRAG behavior)
        # This ensures client is always available, matching raganything/LightRAG
        if embedding_func:
            embedding_dim = getattr(embedding_func, 'embedding_dim', 768)
            # Check if file exists and has incompatible format before creating client
            if os.path.exists(self._vdb_file):
                try:
                    with open(self._vdb_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        # If file has incompatible format {"data": [...], "dim": ...} without "matrix", 
                        # NanoVectorDB won't be able to load it (expects "matrix" key)
                        if isinstance(existing_data, dict) and "data" in existing_data and "matrix" not in existing_data:
                            logger.warning(f"Existing vdb file {self._vdb_file} has incompatible format.")
                            logger.info(f"Backing up and removing incompatible file. NanoVectorDB will create a new file.")
                            # Backup old file
                            backup_file = self._vdb_file + ".backup"
                            import shutil
                            shutil.copy2(self._vdb_file, backup_file)
                            logger.info(f"Backed up incompatible file to {backup_file}")
                            # Delete the incompatible file so NanoVectorDB can create a new one
                            os.remove(self._vdb_file)
                except Exception as e:
                    logger.warning(f"Could not check vdb file format: {e}, proceeding anyway")
            
            # Create NanoVectorDB instance immediately (matching LightRAG behavior)
            # LightRAG creates client in __post_init__ and raises on error
            # We create it here but allow retry in initialize() if it fails
            try:
                self._client = self.nano_vectordb(embedding_dim, storage_file=self._vdb_file)
                self._initialized = True
                logger.info(f"Created NanoVectorDB client for {self.namespace} in __init__ with embedding_dim={embedding_dim}, file={self._vdb_file}")
                logger.debug(f"After __init__: _initialized={self._initialized}, _client={self._client is not None}, id(self)={id(self)}")
            except KeyError as e:
                # This is the 'matrix' KeyError - file format issue
                logger.warning(f"Failed to create NanoVectorDB instance in __init__ for {self.namespace} due to file format issue: {e}")
                logger.info(f"File {self._vdb_file} will be handled in initialize() method")
                self._client = None
                self._initialized = False
            except Exception as e:
                # Other errors - log but allow retry
                logger.warning(f"Failed to create NanoVectorDB instance in __init__ for {self.namespace}: {e}")
                logger.info(f"Will retry in initialize() method")
                import traceback
                logger.debug(traceback.format_exc())
                self._client = None
                self._initialized = False
        else:
            self._client = None
            self._initialized = False
        
        self._lock = asyncio.Lock()
    
    async def _initialize_internal(self):
        """Internal initialization method (assumes lock is already held)"""
        # Log current state for debugging (use INFO level for visibility)
        logger.info(f"_initialize_internal for {self.namespace}: _initialized={self._initialized}, _client={self._client is not None}, _client={repr(self._client) if self._client is not None else 'None'}, id(self)={id(self)}")
        
        # If client exists and initialized flag is set, we're done
        # Only check if client is None, not truthiness (NanoVectorDB may be falsy but valid)
        if self._initialized and self._client is not None:
            logger.info(f"Storage {self.namespace} already initialized, skipping (id={id(self)})")
            logger.info(f"Returning early: _initialized={self._initialized}, _client={self._client is not None}")
            return
        
        # If client exists but initialized flag is not set, just set the flag
        # Only check if client is None, not truthiness (NanoVectorDB may be falsy but valid)
        if self._client is not None and not self._initialized:
            logger.info(f"Client exists but _initialized=False for {self.namespace}, setting flag (id={id(self)})")
            self._initialized = True
            logger.info(f"Set _initialized=True, returning. _client={self._client is not None}")
            return
        
        # If client wasn't created in __init__ (embedding_func was None or creation failed), create it now
        if not self._client:
            logger.info(f"Creating new client for {self.namespace} (id={id(self)})")
            if not self.embedding_func:
                raise ValueError("embedding_func is required for SimpleVectorStorage")
            
            embedding_dim = getattr(self.embedding_func, 'embedding_dim', 768)
            logger.debug(f"Initializing NanoVectorDB for {self.namespace} with embedding_dim={embedding_dim}, file={self._vdb_file}")
            
            # Check if file exists and has incompatible format (not NanoVectorDB format)
            if os.path.exists(self._vdb_file):
                try:
                    with open(self._vdb_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        # If file has incompatible format {"data": [...], "dim": ...} without "matrix", 
                        # NanoVectorDB won't be able to load it (expects "matrix" key)
                        if isinstance(existing_data, dict) and "data" in existing_data and "matrix" not in existing_data:
                            logger.warning(f"Existing vdb file {self._vdb_file} has incompatible format.")
                            logger.info(f"Backing up and removing incompatible file. NanoVectorDB will create a new file.")
                            # Backup old file
                            backup_file = self._vdb_file + ".backup"
                            import shutil
                            shutil.copy2(self._vdb_file, backup_file)
                            logger.info(f"Backed up incompatible file to {backup_file}")
                            # Delete the incompatible file so NanoVectorDB can create a new one
                            os.remove(self._vdb_file)
                        else:
                            logger.debug(f"Existing vdb file {self._vdb_file} format looks compatible")
                except json.JSONDecodeError as e:
                    logger.warning(f"vdb file {self._vdb_file} is not valid JSON: {e}. Will recreate.")
                    os.remove(self._vdb_file)
                except Exception as e:
                    logger.warning(f"Could not check vdb file format: {e}, proceeding anyway")
            
            # Create NanoVectorDB instance (matching LightRAG/raganything behavior)
            try:
                logger.info(f"About to create nano_vectordb for {self.namespace} (id={id(self)})")
                self._client = self.nano_vectordb(embedding_dim, storage_file=self._vdb_file)
                logger.info(f"Created nano_vectordb instance: {type(self._client)}, id={id(self._client)}")
                self._initialized = True  # Set initialized immediately after creating client
                logger.info(f"Successfully initialized NanoVectorDB client for {self.namespace} with embedding_dim={embedding_dim} (id={id(self)})")
                logger.info(f"After initialization: _initialized={self._initialized}, _client={self._client is not None}, _client type={type(self._client)}, _client id={id(self._client)}")
            except KeyError as e:
                # This is likely the 'matrix' KeyError - file format issue
                logger.error(f"Failed to create NanoVectorDB instance for {self.namespace} due to file format issue (KeyError: {e})")
                logger.info(f"Attempting to remove incompatible file and retry...")
                if os.path.exists(self._vdb_file):
                    backup_file = self._vdb_file + ".backup"
                    import shutil
                    try:
                        shutil.copy2(self._vdb_file, backup_file)
                        logger.info(f"Backed up incompatible file to {backup_file}")
                        os.remove(self._vdb_file)
                        # Retry creating client
                        self._client = self.nano_vectordb(embedding_dim, storage_file=self._vdb_file)
                        self._initialized = True  # Set initialized immediately after creating client
                        logger.info(f"Successfully initialized NanoVectorDB client for {self.namespace} after fixing file format")
                        logger.debug(f"After retry initialization: _initialized={self._initialized}, _client={self._client is not None}")
                    except Exception as retry_e:
                        logger.error(f"Failed to retry after fixing file format: {retry_e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        raise RuntimeError(f"Failed to initialize NanoVectorDB storage for {self.namespace} after format fix: {retry_e}")
                else:
                    raise RuntimeError(f"Failed to initialize NanoVectorDB storage for {self.namespace}: KeyError {e}")
            except Exception as e:
                logger.error(f"Failed to create NanoVectorDB instance for {self.namespace}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to initialize NanoVectorDB storage for {self.namespace}: {e}")
        
        # Ensure _initialized is set (in case client was created in __init__)
        # Only check if client is None, not truthiness (NanoVectorDB may be falsy but valid)
        if self._client is not None and not self._initialized:
            self._initialized = True
            logger.info(f"Set _initialized=True for {self.namespace} (client was already created) (id={id(self)})")
        
        logger.info(f"Storage {self.namespace} initialization complete: _initialized={self._initialized}, _client={self._client is not None}, _client type={type(self._client) if self._client else None}, id(self)={id(self)}")
    
    async def initialize(self):
        """Initialize storage (matching LightRAG/raganything behavior)"""
        async with self._lock:
            await self._initialize_internal()
    
    async def query(self, query: str, top_k: int, query_embedding: List[float] = None) -> List[Dict[str, Any]]:
        """Query vectors using NanoVectorDB (matching LightRAG/raganything behavior)"""
        # Ensure initialized - use lock to prevent race conditions
        async with self._lock:
            # Check and initialize if needed
            if not self._initialized or self._client is None:
                await self._initialize_internal()
            
            # Verify client is available (within lock)
            if self._client is None:
                logger.error(f"Client still None after initialization for {self.namespace} (id={id(self)})")
                logger.error(f"State: _initialized={self._initialized}, embedding_func={self.embedding_func is not None}")
                raise RuntimeError(
                    f"NanoVectorDB client not initialized for {self.namespace}. "
                    f"embedding_func available: {self.embedding_func is not None}, "
                    f"embedding_dim: {getattr(self.embedding_func, 'embedding_dim', 'N/A') if self.embedding_func else 'N/A'}"
                )
        
        if not query_embedding:
            if not self.embedding_func:
                return []
            query_embedding = await self.embedding_func.func([query])
            if not query_embedding:
                return []
            query_embedding = query_embedding[0]
        
        try:
            # NanoVectorDB uses query() method, not search()
            results = self._client.query(
                query=query_embedding,
                top_k=top_k,
                better_than_threshold=self.cosine_better_than_threshold,
            )
            
            # Format results to match expected format
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    formatted_result = {
                        **{k: v for k, v in result.items() if k not in ["vector", "__vector__"]},
                        "id": result.get("__id__", result.get("id", "")),
                        "distance": result.get("__metrics__", result.get("distance", float("inf"))),
                        "created_at": result.get("__created_at__", None),
                    }
                    # Include metadata fields
                    for field in self.meta_fields:
                        if field in result:
                            if "metadata" not in formatted_result:
                                formatted_result["metadata"] = {}
                            formatted_result["metadata"][field] = result[field]
                        elif field in result.get("metadata", {}):
                            if "metadata" not in formatted_result:
                                formatted_result["metadata"] = {}
                            formatted_result["metadata"][field] = result["metadata"][field]
                    formatted_results.append(formatted_result)
                else:
                    formatted_results.append({"result": result})
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying NanoVectorDB: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Upsert vectors using NanoVectorDB (matching LightRAG/raganything behavior)"""
        if not data:
            return
        
        # Ensure initialized - use lock to prevent race conditions
        async with self._lock:
            # Log state before initialization check (use INFO level for visibility)
            logger.info(f"upsert for {self.namespace}: _initialized={self._initialized}, _client={self._client is not None}, _client={repr(self._client) if self._client is not None else 'None'}, id(self)={id(self)}")
            
            # Check and initialize if needed
            # Only check if client is None, not truthiness (NanoVectorDB may be falsy but valid)
            check_initialized = not self._initialized
            check_client = self._client is None
            logger.info(f"Initialization check for {self.namespace}: not _initialized={check_initialized}, _client is None={check_client}, condition={check_initialized or check_client}")
            
            if check_initialized or check_client:
                # Call internal initialization method (lock already held)
                logger.info(f"Calling _initialize_internal for {self.namespace} (id={id(self)})")
                await self._initialize_internal()
                logger.info(f"After _initialize_internal: _initialized={self._initialized}, _client={self._client is not None}, _client={repr(self._client) if self._client is not None else 'None'}, id(self)={id(self)}")
            else:
                logger.info(f"Skipping _initialize_internal for {self.namespace} (already initialized) (id={id(self)})")
            
            # Verify client is available (within lock)
            # Only check if client is None, not truthiness (NanoVectorDB may be falsy but valid)
            if self._client is None:
                logger.error(f"Client still None after initialization for {self.namespace} (id={id(self)})")
                logger.error(f"State: _initialized={self._initialized}, embedding_func={self.embedding_func is not None}")
                logger.error(f"embedding_func type: {type(self.embedding_func)}")
                logger.error(f"embedding_func id: {id(self.embedding_func) if self.embedding_func else 'N/A'}")
                raise RuntimeError(
                    f"NanoVectorDB client not initialized for {self.namespace}. "
                    f"embedding_func available: {self.embedding_func is not None}, "
                    f"embedding_dim: {getattr(self.embedding_func, 'embedding_dim', 'N/A') if self.embedding_func else 'N/A'}"
                )
            
            logger.info(f"Client verified for {self.namespace}: _client type={type(self._client)}, _client id={id(self._client)}")
            
            if not self.embedding_func:
                raise ValueError("embedding_func is required for upsert")
            
            # Store client reference locally to use outside lock
            client = self._client
        
        # Process data outside lock (data processing doesn't need lock)
        try:
            import time
            import base64
            import zlib
            current_time = int(time.time())
            
            # Build list_data with only meta_fields (matching LightRAG)
            list_data = [
                {
                    "__id__": k,
                    "__created_at__": current_time,
                    **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
                }
                for k, v in data.items()
            ]
            
            # Check if vectors are already provided (e.g., for Virchow2 features)
            precomputed_vectors = []
            for v in data.values():
                if "__vector__" in v:
                    vec = v["__vector__"]
                    # Ensure vector is a 1D numpy array
                    if isinstance(vec, np.ndarray):
                        # Flatten if needed (handle 2D arrays)
                        if vec.ndim > 1:
                            vec = vec.flatten()
                        elif vec.ndim == 0:
                            # Scalar, convert to 1D array
                            vec = np.array([vec])
                    else:
                        # Convert to numpy array
                        vec = np.array(vec)
                        if vec.ndim > 1:
                            vec = vec.flatten()
                        elif vec.ndim == 0:
                            vec = np.array([vec])
                    precomputed_vectors.append(vec)
                    logger.debug(f"Precomputed vector shape: {vec.shape}, dtype: {vec.dtype}")
                else:
                    precomputed_vectors.append(None)
            
            # If all vectors are precomputed, use them directly
            if all(vec is not None for vec in precomputed_vectors):
                logger.info(f"Using precomputed vectors for {len(list_data)} items in {self.namespace}")
                # Stack vectors into 2D array (num_vectors, embedding_dim)
                embeddings = np.vstack(precomputed_vectors)
                logger.info(f"Stacked embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
            else:
                # Extract contents for embedding
                contents = [v.get("content", "") for v in data.values()]
                
                # Batch embeddings
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                
                # Compute embeddings in batches
                embedding_tasks = [self.embedding_func.func(batch) for batch in batches]
                embeddings_list = await asyncio.gather(*embedding_tasks)
                embeddings = np.concatenate(embeddings_list)
            
            if len(embeddings) == len(list_data):
                # Verify embedding dimensions are consistent
                embedding_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0]
                expected_dim = getattr(self.embedding_func, 'embedding_dim', None)
                if expected_dim and embedding_dim != expected_dim:
                    logger.error(f"Embedding dimension mismatch: expected {expected_dim}, got {embedding_dim}")
                    logger.error(f"Embeddings shape: {embeddings.shape}")
                    raise ValueError(f"Embedding dimension mismatch: expected {expected_dim}, got {embedding_dim}")
                
                # Check if NanoVectorDB storage has existing data with different dimensions
                try:
                    # Try to get existing matrix to check dimensions
                    if hasattr(client, '_NanoVectorDB__storage') and 'matrix' in client._NanoVectorDB__storage:
                        existing_matrix = client._NanoVectorDB__storage['matrix']
                        if existing_matrix.shape[0] > 0:
                            existing_dim = existing_matrix.shape[1]
                            if existing_dim != embedding_dim:
                                logger.error(f"Dimension mismatch with existing data: existing={existing_dim}, new={embedding_dim}")
                                logger.error(f"Existing matrix shape: {existing_matrix.shape}")
                                logger.error(f"New embeddings shape: {embeddings.shape}")
                                raise ValueError(
                                    f"Dimension mismatch: existing data has dimension {existing_dim}, "
                                    f"but new embeddings have dimension {embedding_dim}. "
                                    f"Please clear the storage file and retry."
                                )
                except Exception as check_error:
                    logger.warning(f"Could not check existing data dimensions: {check_error}")
                    # Continue anyway, let NanoVectorDB handle it
                
                for i, d in enumerate(list_data):
                    # Ensure embedding is 1D
                    embedding = embeddings[i] if len(embeddings.shape) > 1 else embeddings
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()
                    
                    # Compress vector using Float16 + zlib + Base64 (matching LightRAG)
                    vector_f16 = embedding.astype(np.float16)
                    compressed_vector = zlib.compress(vector_f16.tobytes())
                    encoded_vector = base64.b64encode(compressed_vector).decode("utf-8")
                    d["vector"] = encoded_vector
                    d["__vector__"] = embedding
                
                logger.debug(f"Upserting {len(list_data)} vectors with dimension {embedding_dim} to {self.namespace}")
                client.upsert(datas=list_data)
            else:
                raise ValueError(
                    f"Embedding count mismatch: {len(embeddings)} != {len(list_data)}"
                )
        except Exception as e:
            logger.error(f"Error upserting to NanoVectorDB: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get vector by ID"""
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        if not self._client:
            raise RuntimeError("NanoVectorDB client not initialized")
        
        try:
            result = self._client.get([id])
            if result:
                return result[0]
            return None
        except Exception as e:
            logger.error(f"Error getting from NanoVectorDB: {e}")
            raise
    
    async def index_done_callback(self) -> None:
        """Save vectors to disk using NanoVectorDB (matching LightRAG/raganything behavior)"""
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        if not self._client:
            raise RuntimeError("NanoVectorDB client not initialized")
        
        try:
            self._client.save()
            logger.info(f"Saved vectors to {self._vdb_file}")
        except Exception as e:
            logger.error(f"Error saving NanoVectorDB: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    async def finalize(self) -> None:
        """Finalize storage"""
        await self.index_done_callback()


class NetworkXGraphStorage(BaseGraphStorage):
    """NetworkX-based graph storage implementation"""
    
    def __init__(self, namespace: str, workspace: str = "", global_config: Dict[str, Any] = None):
        """Initialize NetworkXGraphStorage"""
        super().__init__(namespace=namespace, workspace=workspace, global_config=global_config or {})
        try:
            import networkx as nx
            self.nx = nx
        except ImportError:
            logger.error("NetworkX not available, graph storage will not work")
            self.nx = None
        
        working_dir = self.global_config.get("working_dir", "./rag_storage")
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
        
        os.makedirs(workspace_dir, exist_ok=True)
        self._graphml_file = os.path.join(workspace_dir, f"graph_{self.namespace}.graphml")
        
        self._graph = None
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: Dict[tuple, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize graph from file or create new"""
        async with self._lock:
            if self.nx:
                # Try to load existing graph
                if os.path.exists(self._graphml_file):
                    try:
                        self._graph = self.nx.read_graphml(self._graphml_file)
                        logger.info(f"Loaded graph from {self._graphml_file} with {self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges")
                    except Exception as e:
                        logger.warning(f"Error loading graph from {self._graphml_file}: {e}, creating new graph")
                        self._graph = self.nx.Graph()
                else:
                    self._graph = self.nx.Graph()
                    logger.info(f"Created new empty graph: {self._graphml_file}")
            else:
                logger.error("NetworkX not available! Graph storage will not work. Please install: pip install networkx")
                logger.error("GraphML files will not be saved without NetworkX.")
                logger.error(f"GraphML file path: {self._graphml_file}")
            self._initialized = True
    
    async def get_node(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get node by entity name"""
        async with self._lock:
            if self._graph and self._graph.has_node(entity_name):
                return dict(self._graph.nodes[entity_name])
            return self._nodes.get(entity_name)
    
    async def upsert_node(self, entity_name: str, node_data: Dict[str, Any]) -> None:
        """Insert or update node"""
        async with self._lock:
            self._nodes[entity_name] = node_data
            if self._graph and self.nx:
                self._graph.add_node(entity_name, **node_data)
            elif not self.nx:
                logger.debug(f"NetworkX not available, node {entity_name} stored in memory only")
    
    async def get_edge(self, src_id: str, tgt_id: str) -> Optional[Dict[str, Any]]:
        """Get edge between two nodes"""
        async with self._lock:
            if self._graph and self._graph.has_edge(src_id, tgt_id):
                return dict(self._graph.edges[src_id, tgt_id])
            key = tuple(sorted([src_id, tgt_id]))
            return self._edges.get(key)
    
    async def upsert_edge(self, src_id: str, tgt_id: str, edge_data: Dict[str, Any]) -> None:
        """Insert or update edge"""
        async with self._lock:
            key = tuple(sorted([src_id, tgt_id]))
            self._edges[key] = edge_data
            if self._graph and self.nx:
                self._graph.add_edge(src_id, tgt_id, **edge_data)
            elif not self.nx:
                logger.debug(f"NetworkX not available, edge ({src_id}, {tgt_id}) stored in memory only")
    
    async def has_edge(self, src_id: str, tgt_id: str) -> bool:
        """Check if edge exists"""
        async with self._lock:
            if self._graph:
                return self._graph.has_edge(src_id, tgt_id)
            key = tuple(sorted([src_id, tgt_id]))
            return key in self._edges
    
    async def get_node_edges(self, entity_name: str) -> List[tuple]:
        """Get all edges for a node"""
        async with self._lock:
            if self._graph and self._graph.has_node(entity_name):
                return list(self._graph.edges(entity_name))
            edges = []
            for (src, tgt) in self._edges.keys():
                if src == entity_name or tgt == entity_name:
                    edges.append((src, tgt))
            return edges
    
    async def index_done_callback(self) -> None:
        """Save graph to GraphML file"""
        async with self._lock:
            # Debug: Check NetworkX availability
            if not self.nx:
                logger.warning("NetworkX not available, cannot save graph to GraphML file. Please install: pip install networkx")
                logger.warning(f"GraphML file path would be: {self._graphml_file}")
                logger.warning(f"Nodes in memory: {len(self._nodes)}, Edges in memory: {len(self._edges)}")
                return
            
            # Debug: Check graph initialization
            if not self._graph:
                logger.warning("Graph not initialized, cannot save to GraphML file")
                logger.warning(f"GraphML file path: {self._graphml_file}")
                logger.warning(f"Nodes in memory: {len(self._nodes)}, Edges in memory: {len(self._edges)}")
                # Try to create graph from memory if NetworkX is available
                if self.nx and (self._nodes or self._edges):
                    logger.info("Attempting to create graph from memory data...")
                    self._graph = self.nx.Graph()
                    # Add nodes
                    for node_id, node_data in self._nodes.items():
                        self._graph.add_node(node_id, **node_data)
                    # Add edges
                    for (src_id, tgt_id), edge_data in self._edges.items():
                        self._graph.add_edge(src_id, tgt_id, **edge_data)
                    logger.info(f"Created graph from memory: {self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges")
                else:
                    return
            
            try:
                # Ensure directory exists
                file_dir = os.path.dirname(self._graphml_file)
                if file_dir:
                    os.makedirs(file_dir, exist_ok=True)
                
                # Debug: Log graph state before saving
                num_nodes = self._graph.number_of_nodes()
                num_edges = self._graph.number_of_edges()
                logger.debug(f"About to save graph: {num_nodes} nodes, {num_edges} edges to {self._graphml_file}")
                
                # Save graph to GraphML file (even if empty)
                self.nx.write_graphml(self._graph, self._graphml_file)
                logger.info(f"Saved graph to {self._graphml_file} with {num_nodes} nodes, {num_edges} edges")
            except Exception as e:
                logger.error(f"Error saving graph to {self._graphml_file}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    async def finalize(self) -> None:
        """Finalize storage"""
        await self.index_done_callback()
