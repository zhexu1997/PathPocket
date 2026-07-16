"""
Stage 4: Generate embeddings for text and images (GPU required)
Input: kv_store_text_chunks.json, kv_store_full_entities.json, kv_store_full_relations.json
Output: vdb_chunks.json, vdb_entities.json, vdb_relationships.json, vdb_pathology_images.json
"""


import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import asyncio
import os
import json
import logging
import numpy as np
import base64
import zlib
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import ollama as ollama_lib
from dotenv import load_dotenv

# Try to import sentence-transformers for direct model loading
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    torch = None

def normalize_image_path(image_path: str) -> str:
    """Remap image path prefixes via IMAGE_PATH_PREFIX_FROM/TO (or MAP)."""
    if not image_path or not isinstance(image_path, str):
        return image_path

    mappings = []
    raw = os.getenv("IMAGE_PATH_PREFIX_MAP", "").strip()
    if raw:
        for part in raw.split(","):
            part = part.strip()
            if ":" in part:
                frm, to = part.split(":", 1)
                frm, to = frm.strip().rstrip("/"), to.strip().rstrip("/")
                if frm and to:
                    mappings.append((frm, to))
    frm = os.getenv("IMAGE_PATH_PREFIX_FROM", "").strip().rstrip("/")
    to = os.getenv("IMAGE_PATH_PREFIX_TO", "").strip().rstrip("/")
    if frm and to:
        mappings.append((frm, to))
    mappings.sort(key=lambda x: len(x[0]), reverse=True)

    for old_prefix, new_prefix in mappings:
        if image_path.startswith(old_prefix):
            rel = image_path[len(old_prefix) :].lstrip("/")
            return f"{new_prefix}/{rel}" if rel else new_prefix
    return image_path


from pathpocket import (
    PathPocket,
    PathPocketConfig,
    fix_invalid_doc_status,
)

# Load environment variables
load_dotenv()

# Disable httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)


async def main():
    # ========== Configuration ==========
    # Choose embedding method: "direct" (faster, requires sentence-transformers) or "ollama" (slower but simpler)
    EMBEDDING_METHOD = os.getenv("EMBEDDING_METHOD", "direct").lower()  # "direct" or "ollama"
    
    OLLAMA_EMBEDDING_HOST = os.getenv("OLLAMA_EMBEDDING_HOST", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest")
    # For direct loading: use local path if exists, otherwise use HuggingFace model name
    EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "./models/bge-m3")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", None)  # Will be set based on path existence
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))  # Batch size for direct method
    
    # Check if local model path exists
    if EMBEDDING_METHOD == "direct" and os.path.exists(EMBEDDING_MODEL_PATH):
        EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_PATH
        print(f"✅ Found local model at: {EMBEDDING_MODEL_PATH}")
    elif EMBEDDING_MODEL_NAME is None:
        EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # Fallback to HuggingFace
    
    
    working_dir = os.getenv("WORKING_DIR", "./pathpocket_storage")
    
    # Virchow2 configuration (optional)
    VIRCHOW2_MODEL_PATH = os.getenv("VIRCHOW2_MODEL_PATH", "./models/Virchow2")
    
    print(f"\n{'='*60}")
    print("Stage 4: Generate embeddings for text and images (GPU required)")
    print(f"{'='*60}")
    print(f"Embedding Method: {EMBEDDING_METHOD.upper()}")
    if EMBEDDING_METHOD == "ollama":
        print(f"Embedding Service: Ollama ({OLLAMA_EMBEDDING_HOST})")
        print(f"Embedding Model: {OLLAMA_EMBEDDING_MODEL}")
    else:
        print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
        print(f"Batch Size: {EMBEDDING_BATCH_SIZE}")
    print(f"Embedding Dimension: {EMBEDDING_DIM}")
    print(f"Working directory: {working_dir}")
    print(f"{'='*60}\n")
    
    # Define embedding function based on method
    if EMBEDDING_METHOD == "direct" and SENTENCE_TRANSFORMERS_AVAILABLE:
        # Direct model loading (much faster)
        print("🚀 Using direct model loading (sentence-transformers) - Fast mode")
        # Import torch here to ensure it's available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on device: {device}")
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        
        # Load model once (before async function)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        print("✅ Model loaded successfully")
        
        async def _direct_embed(texts: List[str], **kwargs):
            """Embedding function using direct model loading (batch processing)"""
            # Batch processing for efficiency
            embeddings = embedding_model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # BGE-M3 benefits from normalization
            )
            return np.array(embeddings)
        
        embedding_func_impl = _direct_embed
    else:
        # Fallback to Ollama
        if EMBEDDING_METHOD == "direct":
            print("⚠️  sentence-transformers not available, falling back to Ollama")
            print("   Install with: pip install sentence-transformers")
        
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=2, max=10),
            retry=retry_if_exception_type((Exception,)),
        )
        async def _ollama_embed_with_retry(texts: List[str], **kwargs):
            """Embedding function using Ollama"""
            embeddings = []
            for idx, text in enumerate(texts):
                try:
                    ollama_client = ollama_lib.AsyncClient(
                        host=OLLAMA_EMBEDDING_HOST,
                        timeout=kwargs.get("timeout", 120)
                    )
                    try:
                        data = await ollama_client.embed(
                            model=OLLAMA_EMBEDDING_MODEL,
                            input=[text],
                            options=kwargs.get("options", {})
                        )
                        if data and "embeddings" in data and len(data["embeddings"]) > 0:
                            embeddings.append(data["embeddings"][0])
                        else:
                            raise ValueError(f"Empty embedding response from Ollama for text {idx}")
                        
                        if idx < len(texts) - 1:
                            await asyncio.sleep(0.3)
                    finally:
                        try:
                            await ollama_client._client.aclose()
                        except:
                            pass
                except Exception as e:
                    error_msg = str(e)
                    if "runner process no longer running" in error_msg or "EOF" in error_msg:
                        print(f"⚠️  Ollama runner crashed, waiting for recovery... (text {idx+1}/{len(texts)})")
                        await asyncio.sleep(5)
                    else:
                        print(f"⚠️  Embedding error (text {idx+1}/{len(texts)}): {error_msg[:100]}...")
                        await asyncio.sleep(2)
                    raise
            
            return np.array(embeddings)
        
        embedding_func_impl = _ollama_embed_with_retry
    
    # Create embedding function wrapper
    class SimpleEmbeddingFunc:
        """Simple embedding function wrapper for RAG engine"""
        def __init__(self, embedding_dim, max_token_size, func):
            self.embedding_dim = embedding_dim
            self.max_token_size = max_token_size
            self.func = func
        
        async def __call__(self, texts):
            """Call embedding function"""
            return await self.func(texts)
    
    embedding_func = SimpleEmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,  # BGE-M3 supports up to 8192 tokens
        func=lambda texts: embedding_func_impl(texts, timeout=120),
    )
    
    # Initialize Virchow2 feature extractor for pathology images (optional)
    virchow2_feature_func = None
    try:
        from pathpocket.virchow2_feature_extractor import (
            Virchow2FeatureExtractor,
            Virchow2FeatureExtractorWrapper
        )
        
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device for Virchow2: {device}")
        
        virchow2_extractor = Virchow2FeatureExtractor(
            model_path=VIRCHOW2_MODEL_PATH,
            device=device,
            batch_size=8
        )
        virchow2_feature_func = Virchow2FeatureExtractorWrapper(virchow2_extractor)
        print("Virchow2 feature extractor initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize Virchow2 feature extractor: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing without Virchow2 features...")
    
    # Create PathPocket configuration
    config = PathPocketConfig(
        working_dir=working_dir,
        enable_image_processing=True,
        enable_table_processing=True,
    )
    
    # Dummy LLM function (should not be called)
    async def dummy_llm_func(*args, **kwargs):
        raise RuntimeError("LLM should not be called in Stage 4")
    
    # Get graph storage type from environment or use default
    graph_storage = os.getenv("GRAPH_STORAGE", "HyperNetXStorage")
    # vector_storage = os.getenv("VECTOR_STORAGE", "HyperNetXStorage")

    # Initialize PathPocket with embedding but no LLM
    rag = PathPocket(
        config=config,
        llm_model_func=dummy_llm_func,
        embedding_func=embedding_func,
        conch_feature_func=virchow2_feature_func,
        rag_engine_kwargs={
            "graph_storage": graph_storage,  # Support PGHypergraphStorage  
            "addon_params": {},
            "embedding_func_max_async": 16,
        }
    )
    
    # Display storage configuration
    print(f"Graph Storage: {graph_storage}")
    if graph_storage == "PGHypergraphStorage":
        print(f"  PostgreSQL User: {os.getenv('POSTGRES_USER', 'Not set')}")
        print(f"  PostgreSQL Database: {os.getenv('POSTGRES_DATABASE', 'Not set')}")
        print(f"  PostgreSQL Host: {os.getenv('POSTGRES_HOST', 'localhost')}")
    
    # Fix any invalid doc_status records
    await fix_invalid_doc_status(rag)
    
    # Initialize RAG engine
    init_result = await rag._ensure_rag_engine_initialized()
    if not init_result.get("success"):
        raise RuntimeError(f"Failed to initialize RAG engine: {init_result.get('error')}")
    
    print(f"\n{'='*60}")
    print("Generating embeddings...")
    print(f"{'='*60}\n")
    
    # 1. Generate embeddings for text chunks
    print("1. Generating embeddings for text chunks...")
    
    # Configuration for memory-efficient processing
    CHUNK_READ_BATCH_SIZE = int(os.getenv("CHUNK_READ_BATCH_SIZE", "64"))  # Batch size for reading from DB/JSON
    EMBEDDING_UPSERT_BATCH_SIZE = int(os.getenv("EMBEDDING_UPSERT_BATCH_SIZE", "64"))  # Batch size for embedding upsert
    
    graph_storage = os.getenv("GRAPH_STORAGE", "HyperNetXStorage")
    workspace = "default"
    
    # Try to load from JSON first (faster, less memory)
    json_chunks_file = os.path.join(working_dir, "kv_store_text_chunks.json")
    use_json = os.path.exists(json_chunks_file) 
    
    total_chunks_processed = 0
    
    if use_json:
        # Load from JSON file (streaming for large files)
        print(f"  Loading chunks from JSON file: {json_chunks_file}")
        try:
            import json
            with open(json_chunks_file, 'r', encoding='utf-8') as f:
                all_chunks = json.load(f)
            
            if all_chunks:
                print(f"  Found {len(all_chunks)} chunks in JSON file")
                chunk_items = list(all_chunks.items())
                
                # Process in batches to save memory
                for i in range(0, len(chunk_items), CHUNK_READ_BATCH_SIZE):
                    batch_items = chunk_items[i:i + CHUNK_READ_BATCH_SIZE]
                    batch = dict(batch_items)
                    batch_for_vdb = {}
                    for cid, chunk_data in batch.items():
                        batch_for_vdb[cid] = {
                            "content": chunk_data.get("content", ""),
                            "source_id": cid,
                            "file_path": chunk_data.get("file_path", "unknown"),
                        }
                    try:
                        await rag.rag_engine.chunks_vdb.upsert(batch_for_vdb)
                        total_chunks_processed += len(batch)
                        if (i + CHUNK_READ_BATCH_SIZE) % (CHUNK_READ_BATCH_SIZE * 10) == 0 or i + CHUNK_READ_BATCH_SIZE >= len(chunk_items):
                            print(f"  Processed {total_chunks_processed}/{len(chunk_items)} chunks...")
                    except Exception as e:
                        print(f"  ⚠️  Error processing batch {i // CHUNK_READ_BATCH_SIZE + 1}: {e}")
                
                await rag.rag_engine.chunks_vdb.index_done_callback()
                print(f"✅ Text chunk embeddings completed ({total_chunks_processed} chunks)")
            else:
                print("⚠️  No chunks found in JSON file")
                use_json = False  # Fall back to other methods
        except Exception as e:
            print(f"  ⚠️  Error reading JSON file: {e}")
            use_json = False
    
    
    # if total_chunks_processed == 0:
    #     print("⚠️  No text chunks found or processed")
    
    # # 2. Generate embeddings for entities
    # print("\n2. Generating embeddings for entities...")
    
    # # Direct write to PostgreSQL to avoid memory explosion
    # # Skip VDB and write embeddings directly to VDB_ENTITY table
    # from pathpocket.lightrag_utils import compute_mdhash_id
    
    # # Configuration for batch processing
    # NODE_READ_BATCH_SIZE = int(os.getenv("NODE_READ_BATCH_SIZE", "64"))  # Batch size for reading nodes from DB
    # EMBEDDING_GEN_BATCH_SIZE = int(os.getenv("EMBEDDING_GEN_BATCH_SIZE", "32"))  # Batch size for generating embeddings (smaller to save memory)
    
    # # Check if we're using PGHypergraphStorage (can write directly to PostgreSQL)
    # from pathpocket.lightrag_kg.pg_hypergraph_impl import PGHypergraphStorage
    # is_pg_storage = isinstance(rag.rag_engine.chunk_entity_relation_graph, PGHypergraphStorage)
    
    # if is_pg_storage:
    #     # Direct write to PostgreSQL VDB_ENTITY table
    #     await generate_entity_embeddings_direct_to_postgres(
    #         rag, 
    #         embedding_func, 
    #         EMBEDDING_DIM,
    #         NODE_READ_BATCH_SIZE,
    #         EMBEDDING_GEN_BATCH_SIZE
    #     )
    # else:
    #     # For non-PG storage: use original VDB method (but still in batches)
    #     print("  Using VDB method for non-PG storage...")
    #     ENTITY_EMBEDDING_BATCH_SIZE = int(os.getenv("ENTITY_EMBEDDING_BATCH_SIZE", "64"))
    #     total_entities_processed = 0
    #     processed_entity_ids = set()
        
    #     if hasattr(rag.rag_engine.chunk_entity_relation_graph, 'get_all_nodes'):
    #         try:
    #             all_nodes = await rag.rag_engine.chunk_entity_relation_graph.get_all_nodes()
                
    #             if all_nodes:
    #                 # Process nodes in batches
    #                 for i in range(0, len(all_nodes), ENTITY_EMBEDDING_BATCH_SIZE):
    #                     batch_nodes = all_nodes[i:i + ENTITY_EMBEDDING_BATCH_SIZE]
    #                     batch_for_vdb = {}
                        
    #                     for node in batch_nodes:
    #                         entity_id = node.get("entity_id") or node.get("id", "")
    #                         if not entity_id or entity_id in processed_entity_ids:
    #                             continue
    #                         processed_entity_ids.add(entity_id)
                            
    #                         entity_name = entity_id
    #                         description = node.get("description", "")
                            
    #                         if description:
    #                             content = f"{entity_name}\n{description}"
    #                         else:
    #                             content = entity_name
                            
    #                         eid = compute_mdhash_id(entity_name, prefix="ent-")
                            
    #                         batch_for_vdb[eid] = {
    #                             "entity_name": entity_name,
    #                             "source_id": node.get("source_id", entity_id),
    #                             "content": content,
    #                             "file_path": node.get("file_path", "unknown"),
    #                         }
                        
    #                     if batch_for_vdb:
    #                         try:
    #                             await rag.rag_engine.entities_vdb.upsert(batch_for_vdb)
    #                             total_entities_processed += len(batch_for_vdb)
    #                             batch_num = i // ENTITY_EMBEDDING_BATCH_SIZE + 1
    #                             total_batches = (len(all_nodes) + ENTITY_EMBEDDING_BATCH_SIZE - 1) // ENTITY_EMBEDDING_BATCH_SIZE
    #                             print(f"  Processed batch {batch_num}/{total_batches} ({total_entities_processed} entities)...")
    #                         except Exception as e:
    #                             print(f"  ⚠️  Error processing batch {i // ENTITY_EMBEDDING_BATCH_SIZE + 1}: {e}")
            
    #             if total_entities_processed > 0:
    #                 await rag.rag_engine.entities_vdb.index_done_callback()
    #                 print(f"✅ Entity embeddings completed ({total_entities_processed} entities)")
    #             else:
    #                 print("⚠️  No entities found or processed")
    #         except Exception as e:
    #             print(f"  ⚠️  Error getting nodes: {e}")
    #             print("⚠️  No entities processed")
    
    # # 3. Generate embeddings for relationships
    # print("\n3. Generating embeddings for relationships...")
    # # Get all relations from hypergraph (includes multi-entity relations)
    # all_relations = {}
    # if hasattr(rag.rag_engine.chunk_entity_relation_graph, 'get_all_hyperedges'):
    #     hyperedges = await rag.rag_engine.chunk_entity_relation_graph.get_all_hyperedges()
    #     from pathpocket.operate import GRAPH_FIELD_SEP
        
    #     for edge in hyperedges:
    #         entities = edge.get("entities", [])
            
    #         # Ensure entities is a list (not string)
    #         if isinstance(entities, str):
    #             # If entities is a string, split by SEP
    #             entities = [e.strip() for e in entities.split(GRAPH_FIELD_SEP) if e.strip()]
    #         elif not isinstance(entities, list):
    #             # If entities is not a list, convert to list
    #             entities = list(entities) if entities else []
            
    #         if not entities or len(entities) < 2:
    #             continue
            
    #         # Get relation information
    #         relation = edge.get("relation", "")
    #         keywords = edge.get("keywords", "")
    #         description = edge.get("description", "")
    #         source_id = edge.get("source_id", "")
    #         file_path = edge.get("file_path", "unknown")
            
    #         # Create relation ID from sorted entities
    #         sorted_entities = sorted(entities)
    #         edge_id = edge.get("edge_id", GRAPH_FIELD_SEP.join(sorted_entities))
            
    #         # Build content for embedding
    #         content = f"{keywords}\t{GRAPH_FIELD_SEP.join(entities)}\t{description}"
            
    #         # Only save multi-entity relations (entities list)
    #         all_relations[edge_id] = {
    #             "entities": entities,
    #             "entity_count": len(entities),
    #             "relation": relation,
    #             "keywords": keywords,
    #             "description": description,
    #             "source_id": source_id,
    #             "file_path": file_path,
    #             "content": content,
    #         }
    
    # if all_relations:
    #     relation_items = list(all_relations.items())
    #     batch_size = 64
    #     total_batches = (len(relation_items) + batch_size - 1) // batch_size
        
    #     for i in range(0, len(relation_items), batch_size):
    #         batch = dict(relation_items[i:i + batch_size])
    #         batch_for_vdb = {}
    #         for rid, relation_data in batch.items():
    #             # Get entities list (required for multi-entity relations)
    #             entities = relation_data.get("entities", [])
    #             if not entities or len(entities) < 2:
    #                 continue  # Skip relations without valid entities
                
    #             batch_for_vdb[rid] = {
    #                 "entities": entities,
    #                 "entity_count": len(entities),
    #                 "relation": relation_data.get("relation", ""),
    #                 "keywords": relation_data.get("keywords", ""),
    #                 "description": relation_data.get("description", ""),
    #                 "source_id": relation_data.get("source_id", rid),
    #                 "content": relation_data.get("content", ""),
    #                 "file_path": relation_data.get("file_path", "unknown"),
    #             }
    #         try:
    #             await rag.rag_engine.relationships_vdb.upsert(batch_for_vdb)
    #             batch_num = i // batch_size + 1
    #             print(f"  Processed batch {batch_num}/{total_batches} ({len(batch)} relations)")
    #         except Exception as e:
    #             print(f"  ⚠️  Error processing batch {i // batch_size + 1}: {e}")
        
    #     await rag.rag_engine.relationships_vdb.index_done_callback()
    #     print(f"✅ Relationship embeddings completed ({len(all_relations)} relations)")
    # else:
    #     print("⚠️  No relationships found")
    
    # 4. Generate Virchow2 features for pathology images (if available)
    if virchow2_feature_func and rag.rag_engine.pathology_images_vdb:
        print("\n4. Generating Virchow2 features for pathology images...")
        # Get image entities directly from text_chunks
        image_entities = []
        
        if all_chunks:
            import re
            for chunk_id, chunk_data in all_chunks.items():
                modal_entity_name = chunk_data.get("modal_entity_name", "")
                # Check if this chunk is an image entity
                if modal_entity_name and ("(image)" in modal_entity_name or modal_entity_name.startswith("Image_")):
                    content = chunk_data.get("content", "")
                    image_path = None
                    
                    # Extract path from "Image Content (Path: /path/to/image.jpg):" format
                    if content:
                        # Pattern: "Image Content (Path: /path/to/image.jpg):"
                        path_match = re.search(r'Path:\s*([^\s\)]+\.(jpg|jpeg|png|bmp|tiff|tif))', content, re.IGNORECASE)
                        if path_match:
                            image_path = path_match.group(1).strip()
                            # Normalize image path (replace old prefix with new one)
                            image_path = normalize_image_path(image_path)
                        else:
                            # Fallback: Try to parse content as JSON
                            try:
                                import json
                                content_data = json.loads(content)
                                if isinstance(content_data, dict):
                                    image_path = content_data.get("img_path") or content_data.get("image_path", "")
                                    if image_path:
                                        image_path = normalize_image_path(image_path)
                            except (json.JSONDecodeError, TypeError):
                                pass
                    
                    # Also check chunk_data directly for img_path or image_path
                    if not image_path:
                        image_path = chunk_data.get("img_path") or chunk_data.get("image_path", "")
                        if image_path:
                            image_path = normalize_image_path(image_path)
                    
                    if image_path:
                        # Check if path exists
                        if os.path.exists(image_path):
                            file_path = chunk_data.get("file_path", "unknown")
                            image_entities.append({
                                "entity_name": modal_entity_name,
                                "image_path": image_path,
                                "chunk_id": chunk_id,
                                "file_path": file_path,
                            })
                        else:
                            # Log missing files for debugging (only first few to avoid spam)
                            if len([e for e in image_entities if e.get("entity_name") == modal_entity_name]) == 0:
                                if len([e for e in image_entities if "missing_path" in e]) < 3:
                                    print(f"  ⚠️  Image path not found: {image_path[:80]}... (entity: {modal_entity_name[:50]}...)")
                                    image_entities.append({"missing_path": True, "entity_name": modal_entity_name, "path": image_path})
        
        # Filter out missing_path entries for actual processing
        valid_image_entities = [e for e in image_entities if "missing_path" not in e]
        missing_count = len(image_entities) - len(valid_image_entities)
        
        if valid_image_entities:
            print(f"  Found {len(valid_image_entities)} image entities with valid image paths")
            if missing_count > 0:
                print(f"  ⚠️  {missing_count} image entities have missing image paths")
            
            batch_size = 8  # Smaller batch for image processing
            total_batches = (len(valid_image_entities) + batch_size - 1) // batch_size
            
            for i in range(0, len(valid_image_entities), batch_size):
                batch = valid_image_entities[i:i + batch_size]
                batch_for_vdb = {}
                
                for img_data in batch:
                    image_path = img_data["image_path"]
                    # Normalize image path before processing
                    image_path = normalize_image_path(image_path)
                    img_data["image_path"] = image_path
                    entity_name = img_data["entity_name"]
                    chunk_id = img_data.get("chunk_id", "")
                    file_path = img_data.get("file_path", "unknown")
                    
                    # Generate ID for image
                    if chunk_id:
                        img_id = f"img_virchow2_{chunk_id}"
                    else:
                        import hashlib
                        img_id = f"img_virchow2_{hashlib.md5(entity_name.encode()).hexdigest()[:16]}"
                    
                    try:
                        # Extract Virchow2 features
                        # Virchow2FeatureExtractorWrapper has a 'func' method, not callable directly
                        if hasattr(virchow2_feature_func, 'func'):
                            features_list = await virchow2_feature_func.func([image_path])
                            # func returns a list of arrays, get the first one
                            if features_list and len(features_list) > 0:
                                features = features_list[0]
                            else:
                                raise ValueError("Empty features returned from Virchow2")
                        else:
                            # Fallback: try to call directly (for backward compatibility)
                            features = await virchow2_feature_func([image_path])
                            # If it's a list, get the first element
                            if isinstance(features, list) and len(features) > 0:
                                features = features[0]
                        
                        if features is not None:
                            # Ensure features is 1D array
                            if isinstance(features, np.ndarray):
                                if features.ndim > 1:
                                    features = features.flatten()
                            elif isinstance(features, list):
                                features = np.array(features).flatten()
                            
                            batch_for_vdb[img_id] = {
                                "image_path": image_path,
                                "chunk_id": chunk_id,
                                "file_path": file_path,
                                "entity_name": entity_name,
                                "__vector__": features,  # Precomputed vector
                            }
                    except Exception as e:
                        print(f"  ⚠️  Error extracting features for {image_path}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                if batch_for_vdb:
                    try:
                        # Ensure pathology_images_vdb is initialized before upsert
                        if hasattr(rag.rag_engine.pathology_images_vdb, 'initialize'):
                            if not getattr(rag.rag_engine.pathology_images_vdb, '_initialized', False):
                                await rag.rag_engine.pathology_images_vdb.initialize()
                        
                        await rag.rag_engine.pathology_images_vdb.upsert(batch_for_vdb)
                        batch_num = i // batch_size + 1
                        print(f"  Processed batch {batch_num}/{total_batches} ({len(batch_for_vdb)} images)")
                    except Exception as e:
                        print(f"  ⚠️  Error processing image batch {batch_num}: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Ensure pathology_images_vdb is initialized before calling index_done_callback
            if hasattr(rag.rag_engine.pathology_images_vdb, 'initialize'):
                if not getattr(rag.rag_engine.pathology_images_vdb, '_initialized', False):
                    await rag.rag_engine.pathology_images_vdb.initialize()
            
            # Only call index_done_callback if client is available
            if hasattr(rag.rag_engine.pathology_images_vdb, '_client') and rag.rag_engine.pathology_images_vdb._client:
                try:
                    await rag.rag_engine.pathology_images_vdb.index_done_callback()
                    print(f"✅ Pathology image embeddings completed ({len(valid_image_entities)} images)")
                except Exception as e:
                    print(f"⚠️  Error calling index_done_callback: {e}")
                    print(f"✅ Pathology image embeddings processed ({len(valid_image_entities)} images)")
            else:
                print(f"⚠️  pathology_images_vdb client not available, skipping index_done_callback")
                print(f"✅ Pathology image embeddings processed ({len(valid_image_entities)} images)")
        else:
            if len(image_entities) > 0:
                print(f"  ⚠️  Found {len(image_entities)} image entities but no valid image paths")
                print(f"  This may happen if:")
                print(f"    - Image files are missing or paths are incorrect")
                print(f"    - Chunks don't contain modal_entity_name or image_path information")
            else:
                print("  ⚠️  No image entities found in text_chunks")
    else:
        print("\n4. Skipping Virchow2 features (not configured)")
    
    # 5. Write embeddings to PostgreSQL VDB tables if using PGHypergraphStorage
    if graph_storage == "PGHypergraphStorage":
        print(f"\n{'='*60}")
        print("Writing embeddings to PostgreSQL VDB tables...")
        print(f"{'='*60}\n")
        await write_embeddings_to_postgres_vdb(rag, EMBEDDING_DIM)
    
    # # 6. Export embeddings to JSON files
    # print(f"\n{'='*60}")
    # print("Exporting embeddings to JSON files...")
    # print(f"{'='*60}\n")
    # await export_embeddings_to_json(rag, working_dir, EMBEDDING_DIM)
    
    # print(f"\n{'='*60}")
    # print("✅ Stage 4 completed!")
    # print(f"{'='*60}")
    # print(f"Output files:")
    # print(f"  - {working_dir}/vdb_chunks.json")
    # print(f"  - {working_dir}/vdb_entities.json")
    # print(f"  - {working_dir}/vdb_relationships.json")
    # if virchow2_feature_func and rag.rag_engine.pathology_images_vdb:
    #     print(f"  - {working_dir}/vdb_pathology_images.json")


async def generate_entity_embeddings_direct_to_postgres(
    rag: PathPocket,
    embedding_func,
    embedding_dim: int,
    node_read_batch_size: int,
    embedding_gen_batch_size: int
):
    """
    Generate entity embeddings and write directly to PostgreSQL VDB_ENTITY table.
    This avoids memory explosion by processing in small batches and writing immediately.
    """
    import asyncpg
    import json
    from datetime import datetime, timezone
    from pathpocket.lightrag_utils import compute_mdhash_id
    
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "zhexu")
    password = os.getenv("POSTGRES_PASSWORD", "")
    database = os.getenv("POSTGRES_DATABASE", "pathrag")
    workspace = rag.rag_engine.workspace or "default"
    
    graph = rag.rag_engine.chunk_entity_relation_graph
    node_table_name = graph._node_table_name if hasattr(graph, '_node_table_name') else None
    
    if not node_table_name:
        print("  ⚠️  Cannot find node table name, falling back to VDB method")
        return
    
    print("  Using direct PostgreSQL write (memory-efficient)...")
    
    try:
        # Connect to PostgreSQL
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        current_time = datetime.now(timezone.utc).replace(tzinfo=None)
        
        # Ensure VDB_ENTITY table exists
        try:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS VDB_ENTITY (
                    id TEXT,
                    workspace VARCHAR(255),
                    entity_name VARCHAR(512),
                    content TEXT,
                    content_vector VECTOR({embedding_dim}),
                    chunk_ids TEXT[],
                    file_path TEXT,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT VDB_ENTITY_PK PRIMARY KEY (workspace, id)
                )
            """)
            # Create HNSW index if not exists
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_vdb_entity_hnsw_cosine 
                ON VDB_ENTITY USING HNSW (content_vector vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64);
            """)
        except Exception as e:
            print(f"  ⚠️  Error creating VDB_ENTITY table: {e}")
        
        # Get total node count for progress tracking
        total_nodes = None
        try:
            count_query = f"""
            SELECT COUNT(*) as total
            FROM {node_table_name}
            WHERE workspace = $1
            """
            if hasattr(graph, '_query'):
                count_result = await graph._query(count_query, [workspace], readonly=True)
                total_nodes = count_result[0]["total"] if count_result else 0
                print(f"  Found {total_nodes} nodes in database")
        except Exception as e:
            print(f"  ⚠️  Could not get node count: {e}")
        
        # Process nodes in batches
        offset = 0
        batch_num = 0
        total_entities_processed = 0
        processed_entity_ids = set()
        
        while True:
            batch_num += 1
            
            # Query a batch of nodes
            query = f"""
            SELECT node_id, node_data
            FROM {node_table_name}
            WHERE workspace = $1
            ORDER BY node_id
            LIMIT $2 OFFSET $3
            """
            
            try:
                if hasattr(graph, '_query'):
                    batch_result = await graph._query(
                        query, [workspace, node_read_batch_size, offset], readonly=True
                    )
                    
                    if not batch_result:
                        break  # No more nodes
                    
                    # Prepare entity data for this batch
                    entity_batch = []
                    for row in batch_result:
                        node_id = row["node_id"]
                        if node_id in processed_entity_ids:
                            continue
                        processed_entity_ids.add(node_id)
                        
                        node_data = row["node_data"]
                        if isinstance(node_data, str):
                            import json
                            try:
                                node_data = json.loads(node_data)
                            except json.JSONDecodeError:
                                continue
                        
                        if not isinstance(node_data, dict):
                            continue
                        
                        # Build entity data
                        entity_name = node_id
                        description = node_data.get("description", "")
                        
                        # Create content for embedding
                        if description:
                            content = f"{entity_name}\n{description}"
                        else:
                            content = entity_name
                        
                        # Use entity_id as key (compute hash if needed)
                        eid = compute_mdhash_id(entity_name, prefix="ent-")
                        
                        entity_batch.append({
                            "id": eid,
                            "entity_name": entity_name,
                            "source_id": node_data.get("source_id", node_id),
                            "content": content,
                            "file_path": node_data.get("file_path", "unknown"),
                        })
                    
                    # Generate embeddings in smaller batches to save memory
                    if entity_batch:
                        for i in range(0, len(entity_batch), embedding_gen_batch_size):
                            sub_batch = entity_batch[i:i + embedding_gen_batch_size]
                            
                            # Extract contents for embedding
                            contents = [item["content"] for item in sub_batch]
                            
                            # Generate embeddings (this is the memory-intensive part)
                            try:
                                embeddings = await embedding_func(contents)
                                
                                # Write to database immediately
                                for j, item in enumerate(sub_batch):
                                    try:
                                        # Convert embedding to JSON string for PostgreSQL pgvector
                                        if isinstance(embeddings, np.ndarray):
                                            vector = embeddings[j].tolist()
                                        else:
                                            vector = embeddings[j] if isinstance(embeddings[j], list) else embeddings[j].tolist()
                                        
                                        vector_json = json.dumps(vector)
                                        
                                        await conn.execute("""
                                            INSERT INTO VDB_ENTITY 
                                            (workspace, id, entity_name, content, content_vector, chunk_ids, file_path, create_time, update_time)
                                            VALUES ($1, $2, $3, $4, $5::vector, $6::text[], $7, $8, $9)
                                            ON CONFLICT (workspace, id) DO UPDATE
                                            SET entity_name=EXCLUDED.entity_name,
                                                content=EXCLUDED.content,
                                                content_vector=EXCLUDED.content_vector,
                                                chunk_ids=EXCLUDED.chunk_ids,
                                                file_path=EXCLUDED.file_path,
                                                update_time=EXCLUDED.update_time
                                        """, workspace, item["id"], item["entity_name"], item["content"], 
                                            vector_json, [item["source_id"]], item["file_path"], current_time, current_time)
                                        
                                        total_entities_processed += 1
                                    except Exception as e:
                                        print(f"  ⚠️  Error writing entity {item['id']}: {e}")
                                
                            except Exception as e:
                                print(f"  ⚠️  Error generating embeddings for batch {batch_num} sub-batch {i//embedding_gen_batch_size + 1}: {e}")
                        
                        # Progress update
                        if total_nodes:
                            print(f"  Processed batch {batch_num} ({total_entities_processed}/{total_nodes} entities)...")
                        else:
                            print(f"  Processed batch {batch_num} ({total_entities_processed} entities)...")
                    
                    # Check if we got fewer results than batch size (last batch)
                    if len(batch_result) < node_read_batch_size:
                        break
                    
                    offset += node_read_batch_size
                    
            except Exception as e:
                print(f"  ⚠️  Error querying nodes batch {batch_num}: {e}")
                break
        
        # Also process nodes from hyperedges (may have nodes not in node table)
        try:
            if hasattr(graph, 'get_all_nodes'):
                hyperedge_nodes = await graph.get_all_nodes()
                if hyperedge_nodes:
                    print(f"  Processing {len(hyperedge_nodes)} additional nodes from hyperedges...")
                    entity_batch = []
                    
                    for node in hyperedge_nodes:
                        entity_id = node.get("entity_id") or node.get("id", "")
                        if not entity_id or entity_id in processed_entity_ids:
                            continue
                        processed_entity_ids.add(entity_id)
                        
                        entity_name = entity_id
                        description = node.get("description", "")
                        
                        if description:
                            content = f"{entity_name}\n{description}"
                        else:
                            content = entity_name
                        
                        eid = compute_mdhash_id(entity_name, prefix="ent-")
                        
                        entity_batch.append({
                            "id": eid,
                            "entity_name": entity_name,
                            "source_id": node.get("source_id", entity_id),
                            "content": content,
                            "file_path": node.get("file_path", "unknown"),
                        })
                    
                    # Process hyperedge nodes in batches
                    if entity_batch:
                        for i in range(0, len(entity_batch), embedding_gen_batch_size):
                            sub_batch = entity_batch[i:i + embedding_gen_batch_size]
                            contents = [item["content"] for item in sub_batch]
                            
                            try:
                                embeddings = await embedding_func(contents)
                                
                                for j, item in enumerate(sub_batch):
                                    try:
                                        if isinstance(embeddings, np.ndarray):
                                            vector = embeddings[j].tolist()
                                        else:
                                            vector = embeddings[j] if isinstance(embeddings[j], list) else embeddings[j].tolist()
                                        
                                        vector_json = json.dumps(vector)
                                        
                                        await conn.execute("""
                                            INSERT INTO VDB_ENTITY 
                                            (workspace, id, entity_name, content, content_vector, chunk_ids, file_path, create_time, update_time)
                                            VALUES ($1, $2, $3, $4, $5::vector, $6::text[], $7, $8, $9)
                                            ON CONFLICT (workspace, id) DO UPDATE
                                            SET entity_name=EXCLUDED.entity_name,
                                                content=EXCLUDED.content,
                                                content_vector=EXCLUDED.content_vector,
                                                chunk_ids=EXCLUDED.chunk_ids,
                                                file_path=EXCLUDED.file_path,
                                                update_time=EXCLUDED.update_time
                                        """, workspace, item["id"], item["entity_name"], item["content"], 
                                            vector_json, [item["source_id"]], item["file_path"], current_time, current_time)
                                        
                                        total_entities_processed += 1
                                    except Exception as e:
                                        print(f"  ⚠️  Error writing hyperedge entity {item['id']}: {e}")
                            except Exception as e:
                                print(f"  ⚠️  Error generating embeddings for hyperedge batch: {e}")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not process hyperedge nodes: {e}")
        
        await conn.close()
        print(f"✅ Entity embeddings completed ({total_entities_processed} entities written directly to PostgreSQL)")
        
    except Exception as e:
        print(f"  ❌ Error in direct PostgreSQL write: {e}")
        import traceback
        traceback.print_exc()


def decode_vector(vector_data: Any) -> Optional[List[float]]:
    """
    Decode vector from NanoVectorDB storage format.
    Priority: __vector__ (numpy array) > vector (base64 encoded compressed)
    Returns: List[float] or None
    """
    if vector_data is None:
        return None
    
    # If it's already a numpy array, convert to float32 list
    if isinstance(vector_data, np.ndarray):
        return vector_data.astype(np.float32).tolist()
    
    # If it's already a list, ensure it's float and return
    if isinstance(vector_data, list):
        # Convert to numpy array first to ensure proper type conversion
        try:
            vec_array = np.array(vector_data, dtype=np.float32)
            return vec_array.tolist()
        except (ValueError, TypeError):
            return None
    
    # If it's a string (base64 encoded), decode it
    if isinstance(vector_data, str):
        try:
            # Decode base64
            compressed_vector = base64.b64decode(vector_data)
            # Decompress zlib
            vector_bytes = zlib.decompress(compressed_vector)
            # Convert bytes to numpy array (float16)
            vector_array = np.frombuffer(vector_bytes, dtype=np.float16)
            # Convert to float32 list for PostgreSQL
            return vector_array.astype(np.float32).tolist()
        except Exception as e:
            print(f"  ⚠️  Error decoding vector: {e}")
            return None
    
    return None


async def write_embeddings_to_postgres_vdb(rag: PathPocket, embedding_dim: int):
    """Write embeddings to PostgreSQL VDB tables when using PGHypergraphStorage."""
    import asyncpg
    from datetime import datetime, timezone
    
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "zhexu")
    password = os.getenv("POSTGRES_PASSWORD", "")
    database = os.getenv("POSTGRES_DATABASE", "pathrag")
    workspace = rag.rag_engine.workspace or "default"
    
    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        current_time = datetime.now(timezone.utc).replace(tzinfo=None)
        
        # Create tables if they don't exist
        print("Creating VDB tables if they don't exist...")
        try:
            # Create VDB_CHUNKS table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS VDB_CHUNKS (
                    id TEXT,
                    workspace VARCHAR(255),
                    tokens INTEGER,
                    chunk_order_index INTEGER,
                    full_doc_id TEXT,
                    content TEXT,
                    content_vector VECTOR({embedding_dim}),
                    file_path TEXT,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT VDB_CHUNKS_PK PRIMARY KEY (workspace, id)
                )
            """)
            # Alter existing table if needed (for id and full_doc_id)
            try:
                await conn.execute("""
                    ALTER TABLE VDB_CHUNKS 
                    ALTER COLUMN id TYPE TEXT,
                    ALTER COLUMN full_doc_id TYPE TEXT
                """)
            except Exception:
                pass  # Column might not exist or already correct type
            # Create HNSW index for chunks
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_vdb_chunks_hnsw_cosine ON VDB_CHUNKS USING HNSW (content_vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);
            """)
            
            # Create VDB_ENTITY table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS VDB_ENTITY (
                    id TEXT,
                    workspace VARCHAR(255),
                    entity_name VARCHAR(512),
                    content TEXT,
                    content_vector VECTOR({embedding_dim}),
                    chunk_ids TEXT[],
                    file_path TEXT,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT VDB_ENTITY_PK PRIMARY KEY (workspace, id)
                )
            """)
            # Alter existing table if needed (for id and chunk_ids)
            try:
                await conn.execute("""
                    ALTER TABLE VDB_ENTITY 
                    ALTER COLUMN id TYPE TEXT,
                    ALTER COLUMN chunk_ids TYPE TEXT[]
                """)
            except Exception:
                pass  # Column might not exist or already correct type
            # Create HNSW index for entities
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_vdb_entity_hnsw_cosine ON VDB_ENTITY USING HNSW (content_vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);
            """)
            
            # Create VDB_RELATION table
            # Multi-entity relations only: Use entities array
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS VDB_RELATION (
                    id TEXT,
                    workspace VARCHAR(255),
                    entities TEXT[],
                    entity_count INTEGER,
                    relation TEXT,
                    keywords TEXT,
                    description TEXT,
                    content TEXT,
                    content_vector VECTOR({embedding_dim}),
                    chunk_ids TEXT[],
                    file_path TEXT,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT VDB_RELATION_PK PRIMARY KEY (workspace, id)
                )
            """)
            # Alter existing table if needed (remove source_id/target_id, add entities and entity_count columns)
            try:
                await conn.execute("""
                    ALTER TABLE VDB_RELATION 
                    ALTER COLUMN id TYPE TEXT,
                    ALTER COLUMN chunk_ids TYPE TEXT[]
                """)
                # Remove old columns (source_id, target_id) if they exist
                await conn.execute("""
                    DO $$ 
                    BEGIN
                        IF EXISTS (SELECT 1 FROM information_schema.columns 
                                  WHERE table_name='vdb_relation' AND column_name='source_id') THEN
                            ALTER TABLE VDB_RELATION DROP COLUMN source_id;
                        END IF;
                        IF EXISTS (SELECT 1 FROM information_schema.columns 
                                  WHERE table_name='vdb_relation' AND column_name='target_id') THEN
                            ALTER TABLE VDB_RELATION DROP COLUMN target_id;
                        END IF;
                    END $$;
                """)
                # Add new columns if they don't exist
                await conn.execute("""
                    DO $$ 
                    BEGIN
                        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                      WHERE table_name='vdb_relation' AND column_name='entities') THEN
                            ALTER TABLE VDB_RELATION ADD COLUMN entities TEXT[];
                        END IF;
                        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                      WHERE table_name='vdb_relation' AND column_name='entity_count') THEN
                            ALTER TABLE VDB_RELATION ADD COLUMN entity_count INTEGER;
                        END IF;
                        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                      WHERE table_name='vdb_relation' AND column_name='relation') THEN
                            ALTER TABLE VDB_RELATION ADD COLUMN relation TEXT;
                        END IF;
                        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                      WHERE table_name='vdb_relation' AND column_name='keywords') THEN
                            ALTER TABLE VDB_RELATION ADD COLUMN keywords TEXT;
                        END IF;
                        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                      WHERE table_name='vdb_relation' AND column_name='description') THEN
                            ALTER TABLE VDB_RELATION ADD COLUMN description TEXT;
                        END IF;
                    END $$;
                """)
            except Exception as e:
                print(f"  ⚠️  Warning: Error altering VDB_RELATION table: {e}")
                pass  # Column might not exist or already correct type
            # Create GIN index for entities array for fast queries
            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_vdb_relation_entities_gin 
                    ON VDB_RELATION USING GIN (entities);
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_vdb_relation_entity_count 
                    ON VDB_RELATION (entity_count);
                """)
            except Exception:
                pass  # Index might already exist
            # Create HNSW index for relations
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_vdb_relation_hnsw_cosine ON VDB_RELATION USING HNSW (content_vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);
            """)
            
            print("✅ VDB tables created/verified")
        except Exception as e:
            print(f"  ⚠️  Error creating VDB tables: {e}")
        
        # 1. Write chunks to VDB_CHUNKS
        print("1. Writing chunks to VDB_CHUNKS...")
        if hasattr(rag.rag_engine.chunks_vdb, '_client'):
            # Get all chunks from NanoVectorDB
            client = await rag.rag_engine.chunks_vdb._get_client() if hasattr(rag.rag_engine.chunks_vdb, '_get_client') else rag.rag_engine.chunks_vdb._client
            if client and hasattr(client, '_NanoVectorDB__storage'):
                storage = getattr(client, '_NanoVectorDB__storage')
                if storage and "data" in storage:
                    chunks_data = storage["data"]
                    if chunks_data:
                        # Get chunks metadata from text_chunks
                        all_chunks = {}
                        if hasattr(rag.rag_engine.text_chunks, '_data'):
                            data_dict = rag.rag_engine.text_chunks._data
                            if hasattr(data_dict, '_getvalue'):
                                all_chunks = dict(data_dict._getvalue())
                            else:
                                all_chunks = dict(data_dict)
                        
                        batch_size = 100
                        total = len(chunks_data)
                        for i in range(0, total, batch_size):
                            batch = chunks_data[i:i + batch_size]
                            for item in batch:
                                item_id = item.get("__id__") or item.get("id", "")
                                if not item_id:
                                    continue
                                
                                # Get vector (prefer __vector__, fallback to vector)
                                vector_raw = item.get("__vector__") or item.get("vector")
                                vector = decode_vector(vector_raw)
                                if vector is None:
                                    continue
                                
                                content = item.get("content", "")
                                chunk_meta = all_chunks.get(item_id, {})
                                full_doc_id = chunk_meta.get("full_doc_id", item_id.split("-")[0] if "-" in item_id else item_id)
                                chunk_order_index = chunk_meta.get("chunk_order_index", 0)
                                tokens = chunk_meta.get("tokens", len(content.split()))
                                file_path = chunk_meta.get("file_path", item.get("file_path", "unknown"))
                                
                                try:
                                    # Convert vector to JSON string format for PostgreSQL pgvector
                                    vector_json = json.dumps(vector)
                                    await conn.execute("""
                                        INSERT INTO VDB_CHUNKS 
                                        (workspace, id, tokens, chunk_order_index, full_doc_id, content, content_vector, file_path, create_time, update_time)
                                        VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, $9, $10)
                                        ON CONFLICT (workspace, id) DO UPDATE
                                        SET tokens=EXCLUDED.tokens,
                                            chunk_order_index=EXCLUDED.chunk_order_index,
                                            full_doc_id=EXCLUDED.full_doc_id,
                                            content=EXCLUDED.content,
                                            content_vector=EXCLUDED.content_vector,
                                            file_path=EXCLUDED.file_path,
                                            update_time=EXCLUDED.update_time
                                    """, workspace, item_id, tokens, chunk_order_index, full_doc_id, content, vector_json, file_path, current_time, current_time)
                                except Exception as e:
                                    print(f"  ⚠️  Error writing chunk {item_id}: {e}")
                            
                            if i + batch_size < total:
                                print(f"  Processed {min(i + batch_size, total)}/{total} chunks...")
                        
                        print(f"✅ Wrote {total} chunks to VDB_CHUNKS")
        
        # 2. Write entities to VDB_ENTITY
        print("\n2. Writing entities to VDB_ENTITY...")
        if hasattr(rag.rag_engine.entities_vdb, '_client'):
            client = await rag.rag_engine.entities_vdb._get_client() if hasattr(rag.rag_engine.entities_vdb, '_get_client') else rag.rag_engine.entities_vdb._client
            if client and hasattr(client, '_NanoVectorDB__storage'):
                storage = getattr(client, '_NanoVectorDB__storage')
                if storage and "data" in storage:
                    entities_data = storage["data"]
                    if entities_data:
                        batch_size = 100
                        total = len(entities_data)
                        for i in range(0, total, batch_size):
                            batch = entities_data[i:i + batch_size]
                            for item in batch:
                                item_id = item.get("__id__") or item.get("id", "")
                                if not item_id:
                                    continue
                                
                                # Get vector (prefer __vector__, fallback to vector)
                                vector_raw = item.get("__vector__") or item.get("vector")
                                vector = decode_vector(vector_raw)
                                if vector is None:
                                    continue
                                
                                entity_name = item.get("entity_name", item_id)
                                content = item.get("content", entity_name)
                                file_path = item.get("file_path", "unknown")
                                source_id = item.get("source_id", item_id)
                                
                                try:
                                    # Convert vector to JSON string format for PostgreSQL pgvector
                                    vector_json = json.dumps(vector)
                                    await conn.execute("""
                                        INSERT INTO VDB_ENTITY 
                                        (workspace, id, entity_name, content, content_vector, chunk_ids, file_path, create_time, update_time)
                                        VALUES ($1, $2, $3, $4, $5::vector, $6::text[], $7, $8, $9)
                                        ON CONFLICT (workspace, id) DO UPDATE
                                        SET entity_name=EXCLUDED.entity_name,
                                            content=EXCLUDED.content,
                                            content_vector=EXCLUDED.content_vector,
                                            chunk_ids=EXCLUDED.chunk_ids,
                                            file_path=EXCLUDED.file_path,
                                            update_time=EXCLUDED.update_time
                                    """, workspace, item_id, entity_name, content, vector_json, [source_id], file_path, current_time, current_time)
                                except Exception as e:
                                    print(f"  ⚠️  Error writing entity {item_id}: {e}")
                            
                            if i + batch_size < total:
                                print(f"  Processed {min(i + batch_size, total)}/{total} entities...")
                        
                        print(f"✅ Wrote {total} entities to VDB_ENTITY")
        
        # 3. Write relationships to VDB_RELATION
        print("\n3. Writing relationships to VDB_RELATION...")
        written_count = 0
        skipped_no_id = 0
        skipped_no_vector = 0
        skipped_no_entities = 0
        error_count = 0
        
        if hasattr(rag.rag_engine.relationships_vdb, '_client'):
            client = await rag.rag_engine.relationships_vdb._get_client() if hasattr(rag.rag_engine.relationships_vdb, '_get_client') else rag.rag_engine.relationships_vdb._client
            if client and hasattr(client, '_NanoVectorDB__storage'):
                storage = getattr(client, '_NanoVectorDB__storage')
                if storage and "data" in storage:
                    relations_data = storage["data"]
                    if relations_data:
                        batch_size = 100
                        total = len(relations_data)
                        print(f"  Found {total} relations in relationships_vdb")
                        
                        # Debug: show first item structure
                        if total > 0:
                            first_item = relations_data[0]
                            print(f"  Sample relation keys: {list(first_item.keys())[:10]}")
                            if "entities" in first_item:
                                print(f"  Sample entities: {first_item.get('entities', [])[:5]}")
                            elif "content" in first_item:
                                content_sample = first_item.get('content', '')
                                print(f"  Sample content (first 200 chars): {content_sample[:200] if content_sample else 'N/A'}")
                        
                        for i in range(0, total, batch_size):
                            batch = relations_data[i:i + batch_size]
                            for item in batch:
                                item_id = item.get("__id__") or item.get("id", "")
                                if not item_id:
                                    skipped_no_id += 1
                                    continue
                                
                                # Get vector (prefer __vector__, fallback to vector)
                                vector_raw = item.get("__vector__") or item.get("vector")
                                vector = decode_vector(vector_raw)
                                if vector is None:
                                    skipped_no_vector += 1
                                    continue
                                
                                # Get entities array - try multiple sources
                                entities = item.get("entities", [])
                                
                                # If no entities array, try to construct from various sources
                                if not entities or len(entities) < 2:
                                    source_id = item.get("source_id", "")
                                    target_id = item.get("target_id", "")
                                    content = item.get("content", "")
                                    
                                    # Method 1: Try to parse entities from content field
                                    # Format: "keywords\tentities\ndescription" where entities are separated by GRAPH_FIELD_SEP
                                    if content and not entities:
                                        try:
                                            from pathpocket.operate import GRAPH_FIELD_SEP
                                            parts = content.split("\t", 1)
                                            if len(parts) >= 2:
                                                entities_part = parts[1].split("\n")[0]  # Get entities before newline
                                                entities = [e.strip() for e in entities_part.split(GRAPH_FIELD_SEP) if e.strip()]
                                        except Exception:
                                            pass
                                    
                                    # Method 2: Try to extract entities from source_id/target_id
                                    if not entities or len(entities) < 2:
                                        if source_id and target_id:
                                            # If source_id and target_id are entity names, use them
                                            entities = [source_id, target_id]
                                        elif source_id:
                                            # Try to parse entities from source_id (might be separated)
                                            from pathpocket.operate import GRAPH_FIELD_SEP
                                            entities = [e.strip() for e in source_id.split(GRAPH_FIELD_SEP) if e.strip()]
                                    
                                    # Method 3: Try to get entities from graph storage using chunk_id
                                    if not entities or len(entities) < 2:
                                        if source_id and hasattr(rag.rag_engine, 'chunk_entity_relation_graph'):
                                            try:
                                                # source_id might be a chunk_id, try to find entities in graph
                                                # Look for edges containing this chunk_id
                                                graph = rag.rag_engine.chunk_entity_relation_graph
                                                if hasattr(graph, 'get_node_edges'):
                                                    # Get edges for this node (if source_id is an entity)
                                                    edges = await graph.get_node_edges(source_id)
                                                    if edges:
                                                        # Extract unique entities from edges
                                                        entity_set = set()
                                                        for src, tgt in edges:
                                                            entity_set.add(src)
                                                            entity_set.add(tgt)
                                                        if len(entity_set) >= 2:
                                                            entities = list(entity_set)[:10]  # Limit to 10 entities
                                            except Exception as e:
                                                if skipped_no_entities <= 3:
                                                    print(f"  ⚠️  Error querying graph for {source_id}: {e}")
                                    
                                    # If still no valid entities, skip
                                    if not entities or len(entities) < 2:
                                        skipped_no_entities += 1
                                        if skipped_no_entities <= 3:  # Show first 3 examples
                                            print(f"  ⚠️  Skipping relation {item_id}: no valid entities (source_id={source_id[:50] if source_id else 'N/A'}, content_len={len(content) if content else 0})")
                                        continue
                                
                                entity_count = len(entities)
                                content = item.get("content", "")
                                relation = item.get("relation", "")
                                keywords = item.get("keywords", "")
                                description = item.get("description", "")
                                file_path = item.get("file_path", "unknown")
                                
                                # Get chunk_ids directly from item (preferred) or parse from source_id as fallback
                                chunk_ids = item.get("chunk_ids", [])
                                if not chunk_ids:
                                    # Fallback: try to parse from source_id if chunk_ids not available
                                    source_id = item.get("source_id", "")
                                    if source_id:
                                        from pathpocket.operate import GRAPH_FIELD_SEP
                                        chunk_ids = [cid.strip() for cid in source_id.split(GRAPH_FIELD_SEP) if cid.strip()]
                                # Ensure chunk_ids is a list
                                if not isinstance(chunk_ids, list):
                                    chunk_ids = []
                                
                                try:
                                    # Convert vector to JSON string format for PostgreSQL pgvector
                                    vector_json = json.dumps(vector)
                                    await conn.execute("""
                                        INSERT INTO VDB_RELATION 
                                        (workspace, id, entities, entity_count, relation, keywords, description, content, content_vector, chunk_ids, file_path, create_time, update_time)
                                        VALUES ($1, $2, $3::text[], $4, $5, $6, $7, $8, $9::vector, $10::text[], $11, $12, $13)
                                        ON CONFLICT (workspace, id) DO UPDATE
                                        SET entities=EXCLUDED.entities,
                                            entity_count=EXCLUDED.entity_count,
                                            relation=EXCLUDED.relation,
                                            keywords=EXCLUDED.keywords,
                                            description=EXCLUDED.description,
                                            content=EXCLUDED.content,
                                            content_vector=EXCLUDED.content_vector,
                                            chunk_ids=EXCLUDED.chunk_ids,
                                            file_path=EXCLUDED.file_path,
                                            update_time=EXCLUDED.update_time
                                    """, workspace, item_id, entities, entity_count, relation, keywords, description, content, vector_json, chunk_ids, file_path, current_time, current_time)
                                    written_count += 1
                                except Exception as e:
                                    error_count += 1
                                    if error_count <= 5:  # Show first 5 errors
                                        print(f"  ⚠️  Error writing relation {item_id}: {e}")
                            
                            if i + batch_size < total:
                                print(f"  Processed {min(i + batch_size, total)}/{total} relations (written: {written_count}, skipped: {skipped_no_id + skipped_no_vector + skipped_no_entities}, errors: {error_count})...")
                        
                        print(f"✅ Relations processing complete:")
                        print(f"   Total in storage: {total}")
                        print(f"   Written: {written_count}")
                        print(f"   Skipped (no ID): {skipped_no_id}")
                        print(f"   Skipped (no vector): {skipped_no_vector}")
                        print(f"   Skipped (no entities): {skipped_no_entities}")
                        print(f"   Errors: {error_count}")
                    else:
                        print("  ⚠️  No relations data found in relationships_vdb storage")
                else:
                    print("  ⚠️  relationships_vdb storage structure not found")
            else:
                print("  ⚠️  relationships_vdb client not found")
        else:
            print("  ⚠️  relationships_vdb not available")
        
        # 4. Create and write images to VDB_IMAGES (if pathology_images_vdb exists)
        print("\n4. Writing images to VDB_IMAGES...")
        if rag.rag_engine.pathology_images_vdb:
            # Ensure pathology_images_vdb is initialized
            try:
                if hasattr(rag.rag_engine.pathology_images_vdb, 'initialize'):
                    if not getattr(rag.rag_engine.pathology_images_vdb, '_initialized', False):
                        await rag.rag_engine.pathology_images_vdb.initialize()
            except Exception as e:
                print(f"  ⚠️  Error initializing pathology_images_vdb: {e}")
            
            # First, create table if it doesn't exist
            # Virchow2 embedding dimension is 2560
            virchow2_dim = 2560
            try:
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS VDB_IMAGES (
                        id TEXT,
                        workspace VARCHAR(255),
                        image_path TEXT,
                        entity_name VARCHAR(512),
                        chunk_id TEXT,
                        content TEXT,
                        content_vector VECTOR({virchow2_dim}),
                        file_path TEXT NULL,
                        create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT VDB_IMAGES_PK PRIMARY KEY (workspace, id)
                    )
                """)
                # Alter existing table if needed (for id and chunk_id)
                try:
                    await conn.execute("""
                        ALTER TABLE VDB_IMAGES 
                        ALTER COLUMN id TYPE TEXT,
                        ALTER COLUMN chunk_id TYPE TEXT
                    """)
                except Exception:
                    pass  # Column might not exist or already correct type
                print(f"  ✅ Table VDB_IMAGES created/verified (vector dimension: {virchow2_dim})")
            except Exception as e:
                print(f"  ⚠️  Error creating VDB_IMAGES table: {e}")
            
            # Get all images from NanoVectorDB
            # Try multiple methods to access the data
            images_data = None
            
            # Method 1: Try using get_all() method if available (most reliable)
            if hasattr(rag.rag_engine.pathology_images_vdb, 'get_all'):
                try:
                    all_items = await rag.rag_engine.pathology_images_vdb.get_all()
                    if all_items:
                        # If get_all returns a dict, convert to list
                        if isinstance(all_items, dict):
                            images_data = []
                            for item_id, item_data in all_items.items():
                                # Get full data by ID to include vector
                                if hasattr(rag.rag_engine.pathology_images_vdb, 'get_by_id'):
                                    full_item = await rag.rag_engine.pathology_images_vdb.get_by_id(item_id)
                                    if full_item:
                                        images_data.append(full_item)
                                else:
                                    # If no get_by_id, use the item_data directly
                                    item_data['__id__'] = item_id
                                    images_data.append(item_data)
                        elif isinstance(all_items, list):
                            images_data = all_items
                except Exception as e:
                    print(f"  ⚠️  Error using get_all() method: {e}")
            
            # Method 2: Try to access via _NanoVectorDB__storage attribute
            if not images_data:
                client = None
                if hasattr(rag.rag_engine.pathology_images_vdb, '_get_client'):
                    try:
                        client = await rag.rag_engine.pathology_images_vdb._get_client()
                    except Exception as e:
                        print(f"  ⚠️  Error getting client via _get_client(): {e}")
                elif hasattr(rag.rag_engine.pathology_images_vdb, '_client'):
                    client = rag.rag_engine.pathology_images_vdb._client
                
                if client:
                    # Try different attribute names for storage
                    storage_attrs = ['_NanoVectorDB__storage', '_storage', 'storage', '__storage__']
                    for attr in storage_attrs:
                        if hasattr(client, attr):
                            try:
                                storage = getattr(client, attr)
                                if storage and isinstance(storage, dict) and "data" in storage:
                                    images_data = storage["data"]
                                    if images_data:
                                        break
                            except Exception:
                                continue
                
                # Method 3: Try to read from JSON file directly (NanoVectorDB uses JSON storage)
                if not images_data and hasattr(rag.rag_engine.pathology_images_vdb, '_vdb_file'):
                    try:
                        vdb_file = rag.rag_engine.pathology_images_vdb._vdb_file
                        if os.path.exists(vdb_file):
                            with open(vdb_file, 'r', encoding='utf-8') as f:
                                vdb_data = json.load(f)
                                # Handle different JSON formats
                                if isinstance(vdb_data, dict):
                                    # Try "data" key first
                                    if "data" in vdb_data:
                                        images_data = vdb_data["data"]
                                    # Try "matrix" key (NanoVectorDB format)
                                    elif "matrix" in vdb_data and isinstance(vdb_data.get("matrix"), dict):
                                        # NanoVectorDB stores data in matrix format, need to extract
                                        matrix = vdb_data["matrix"]
                                        if "data" in matrix:
                                            images_data = matrix["data"]
                                elif isinstance(vdb_data, list):
                                    # If it's a list directly, use it
                                    images_data = vdb_data
                    except Exception as e:
                        print(f"  ⚠️  Error reading from JSON file: {e}")
            
            if images_data:
                batch_size = 50  # Smaller batch for images
                total = len(images_data)
                for i in range(0, total, batch_size):
                    batch = images_data[i:i + batch_size]
                    for item in batch:
                        item_id = item.get("__id__") or item.get("id", "")
                        if not item_id:
                            continue
                        
                        # Get vector (prefer __vector__, fallback to vector)
                        vector_raw = item.get("__vector__") or item.get("vector")
                        vector = decode_vector(vector_raw)
                        if vector is None:
                            continue
                        
                        # Check vector dimension
                        if len(vector) != virchow2_dim:
                            print(f"  ⚠️  Warning: Image {item_id} vector dimension is {len(vector)}, expected {virchow2_dim}")
                            continue
                        
                        image_path = item.get("image_path", "")
                        # Normalize image path
                        if image_path:
                            image_path = normalize_image_path(image_path)
                        entity_name = item.get("entity_name", "")
                        chunk_id = item.get("chunk_id", "")
                        file_path = item.get("file_path", "unknown")
                        content = item.get("content", f"Image: {image_path}")
                        
                        try:
                            # Convert vector to JSON string format for PostgreSQL pgvector
                            vector_json = json.dumps(vector)
                            await conn.execute("""
                                INSERT INTO VDB_IMAGES 
                                (workspace, id, image_path, entity_name, chunk_id, content, content_vector, file_path, create_time, update_time)
                                VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, $9, $10)
                                ON CONFLICT (workspace, id) DO UPDATE
                                SET image_path=EXCLUDED.image_path,
                                    entity_name=EXCLUDED.entity_name,
                                    chunk_id=EXCLUDED.chunk_id,
                                    content=EXCLUDED.content,
                                    content_vector=EXCLUDED.content_vector,
                                    file_path=EXCLUDED.file_path,
                                    update_time=EXCLUDED.update_time
                            """, workspace, item_id, image_path, entity_name, chunk_id, content, vector_json, file_path, current_time, current_time)
                        except Exception as e:
                            print(f"  ⚠️  Error writing image {item_id}: {e}")
                    
                    if i + batch_size < total:
                        print(f"  Processed {min(i + batch_size, total)}/{total} images...")
                
                print(f"✅ Wrote {total} images to VDB_IMAGES")
            else:
                if client is None:
                    print("  ⚠️  pathology_images_vdb client is None (may not be initialized or no images processed)")
                else:
                    print(f"  ⚠️  Could not access pathology_images_vdb data (tried get_all(), storage attributes, and JSON file)")
        else:
            print("  ⚠️  pathology_images_vdb not initialized, skipping image storage")
        
        await conn.close()
        print("\n✅ PostgreSQL VDB tables updated successfully")
        
    except Exception as e:
        print(f"❌ Error writing to PostgreSQL VDB tables: {e}")
        import traceback
        traceback.print_exc()


async def export_embeddings_to_json(rag: PathPocket, working_dir: str, embedding_dim: int):
    """Export vector database embeddings to JSON files in the reference format."""
    import os
    
    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)
    
    # Helper function to get all data from vector storage
    async def get_all_vdb_data(vdb_storage, storage_name: str) -> List[Dict[str, Any]]:
        """Get all data from vector storage."""
        all_data = []
        try:
            # Method 1: Try to get all data using get_all or similar method
            if hasattr(vdb_storage, 'get_all'):
                all_items = await vdb_storage.get_all()
                if all_items:
                    # If get_all returns a dict, convert to list
                    if isinstance(all_items, dict):
                        for item_id, item_data in all_items.items():
                            # Get full data by ID to include vector
                            full_item = await vdb_storage.get_by_id(item_id)
                            if full_item:
                                all_data.append(full_item)
                    elif isinstance(all_items, list):
                        all_data = all_items
            # Method 2: For NanoVectorDBStorage, access internal storage
            elif hasattr(vdb_storage, 'client_storage'):
                storage = await vdb_storage.client_storage
                if storage and "data" in storage:
                    all_data = storage["data"]
            # Method 3: Try to get data from internal storage directly
            elif hasattr(vdb_storage, '_client'):
                client = await vdb_storage._get_client() if hasattr(vdb_storage, '_get_client') else vdb_storage._client
                if client:
                    # Try to get all IDs first
                    if hasattr(client, '__len__'):
                        # Get all items by querying with empty query or using internal storage
                        if hasattr(client, '_NanoVectorDB__storage'):
                            storage = getattr(client, '_NanoVectorDB__storage')
                            if storage and "data" in storage:
                                all_data = storage["data"]
                    elif hasattr(client, 'get_all'):
                        all_data = await client.get_all() if asyncio.iscoroutinefunction(client.get_all) else client.get_all()
        except Exception as e:
            print(f"  ⚠️  Warning: Could not get all data from {storage_name}: {e}")
            # Try alternative: get by IDs if we can get the IDs
            try:
                # For NanoVectorDBStorage, try to get all IDs from internal storage
                if hasattr(vdb_storage, 'client_storage'):
                    storage = await vdb_storage.client_storage
                    if storage and "data" in storage:
                        all_ids = [item.get("__id__") or item.get("id", "") for item in storage["data"] if item.get("__id__") or item.get("id")]
                        # Get items in batches
                        batch_size = 100
                        for i in range(0, len(all_ids), batch_size):
                            batch_ids = all_ids[i:i + batch_size]
                            if hasattr(vdb_storage, 'get_by_ids'):
                                batch_items = await vdb_storage.get_by_ids(batch_ids)
                                all_data.extend(batch_items)
                            else:
                                for item_id in batch_ids:
                                    item = await vdb_storage.get_by_id(item_id)
                                    if item:
                                        all_data.append(item)
                elif hasattr(vdb_storage, 'get_all_ids') or hasattr(vdb_storage, '_get_all_ids'):
                    get_ids_method = getattr(vdb_storage, 'get_all_ids', None) or getattr(vdb_storage, '_get_all_ids', None)
                    if get_ids_method:
                        all_ids = await get_ids_method() if asyncio.iscoroutinefunction(get_ids_method) else get_ids_method()
                        # Get items in batches
                        batch_size = 100
                        for i in range(0, len(all_ids), batch_size):
                            batch_ids = all_ids[i:i + batch_size]
                            if hasattr(vdb_storage, 'get_by_ids'):
                                batch_items = await vdb_storage.get_by_ids(batch_ids)
                                all_data.extend(batch_items)
                            else:
                                for item_id in batch_ids:
                                    item = await vdb_storage.get_by_id(item_id)
                                    if item:
                                        all_data.append(item)
            except Exception as e2:
                print(f"  ⚠️  Warning: Alternative method also failed for {storage_name}: {e2}")
        
        return all_data
    
    # Helper function to convert vector storage data to JSON format
    def convert_to_json_format(data_list: List[Dict[str, Any]], embedding_dim: int) -> Dict[str, Any]:
        """Convert vector storage data to reference JSON format."""
        json_data = []
        for item in data_list:
            # Extract vector (may be in different formats)
            vector = None
            if "vector" in item:
                vector = item["vector"]
            elif "__vector__" in item:
                # Convert numpy array to base64 encoded compressed format
                vec_array = item["__vector__"]
                if isinstance(vec_array, np.ndarray):
                    # Compress vector using Float16 + zlib + Base64
                    vector_f16 = vec_array.astype(np.float16)
                    compressed_vector = zlib.compress(vector_f16.tobytes())
                    vector = base64.b64encode(compressed_vector).decode("utf-8")
                else:
                    vector = vec_array
            
            # Build JSON entry
            json_entry = {
                "__id__": item.get("__id__") or item.get("id", ""),
                "__created_at__": item.get("__created_at__") or item.get("create_time", 0),
            }
            
            # Add content and metadata fields
            for key in ["content", "entity_name", "image_path", "chunk_id", "source_id", "file_path", 
                       "full_doc_id", "entities", "entity_count", "relation", "keywords", "description"]:
                if key in item:
                    value = item[key]
                    # Handle entities: ensure it's a list, not a string
                    if key == "entities":
                        if isinstance(value, str):
                            # If it's a string, split by <SEP>
                            from pathpocket.operate import GRAPH_FIELD_SEP
                            json_entry[key] = [e.strip() for e in value.split(GRAPH_FIELD_SEP) if e.strip()]
                        elif isinstance(value, list):
                            json_entry[key] = value
                        else:
                            # Skip if not a valid format
                            continue
                    else:
                        json_entry[key] = value
            
            # Add vector if available
            if vector:
                json_entry["vector"] = vector
            
            json_data.append(json_entry)
        
        return {
            "embedding_dim": embedding_dim,
            "data": json_data
        }
    
    # 1. Export vdb_chunks.json
    print("Exporting vdb_chunks.json...")
    try:
        if rag.rag_engine.chunks_vdb:
            chunks_data = await get_all_vdb_data(rag.rag_engine.chunks_vdb, "chunks_vdb")
            if chunks_data:
                json_output = convert_to_json_format(chunks_data, embedding_dim)
                output_file = os.path.join(working_dir, "vdb_chunks.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, ensure_ascii=False, indent=2)
                print(f"  ✅ Exported {len(json_output['data'])} chunks to {output_file}")
            else:
                print(f"  ⚠️  No chunks data found")
        else:
            print(f"  ⚠️  chunks_vdb not available")
    except Exception as e:
        print(f"  ❌ Error exporting chunks: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Export vdb_entities.json
    print("\nExporting vdb_entities.json...")
    try:
        if rag.rag_engine.entities_vdb:
            entities_data = await get_all_vdb_data(rag.rag_engine.entities_vdb, "entities_vdb")
            if entities_data:
                json_output = convert_to_json_format(entities_data, embedding_dim)
                output_file = os.path.join(working_dir, "vdb_entities.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, ensure_ascii=False, indent=2)
                print(f"  ✅ Exported {len(json_output['data'])} entities to {output_file}")
            else:
                print(f"  ⚠️  No entities data found")
        else:
            print(f"  ⚠️  entities_vdb not available")
    except Exception as e:
        print(f"  ❌ Error exporting entities: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Export vdb_relationships.json
    print("\nExporting vdb_relationships.json...")
    try:
        if rag.rag_engine.relationships_vdb:
            relationships_data = await get_all_vdb_data(rag.rag_engine.relationships_vdb, "relationships_vdb")
            if relationships_data:
                json_output = convert_to_json_format(relationships_data, embedding_dim)
                output_file = os.path.join(working_dir, "vdb_relationships.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, ensure_ascii=False, indent=2)
                print(f"  ✅ Exported {len(json_output['data'])} relationships to {output_file}")
            else:
                print(f"  ⚠️  No relationships data found")
        else:
            print(f"  ⚠️  relationships_vdb not available")
    except Exception as e:
        print(f"  ❌ Error exporting relationships: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Export vdb_pathology_images.json (if available)
    print("\nExporting vdb_pathology_images.json...")
    try:
        if rag.rag_engine.pathology_images_vdb:
            images_data = await get_all_vdb_data(rag.rag_engine.pathology_images_vdb, "pathology_images_vdb")
            if images_data:
                # Virchow2 features have different dimension (2560)
                virchow2_dim = 2560
                json_output = convert_to_json_format(images_data, virchow2_dim)
                output_file = os.path.join(working_dir, "vdb_pathology_images.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, ensure_ascii=False, indent=2)
                print(f"  ✅ Exported {len(json_output['data'])} images to {output_file}")
            else:
                print(f"  ⚠️  No pathology images data found")
        else:
            print(f"  ⚠️  pathology_images_vdb not available")
    except Exception as e:
        print(f"  ⚠️  Error exporting pathology images (may not be configured): {e}")


if __name__ == "__main__":
    asyncio.run(main())

