"""
Stage 4 (Streaming): Generate embeddings for entities using JSON file as intermediate storage
This script avoids memory explosion by:
1. First exporting node data to JSON file (batch processing)
2. Then loading JSON file and generating embeddings (batch processing)

Usage:
    # Option 1: Export all nodes to JSON
    python stage4_embed_entities_streaming.py --export
    
    # Step 2: Generate embeddings from JSON
    python stage4_embed_entities_streaming.py --embed
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
import argparse
from typing import List, Dict, Any, Optional
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

from pathpocket import (
    PathPocket,
    PathPocketConfig,
    fix_invalid_doc_status,
)

# Load environment variables
load_dotenv()

# Disable httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)


async def export_nodes_to_json(rag: PathPocket, output_file: str, batch_size: int = 500):
    """
    Export all nodes from PGHypergraphStorage to JSON file in batches.
    Uses streaming write to avoid memory explosion.
    """
    from pathpocket.lightrag_kg.pg_hypergraph_impl import PGHypergraphStorage
    
    graph = rag.rag_engine.chunk_entity_relation_graph
    if not isinstance(graph, PGHypergraphStorage):
        print("⚠️  This script only works with PGHypergraphStorage")
        return False
    
    if not hasattr(graph, '_node_table_name'):
        print("⚠️  Cannot find node table name")
        return False
    
    node_table_name = graph._node_table_name
    workspace = graph.workspace
    
    print(f"Exporting nodes from {node_table_name} (workspace: {workspace})...")
    print(f"Output file: {output_file}")
    print(f"Batch size: {batch_size}")
    
    # Get total count
    try:
        count_query = f"""
        SELECT COUNT(*) as total
        FROM {node_table_name}
        WHERE workspace = $1
        """
        if hasattr(graph, '_query'):
            count_result = await graph._query(count_query, [workspace], readonly=True)
            total_nodes = count_result[0]["total"] if count_result else 0
            print(f"Found {total_nodes} nodes in database")
    except Exception as e:
        print(f"⚠️  Could not get node count: {e}")
        total_nodes = None
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    # Use a more memory-efficient approach: use database query to check for duplicates
    # instead of keeping all node IDs in memory
    # We'll use a temporary file to track the last processed node_id for resumability
    temp_tracking_file = output_file + ".last_id"
    last_processed_id = None
    
    # Try to resume from last processed ID if file exists
    if os.path.exists(temp_tracking_file):
        try:
            with open(temp_tracking_file, 'r', encoding='utf-8') as tf:
                last_processed_id = tf.read().strip()
                if last_processed_id:
                    print(f"  Resuming from node_id: {last_processed_id}")
        except Exception:
            pass
    
    total_exported = 0
    
    # Open output file for streaming write
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write JSON header
        f.write('{\n')
        f.write(f'  "workspace": {json.dumps(workspace)},\n')
        f.write('  "nodes": [\n')
        
        first_node = True
        offset = 0
        batch_num = 0
        
        # Process nodes from node table in batches
        # Use ORDER BY node_id to ensure consistent ordering and enable resumability
        while True:
            batch_num += 1
            
            # Optimized query: use WHERE clause to skip already processed nodes if resuming
            # When resuming, we use node_id > last_processed_id, so offset should always be 0
            if last_processed_id:
                query = f"""
                SELECT node_id, node_data
                FROM {node_table_name}
                WHERE workspace = $1 AND node_id > $2
                ORDER BY node_id
                LIMIT $3
                """
                query_params = [workspace, last_processed_id, batch_size]
            else:
                query = f"""
                SELECT node_id, node_data
                FROM {node_table_name}
                WHERE workspace = $1
                ORDER BY node_id
                LIMIT $2 OFFSET $3
                """
                query_params = [workspace, batch_size, offset]
            
            try:
                if hasattr(graph, '_query'):
                    batch_result = await graph._query(query, query_params, readonly=True)
                    
                    if not batch_result:
                        break  # No more nodes
                    
                    # Process and write this batch immediately
                    batch_last_id = None
                    for row in batch_result:
                        node_id = row["node_id"]
                        batch_last_id = node_id  # Track last ID in this batch
                        
                        node_data = row["node_data"]
                        
                        # Parse JSON if needed
                        if isinstance(node_data, str):
                            try:
                                node_data = json.loads(node_data)
                            except json.JSONDecodeError:
                                continue
                        
                        if not isinstance(node_data, dict):
                            continue
                        
                        # Write node data immediately (streaming)
                        if not first_node:
                            f.write(',\n')
                        first_node = False
                        
                        node_obj = {
                            "node_id": node_id,
                            "entity_id": node_id,
                            "description": node_data.get("description", ""),
                            "entity_type": node_data.get("entity_type", ""),
                            "source_id": node_data.get("source_id", ""),
                            "file_path": node_data.get("file_path", "unknown"),
                        }
                        f.write('    ' + json.dumps(node_obj, ensure_ascii=False))
                        total_exported += 1
                    
                    # Update last processed ID for resumability and next iteration
                    if batch_last_id:
                        last_processed_id = batch_last_id
                        # Once we start using WHERE clause, always use it (no offset)
                        offset = 0
                        try:
                            with open(temp_tracking_file, 'w', encoding='utf-8') as tf:
                                tf.write(last_processed_id)
                        except Exception:
                            pass
                    
                    # Progress update
                    if total_nodes:
                        print(f"  Exported batch {batch_num} ({total_exported}/{total_nodes} nodes)...")
                    else:
                        print(f"  Exported batch {batch_num} ({total_exported} nodes)...")
                    
                    # Check if we got fewer results than batch size (last batch)
                    if len(batch_result) < batch_size:
                        break
                    
                    # Update offset only when not using WHERE clause (normal mode)
                    if not last_processed_id:
                        offset += batch_size
                    
            except Exception as e:
                print(f"⚠️  Error querying nodes batch {batch_num}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Skip hyperedge nodes processing - they should already be in node table
        # If needed, they can be added later via a separate optimized query
        print(f"  Skipping hyperedge nodes check (assuming all nodes are in node table)")
        
        # Close JSON structure
        f.write('\n  ],\n')
        f.write(f'  "total_nodes": {total_exported}\n')
        f.write('}\n')
    
    # Clean up temporary tracking file
    temp_tracking_file = output_file + ".last_id"
    if os.path.exists(temp_tracking_file):
        try:
            os.remove(temp_tracking_file)
        except Exception:
            pass
    
    print(f"\n✅ Exported {total_exported} nodes to {output_file}")
    return True


async def generate_embeddings_from_json(
    rag: PathPocket,
    input_file: str,
    embedding_func,
    embedding_dim: int,
    batch_size: int = 32
):
    """
    Load nodes from JSON file and generate embeddings in batches.
    Write embeddings directly to PostgreSQL VDB_ENTITY table.
    """
    import asyncpg
    from datetime import datetime, timezone
    from pathpocket.lightrag_utils import compute_mdhash_id
    print(f"Streaming nodes from {input_file}...")

    # Helper generator: stream node objects from large JSON file
    def stream_nodes(path: str):
        """
        Stream `nodes` array objects from the JSON file without loading whole file.
        Assumes file structure created by export_nodes_to_json().
        """
        with open(path, "r", encoding="utf-8") as f:
            in_nodes_array = False
            for line in f:
                stripped = line.strip()
                if not in_nodes_array:
                    # Find the `"nodes": [` line
                    if stripped.startswith('"nodes"') or '"nodes"' in stripped:
                        # Consume until '[' appears; can be on same line or next
                        if "[" in stripped:
                            in_nodes_array = True
                        else:
                            # Next non-empty line should contain '['
                            continue
                    continue

                # We are inside nodes array
                if not stripped:
                    continue

                # Skip array delimiters and commas between elements
                if stripped == "[":
                    continue
                if stripped == "],":
                    break
                if stripped == "]":
                    break
                if stripped == ",":
                    continue

                # Remove trailing comma if present
                if stripped.endswith(","):
                    stripped = stripped[:-1].rstrip()

                # Only JSON objects are expected here
                try:
                    node = json.loads(stripped)
                except json.JSONDecodeError:
                    # If for some reason objects span multiple lines, fall back to
                    # accumulating until a full JSON object is read.
                    buffer = [stripped]
                    for cont_line in f:
                        cont_stripped = cont_line.strip()
                        if not cont_stripped:
                            continue
                        buffer.append(cont_stripped.rstrip(","))
                        try:
                            node = json.loads(" ".join(buffer))
                            break
                        except json.JSONDecodeError:
                            continue
                    else:
                        # Could not parse a valid JSON object, stop streaming
                        break

                yield node

    # We still need workspace; read it from small header instead of whole file
    workspace = rag.rag_engine.workspace or "default"
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('"workspace"'):
                    # Example: "workspace": "xxx",
                    try:
                        key, value = stripped.split(":", 1)
                        workspace = json.loads(value.rstrip(","))
                    except Exception:
                        pass
                    break
    except Exception:
        pass

    print(f"Workspace: {workspace}")
    print(f"Embedding batch size: {batch_size}")
    
    # Connect to PostgreSQL
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "zhexu")
    password = os.getenv("POSTGRES_PASSWORD", "")
    database = os.getenv("POSTGRES_DATABASE", "pathrag")
    
    try:
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
            print("✅ VDB_ENTITY table created/verified")
        except Exception as e:
            print(f"⚠️  Error creating VDB_ENTITY table: {e}")
        
        # Stream nodes and process in batches without loading full JSON
        print("Checking/processing entities in streaming mode...")

        existing_entity_ids = set()
        total_processed = 0
        total_skipped = 0
        total_failed = 0
        processed_ids = set()

        nodes_iter = stream_nodes(input_file)
        check_batch_size = 1000  # how many nodes to check against DB at once
        batch_index = 0

        while True:
            # Collect a chunk of nodes for existence check
            raw_nodes = []
            for _ in range(check_batch_size):
                try:
                    node = next(nodes_iter)
                except StopIteration:
                    break
                raw_nodes.append(node)

            if not raw_nodes:
                break

            batch_index += 1

            # Compute entity IDs for this chunk
            batch_entity_ids = []
            node_infos = []  # keep (node, eid, node_id) to reuse later
            for node in raw_nodes:
                node_id = node.get("node_id") or node.get("entity_id", "")
                if not node_id:
                    continue
                entity_name = node_id
                eid = compute_mdhash_id(entity_name, prefix="ent-")
                batch_entity_ids.append(eid)
                node_infos.append((node, eid, node_id))

            if not batch_entity_ids:
                continue

            # Query existing entity IDs from database for this chunk
            try:
                query = f"""
                SELECT id
                FROM VDB_ENTITY
                WHERE workspace = $1 AND id = ANY($2::text[])
                """
                result = await conn.fetch(query, workspace, batch_entity_ids)
                for row in result:
                    existing_entity_ids.add(row["id"])
            except Exception as e:
                print(f"⚠️  Error checking existing entities batch {batch_index}: {e}")

            # Now process this chunk further in smaller embedding batches
            for start in range(0, len(node_infos), batch_size):
                sub_infos = node_infos[start : start + batch_size]

                # Prepare entity data for this embedding batch
                entity_batch = []
                for node, eid, node_id in sub_infos:
                    if not node_id or node_id in processed_ids:
                        continue
                    processed_ids.add(node_id)

                    # Skip if already exists in database
                    if eid in existing_entity_ids:
                        total_skipped += 1
                        continue

                    entity_name = node_id
                    description = node.get("description", "")

                    if description:
                        content = f"{entity_name}\n{description}"
                    else:
                        content = entity_name

                    entity_batch.append(
                        {
                            "id": eid,
                            "entity_name": entity_name,
                            "source_id": node.get("source_id", node_id),
                            "content": content,
                            "file_path": node.get("file_path", "unknown"),
                        }
                    )

                if not entity_batch:
                    continue

                contents = [item["content"] for item in entity_batch]

                # Generate embeddings
                try:
                    embeddings = await embedding_func(contents)

                    # Write to database immediately
                    for j, item in enumerate(entity_batch):
                        try:
                            # Convert embedding to JSON string for PostgreSQL pgvector
                            if isinstance(embeddings, np.ndarray):
                                vector = embeddings[j].tolist()
                            elif isinstance(embeddings, list):
                                vector = (
                                    embeddings[j]
                                    if isinstance(embeddings[j], list)
                                    else embeddings[j].tolist()
                                )
                            else:
                                vector = embeddings[j].tolist()

                            vector_json = json.dumps(vector)

                            await conn.execute(
                                """
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
                                """,
                                workspace,
                                item["id"],
                                item["entity_name"],
                                item["content"],
                                vector_json,
                                [item["source_id"]],
                                item["file_path"],
                                current_time,
                                current_time,
                            )

                            total_processed += 1
                        except Exception as e:
                            print(f"⚠️  Error writing entity {item['id']}: {e}")
                            total_failed += 1

                    print(
                        f"  Batch {batch_index}, sub-batch {start//batch_size + 1} "
                        f"(processed: {total_processed}, skipped: {total_skipped}, failed: {total_failed})..."
                    )

                except Exception as e:
                    print(
                        f"⚠️  Error generating embeddings for batch {batch_index}, "
                        f"sub-batch {start//batch_size + 1}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    total_failed += len(entity_batch)
        
        await conn.close()
        print(f"\n✅ Entity embeddings completed:")
        print(f"   Total processed: {total_processed}")
        print(f"   Skipped (already exists): {total_skipped}")
        print(f"   Failed: {total_failed}")
        return True
        
    except Exception as e:
        print(f"❌ Error in embedding generation: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    parser = argparse.ArgumentParser(description='Generate entity embeddings using JSON file as intermediate storage')
    parser.add_argument('--export', action='store_true', help='Export all nodes to JSON file')
    parser.add_argument('--embed', action='store_true', help='Generate embeddings from JSON file')
    parser.add_argument('--json-file', type=str, default=None, help='JSON file path (default: working_dir/entities_nodes.json)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for embedding generation (default: 32)')
    parser.add_argument('--export_batch_size', type=int, default=64, help='Batch size for node export (default: 500)')
    
    args = parser.parse_args()
    
    if not args.export and not args.embed:
        parser.print_help()
        return
    
    # Configuration
    EMBEDDING_METHOD = os.getenv("EMBEDDING_METHOD", "direct").lower()
    EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "./models/bge-m3")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", None)
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    
    working_dir = os.getenv("WORKING_DIR", "./pathpocket_storage")
    
    # Set JSON file path
    if args.json_file:
        json_file = args.json_file
    else:
        json_file = os.path.join(working_dir, "entities_nodes.json")
    
    # Initialize RAG (minimal, just for graph access)
    config = PathPocketConfig(
        working_dir=working_dir,
        enable_image_processing=False,
        enable_table_processing=False,
    )
    
    async def dummy_llm_func(*args, **kwargs):
        raise RuntimeError("LLM should not be called")
    
    # Get graph storage type
    graph_storage = os.getenv("GRAPH_STORAGE", "PGHypergraphStorage")
    
    # Get embedding dimension
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
    
    # Create a dummy embedding function wrapper for export step (not actually used)
    async def dummy_embedding_func_impl(texts: List[str], **kwargs):
        """Dummy embedding function implementation for export step"""
        raise RuntimeError("Embedding function should not be called during export")
    
    # Create embedding function wrapper class (same as in stage4_embed_gpu.py)
    class SimpleEmbeddingFunc:
        """Simple embedding function wrapper for RAG engine"""
        def __init__(self, embedding_dim, max_token_size, func):
            self.embedding_dim = embedding_dim
            self.max_token_size = max_token_size
            self.func = func
        
        async def __call__(self, texts):
            """Call embedding function"""
            return await self.func(texts)
    
    # Create dummy embedding function wrapper
    dummy_embedding_func = SimpleEmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        func=dummy_embedding_func_impl,
    )
    
    # Initialize PathPocket
    # For export step, we need a dummy embedding_func to satisfy RAG engine initialization
    rag = PathPocket(
        config=config,
        llm_model_func=dummy_llm_func,
        embedding_func=dummy_embedding_func,  # Dummy wrapper for export step
        rag_engine_kwargs={
            "graph_storage": graph_storage,
            "addon_params": {},
        }
    )
    
    # Fix invalid doc_status
    await fix_invalid_doc_status(rag)
    
    # Initialize RAG engine
    init_result = await rag._ensure_rag_engine_initialized()
    if not init_result.get("success"):
        raise RuntimeError(f"Failed to initialize RAG engine: {init_result.get('error')}")
    
    # Step 1: Export nodes to JSON
    if args.export:
        print(f"\n{'='*60}")
        print("Step 1: Exporting all nodes to JSON file")
        print(f"{'='*60}\n")
        
        success = await export_nodes_to_json(rag, json_file, args.export_batch_size)
        if success:
            print(f"\n✅ Export completed! JSON file: {json_file}")
            print(f"Next step: Run with --embed to generate embeddings")
        else:
            print("\n❌ Export failed!")
        return
    
    # Step 2: Generate embeddings from JSON
    if args.embed:
        print(f"\n{'='*60}")
        print("Step 2: Generating embeddings from JSON file")
        print(f"{'='*60}\n")
        
        # Check if JSON file exists
        if not os.path.exists(json_file):
            print(f"❌ JSON file not found: {json_file}")
            print("Please run with --export first to create the JSON file")
            return
        
        # Setup embedding function
        if EMBEDDING_METHOD == "direct" and SENTENCE_TRANSFORMERS_AVAILABLE:
            print("🚀 Using direct model loading (sentence-transformers)")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading model on device: {device}")
            
            if os.path.exists(EMBEDDING_MODEL_PATH):
                EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_PATH
            elif EMBEDDING_MODEL_NAME is None:
                EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
            
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            print("✅ Model loaded successfully")
            
            async def _direct_embed(texts: List[str], **kwargs):
                embeddings = embedding_model.encode(
                    texts,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                return np.array(embeddings)
            
            embedding_func = _direct_embed
        else:
            print("❌ Direct embedding method requires sentence-transformers")
            print("Install with: pip install sentence-transformers")
            return
        
        # Generate embeddings
        success = await generate_embeddings_from_json(
            rag,
            json_file,
            embedding_func,
            EMBEDDING_DIM,
            args.batch_size
        )
        
        if success:
            print(f"\n✅ Embedding generation completed!")
        else:
            print("\n❌ Embedding generation failed!")


if __name__ == "__main__":
    asyncio.run(main())

