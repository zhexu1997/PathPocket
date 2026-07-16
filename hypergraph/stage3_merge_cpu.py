"""
Stage 3 Optimized: Merge entities and relations with optimized filtering
Input: kv_store_llm_response_cache.json, kv_store_text_chunks.json
Output: hypergraph_chunk_entity_relation.json, kv_store_entity_chunks.json,
        kv_store_full_entities.json, kv_store_full_relations.json,
        kv_store_relation_chunks.json

Optimizations:
1. Process entities first
2. Filter relations before merging: remove entities not in entity list, remove relations with < 5 entities
3. Multimodal entities only added to relations from current chunk
4. No binary relations between multimodal and text entities
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
from typing import List, Dict, Any, Tuple
import time as time_module

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger = logging.getLogger(__name__)
    if os.getenv("GRAPH_STORAGE"):
        logger.info(f"Loaded GRAPH_STORAGE from .env: {os.getenv('GRAPH_STORAGE')}")
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass
except Exception as e:
    # .env file parsing error, continue without it
    pass

from pathpocket import (
    PathPocket,
    PathPocketConfig,
    fix_invalid_doc_status,
)
from pathpocket.operate import (
    GRAPH_FIELD_SEP,
    merge_source_ids,
    apply_source_ids_limit,
    _get_cached_extraction_results,
    _rebuild_from_extraction_result,
)
from pathpocket.shared_storage import get_storage_keyed_lock
from collections import defaultdict

# Disable httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ========== Helper functions ==========

def deduplicate_entities_by_description(entity_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fast deduplication of entities by description"""
    if not entity_list:
        return []
    
    unique_entities = {}
    for entity_data in entity_list:
        desc = entity_data.get("description", "")
        if desc and desc not in unique_entities:
            unique_entities[desc] = entity_data
    
    return sorted(
        unique_entities.values(),
        key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
    )


def deduplicate_relations_by_description(rel_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fast deduplication of relations by description"""
    if not rel_list:
        return []
    
    unique_relations = {}
    for rel_data in rel_list:
        desc = rel_data.get("description", "")
        if desc and desc not in unique_relations:
            unique_relations[desc] = rel_data
    
    return sorted(
        unique_relations.values(),
        key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
    )


def merge_entities_fast(
    target: Dict[str, List[Dict[str, Any]]],
    source: Dict[str, List[Dict[str, Any]]],
    fix_multimodal_types: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fast merge of entities with deduplication"""
    for entity_name, entity_list in source.items():
        deduplicated = deduplicate_entities_by_description(entity_list)
        
        if entity_name not in target:
            target[entity_name] = []
        
        existing_descriptions = {e.get("description", "") for e in target[entity_name]}
        
        for entity_data in deduplicated:
            desc = entity_data.get("description", "")
            if desc and desc not in existing_descriptions:
                if fix_multimodal_types:
                    if "(image)" in entity_name:
                        entity_data["entity_type"] = "Image"
                    elif "(table)" in entity_name:
                        entity_data["entity_type"] = "Table"
                target[entity_name].append(entity_data)
                existing_descriptions.add(desc)
    
    return target


def merge_relations_fast(
    target: Dict[tuple, List[Dict[str, Any]]],
    source: Dict[tuple, List[Dict[str, Any]]],
) -> Dict[tuple, List[Dict[str, Any]]]:
    """Fast merge of relations with deduplication"""
    for edge_key, rel_list in source.items():
        deduplicated = deduplicate_relations_by_description(rel_list)
        
        if edge_key not in target:
            target[edge_key] = []
        
        existing_descriptions = {r.get("description", "") for r in target[edge_key]}
        
        for rel_data in deduplicated:
            desc = rel_data.get("description", "")
            if desc and desc not in existing_descriptions:
                target[edge_key].append(rel_data)
                existing_descriptions.add(desc)
    
    return target


async def extract_entities_and_relations_from_cache(
    cached_results: Dict[str, List[tuple]],
    chunk_id: str,
    text_chunks_storage,
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[tuple, List[Dict[str, Any]]]]:
    """Extract entities and relations from cached LLM results"""
    if chunk_id not in cached_results:
        return {}, {}
    
    chunk_entities = defaultdict(list)
    chunk_relationships = defaultdict(list)
    
    for result in cached_results[chunk_id]:
        entities, relationships = await _rebuild_from_extraction_result(
            text_chunks_storage=text_chunks_storage,
            chunk_id=chunk_id,
            extraction_result=result[0],
            timestamp=result[1],
        )
        
        merge_entities_fast(chunk_entities, entities)
        merge_relations_fast(chunk_relationships, relationships)
    
    return dict(chunk_entities), dict(chunk_relationships)


def filter_relations_by_entities(
    relations: Dict[tuple, List[Dict[str, Any]]],
    valid_entity_names: set,
    min_entities: int = 2,
) -> Dict[tuple, List[Dict[str, Any]]]:
    """
    Filter relations:
    1. Remove entities not in valid_entity_names (EXCEPT multimodal entities)
    2. Remove relations with < min_entities after filtering
    
    CRITICAL: Multimodal entities (containing "(image)" or "(table)") are always kept,
    even if they're not in valid_entity_names.
    
    Note: min_entities=2 means at least 2 entities (relation format: relation<|#|>entity1<|#|>entity2<|#|>keywords<|#|>description = 5 fields total)
    """
    filtered = {}
    removed_count = 0
    removed_entities_count = 0
    
    for edge_key, rel_list in relations.items():
        # Get entities from edge_key or from first relation
        if isinstance(edge_key, tuple):
            entities = list(edge_key)
        else:
            entities = []
            if rel_list:
                first_rel = rel_list[0]
                entities_str = first_rel.get("entities", "")
                if entities_str:
                    entities = entities_str.split(GRAPH_FIELD_SEP) if isinstance(entities_str, str) else entities_str
        
        # Filter entities: keep those in valid_entity_names OR multimodal entities
        # Multimodal entities are identified by containing "(image)" or "(table)"
        filtered_entities = [
            e for e in entities 
            if e in valid_entity_names or "(image)" in e or "(table)" in e
        ]
        removed_entities_count += len(entities) - len(filtered_entities)
        
        # Remove relation if not enough entities after filtering
        if len(filtered_entities) < min_entities:
            removed_count += 1
            continue
        
        # Update relation data with filtered entities
        new_edge_key = tuple(sorted(filtered_entities))
        updated_rel_list = []
        for rel_data in rel_list:
            new_rel = rel_data.copy()
            new_rel["entities"] = GRAPH_FIELD_SEP.join(filtered_entities)
            new_rel["entity_count"] = len(filtered_entities)
            updated_rel_list.append(new_rel)
        
        if new_edge_key not in filtered:
            filtered[new_edge_key] = []
        filtered[new_edge_key].extend(updated_rel_list)
    
    if removed_count > 0 or removed_entities_count > 0:
        logger.info(f"Filtered relations: removed {removed_count} relations, {removed_entities_count} invalid entities")
    
    return filtered


async def merge_entities_optimized(
    all_entities: Dict[str, List[Dict[str, Any]]],
    knowledge_graph_inst,
    entity_vdb,
    global_config: Dict[str, Any],
    entity_chunks_storage=None,
    doc_id: str = None,
    entity_max_async: int = 32,
):
    """Merge entities first (before relations)"""
    from pathpocket.operate import _merge_nodes_then_upsert
    
    entity_items = list(all_entities.items())
    total_entities = len(entity_items)
    logger.info(f"Merging {total_entities} entities...")
    
    # Get predefined entity types from config
    custom_entity_types = global_config.get("entity_types", [])
    if not custom_entity_types:
        # Fallback: try to get from rag_engine if available
        # This will be set in main() function
        custom_entity_types = []
    
    entity_semaphore = asyncio.Semaphore(entity_max_async)
    workspace = global_config.get("workspace", "default")
    namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
    
    processed_entities = []
    
    # Progress tracking
    processed_count = 0
    progress_lock = asyncio.Lock()
    
    async def merge_entity(entity_name, entities, index):
        nonlocal processed_count
        
        async with entity_semaphore:
            sorted_key = sorted([entity_name])
            lock_context = get_storage_keyed_lock(sorted_key, namespace=namespace, enable_logging=False)
            
            async with lock_context:
                try:
                    # Ensure all entities have required fields
                    normalized_entities = []
                    for entity in entities:
                        normalized_entity = dict(entity)
                        entity_type = normalized_entity.get("entity_type", "").strip()
                        
                        # Rule 1: Multimodal entities - always use "Image" or "Table" (capitalized)
                        if "(image)" in entity_name:
                            normalized_entity["entity_type"] = "Image"
                        elif "(table)" in entity_name:
                            normalized_entity["entity_type"] = "Table"
                        # Rule 2: Normalize old lowercase "image"/"table" to capitalized
                        elif entity_type.lower() == "image":
                            normalized_entity["entity_type"] = "Image"
                        elif entity_type.lower() == "table":
                            normalized_entity["entity_type"] = "Table"
                        # Rule 3: If entity_type is already "Image" or "Table" (multimodal), keep it
                        elif entity_type in ["Image", "Table"]:
                            # Keep multimodal types
                            pass
                        # Rule 4: If entity_type is missing, default to "Other"
                        elif not entity_type:
                            normalized_entity["entity_type"] = "Other"
                        # Rule 5: Fix old "generic" type to "Other"
                        elif entity_type.lower() in ["generic"]:
                            normalized_entity["entity_type"] = "Other"
                        # Rule 6: If entity_type is in custom_entity_types, keep it (priority: preserve LLM extracted type)
                        # Use case-insensitive comparison to match predefined types
                        elif custom_entity_types:
                            # Check if entity_type matches any predefined type (case-insensitive)
                            entity_type_lower = entity_type.lower()
                            matching_predefined = None
                            for predefined_type in custom_entity_types:
                                if predefined_type.lower() == entity_type_lower:
                                    matching_predefined = predefined_type
                                    break
                            
                            if matching_predefined:
                                # Use the predefined type format (preserve original case from list)
                                normalized_entity["entity_type"] = matching_predefined
                            else:
                                # Not in predefined list, change to "Other"
                                normalized_entity["entity_type"] = "Other"
                        # Rule 7: If custom_entity_types is not available and not multimodal, use "Other"
                        else:
                            # Change to "Other" if not in predefined list
                            normalized_entity["entity_type"] = "Other"
                        # Ensure other required fields exist
                        normalized_entity.setdefault("description", "")
                        normalized_entity.setdefault("source_id", "")
                        normalized_entity.setdefault("file_path", "unknown_source")
                        normalized_entity.setdefault("timestamp", int(time_module.time()))
                        normalized_entities.append(normalized_entity)
                    
                    result = await _merge_nodes_then_upsert(
                        entity_name=entity_name,
                        nodes_data=normalized_entities,
                        entity_vdb=entity_vdb,
                        knowledge_graph_inst=knowledge_graph_inst,
                        global_config=global_config,
                        entity_chunks_storage=entity_chunks_storage,
                    )
                    
                    # Update progress
                    async with progress_lock:
                        processed_count += 1
                        if processed_count % 100 == 0 or processed_count == total_entities:
                            progress_pct = (processed_count / total_entities) * 100
                            logger.info(f"Entity merge progress: {processed_count}/{total_entities} ({progress_pct:.1f}%)")
                    
                    return result
                except Exception as e:
                    logger.error(f"Error merging entity {entity_name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # Update progress even on error
                    async with progress_lock:
                        processed_count += 1
                    
                    return None
    
    # Process entities in parallel
    tasks = [merge_entity(name, entities, idx) for idx, (name, entities) in enumerate(entity_items)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if result is not None and not isinstance(result, Exception):
            processed_entities.append(result)
    
    logger.info(f"Merged {len(processed_entities)} entities (completed {processed_count}/{total_entities})")
    return processed_entities


async def merge_relations_optimized(
    all_relations: Dict[tuple, List[Dict[str, Any]]],
    valid_entity_names: set,
    knowledge_graph_inst,
    global_config: Dict[str, Any],
    relation_max_async: int = 8,
    min_entities_per_relation: int = 2,
):
    """
    Merge relations with filtering.
    """
    from pathpocket.operate import merge_source_ids, apply_source_ids_limit
    
    # Filter relations before merging
    logger.info(f"Filtering {len(all_relations)} relations (min entities: {min_entities_per_relation})...")
    filtered_relations = filter_relations_by_entities(
        all_relations,
        valid_entity_names,
        min_entities=min_entities_per_relation,
    )
    logger.info(f"After filtering: {len(filtered_relations)} relations")
    
    if not filtered_relations:
        return []
    
    relation_semaphore = asyncio.Semaphore(relation_max_async)
    workspace = global_config.get("workspace", "")
    namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
    
    processed_edges = []
    relation_items = list(filtered_relations.items())
    total_relations = len(relation_items)
    
    # Progress tracking
    processed_count = 0
    progress_lock = asyncio.Lock()
    
    async def merge_relation(edge_key, relations, index):
        nonlocal processed_count
        
        # Extract entities and deduplicate
        entities = list(edge_key) if isinstance(edge_key, tuple) else []
        entities_unique = list(dict.fromkeys(entities))
        
        edge_id = relations[0].get("edge_id", f"rel_{'_'.join(sorted(entities_unique))}") if relations else "unknown"
        sorted_edge_key = sorted(entities_unique)
        
        async with relation_semaphore:
            lock_context = get_storage_keyed_lock(sorted_edge_key, namespace=namespace, enable_logging=False)
            
            async with lock_context:
                try:
                    has_hyperedge_support = hasattr(knowledge_graph_inst, 'get_hyperedge')
                    
                    # Read existing data and merge (preserves all data)
                    # Read existing hyperedge
                    existing_he = None
                    if has_hyperedge_support:
                        existing_he = await knowledge_graph_inst.get_hyperedge(edge_id)
                    
                    # Merge data
                        already_description = []
                        already_keywords = []
                        already_source_ids = []
                        already_file_paths = []
                        already_weight = 0.0
                        
                        if existing_he:
                            if existing_he.get("description"):
                                already_description = existing_he["description"].split(GRAPH_FIELD_SEP)
                            if existing_he.get("keywords"):
                                already_keywords = existing_he["keywords"].split(",")
                            if existing_he.get("source_id"):
                                already_source_ids = existing_he["source_id"].split(GRAPH_FIELD_SEP)
                            if existing_he.get("file_path"):
                                already_file_paths = existing_he["file_path"].split(GRAPH_FIELD_SEP)
                            already_weight = existing_he.get("weight", 0.0)
                        
                        # Merge source_ids
                        new_source_ids = [he.get("source_id") for he in relations if he.get("source_id")]
                        existing_full_source_ids = [chunk_id for chunk_id in already_source_ids if chunk_id]
                        full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)
                        
                        limit_method = global_config.get("source_ids_limit_method", "FIFO")
                        max_source_limit = global_config.get("max_source_ids_per_relation", 30)
                        source_ids = apply_source_ids_limit(
                            full_source_ids,
                            max_source_limit,
                            limit_method,
                            identifier=f"Hyperedge `{edge_id}`",
                        )
                        
                        # Merge keywords
                        all_keywords = set(k.strip() for k in already_keywords if k.strip())
                        for he in relations:
                            if he.get("keywords"):
                                all_keywords.update(k.strip() for k in he["keywords"].split(",") if k.strip())
                        keywords = ",".join(sorted(all_keywords))
                        
                        # Merge weight
                        weight = already_weight + sum(he.get("weight", 1.0) for he in relations)
                        
                        # Merge descriptions
                        unique_hyperedges = {}
                        for he in relations:
                            description_value = he.get("description")
                            if not description_value:
                                continue
                            if description_value not in unique_hyperedges:
                                unique_hyperedges[description_value] = he
                        
                        sorted_hyperedges = sorted(
                            unique_hyperedges.values(),
                            key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
                        )
                        sorted_descriptions = [he["description"] for he in sorted_hyperedges]
                        
                        all_descriptions = already_description + sorted_descriptions
                        unique_descriptions = list(dict.fromkeys(all_descriptions))
                        
                        if not unique_descriptions:
                            logger.warning(f"Hyperedge {edge_id} has no description")
                            return None
                        
                        description = GRAPH_FIELD_SEP.join(unique_descriptions)
                        
                        # Merge file_paths
                        new_file_paths = [he.get("file_path", "") for he in relations if he.get("file_path")]
                        all_file_paths = already_file_paths + new_file_paths
                        unique_file_paths = list(dict.fromkeys(all_file_paths))
                        
                        # Create final edge data
                        edge_data = {
                            "weight": weight,
                            "description": description,
                            "keywords": keywords,
                            "source_id": GRAPH_FIELD_SEP.join(source_ids),
                            "file_path": GRAPH_FIELD_SEP.join(unique_file_paths[:10]),
                            "entities": GRAPH_FIELD_SEP.join(entities_unique),
                            "entity_count": len(entities_unique),
                        }
                    
                    # Write
                    if has_hyperedge_support:
                        await knowledge_graph_inst.upsert_hyperedge(edge_id, entities_unique, edge_data)
                    else:
                        for i, e1 in enumerate(entities_unique):
                            for e2 in entities_unique[i+1:]:
                                await knowledge_graph_inst.upsert_edge(e1, e2, edge_data)
                    
                    # Update progress
                    async with progress_lock:
                        processed_count += 1
                        if processed_count % 10 == 0 or processed_count == total_relations:
                            progress_pct = (processed_count / total_relations) * 100
                            logger.info(f"Merging relations: {processed_count}/{total_relations} ({progress_pct:.1f}%)")
                    
                    return edge_data
                except Exception as e:
                    logger.error(f"Error merging relation {edge_id}: {e}")
                    # Update progress even on error
                    async with progress_lock:
                        processed_count += 1
                    return None
    
    # Process relations in parallel
    logger.info(f"Starting to merge {total_relations} relations...")
    tasks = [merge_relation(key, relations, idx) for idx, (key, relations) in enumerate(relation_items)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if result is not None and not isinstance(result, Exception):
            processed_edges.append(result)
    
    logger.info(f"Completed merging relations: {len(processed_edges)}/{total_relations} successful")
    
    logger.info(f"Merged {len(processed_edges)} relations")
    return processed_edges


async def process_document_optimized(
    doc_id: str,
    chunks_list: List[str],
    rag: PathPocket,
    cached_results: Dict[str, List[tuple]],
    chunk_data_cache: Dict[str, Dict],
    multimodal_entity_names: List[str],
    global_config: Dict[str, Any],
):
    """Process a single document with optimized logic"""
    
    # Separate text and multimodal chunks
    text_chunk_ids = []
    multimodal_chunk_ids = []
    chunk_to_modal_entity = {}
    
    for chunk_id in chunks_list:
        chunk_data = chunk_data_cache.get(chunk_id)
        if chunk_data:
            modal_entity_name = chunk_data.get("modal_entity_name")
            if modal_entity_name:
                multimodal_chunk_ids.append(chunk_id)
                chunk_to_modal_entity[chunk_id] = modal_entity_name
            else:
                text_chunk_ids.append(chunk_id)
    
    logger.info(f"Processing document {doc_id}: {len(text_chunk_ids)} text chunks, {len(multimodal_chunk_ids)} multimodal chunks")
    
    # Step 1: Process text chunks - extract entities and relations
    all_entities = defaultdict(list)
    all_relations = defaultdict(list)
    
    async def process_text_chunk(chunk_id):
        if chunk_id not in cached_results:
            return None
        
        chunk_entities, chunk_relationships = await extract_entities_and_relations_from_cache(
            cached_results=cached_results,
            chunk_id=chunk_id,
            text_chunks_storage=rag.rag_engine.text_chunks,
        )
        
        return (chunk_entities, chunk_relationships)
    
    # Process all text chunks in parallel
    if text_chunk_ids:
        text_tasks = [process_text_chunk(chunk_id) for chunk_id in text_chunk_ids]
        text_results = await asyncio.gather(*text_tasks, return_exceptions=True)
        
        for result in text_results:
            if isinstance(result, Exception) or result is None:
                continue
            chunk_entities, chunk_relationships = result
            merge_entities_fast(all_entities, chunk_entities)
            merge_relations_fast(all_relations, chunk_relationships)
    
    # Step 2: Process multimodal chunks - add multimodal entities to relations from current chunk only
    multimodal_relations_added = 0
    for chunk_id in multimodal_chunk_ids:
        modal_entity_name = chunk_to_modal_entity.get(chunk_id)
        if not modal_entity_name:
            continue
        
        # Get relations from THIS chunk only
        if chunk_id not in cached_results:
            continue
        
        chunk_entities, chunk_relationships = await extract_entities_and_relations_from_cache(
            cached_results={chunk_id: cached_results[chunk_id]},
            chunk_id=chunk_id,
            text_chunks_storage=rag.rag_engine.text_chunks,
        )
        
        # CRITICAL: Filter relations from this chunk first (remove invalid entities)
        # But allow relations with >= 1 entity (will become 2+ after adding multimodal entity)
        # This ensures that even if a relation has only 1 valid entity after filtering,
        # it can still be used by adding the multimodal entity
        valid_entity_names_for_filtering = set(all_entities.keys())
        if multimodal_entity_names:
            valid_entity_names_for_filtering |= set(multimodal_entity_names)
        
        filtered_chunk_relations = {}
        for edge_key, rel_list in chunk_relationships.items():
            for rel_data in rel_list:
                rel_entities = rel_data.get("entities", [])
                if isinstance(rel_entities, str):
                    rel_entities = [e.strip() for e in rel_entities.split(GRAPH_FIELD_SEP) if e.strip()]
                elif not isinstance(rel_entities, list):
                    rel_entities = list(edge_key) if edge_key else []
                
                # Filter entities: keep valid entities OR multimodal entities
                filtered_entities = [
                    e for e in rel_entities 
                    if e in valid_entity_names_for_filtering or "(image)" in e or "(table)" in e
                ]
                
                # CRITICAL: For multimodal chunks, allow relations with >= 1 entity
                # (will become 2+ after adding multimodal entity)
                if len(filtered_entities) >= 1:
                    # Store filtered relation for adding multimodal entity
                    filtered_edge_key = tuple(sorted(filtered_entities))
                    if filtered_edge_key not in filtered_chunk_relations:
                        filtered_chunk_relations[filtered_edge_key] = []
                    
                    filtered_rel_data = rel_data.copy()
                    filtered_rel_data["entities"] = GRAPH_FIELD_SEP.join(filtered_entities)
                    filtered_rel_data["entity_count"] = len(filtered_entities)
                    filtered_chunk_relations[filtered_edge_key].append(filtered_rel_data)
        
        # Add multimodal entity to filtered relations from this chunk
        chunk_relations_added = 0
        for edge_key, rel_list in filtered_chunk_relations.items():
            for rel_data in rel_list:
                rel_entities = rel_data.get("entities", [])
                if isinstance(rel_entities, str):
                    rel_entities = [e.strip() for e in rel_entities.split(GRAPH_FIELD_SEP) if e.strip()]
                elif not isinstance(rel_entities, list):
                    rel_entities = list(edge_key) if edge_key else []
                
                # Skip if multimodal entity already in relation
                if modal_entity_name in rel_entities:
                    continue
                
                # Add multimodal entity to relations
                # CRITICAL: Even if relation has only 1 entity after filtering, adding multimodal entity makes it 2 entities
                # Format: relation<|#|>entity1<|#|>modal_entity<|#|>keywords<|#|>description = 5 fields total
                if len(rel_entities) >= 1:  # At least 1 entity (will become 2 with multimodal entity)
                    new_entities = list(rel_entities) + [modal_entity_name]
                    
                    # Create relation if it will have >= 2 entities after adding multimodal entity
                    # This allows relations with 1 entity to become valid (2 entities = 5 fields total)
                    if len(new_entities) >= 2:
                        new_edge_key = tuple(sorted(new_entities))
                        
                        new_rel_data = rel_data.copy()
                        new_rel_data["entities"] = GRAPH_FIELD_SEP.join(new_entities)
                        new_rel_data["entity_count"] = len(new_entities)
                        
                        if new_edge_key not in all_relations:
                            all_relations[new_edge_key] = []
                        all_relations[new_edge_key].append(new_rel_data)
                        chunk_relations_added += 1
        
        multimodal_relations_added += chunk_relations_added
        
        # Add multimodal entity to entities
        chunk_data = chunk_data_cache.get(chunk_id)
        if modal_entity_name not in all_entities:
            content_type = chunk_data.get("content_type", "") if chunk_data else ""
            # Multimodal entities should be "Image" or "Table" (capitalized)
            entity_type = "Image" if content_type == "image" else "Table" if content_type == "table" else "Image"  # Default to "Image" if unknown
            
            entity_data = {
                "entity_name": modal_entity_name,
                "entity_type": entity_type,
                "description": chunk_data.get("content", "") if chunk_data else "",
                "source_id": chunk_id,
                "file_path": chunk_data.get("file_path", "unknown_source") if chunk_data else "unknown_source",
                "timestamp": int(time_module.time()),
            }
            all_entities[modal_entity_name] = [entity_data]
    
    if multimodal_relations_added > 0:
        logger.info(f"Added {multimodal_relations_added} relations with multimodal entities")
    
    # Step 3: Merge entities first
    # CRITICAL: Include all multimodal entities in valid_entity_names before filtering relations
    valid_entity_names = set(all_entities.keys())
    if multimodal_entity_names:
        valid_entity_names |= set(multimodal_entity_names)
    
    logger.info(f"Total entities before merging: {len(all_entities)} (including {len(multimodal_entity_names)} multimodal entities)")
    logger.info(f"Total relations before filtering: {len(all_relations)}")
    
    processed_entities = await merge_entities_optimized(
        all_entities=all_entities,
        knowledge_graph_inst=rag.rag_engine.chunk_entity_relation_graph,
        entity_vdb=None,  # Skip VDB in Stage 3
        global_config=global_config,
        entity_chunks_storage=rag.rag_engine.entity_chunks,
        doc_id=doc_id,
        entity_max_async=global_config.get("llm_model_max_async", 32),
    )
    
    # Step 4: Filter and merge relations
    processed_edges = await merge_relations_optimized(
        all_relations=all_relations,
        valid_entity_names=valid_entity_names,
        knowledge_graph_inst=rag.rag_engine.chunk_entity_relation_graph,
        global_config=global_config,
        relation_max_async=global_config.get("relation_max_async", 8),
        min_entities_per_relation=2,
    )
    
    # Step 5: Update full_entities and full_relations
    if rag.rag_engine.full_entities and rag.rag_engine.full_relations:
        try:
            existing_entities_data = await rag.rag_engine.full_entities.get_by_id(doc_id)
            existing_relations_data = await rag.rag_engine.full_relations.get_by_id(doc_id)
            
            existing_entity_names = set(existing_entities_data.get("entity_names", [])) if existing_entities_data else set()
            existing_relation_pairs = set(
                tuple(pair) if isinstance(pair, list) else pair
                for pair in (existing_relations_data.get("relation_pairs", []) if existing_relations_data else [])
            )
            
            # Collect new entities
            new_entity_names = set()
            for entity_data in processed_entities:
                if entity_data and entity_data.get("entity_id"):
                    new_entity_names.add(entity_data["entity_id"])
            
            if multimodal_entity_names:
                new_entity_names |= set(multimodal_entity_names)
            
            # Collect new relations (at least 2 entities)
            new_relations = []
            for edge_data in processed_edges:
                if edge_data:
                    entities_str = edge_data.get("entities", "")
                    if entities_str:
                        entities_list = entities_str.split(GRAPH_FIELD_SEP) if isinstance(entities_str, str) else entities_str
                        if len(entities_list) >= 2:  # At least 2 entities (binary relation minimum)
                            new_relations.append(tuple(sorted(entities_list)))
            
            # Merge with existing
            all_new_relations = set(new_relations)
            all_relations = existing_relation_pairs | all_new_relations
            
            # CRITICAL: Filter out binary relations between multimodal entities and text entities
            # if they are already in n-ary relations together
            # This ensures kv_store_full_relations matches hypergraph_chunk_entity_relation
            filtered_relations = set()
            nary_relations = set()  # Store n-ary relations (length > 2)
            binary_relations = []  # Store binary relations for filtering
            
            # First pass: separate n-ary and binary relations from ALL relations
            for rel in all_relations:
                # Normalize relation to tuple
                if isinstance(rel, list):
                    rel = tuple(rel)
                elif not isinstance(rel, tuple):
                    continue
                
                if len(rel) > 2:
                    # This is an n-ary relation
                    sorted_rel = tuple(sorted(rel))
                    nary_relations.add(sorted_rel)
                    filtered_relations.add(sorted_rel)  # Always keep n-ary relations
                elif len(rel) == 2:
                    # This is a binary relation - check if it should be filtered
                    binary_relations.append(tuple(sorted(rel)))
                else:
                    # Invalid relation, skip
                    continue
            
            # Second pass: filter binary relations
            # If a binary relation between a multimodal entity and a text entity
            # already exists in an n-ary relation, skip the binary relation
            multimodal_entity_set = set()
            if multimodal_entity_names:
                multimodal_entity_set = set(multimodal_entity_names)
            
            filtered_binary_count = 0
            for binary_rel in binary_relations:
                entity1, entity2 = binary_rel
                is_modal1 = entity1 in multimodal_entity_set or "(image)" in entity1 or "(table)" in entity1
                is_modal2 = entity2 in multimodal_entity_set or "(image)" in entity2 or "(table)" in entity2
                
                # If this is a binary relation between a multimodal entity and a text entity
                if (is_modal1 and not is_modal2) or (is_modal2 and not is_modal1):
                    # Check if both entities are already in an n-ary relation together
                    should_filter = False
                    for nary_rel in nary_relations:
                        if entity1 in nary_rel and entity2 in nary_rel:
                            # Both entities are already in this n-ary relation, filter out the binary relation
                            should_filter = True
                            filtered_binary_count += 1
                            break
                    
                    if not should_filter:
                        # Not in n-ary relation, keep the binary relation
                        filtered_relations.add(binary_rel)
                else:
                    # Not a multimodal-text binary relation, keep it
                    filtered_relations.add(binary_rel)
            
            if filtered_binary_count > 0:
                logger.info(f"Filtered {filtered_binary_count} binary relations that are already in n-ary relations")
            
            # Merge
            final_entity_names = existing_entity_names | new_entity_names
            final_relations = filtered_relations
            
            # Save
            create_time_entities = existing_entities_data.get("create_time") if existing_entities_data else int(time_module.time())
            create_time_relations = existing_relations_data.get("create_time") if existing_relations_data else int(time_module.time())
            
            await rag.rag_engine.full_entities.upsert({
                doc_id: {
                    "entity_names": sorted(list(final_entity_names)),
                    "count": len(final_entity_names),
                    "create_time": create_time_entities,
                    "update_time": int(time_module.time()),
                    "_id": doc_id,
                }
            })
            
            await rag.rag_engine.full_relations.upsert({
                doc_id: {
                    "relation_pairs": [list(pair) for pair in final_relations],
                    "count": len(final_relations),
                    "create_time": create_time_relations,
                    "update_time": int(time_module.time()),
                    "_id": doc_id,
                }
            })
            
            logger.info(f"Updated full_entities ({len(final_entity_names)} entities) and full_relations ({len(final_relations)} relations)")
        except Exception as e:
            logger.error(f"Error updating full_entities/full_relations: {e}")
            import traceback
            logger.error(traceback.format_exc())


async def export_to_json_files(rag: PathPocket, working_dir: str):
    """Export PGHypergraphStorage and KV storage data to JSON files for inspection."""
    try:
        from pathpocket.lightrag_namespace import NameSpace
        
        # Ensure working directory exists
        os.makedirs(working_dir, exist_ok=True)
        
        # 1. Export hypergraph_chunk_entity_relation.json
        print("Exporting hypergraph_chunk_entity_relation.json...")
        try:
            if hasattr(rag.rag_engine.chunk_entity_relation_graph, 'get_all_hyperedges'):
                # PGHypergraphStorage
                hyperedges = await rag.rag_engine.chunk_entity_relation_graph.get_all_hyperedges()
                
                # Get all nodes for node_attrs
                all_nodes = await rag.rag_engine.chunk_entity_relation_graph.get_all_nodes()
                
                # Convert to HyperNetX format for compatibility
                hypergraph_data = {
                    "hyperedges": {},
                    "node_attrs": {},
                    "edge_attrs": {}
                }
                
                # Populate node_attrs from all nodes
                # For PGHypergraphStorage, get_all_nodes() only returns entity_id
                # We need to get full node attributes from the graph storage
                # Collect all unique nodes first
                all_node_ids = set()
                for node in all_nodes:
                    node_id = node.get("entity_id") or node.get("id", "")
                    if node_id:
                        all_node_ids.add(node_id)
                
                # Get node data from graph storage (batch operation if available)
                if hasattr(rag.rag_engine.chunk_entity_relation_graph, 'get_nodes_batch'):
                    # Use batch operation for better performance
                    node_ids_list = list(all_node_ids)
                    nodes_dict = await rag.rag_engine.chunk_entity_relation_graph.get_nodes_batch(node_ids_list)
                    for node_id in all_node_ids:
                        full_node_data = nodes_dict.get(node_id)
                        if full_node_data:
                            # Store node attributes (exclude entity_id/id)
                            node_attrs = {k: v for k, v in full_node_data.items() 
                                        if k not in ["entity_id", "id"]}
                            hypergraph_data["node_attrs"][node_id] = node_attrs
                        else:
                            # If no full data, store empty dict (node exists but no attributes)
                            hypergraph_data["node_attrs"][node_id] = {}
                else:
                    # Fallback to individual queries
                    for node_id in all_node_ids:
                        try:
                            full_node_data = await rag.rag_engine.chunk_entity_relation_graph.get_node(node_id)
                            if full_node_data:
                                # Store node attributes (exclude entity_id/id)
                                node_attrs = {k: v for k, v in full_node_data.items() 
                                            if k not in ["entity_id", "id"]}
                                hypergraph_data["node_attrs"][node_id] = node_attrs
                            else:
                                # If no full data, store empty dict (node exists but no attributes)
                                hypergraph_data["node_attrs"][node_id] = {}
                        except Exception as e:
                            # If get_node fails, at least record that the node exists
                            hypergraph_data["node_attrs"][node_id] = {}
                            logger.debug(f"Could not get full node data for {node_id}: {e}")
                
                # Populate hyperedges and edge_attrs
                from pathpocket.operate import GRAPH_FIELD_SEP
                for edge in hyperedges:
                    edge_id = edge.get("edge_id", "")
                    entities = edge.get("entities", [])
                    entity_count = edge.get("entity_count", len(entities) if entities else 0)
                    
                    # Ensure entities is a list (not string)
                    if isinstance(entities, str):
                        # If entities is a string, split by SEP
                        entities = [e.strip() for e in entities.split(GRAPH_FIELD_SEP) if e.strip()]
                    elif not isinstance(entities, list):
                        # If entities is not a list, convert to list
                        entities = list(entities) if entities else []
                    
                    if edge_id and entities:
                        # Store hyperedges as arrays - format: {"edge_id": ["entity1", "entity2", ...]}
                        hypergraph_data["hyperedges"][edge_id] = entities
                        
                        # Store edge attributes including entities and entity_count
                        edge_attrs = {k: v for k, v in edge.items() 
                                    if k not in ["edge_id", "entities", "entity_count"]}
                        # Ensure entities (as array) and entity_count are included in edge_attrs
                        # Format: ["entity1", "entity2", ...] (array of entity names)
                        edge_attrs["entities"] = entities  # Keep as array, not string
                        edge_attrs["entity_count"] = entity_count
                        hypergraph_data["edge_attrs"][edge_id] = edge_attrs
                
                output_file = os.path.join(working_dir, "hypergraph_chunk_entity_relation.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(hypergraph_data, f, ensure_ascii=False, indent=2)
                print(f"  ✅ Exported {len(hypergraph_data['hyperedges'])} hyperedges to {output_file}")
            else:
                # HyperNetXStorage - already in JSON format
                if hasattr(rag.rag_engine.chunk_entity_relation_graph, '_hypergraph_file'):
                    source_file = rag.rag_engine.chunk_entity_relation_graph._hypergraph_file
                    if os.path.exists(source_file):
                        import shutil
                        output_file = os.path.join(working_dir, "hypergraph_chunk_entity_relation.json")
                        shutil.copy2(source_file, output_file)
                        print(f"  ✅ Copied hypergraph from {source_file} to {output_file}")
        except Exception as e:
            print(f"  ⚠️  Error exporting hypergraph: {e}")
            logger.error(f"Error exporting hypergraph: {e}")
        
        # 2. Export kv_store_entity_chunks.json
        print("Exporting kv_store_entity_chunks.json...")
        try:
            entity_chunks_data = {}
            if hasattr(rag.rag_engine.entity_chunks, '_data'):
                # JsonKVStorage - access internal data
                data_dict = rag.rag_engine.entity_chunks._data
                if hasattr(data_dict, '_getvalue'):
                    data_dict = dict(data_dict._getvalue())
                entity_chunks_data = dict(data_dict)
            elif hasattr(rag.rag_engine.entity_chunks, '_file_name'):
                # Load from file if exists
                source_file = rag.rag_engine.entity_chunks._file_name
                if os.path.exists(source_file):
                    with open(source_file, 'r', encoding='utf-8') as f:
                        entity_chunks_data = json.load(f)
            
            output_file = os.path.join(working_dir, "kv_store_entity_chunks.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entity_chunks_data, f, ensure_ascii=False, indent=2)
            print(f"  ✅ Exported {len(entity_chunks_data)} entity chunks to {output_file}")
        except Exception as e:
            print(f"  ⚠️  Error exporting entity_chunks: {e}")
            logger.error(f"Error exporting entity_chunks: {e}")
        
        # 3. Export kv_store_full_entities.json
        print("Exporting kv_store_full_entities.json...")
        try:
            full_entities_data = {}
            if hasattr(rag.rag_engine.full_entities, '_data'):
                data_dict = rag.rag_engine.full_entities._data
                if hasattr(data_dict, '_getvalue'):
                    data_dict = dict(data_dict._getvalue())
                full_entities_data = dict(data_dict)
            elif hasattr(rag.rag_engine.full_entities, '_file_name'):
                source_file = rag.rag_engine.full_entities._file_name
                if os.path.exists(source_file):
                    with open(source_file, 'r', encoding='utf-8') as f:
                        full_entities_data = json.load(f)
            
            output_file = os.path.join(working_dir, "kv_store_full_entities.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_entities_data, f, ensure_ascii=False, indent=2)
            print(f"  ✅ Exported {len(full_entities_data)} full entities to {output_file}")
        except Exception as e:
            print(f"  ⚠️  Error exporting full_entities: {e}")
            logger.error(f"Error exporting full_entities: {e}")
        
        # 4. Export kv_store_full_relations.json
        print("Exporting kv_store_full_relations.json...")
        try:
            full_relations_data = {}
            if hasattr(rag.rag_engine.full_relations, '_data'):
                data_dict = rag.rag_engine.full_relations._data
                if hasattr(data_dict, '_getvalue'):
                    data_dict = dict(data_dict._getvalue())
                full_relations_data = dict(data_dict)
            elif hasattr(rag.rag_engine.full_relations, '_file_name'):
                source_file = rag.rag_engine.full_relations._file_name
                if os.path.exists(source_file):
                    with open(source_file, 'r', encoding='utf-8') as f:
                        full_relations_data = json.load(f)
            
            output_file = os.path.join(working_dir, "kv_store_full_relations.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_relations_data, f, ensure_ascii=False, indent=2)
            print(f"  ✅ Exported {len(full_relations_data)} full relations to {output_file}")
        except Exception as e:
            print(f"  ⚠️  Error exporting full_relations: {e}")
            logger.error(f"Error exporting full_relations: {e}")
        
        # 5. Export kv_store_relation_chunks.json (new)
        print("Exporting kv_store_relation_chunks.json...")
        try:
            relation_chunks_data = {}
            
            # Always build from hyperedges to ensure correct format
            # For hyperedges, use all entities joined by SEP as the key (matching hyperedges format)
            # Format: "entity1<SEP>entity2<SEP>entity3..." (e.g., "Breast Cancer<SEP>Ductal Carcinoma In Situ<SEP>Lobular Carcinoma In Situ")
            if hasattr(rag.rag_engine.chunk_entity_relation_graph, 'get_all_hyperedges'):
                from pathpocket.operate import GRAPH_FIELD_SEP
                hyperedges = await rag.rag_engine.chunk_entity_relation_graph.get_all_hyperedges()
                
                for edge in hyperedges:
                    entities = edge.get("entities", [])
                    source_id = edge.get("source_id", "")
                    
                    if not entities or not source_id:
                        continue
                    
                    # Ensure entities is a list (not string)
                    if isinstance(entities, str):
                        # If entities is a string, split by SEP
                        entities = [e.strip() for e in entities.split(GRAPH_FIELD_SEP) if e.strip()]
                    elif not isinstance(entities, list):
                        # If entities is not a list, convert to list
                        entities = list(entities) if entities else []
                    
                    if not entities:
                        continue
                    
                    # Extract chunk_ids from source_id
                    chunk_ids = [cid.strip() for cid in source_id.split(GRAPH_FIELD_SEP) if cid.strip()]
                    if not chunk_ids:
                        continue
                    
                    # Create key using all entities joined by SEP (matching hyperedges format)
                    # Format: "entity1<SEP>entity2<SEP>entity3..." (e.g., "Breast Cancer<SEP>Ductal Carcinoma In Situ<SEP>Lobular Carcinoma In Situ")
                    sorted_entities = sorted(entities)  # Sort for consistency
                    storage_key = GRAPH_FIELD_SEP.join(sorted_entities)
                    
                    # Merge chunk_ids if key already exists (same entity combination from different edges)
                    if storage_key in relation_chunks_data:
                        existing_chunk_ids = set(relation_chunks_data[storage_key].get("chunk_ids", []))
                        existing_chunk_ids.update(chunk_ids)
                        chunk_ids = sorted(list(existing_chunk_ids))
                    
                    relation_chunks_data[storage_key] = {
                        "chunk_ids": chunk_ids,
                        "count": len(chunk_ids),
                        "update_time": int(time_module.time()),
                        "_id": storage_key  # _id should match the key (all entities joined by SEP)
                    }
            else:
                # Fallback: Try to get data from storage if hyperedges not available
                if hasattr(rag.rag_engine.relation_chunks, '_data'):
                    data_dict = rag.rag_engine.relation_chunks._data
                    if hasattr(data_dict, '_getvalue'):
                        data_dict = dict(data_dict._getvalue())
                    relation_chunks_data = dict(data_dict)
                elif hasattr(rag.rag_engine.relation_chunks, '_file_name'):
                    source_file = rag.rag_engine.relation_chunks._file_name
                    if os.path.exists(source_file):
                        with open(source_file, 'r', encoding='utf-8') as f:
                            relation_chunks_data = json.load(f)
            
            output_file = os.path.join(working_dir, "kv_store_relation_chunks.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(relation_chunks_data, f, ensure_ascii=False, indent=2)
            print(f"  ✅ Exported {len(relation_chunks_data)} relation chunks to {output_file}")
        except Exception as e:
            print(f"  ⚠️  Error exporting relation_chunks: {e}")
            logger.error(f"Error exporting relation_chunks: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        logger.error(f"Error during export: {e}")
        import traceback
        logger.error(traceback.format_exc())


async def save_text_chunks_to_postgres(working_dir: str):
    """Save kv_store_text_chunks.json content to PostgreSQL DOC_CHUNKS table."""
    import asyncpg
    from pathlib import Path
    from datetime import datetime, timezone
    
    # Get PostgreSQL connection info
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "zhexu")
    password = os.getenv("POSTGRES_PASSWORD", "")
    database = os.getenv("POSTGRES_DATABASE", "pathrag")
    
    # Load text chunks from JSON file
    text_chunks_file = Path(working_dir) / "kv_store_text_chunks.json"
    workspace =  "default"

    if not text_chunks_file.exists():
        print(f"⚠️  Text chunks file not found: {text_chunks_file}")
        return
    
    print(f"\n{'='*60}")
    print("Saving text chunks to PostgreSQL DOC_CHUNKS table")
    print(f"{'='*60}")
    print(f"Working directory: {working_dir}")
    print(f"Text chunks file: {text_chunks_file}")
    
    try:
        # Load JSON data
        with open(text_chunks_file, 'r', encoding='utf-8') as f:
            text_chunks = json.load(f)
        
        if not text_chunks:
            print("⚠️  No text chunks found in JSON file")
            return
        
        print(f"Found {len(text_chunks)} text chunks in JSON file")
        
        # Connect to PostgreSQL
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        try:
            # Create DOC_CHUNKS table if it doesn't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS DOC_CHUNKS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    file_path TEXT NULL,
                    llm_cache_list JSONB NULL DEFAULT '[]'::jsonb,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT DOC_CHUNKS_PK PRIMARY KEY (workspace, id)
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_chunks_workspace ON DOC_CHUNKS (workspace);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_chunks_full_doc_id ON DOC_CHUNKS (full_doc_id);
            """)
            
            print("✅ DOC_CHUNKS table created/verified")
            
            # Get current time
            current_time = datetime.now(timezone.utc).replace(tzinfo=None)
            
            # Batch insert chunks
            batch_size = 100
            total = len(text_chunks)
            inserted_count = 0
            error_count = 0
            
            chunks_items = list(text_chunks.items())
            for i in range(0, total, batch_size):
                batch = chunks_items[i:i + batch_size]
                
                for chunk_id, chunk_data in batch:
                    try:
                        # Convert create_time and update_time from timestamp to datetime
                        create_time = current_time
                        update_time = current_time
                        
                        if "create_time" in chunk_data:
                            try:
                                create_time = datetime.fromtimestamp(
                                    chunk_data["create_time"], tz=timezone.utc
                                ).replace(tzinfo=None)
                            except (ValueError, TypeError, OSError):
                                pass
                        
                        if "update_time" in chunk_data:
                            try:
                                update_time = datetime.fromtimestamp(
                                    chunk_data["update_time"], tz=timezone.utc
                                ).replace(tzinfo=None)
                            except (ValueError, TypeError, OSError):
                                pass
                        
                        # Prepare llm_cache_list
                        llm_cache_list = chunk_data.get("llm_cache_list", [])
                        if not isinstance(llm_cache_list, list):
                            llm_cache_list = []
                        
                        await conn.execute("""
                            INSERT INTO DOC_CHUNKS 
                            (workspace, id, full_doc_id, chunk_order_index, tokens, content, file_path, llm_cache_list, create_time, update_time)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10)
                            ON CONFLICT (workspace, id) DO UPDATE
                            SET full_doc_id=EXCLUDED.full_doc_id,
                                chunk_order_index=EXCLUDED.chunk_order_index,
                                tokens=EXCLUDED.tokens,
                                content=EXCLUDED.content,
                                file_path=EXCLUDED.file_path,
                                llm_cache_list=EXCLUDED.llm_cache_list,
                                update_time=EXCLUDED.update_time
                        """, 
                            workspace,
                            chunk_id,
                            chunk_data.get("full_doc_id", ""),
                            chunk_data.get("chunk_order_index", 0),
                            chunk_data.get("tokens", 0),
                            chunk_data.get("content", ""),
                            chunk_data.get("file_path", ""),
                            json.dumps(llm_cache_list),
                            create_time,
                            update_time
                        )
                        inserted_count += 1
                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:  # Show first 5 errors
                            print(f"  ⚠️  Error writing chunk {chunk_id}: {e}")
                
                if i + batch_size < total:
                    print(f"  Processed {min(i + batch_size, total)}/{total} chunks...")
            
            print(f"\n✅ Text chunks saved to DOC_CHUNKS:")
            print(f"   Total in JSON: {total}")
            print(f"   Inserted/Updated: {inserted_count}")
            print(f"   Errors: {error_count}")
            
        finally:
            await conn.close()
            
    except Exception as e:
        print(f"❌ Error saving text chunks to PostgreSQL: {e}")
        import traceback
        traceback.print_exc()


async def main():
    # ========== Configuration (override via env) ==========
    _language = os.getenv("PIPELINE_LANGUAGE", "English")
    max_tokens = int(os.getenv("CHUNK_TOKEN_SIZE", "2400"))
    working_dir = os.getenv("WORKING_DIR", "./pathpocket_storage")
    
    custom_entity_types = [
        "Disease", "Symptom", "PathologicalFinding", "AnatomicalSite", "CellType",
        "HistologicalPattern", "Gene", "GeneticMutation", "Biomarker", "MolecularPathway",
        "Pathogen", "RiskFactor", "Pathogenesis", "DiagnosticMethod", "StagingMethod",
        "LabTest", "DiagnosticCriteria", "Treatment", "Drug", "MedicalDevice",
        "Prognosis", "Organization", "Location", "Person",
    ]
    
    print(f"\n{'='*60}")
    print("Stage 3 Optimized: Merge entities and relations")
    print(f"{'='*60}")
    print(f"Working directory: {working_dir}")
    
    # Display storage configuration
    graph_storage = os.getenv("GRAPH_STORAGE", "HyperNetXStorage")
    print(f"Graph Storage: {graph_storage}")
    if graph_storage == "PGHypergraphStorage":
        print(f"  PostgreSQL User: {os.getenv('POSTGRES_USER', 'Not set')}")
        print(f"  PostgreSQL Database: {os.getenv('POSTGRES_DATABASE', 'Not set')}")
        print(f"  PostgreSQL Host: {os.getenv('POSTGRES_HOST', 'localhost')}")
    print(f"{'='*60}\n")
    
    # Create config
    config = PathPocketConfig(
        working_dir=working_dir,
        enable_image_processing=True,
        enable_table_processing=True,
    )
    
    # Dummy LLM function
    async def dummy_llm_func(prompt, *args, **kwargs):
        return "Descriptions merged without LLM summarization in Stage 3"
    
    # Dummy embedding function
    class DummyEmbeddingFunc:
        def __init__(self):
            self.embedding_dim = 1024
            self.max_token_size = 4096
        
        async def __call__(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            num_texts = len(texts) if isinstance(texts, (list, tuple)) else 1
            unit_value = 1.0 / np.sqrt(self.embedding_dim)
            return np.full((num_texts, self.embedding_dim), unit_value, dtype=np.float32)
    
    dummy_embedding_func = DummyEmbeddingFunc()
    
    # Initialize PathPocket
    rag = PathPocket(
        config=config,
        llm_model_func=dummy_llm_func,
        embedding_func=dummy_embedding_func,
        rag_engine_kwargs={
            # Use PGHypergraphStorage for high performance with native hyperedge support
            # Requires: export POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE
            # Fallback to HyperNetXStorage if PostgreSQL not configured
            "graph_storage": os.getenv("GRAPH_STORAGE", "HyperNetXStorage"),  # Can be set to "PGHypergraphStorage"
            "vector_storage": "NanoVectorDBStorage",
            "addon_params": {
                "entity_types": custom_entity_types,
                "language": _language,
            },
            "chunk_token_size": max_tokens,
            "enable_llm_cache_for_entity_extract": False,
            "force_llm_summary_on_merge": 999999,
        }
    )
    
    # Fix invalid doc_status
    await fix_invalid_doc_status(rag)
    
    # # Save text chunks to PostgreSQL DOC_CHUNKS table
    # if os.getenv("GRAPH_STORAGE") == "PGHypergraphStorage":
    #     await save_text_chunks_to_postgres(working_dir)
    
    # Initialize RAG engine
    init_result = await rag._ensure_rag_engine_initialized()
    if not init_result.get("success"):
        raise RuntimeError(f"Failed to initialize RAG engine: {init_result.get('error')}")
    
    # Get documents to process
    from pathpocket.lightrag_base import DocStatus
    preprocessed_docs = await rag.rag_engine.doc_status.get_docs_by_status(DocStatus.PREPROCESSED)
    docs_to_process = list(preprocessed_docs.keys())
    
    print(f"Found {len(docs_to_process)} documents to process\n")
    
    if not docs_to_process:
        print("✅ No documents need merging!")
        return
    
    # Global config
    global_config = {
        **rag.rag_engine.__dict__,
        "llm_model_func": dummy_llm_func,
        "force_llm_summary_on_merge": 999999,
        "relation_max_async": 32,
        "llm_model_max_async": 96,
        "max_source_ids_per_relation": 30,
        "max_source_ids_per_entity": 100,
        "source_ids_limit_method": "KEEP",
        "entity_types": custom_entity_types,  # Add predefined entity types to global_config
        "addon_params": {
            "entity_types": custom_entity_types,
            "language": _language,
        },
    }
    
    # Process each document
    total_docs = len(docs_to_process)
    for idx, doc_id in enumerate(docs_to_process, 1):
        print(f"\n{'='*60}")
        print(f"Processing document {idx}/{total_docs}: {doc_id[:30]}...")
        print(f"{'='*60}")
        
        try:
            doc_status = await rag.rag_engine.doc_status.get_by_id(doc_id)
            if not doc_status:
                print(f"⚠️  Document {doc_id} not found in doc_status, skipping")
                continue
            
            chunks_list = doc_status.get("chunks_list", [])
            if not chunks_list:
                print(f"⚠️  No chunks_list found for document {doc_id}, skipping")
                continue
            
            print(f"Merging entities and relations from {len(chunks_list)} chunks...")
            
            # Get cached results
            cached_results = await _get_cached_extraction_results(
                llm_response_cache=rag.rag_engine.llm_response_cache,
                chunk_ids=set(chunks_list),
                text_chunks_storage=rag.rag_engine.text_chunks,
            )
            
            if not cached_results:
                print(f"⚠️  No cached LLM results found for document {doc_id}, skipping")
                continue
            
            # Batch fetch chunk data
            chunk_data_cache = {}
            try:
                if hasattr(rag.rag_engine.text_chunks, 'get_by_ids'):
                    all_chunk_data = await rag.rag_engine.text_chunks.get_by_ids(chunks_list)
                    chunk_data_cache = {chunk.get("_id") or chunk.get("chunk_id"): chunk 
                                       for chunk in all_chunk_data if chunk}
                else:
                    fetch_tasks = [rag.rag_engine.text_chunks.get_by_id(chunk_id) for chunk_id in chunks_list]
                    all_chunk_data = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                    chunk_data_cache = {chunks_list[i]: data for i, data in enumerate(all_chunk_data) 
                                       if not isinstance(data, Exception) and data}
            except Exception as e:
                print(f"⚠️  Warning: Batch fetch failed: {e}")
                for chunk_id in chunks_list:
                    try:
                        chunk_data = await rag.rag_engine.text_chunks.get_by_id(chunk_id)
                        if chunk_data:
                            chunk_data_cache[chunk_id] = chunk_data
                    except Exception:
                        continue
            
            # Collect multimodal entity names
            multimodal_entity_names = []
            for chunk_id, chunk_data in chunk_data_cache.items():
                modal_entity_name = chunk_data.get("modal_entity_name")
                if modal_entity_name and modal_entity_name not in multimodal_entity_names:
                    multimodal_entity_names.append(modal_entity_name)
            
            # Process document
            await process_document_optimized(
                doc_id=doc_id,
                chunks_list=chunks_list,
                rag=rag,
                cached_results=cached_results,
                chunk_data_cache=chunk_data_cache,
                multimodal_entity_names=multimodal_entity_names,
                global_config=global_config,
            )
            
            # Update doc_status to "processed"
            import time as time_module
            await rag.rag_engine.doc_status.upsert({
                doc_id: {
                    **doc_status,
                    "status": "processed",
                    "updated_at": int(time_module.time()),
                }
            })
            
            print(f"✅ Completed document {idx}/{total_docs} (status updated to 'processed')")
            
        except Exception as e:
            print(f"❌ Error processing document {doc_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("✅ Stage 3 Optimized completed!")
    print(f"{'='*60}\n")
    
    # # Export data to JSON files for inspection
    # print(f"\n{'='*60}")
    # print("Exporting data to JSON files...")
    # print(f"{'='*60}\n")
    # await export_to_json_files(rag, working_dir)
    # print(f"\n{'='*60}")
    # print("✅ Export completed!")
    # print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())

