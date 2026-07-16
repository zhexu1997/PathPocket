"""
Custom merge functions for PathPocket
Ensures multimodal entities are properly saved to full_entities and full_relations
"""

import asyncio
import time
from typing import List, Tuple, Dict, Any
from collections import defaultdict

# Import from PathPocket core and operate
from pathpocket.core import logger
from pathpocket.operate import merge_nodes_and_edges


async def merge_nodes_and_edges_with_multimodal(
    chunk_results: List[Tuple],
    knowledge_graph_inst,
    entity_vdb,
    relationships_vdb,
    global_config: Dict[str, Any],
    full_entities_storage,
    full_relations_storage,
    doc_id: str,
    pipeline_status: Dict = None,
    pipeline_status_lock=None,
    llm_response_cache=None,
    entity_chunks_storage=None,
    relation_chunks_storage=None,
    current_file_number: int = 0,
    total_files: int = 0,
    file_path: str = "unknown_source",
    multimodal_entity_names: List[str] = None,
):
    """
    Custom merge function that ensures multimodal entities are saved to full_entities
    
    Args:
        chunk_results: List of (maybe_nodes, maybe_edges) tuples
        multimodal_entity_names: List of multimodal entity names that must be saved
        ... (other args same as merge_nodes_and_edges)
    """
    # First, call the original merge function
    try:
        await merge_nodes_and_edges(
            chunk_results=chunk_results,
            knowledge_graph_inst=knowledge_graph_inst,
            entity_vdb=entity_vdb,
            relationships_vdb=relationships_vdb,
            global_config=global_config,
            full_entities_storage=full_entities_storage,
            full_relations_storage=full_relations_storage,
            doc_id=doc_id,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=llm_response_cache,
            entity_chunks_storage=entity_chunks_storage,
            relation_chunks_storage=relation_chunks_storage,
            current_file_number=current_file_number,
            total_files=total_files,
            file_path=file_path,
        )
    except Exception as e:
        logger.error(f"Error in merge_nodes_and_edges: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Then, ensure multimodal entities are in full_entities
    if multimodal_entity_names and full_entities_storage:
        try:
            logger.info(f"[Multimodal Merge] Checking multimodal entities for doc_id: {doc_id}")
            logger.info(f"[Multimodal Merge] Multimodal entity names ({len(multimodal_entity_names)}): {multimodal_entity_names[:5] if len(multimodal_entity_names) > 5 else multimodal_entity_names}")
            
            # Wait a bit for async operations to complete
            await asyncio.sleep(0.3)
            
            # Get current full_entities AFTER merge
            doc_entities = await full_entities_storage.get_by_id(doc_id)
            
            if doc_entities:
                existing_entity_names = set(doc_entities.get("entity_names", []))
                logger.info(f"[Multimodal Merge] Current full_entities has {len(existing_entity_names)} entities")
            else:
                existing_entity_names = set()
                logger.warning(f"[Multimodal Merge] No existing full_entities found for doc_id: {doc_id}, creating new entry")
            
            # Add multimodal entities that are missing
            multimodal_set = set(multimodal_entity_names)
            missing_multimodal = multimodal_set - existing_entity_names
            
            logger.info(f"[Multimodal Merge] Multimodal check: total={len(multimodal_set)}, existing={len(existing_entity_names & multimodal_set)}, missing={len(missing_multimodal)}")
            
            if missing_multimodal:
                logger.warning(f"[Multimodal Merge] Missing {len(missing_multimodal)} multimodal entities: {list(missing_multimodal)[:5]}")
                
                # Merge with existing entities
                final_entity_names = existing_entity_names | multimodal_set
                
                # Preserve existing metadata
                create_time = doc_entities.get("create_time") if doc_entities else int(time.time())
                
                # Update full_entities - use upsert which should merge/update
                # IMPORTANT: upsert replaces the entire document, so we must include ALL entities
                update_data = {
                    doc_id: {
                        "entity_names": sorted(list(final_entity_names)),  # Sort for consistency
                        "count": len(final_entity_names),
                        "create_time": create_time,
                        "update_time": int(time.time()),
                        "_id": doc_id,
                    }
                }
                logger.info(f"[Multimodal Merge] Upserting full_entities: {len(existing_entity_names)} existing + {len(missing_multimodal)} new = {len(final_entity_names)} total")
                logger.debug(f"[Multimodal Merge] Entity names sample: {list(final_entity_names)[:10]}")
                
                # Perform upsert
                await full_entities_storage.upsert(update_data)
                
                # Call index_done_callback to ensure data is persisted
                if hasattr(full_entities_storage, 'index_done_callback'):
                    await full_entities_storage.index_done_callback()
                
                # Force flush if available
                if hasattr(full_entities_storage, 'flush'):
                    await full_entities_storage.flush()
                
                # Verify update
                await asyncio.sleep(0.2)
                verify_entities = await full_entities_storage.get_by_id(doc_id)
                if verify_entities:
                    verify_names = set(verify_entities.get("entity_names", []))
                    verify_multimodal = verify_names & multimodal_set
                    if len(verify_multimodal) == len(multimodal_set):
                        logger.info(f"✅ [Multimodal Merge] Successfully added all {len(multimodal_set)} multimodal entities to full_entities")
                    else:
                        logger.error(f"❌ [Multimodal Merge] Only {len(verify_multimodal)}/{len(multimodal_set)} multimodal entities in full_entities after update!")
                        logger.error(f"   Missing: {list(multimodal_set - verify_names)[:5]}")
                else:
                    logger.error(f"❌ [Multimodal Merge] Verification failed: full_entities is None after update!")
            else:
                logger.info(f"[Multimodal Merge] ✅ All {len(multimodal_set)} multimodal entities already in full_entities")
            
            # Also ensure multimodal entities appear in relations if they have edges
            if full_relations_storage:
                try:
                    doc_relations = await full_relations_storage.get_by_id(doc_id)
                    if doc_relations:
                        existing_relation_pairs = set(
                            tuple(pair) if isinstance(pair, list) else pair
                            for pair in doc_relations.get("relation_pairs", [])
                        )
                    else:
                        existing_relation_pairs = set()
                    
                    # Check if any multimodal entities have relations
                    # Get all edges involving multimodal entities from knowledge graph
                    multimodal_relations = set()
                    for modal_entity_name in multimodal_set:
                        try:
                            edges = await knowledge_graph_inst.get_node_edges(modal_entity_name)
                            for src_id, tgt_id in edges:
                                # Add both directions
                                multimodal_relations.add(tuple(sorted([src_id, tgt_id])))
                        except Exception as e:
                            logger.debug(f"Could not get edges for {modal_entity_name}: {e}")
                    
                    if multimodal_relations:
                        missing_relations = multimodal_relations - existing_relation_pairs
                        if missing_relations:
                            logger.info(f"Adding {len(missing_relations)} missing multimodal relations to full_relations")
                            final_relation_pairs = existing_relation_pairs | multimodal_relations
                            
                            create_time = doc_relations.get("create_time") if doc_relations else int(time.time())
                            
                            await full_relations_storage.upsert(
                                {
                                    doc_id: {
                                        "relation_pairs": [list(pair) for pair in final_relation_pairs],
                                        "count": len(final_relation_pairs),
                                        "create_time": create_time,
                                        "update_time": int(time.time()),
                                        "_id": doc_id,
                                    }
                                }
                            )
                            logger.info(f"✅ Added {len(missing_relations)} multimodal relations to full_relations")
                except Exception as e:
                    logger.warning(f"Error ensuring multimodal relations in full_relations: {e}")
                
        except Exception as e:
            logger.error(f"Error ensuring multimodal entities in full_entities: {e}")
            import traceback
            logger.error(traceback.format_exc())
