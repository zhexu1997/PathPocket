"""
PathPocket Processor Module
Handles multimodal content processing using caption-based extraction (no vision model)
"""

import os
import asyncio
import time
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import time as time_module

from pathpocket.core import logger, compute_mdhash_id
from pathpocket.operate import extract_entities, chunking_by_token_size
from pathpocket.merge import merge_nodes_and_edges_with_multimodal


class ProcessorMixin:
    """ProcessorMixin for PathPocket multimodal content processing
    Copied from raganything.processor and adapted for PathPocket
    """

    async def _process_multimodal_content(
        self,
        multimodal_items: List[Dict[str, Any]],
        file_path: str,
        doc_id: str,
        pipeline_status: Optional[Any] = None,
        pipeline_status_lock: Optional[Any] = None,
    ):
        """
        Process multimodal content (using specialized processors)

        Args:
            multimodal_items: List of multimodal items
            file_path: File path (for reference)
            doc_id: Document ID for proper chunk association
            pipeline_status: Pipeline status object
            pipeline_status_lock: Pipeline status lock
        """
        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

        # Check multimodal processing status
        try:
            existing_doc_status = await self.rag_engine.doc_status.get_by_id(doc_id)
            if existing_doc_status:
                multimodal_processed = existing_doc_status.get(
                    "multimodal_processed", False
                )
                if multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} multimodal content is already processed"
                    )
                    return
        except Exception as e:
            self.logger.debug(f"Error checking document status for {doc_id}: {e}")

        # Use batch processing
        try:
            await self._ensure_rag_engine_initialized()
            await self._process_multimodal_content_batch(
                multimodal_items=multimodal_items, file_path=file_path, doc_id=doc_id
            )
            await self._mark_multimodal_processing_complete(doc_id)
        except Exception as e:
            self.logger.error(f"Error in multimodal processing: {e}")
            raise

    async def _mark_multimodal_processing_complete(self, doc_id: str):
        """Mark multimodal content processing as complete in the document status."""
        try:
            current_doc_status = await self.rag_engine.doc_status.get_by_id(doc_id)
            if current_doc_status:
                await self.rag_engine.doc_status.upsert(
                    {
                        doc_id: {
                            **current_doc_status,
                            "multimodal_processed": True,
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        }
                    }
                )
                await self.rag_engine.doc_status.index_done_callback()
                self.logger.debug(
                    f"Marked multimodal content processing as complete for document {doc_id}"
                )
        except Exception as e:
            self.logger.warning(
                f"Error marking multimodal processing as complete for document {doc_id}: {e}"
            )

    def _get_processor_for_type(self, content_type: str):
        """Get the appropriate processor for content type"""
        if content_type == "image" and "image" in self.modal_processors:
            return self.modal_processors["image"]
        elif content_type == "table" and "table" in self.modal_processors:
            return self.modal_processors["table"]
        else:
            return self.modal_processors.get("generic")

    def _apply_chunk_template(
        self, content_type: str, original_item: Dict[str, Any], description: str
    ) -> str:
        """Apply the appropriate chunk template based on content type (simplified, no duplicate content)"""
        try:
            if content_type == "image":
                image_path = original_item.get("img_path", "")
                # Description already contains captions and footnotes, no need to repeat
                # Include image path for reference
                if image_path:
                    return f"Image Content (Path: {image_path}):\n{description}"
                else:
                    return f"Image Content:\n{description}"

            elif content_type == "table":
                table_img_path = original_item.get("img_path", "")
                table_body = original_item.get("table_body", "")
                # Description already contains caption and footnotes
                # Include table body if not already in description
                parts = [description]
                if table_body and table_body not in description:
                    parts.append(f"Table Data: {table_body}")
                if table_img_path:
                    parts.insert(0, f"Table Content (Path: {table_img_path}):")
                return "\n".join(parts)

            else:
                content = str(original_item.get("content", original_item))
                return f"{content_type.title()} Content:\nContent: {description}"

        except Exception as e:
            self.logger.warning(
                f"Error applying chunk template for {content_type}: {e}"
            )
            return description

    async def _process_multimodal_content_batch(
        self, multimodal_items: List[Dict[str, Any]], file_path: str, doc_id: str
    ):
        """
        Batch process multimodal content using caption-based extraction (no vision model)

        Args:
            multimodal_items: List of multimodal items with different types
            file_path: File path for citation
            doc_id: Document ID for proper association
        """
        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

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
                    return
        except Exception as e:
            self.logger.debug(f"Error checking document status for {doc_id}: {e}")
            # Continue with processing if check fails

        # Get existing chunks count for proper order indexing
        try:
            existing_doc_status = await self.rag_engine.doc_status.get_by_id(doc_id)
            existing_chunks_count = (
                existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
            )
        except Exception:
            existing_chunks_count = 0

        # Use RAG core's concurrency control for description generation
        # Note: This only controls description generation (which doesn't use LLM now)
        # Entity extraction uses llm_model_max_async from global_config
        semaphore = asyncio.Semaphore(
            getattr(self.rag_engine, "max_parallel_insert", 2)
        )

        # Progress tracking
        total_items = len(multimodal_items)
        completed_count = 0
        progress_lock = asyncio.Lock()

        self.logger.info(f"Starting to process {total_items} multimodal content items")

        # Stage 1: Generate descriptions using caption-based processors
        async def process_single_item(item: Dict[str, Any], index: int, file_path: str):
            """Process single item using the correct processor"""
            nonlocal completed_count
            async with semaphore:
                try:
                    content_type = item.get("type", "unknown")

                    # Select the correct processor
                    processor = self._get_processor_for_type(content_type)

                    if not processor:
                        self.logger.warning(
                            f"No processor found for type: {content_type}"
                        )
                        return None

                    item_info = {
                        "page_idx": item.get("page_idx", 0),
                        "index": index,
                        "type": content_type,
                    }

                    # Generate description based on caption (no vision model)
                    description, entity_info = await processor.generate_description_only(
                        modal_content=item,
                        content_type=content_type,
                        item_info=item_info,
                        entity_name=None,
                    )

                    # Update progress
                    async with progress_lock:
                        completed_count += 1
                        if (
                            completed_count % max(1, total_items // 10) == 0
                            or completed_count == total_items
                        ):
                            progress_percent = (completed_count / total_items) * 100
                            self.logger.info(
                                f"Multimodal description generation: {completed_count}/{total_items} ({progress_percent:.1f}%)"
                            )

                    return {
                        "index": index,
                        "content_type": content_type,
                        "description": description,
                        "entity_info": entity_info,
                        "original_item": item,
                        "item_info": item_info,
                        "chunk_order_index": existing_chunks_count + index,
                        "file_path": file_path,
                    }

                except Exception as e:
                    async with progress_lock:
                        completed_count += 1
                    self.logger.error(
                        f"Error processing {content_type} item {index}: {e}"
                    )
                    return None

        # Process all items concurrently
        tasks = [
            asyncio.create_task(process_single_item(item, i, file_path))
            for i, item in enumerate(multimodal_items)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        multimodal_data_list = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Task failed: {result}")
                continue
            if result is not None:
                multimodal_data_list.append(result)

        if not multimodal_data_list:
            self.logger.warning("No valid multimodal descriptions generated")
            return

        self.logger.info(
            f"Generated descriptions for {len(multimodal_data_list)}/{len(multimodal_items)} multimodal items"
        )

        # Stage 2: Convert to RAG core chunks format
        rag_chunks = self._convert_to_rag_chunks(
            multimodal_data_list, file_path, doc_id
        )

        # Stage 3: Store chunks to RAG core storage
        await self._store_chunks_to_rag_storage(rag_chunks)

        # Stage 3.5: Extract and store Virchow2 features for ALL images
        # Note: Text vectors are already stored in chunks_vdb (Stage 3)
        # This stage stores Virchow2 visual features in pathology_images_vdb
        # Each image will have TWO vectors: text vector (chunks_vdb) + Virchow2 vector (pathology_images_vdb)
        await self._extract_and_store_conch_features(
            multimodal_data_list, rag_chunks, file_path
        )

        # Stage 4: Store multimodal main entities
        await self._store_multimodal_main_entities(
            multimodal_data_list, rag_chunks, file_path, doc_id
        )

        # Track chunk IDs
        chunk_ids = list(rag_chunks.keys())

        # Stage 5: Extract entities and relations using RAG core
        chunk_results = await self._batch_extract_entities(rag_chunks)

        # Stage 6: Add belongs_to relations
        enhanced_chunk_results = await self._batch_add_belongs_to_relations(
            chunk_results, multimodal_data_list, rag_chunks
        )

        # Stage 7: Merge entities and relations
        await self._batch_merge(enhanced_chunk_results, file_path, doc_id)

        # Stage 8: Update doc_status
        await self._update_doc_status(doc_id, chunk_ids)

        # Stage 9: Mark multimodal processing as complete
        await self._mark_multimodal_processing_complete(doc_id)
        
        # Stage 10: Save all storages to disk
        await self.rag_engine.text_chunks.index_done_callback()
        if self.rag_engine.chunks_vdb:
            await self.rag_engine.chunks_vdb.index_done_callback()
        if self.rag_engine.entities_vdb:
            await self.rag_engine.entities_vdb.index_done_callback()
        if self.rag_engine.relationships_vdb:
            await self.rag_engine.relationships_vdb.index_done_callback()
        if self.rag_engine.pathology_images_vdb:
            await self.rag_engine.pathology_images_vdb.index_done_callback()
        if self.rag_engine.full_docs:
            await self.rag_engine.full_docs.index_done_callback()
        await self.rag_engine.full_entities.index_done_callback()
        await self.rag_engine.full_relations.index_done_callback()
        if self.rag_engine.entity_chunks:
            await self.rag_engine.entity_chunks.index_done_callback()
        if self.rag_engine.relation_chunks:
            await self.rag_engine.relation_chunks.index_done_callback()
        if self.rag_engine.chunk_entity_relation_graph:
            await self.rag_engine.chunk_entity_relation_graph.index_done_callback()
        await self.rag_engine.doc_status.index_done_callback()

    async def _process_text_content(
        self,
        text: str,
        file_path: str,
        doc_id: str,
        split_by_character: str = "\n\n",
        split_by_character_only: bool = False,
    ):
        """Process text content: chunk, store, extract entities, and merge"""
        # Chunk text
        chunk_token_size = getattr(self.rag_core, "chunk_token_size", 1200)
        chunk_overlap_token_size = getattr(self.rag_core, "chunk_overlap_token_size", 100)
        
        chunks_list = chunking_by_token_size(
            tokenizer=self.rag_engine.tokenizer,
            content=text,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            chunk_overlap_token_size=chunk_overlap_token_size,
            chunk_token_size=chunk_token_size,
        )
        
        # Convert to chunks dict format
        chunks = {}
        for chunk_data in chunks_list:
            chunk_id = compute_mdhash_id(chunk_data["content"], prefix="chunk-")
            chunks[chunk_id] = {
                **chunk_data,
                "full_doc_id": doc_id,
                "file_path": os.path.basename(file_path),
                "llm_cache_list": [],
            }
        
        # Store chunks
        await self.rag_engine.text_chunks.upsert(chunks)
        
        # Store full document content in full_docs
        if self.rag_engine.full_docs:
            full_doc_content = text  # Store original text
            await self.rag_engine.full_docs.upsert({
                doc_id: {
                    "content": full_doc_content,
                    "file_path": file_path,
                    "chunks_count": len(chunks),
                    "create_time": int(time_module.time()),
                    "update_time": int(time_module.time()),
                    "_id": doc_id,
                }
            })
        
        # Store to vector DB
        if self.rag_engine.chunks_vdb and self.rag_engine.embedding_func:
            batch_size = 5
            chunk_items = list(chunks.items())
            for i in range(0, len(chunk_items), batch_size):
                batch = dict(chunk_items[i:i + batch_size])
                batch_for_vdb = {}
                for cid, chunk_data in batch.items():
                    batch_for_vdb[cid] = {
                        "content": chunk_data["content"],
                        "source_id": cid,
                        "file_path": chunk_data.get("file_path", file_path),
                    }
                try:
                    await self.rag_engine.chunks_vdb.upsert(batch_for_vdb)
                except Exception as e:
                    self.logger.warning(f"Error storing chunks to VDB: {e}")
        
        # Extract entities
        chunk_results = await self._batch_extract_entities(chunks)
        
        # Merge entities and relations
        await self._batch_merge(chunk_results, file_path, doc_id)
        
        # Update doc_status
        await self._update_doc_status(doc_id, list(chunks.keys()))
        
        # Save all storages
        await self.rag_engine.text_chunks.index_done_callback()
        if self.rag_engine.chunks_vdb:
            await self.rag_engine.chunks_vdb.index_done_callback()
        if self.rag_engine.full_docs:
            await self.rag_engine.full_docs.index_done_callback()
        await self.rag_engine.full_entities.index_done_callback()
        await self.rag_engine.full_relations.index_done_callback()
        if self.rag_engine.entity_chunks:
            await self.rag_engine.entity_chunks.index_done_callback()
        if self.rag_engine.relation_chunks:
            await self.rag_engine.relation_chunks.index_done_callback()
        if self.rag_engine.chunk_entity_relation_graph:
            await self.rag_engine.chunk_entity_relation_graph.index_done_callback()
        await self.rag_engine.doc_status.index_done_callback()
    
    def _convert_to_rag_chunks(
        self, multimodal_data_list: List[Dict[str, Any]], file_path: str, doc_id: str
    ) -> Dict[str, Any]:
        """Convert multimodal data to RAG core standard chunks format"""
        chunks = {}

        for data in multimodal_data_list:
            description = data["description"]
            entity_info = data["entity_info"]
            chunk_order_index = data["chunk_order_index"]
            content_type = data["content_type"]
            original_item = data["original_item"]
            entity_name = entity_info["entity_name"]

            # Apply chunk template
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )
            
            # Ensure entity name appears in chunk content so LLM can extract it
            # Add entity name at the beginning to make it prominent
            if entity_name not in formatted_chunk_content:
                formatted_chunk_content = f"Entity: {entity_name}\n\n{formatted_chunk_content}"

            # Generate chunk_id
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # Calculate tokens
            tokens = len(self.rag_engine.tokenizer.encode(formatted_chunk_content))

            # Build RAG core standard chunk format
            chunks[chunk_id] = {
                "content": formatted_chunk_content,
                "tokens": tokens,
                "full_doc_id": doc_id,
                "chunk_order_index": chunk_order_index,
                "file_path": os.path.basename(file_path),
                "llm_cache_list": [],
                "is_multimodal": True,
                "modal_entity_name": entity_name,
                "original_type": content_type,
                "page_idx": data["item_info"].get("page_idx", 0),
            }

        return chunks

    async def _store_chunks_to_rag_storage(self, chunks: Dict[str, Any]):
        """Store chunks to RAG core storage with error handling for embedding timeout"""
        try:
            await self.rag_engine.text_chunks.upsert(chunks)
            
            # Store chunks to vector DB with retry and batch processing
            # Split into smaller batches to avoid timeout
            if self.rag_engine.chunks_vdb and self.rag_engine.embedding_func:
                chunk_items = list(chunks.items())
                batch_size = 5  # Process 5 chunks at a time
                
                for i in range(0, len(chunk_items), batch_size):
                    batch = dict(chunk_items[i:i + batch_size])
                    batch_for_vdb = {}
                    for cid, chunk_data in batch.items():
                        batch_for_vdb[cid] = {
                            "content": chunk_data["content"],
                            "source_id": cid,
                            "file_path": chunk_data.get("file_path", "unknown"),
                        }
                    try:
                        await self.rag_engine.chunks_vdb.upsert(batch_for_vdb)
                        self.logger.debug(f"Stored batch {i//batch_size + 1} of multimodal chunks ({len(batch)} chunks)")
                    except Exception as batch_error:
                        self.logger.warning(f"Error storing batch {i//batch_size + 1}: {batch_error}")
                        # Continue with next batch instead of failing completely
                        continue
            
            self.logger.debug(f"Stored {len(chunks)} multimodal chunks to storage")
        except Exception as e:
            self.logger.error(f"Error storing chunks: {e}")
            # Don't raise - allow processing to continue even if some chunks fail
            self.logger.warning("Continuing despite chunk storage error")

    async def _extract_and_store_conch_features(
        self,
        multimodal_data_list: List[Dict[str, Any]],
        rag_chunks: Dict[str, Any],
        file_path: str,
    ):
        """Extract and store Virchow2 features for ALL images (not just pathology images)"""
        try:
            # Check if Virchow2 feature extraction is available
            image_processor = self.modal_processors.get("image")
            if not image_processor or not hasattr(image_processor, "conch_feature_func"):
                self.logger.debug("Image processor or conch_feature_func not available, skipping Virchow2 extraction")
                return
                
            if not image_processor.conch_feature_func:
                self.logger.debug("conch_feature_func is None, skipping Virchow2 extraction")
                return
                
            if not self.rag_engine.pathology_images_vdb:
                self.logger.warning("pathology_images_vdb is not initialized, skipping Virchow2 extraction")
                self.logger.warning("This may happen if conch_feature_func was not provided in addon_params")
                return
            
            self.logger.info(f"Starting Virchow2 feature extraction for images in {file_path}")
            
            # Process each image item - process ALL images, not just pathology images
            images_processed = 0
            for data in multimodal_data_list:
                if data["content_type"] != "image":
                    continue
                    
                original_item = data["original_item"]
                entity_info = data["entity_info"]
                entity_name = entity_info["entity_name"]
                
                # Parse image content
                if isinstance(original_item, str):
                    try:
                        import json
                        content_data = json.loads(original_item)
                    except json.JSONDecodeError:
                        content_data = {"description": original_item}
                else:
                    content_data = original_item
                    
                image_path = content_data.get("img_path", "")
                if not image_path:
                    continue
                
                # Process ALL images - removed pathology image check
                # All images will get Virchow2 features extracted
                
                # Find corresponding chunk_id
                chunk_id = None
                for cid, chunk_data in rag_chunks.items():
                    if chunk_data.get("modal_entity_name") == entity_name:
                        chunk_id = cid
                        break
                
                if chunk_id:
                    # Extract and store Virchow2 features for this image
                    success = await image_processor.extract_and_store_conch_features(
                        image_path=image_path,
                        chunk_id=chunk_id,
                        entity_name=entity_name,
                        file_path=file_path,
                    )
                    if success:
                        images_processed += 1
            
            if images_processed > 0:
                # Save image features to disk
                await self.rag_engine.pathology_images_vdb.index_done_callback()
                # Get the file path where features are saved
                vdb_file = getattr(self.rag_engine.pathology_images_vdb, "_vdb_file", "unknown")
                self.logger.info(
                    f"Extracted and stored Virchow2 features for {images_processed} images "
                    f"(text vectors already stored in chunks_vdb)"
                )
                self.logger.info(f"Virchow2 features saved to: {vdb_file}")
            else:
                self.logger.warning("No images were processed for Virchow2 feature extraction")
                
        except Exception as e:
            self.logger.error(f"Error extracting/storing Virchow2 features: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Don't raise - allow processing to continue even if Virchow2 extraction fails
            self.logger.warning("Continuing despite Virchow2 feature extraction error")

    async def _store_multimodal_main_entities(
        self,
        multimodal_data_list: List[Dict[str, Any]],
        rag_chunks: Dict[str, Any],
        file_path: str,
        doc_id: str,
    ):
        """Store multimodal main entities to entities_vdb and full_entities"""
        try:
            entities_to_store = {}
            entity_names_list = []

            for data in multimodal_data_list:
                entity_info = data["entity_info"]
                entity_name = entity_info["entity_name"]
                entity_description = entity_info.get("summary", "")

                # Find corresponding chunk_id
                chunk_id = None
                for cid, chunk_data in rag_chunks.items():
                    if chunk_data.get("modal_entity_name") == entity_name:
                        chunk_id = cid
                        break

                if chunk_id:
                    # Generate entity_vdb_id using compute_mdhash_id with "ent-" prefix
                    entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
                    
                    # Create entity content (required by entities_vdb)
                    entity_content = f"{entity_name}\n{entity_description}"
                    
                    # Format entity data according to RAG core's expected format
                    entity_data = {
                        "content": entity_content,  # Required field
                        "entity_name": entity_name,
                        "entity_type": entity_info.get("entity_type", "generic"),
                        "description": entity_description,
                        "source_id": chunk_id,
                        "file_path": file_path,
                    }
                    
                    # Use entity_vdb_id as key (not entity_name)
                    entities_to_store[entity_vdb_id] = entity_data
                    entity_names_list.append(entity_name)

            # Store entities to knowledge graph, entities_vdb, and full_entities
            if entities_to_store:
                import time
                
                # Store entities in knowledge graph
                for entity_vdb_id, entity_data in entities_to_store.items():
                    entity_name = entity_data["entity_name"]
                    
                    # Create node data for knowledge graph
                    node_data = {
                        "entity_id": entity_name,
                        "entity_type": entity_data["entity_type"],
                        "description": entity_data["description"],
                        "source_id": entity_data["source_id"],
                        "file_path": os.path.basename(entity_data["file_path"]),
                        "created_at": int(time.time()),
                    }
                    
                    # Store in knowledge graph
                    await self.rag_engine.chunk_entity_relation_graph.upsert_node(
                        entity_name, node_data
                    )
                
                # Store in entities_vdb
                if self.rag_engine.entities_vdb:
                    await self.rag_engine.entities_vdb.upsert(entities_to_store)
                    await self.rag_engine.entities_vdb.index_done_callback()

                # Note: full_entities will be automatically updated by merge_nodes_and_edges
                # So we don't need to manually update it here to avoid duplication

            self.logger.debug(
                f"Stored {len(entities_to_store)} multimodal main entities to knowledge graph, entities_vdb, and full_entities"
            )

        except Exception as e:
            self.logger.error(f"Error storing multimodal entities: {e}")
            raise

    async def _batch_extract_entities(
        self, rag_chunks: Dict[str, Any]
    ) -> List[Tuple]:
        """Use RAG core's extract_entities for batch entity relation extraction"""
        chunk_results = await extract_entities(
            chunks=rag_chunks,
            global_config=self.rag_engine.__dict__,
            pipeline_status=None,
            pipeline_status_lock=None,
            llm_response_cache=self.rag_engine.llm_response_cache,
            text_chunks_storage=self.rag_engine.text_chunks,
        )

        self.logger.info(
            f"Extracted entities from {len(rag_chunks)} multimodal chunks "
            f"(concurrency: {self.rag_engine.llm_model_max_async})"
        )
        return chunk_results

    async def _batch_add_belongs_to_relations(
        self, 
        chunk_results: List[Tuple], 
        multimodal_data_list: List[Dict[str, Any]],
        rag_chunks: Dict[str, Any]
    ) -> List[Tuple]:
        """Add belongs_to relations for multimodal entities and ensure multimodal main entities are included.
        Also adds multimodal entities to existing n-ary relations extracted from the chunk."""
        # Create mapping from chunk_id to modal_entity_name using rag_chunks
        chunk_to_modal_entity = {}
        chunk_to_entity_info = {}
        for chunk_id, chunk_data in rag_chunks.items():
            modal_entity_name = chunk_data.get("modal_entity_name")
            if modal_entity_name:
                chunk_to_modal_entity[chunk_id] = modal_entity_name
                # Find corresponding entity_info
                for data in multimodal_data_list:
                    if data["entity_info"]["entity_name"] == modal_entity_name:
                        chunk_to_entity_info[chunk_id] = data["entity_info"]
                        break
        
        # Debug: Log multimodal entities found
        if chunk_to_modal_entity:
            self.logger.info(f"Found {len(chunk_to_modal_entity)} multimodal chunks with entity names: {list(chunk_to_modal_entity.values())[:5]}")
        else:
            self.logger.warning("No multimodal chunks found in rag_chunks!")
        
        # Add belongs_to relations and ensure multimodal main entities are in the results
        enhanced_results = []
        # chunk_results is a list of tuples: (maybe_nodes, maybe_edges)
        # We need to match each result with its corresponding chunk_id
        chunk_ids_list = list(rag_chunks.keys())
        
        # Debug: Check if we have multimodal chunks
        if not chunk_to_modal_entity:
            self.logger.warning("No multimodal chunks found! Cannot add multimodal entities.")
        
        for idx, chunk_result in enumerate(chunk_results):
            # Unified format: (nodes, relations)
            maybe_nodes, maybe_relations = chunk_result
            # Get corresponding chunk_id (assuming same order)
            if idx < len(chunk_ids_list):
                chunk_id = chunk_ids_list[idx]
                modal_entity_name = chunk_to_modal_entity.get(chunk_id)
                entity_info = chunk_to_entity_info.get(chunk_id)
                
                if modal_entity_name:
                    # Ensure multimodal main entity is in maybe_nodes (even if LLM didn't extract it)
                    if modal_entity_name not in maybe_nodes:
                        # Add multimodal main entity to nodes
                        # Format matches RAG core's entity extraction output
                        # Ensure description is not empty (required by _merge_nodes_then_upsert)
                        entity_description = entity_info.get("summary", "") if entity_info else ""
                        if not entity_description.strip():
                            # Use a fallback description if summary is empty
                            entity_description = f"{entity_info.get('entity_type', 'multimodal')} content"
                        
                        entity_data = {
                            "entity_name": modal_entity_name,
                            "entity_type": entity_info.get("entity_type", "generic") if entity_info else "generic",
                            "description": entity_description,
                            "source_id": chunk_id,
                            "file_path": rag_chunks[chunk_id].get("file_path", "unknown"),
                            "timestamp": int(time.time()),
                        }
                        maybe_nodes[modal_entity_name] = [entity_data]
                        self.logger.info(f"Added multimodal main entity '{modal_entity_name}' to extraction results: {entity_data}")
                    else:
                        self.logger.debug(f"Multimodal main entity '{modal_entity_name}' already in maybe_nodes")
                    
                    # Step 1: Add multimodal entity to existing n-ary relations (entity_count > 2)
                    # This allows image/table entities to participate in multi-entity relations
                    relations_to_update = []
                    for edge_key, relation_list in maybe_relations.items():
                        for relation in relation_list:
                            # Check if this is an n-ary relation (more than 2 entities)
                            entity_count = relation.get("entity_count", len(relation.get("entities", [])))
                            entities = relation.get("entities", [])
                            
                            if entity_count > 2 and modal_entity_name not in entities:
                                # This is an n-ary relation that doesn't include the multimodal entity yet
                                # Add the multimodal entity to this relation
                                new_entities = entities + [modal_entity_name]
                                new_entity_count = len(new_entities)
                                
                                # Create updated relation
                                updated_relation = relation.copy()
                                updated_relation["entities"] = new_entities
                                updated_relation["entity_count"] = new_entity_count
                                
                                # Update description to mention the multimodal entity
                                original_desc = relation.get("description", "")
                                if modal_entity_name not in original_desc:
                                    updated_relation["description"] = f"{original_desc} (context: {modal_entity_name})"
                                
                                # Update edge_id to reflect new entity count
                                # Generate new edge_id based on sorted entities
                                original_edge_id = relation.get("edge_id", "")
                                # Sanitize modal_entity_name for use in edge_id (remove special chars)
                                sanitized_modal_name = modal_entity_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                                if original_edge_id:
                                    # Try to preserve original edge_id structure if possible
                                    updated_relation["edge_id"] = f"{original_edge_id}_with_{sanitized_modal_name}"
                                else:
                                    # Generate new edge_id
                                    updated_relation["edge_id"] = f"rel_{chunk_id}_{hash('_'.join(sorted(new_entities))) % 10000}"
                                
                                relations_to_update.append((edge_key, relation, updated_relation))
                    
                    # Apply updates to relations
                    for old_edge_key, old_relation, updated_relation in relations_to_update:
                        # Remove old relation from the list
                        if old_relation in maybe_relations[old_edge_key]:
                            maybe_relations[old_edge_key].remove(old_relation)
                            if not maybe_relations[old_edge_key]:
                                del maybe_relations[old_edge_key]
                        
                        # Add updated relation with new edge_key (sorted entities)
                        new_edge_key = tuple(sorted(updated_relation["entities"]))
                        if new_edge_key not in maybe_relations:
                            maybe_relations[new_edge_key] = []
                        maybe_relations[new_edge_key].append(updated_relation)
                        
                        self.logger.debug(
                            f"Added multimodal entity '{modal_entity_name}' to n-ary relation "
                            f"({old_relation.get('entity_count', 0)} -> {updated_relation['entity_count']} entities)"
                        )
                    
                    if relations_to_update:
                        self.logger.info(
                            f"Updated {len(relations_to_update)} n-ary relations to include multimodal entity '{modal_entity_name}'"
                        )
                    
                    # Step 2: Track which entities are already in n-ary relations with the multimodal entity
                    # This helps us decide whether to create binary belongs_to relations
                    entities_in_nary_relations = set()
                    for edge_key, relation_list in maybe_relations.items():
                        for relation in relation_list:
                            entities = relation.get("entities", [])
                            entity_count = relation.get("entity_count", len(entities))
                            # If this relation includes the multimodal entity and has more than 2 entities,
                            # all other entities in this relation are already connected via n-ary relation
                            if modal_entity_name in entities and entity_count > 2:
                                entities_in_nary_relations.update(entities)
                    
                    # Step 3: Add belongs_to relations only for entities NOT already in n-ary relations
                    # This avoids redundancy while ensuring all entities are connected to the multimodal entity
                    entities_needing_binary_relation = []
                    for entity_name in maybe_nodes.keys():
                        if entity_name != modal_entity_name and entity_name not in entities_in_nary_relations:
                            entities_needing_binary_relation.append(entity_name)
                    
                    if entities_needing_binary_relation:
                        self.logger.debug(
                            f"Creating binary belongs_to relations for {len(entities_needing_binary_relation)} entities "
                            f"not in n-ary relations with '{modal_entity_name}'"
                        )
                    
                    for entity_name in entities_needing_binary_relation:
                        belongs_to_relation = {
                            "edge_id": f"rel_belongs_{entity_name}_{modal_entity_name}",
                            "entities": [entity_name, modal_entity_name],
                            "entity_count": 2,
                            "description": f"Entity {entity_name} belongs to {modal_entity_name}",
                            "keywords": "belongs_to,part_of,contained_in",
                            "weight": 10.0,
                        }
                        # Add to maybe_relations with sorted tuple key
                        edge_key = tuple(sorted([entity_name, modal_entity_name]))
                        if edge_key not in maybe_relations:
                            maybe_relations[edge_key] = []
                        maybe_relations[edge_key].append(belongs_to_relation)
                    
                    # Log summary
                    total_entities = len([e for e in maybe_nodes.keys() if e != modal_entity_name])
                    if total_entities > 0:
                        # Count entities in n-ary relations (excluding modal_entity_name itself)
                        entities_in_nary = entities_in_nary_relations - {modal_entity_name}
                        nary_count = len(entities_in_nary)
                        binary_count = len(entities_needing_binary_relation)
                        self.logger.info(
                            f"Entity connection summary for '{modal_entity_name}': "
                            f"{nary_count} entities in n-ary relations, "
                            f"{binary_count} entities in binary belongs_to relations, "
                            f"{total_entities} total entities"
                        )

            enhanced_results.append((maybe_nodes, maybe_relations))

        return enhanced_results

    async def _batch_merge(
        self, chunk_results: List[Tuple], file_path: str, doc_id: str
    ):
        """Merge entities and relations using RAG core's merge function"""
        # Debug: Log all entities in chunk_results before merging
        all_entities = set()
        for chunk_result in chunk_results:
            maybe_nodes = chunk_result[0]
            maybe_edges = chunk_result[1]
            all_entities.update(maybe_nodes.keys())
        
        multimodal_entities = [e for e in all_entities if "(image)" in e or "(table)" in e]
        if multimodal_entities:
            self.logger.info(f"Multimodal entities in chunk_results before merge ({len(multimodal_entities)}): {multimodal_entities[:5]}")
        else:
            self.logger.warning(f"No multimodal entities found in chunk_results! Total entities: {list(all_entities)[:10]}")
        
        # Store multimodal entity names for verification after merge
        multimodal_entity_names_set = set(multimodal_entities)
        
        # Use custom merge function that ensures multimodal entities are saved
        await merge_nodes_and_edges_with_multimodal(
            chunk_results=chunk_results,
            knowledge_graph_inst=self.rag_engine.chunk_entity_relation_graph,
            entity_vdb=self.rag_engine.entities_vdb,
            relationships_vdb=self.rag_engine.relationships_vdb,
            global_config=self.rag_engine.__dict__,
            full_entities_storage=self.rag_engine.full_entities,
            full_relations_storage=self.rag_engine.full_relations,
            doc_id=doc_id,
            pipeline_status=None,
            pipeline_status_lock=None,
            llm_response_cache=self.rag_engine.llm_response_cache,
            entity_chunks_storage=self.rag_engine.entity_chunks,
            relation_chunks_storage=self.rag_engine.relation_chunks,
            current_file_number=1,
            total_files=1,
            file_path=file_path,
            multimodal_entity_names=list(multimodal_entity_names_set),
        )
        
        # Verify multimodal entities were saved to full_entities
        try:
            # Wait a bit for async operations to complete
            await asyncio.sleep(0.5)
            
            doc_entities = await self.rag_engine.full_entities.get_by_id(doc_id)
            if doc_entities:
                saved_entity_names = set(doc_entities.get("entity_names", []))
                saved_multimodal = saved_entity_names.intersection(multimodal_entity_names_set)
                if saved_multimodal:
                    self.logger.info(f"✅ Multimodal entities saved to full_entities ({len(saved_multimodal)}/{len(multimodal_entity_names_set)}): {list(saved_multimodal)[:5]}")
                else:
                    self.logger.warning(f"❌ Multimodal entities NOT found in full_entities!")
                    self.logger.warning(f"   Expected ({len(multimodal_entity_names_set)}): {list(multimodal_entity_names_set)[:5]}")
                    self.logger.warning(f"   Saved entities sample: {list(saved_entity_names)[:10]}")
            else:
                self.logger.warning(f"❌ No doc_entities found for doc_id: {doc_id}")
        except Exception as e:
            self.logger.error(f"Error verifying multimodal entities in full_entities: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    async def _update_doc_status(self, doc_id: str, chunk_ids: List[str]):
        """Update doc_status with chunks"""
        try:
            import time as time_module
            current_doc_status_data = await self.rag_engine.doc_status.get_by_id(doc_id)

            if current_doc_status_data:
                existing_chunks_list = current_doc_status_data.get("chunks_list", [])
                existing_chunks_count = current_doc_status_data.get("chunks_count", 0)

                updated_chunks_list = existing_chunks_list + chunk_ids
                updated_chunks_count = existing_chunks_count + len(chunk_ids)

                await self.rag_engine.doc_status.upsert(
                    {
                        doc_id: {
                            **current_doc_status_data,
                            "chunks_list": updated_chunks_list,
                            "chunks_count": updated_chunks_count,
                            "update_time": int(time_module.time()),
                        }
                    }
                )

                self.logger.info(
                    f"Updated doc_status with {len(chunk_ids)} chunks (total: {updated_chunks_count})"
                )
            else:
                # Create new doc_status entry if it doesn't exist
                await self.rag_engine.doc_status.upsert(
                    {
                        doc_id: {
                            "chunks_list": chunk_ids,
                            "chunks_count": len(chunk_ids),
                            "status": "processing",
                            "multimodal_processed": False,
                            "create_time": int(time_module.time()),
                            "update_time": int(time_module.time()),
                            "_id": doc_id,
                        }
                    }
                )
                self.logger.info(
                    f"Created new doc_status for {doc_id} with {len(chunk_ids)} chunks"
                )

        except Exception as e:
            self.logger.warning(f"Error updating doc_status: {e}")
            import traceback
            self.logger.warning(traceback.format_exc())

    async def _mark_multimodal_processing_complete(self, doc_id: str):
        """Mark multimodal processing as complete in doc_status"""
        try:
            import time
            
            current_doc_status_data = await self.rag_engine.doc_status.get_by_id(doc_id)

            if current_doc_status_data:
                # Ensure status field exists (fix for invalid records)
                if "status" not in current_doc_status_data:
                    # If status is missing, set it based on multimodal_processed
                    if current_doc_status_data.get("multimodal_processed", False):
                        current_doc_status_data["status"] = "completed"
                    else:
                        current_doc_status_data["status"] = "processing"
                    self.logger.warning(
                        f"Fixed missing 'status' field in doc_status for {doc_id}"
                    )
                
                # Update existing doc_status with multimodal_processed flag
                await self.rag_engine.doc_status.upsert(
                    {
                        doc_id: {
                            **current_doc_status_data,
                            "multimodal_processed": True,
                            "update_time": int(time.time()),
                        }
                    }
                )
                # Ensure status is persisted to disk
                await self.rag_engine.doc_status.index_done_callback()

                self.logger.info(
                    f"Marked multimodal processing as complete for document {doc_id}"
                )
            else:
                # If doc_status doesn't exist, we shouldn't create it here
                # It should be created by RAG core's text processing first
                # Just log a warning
                self.logger.warning(
                    f"Document {doc_id} status not found. Cannot mark multimodal processing complete. "
                    f"This may happen if text content was not processed first."
                )

        except Exception as e:
            self.logger.warning(f"Error marking multimodal processing complete: {e}")
