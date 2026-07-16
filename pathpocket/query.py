"""
Query functionality for PathPocket
Supports multimodal queries with vision model for image analysis
"""

import base64
import hashlib
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from dataclasses import dataclass
from typing import Literal
from pathpocket.core import logger

@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    only_need_context: bool = False
    only_need_prompt: bool = False

from pathpocket.prompt import PROMPTS


def get_processor_for_type(modal_processors: Dict[str, Any], content_type: str):
    """Get the appropriate processor for content type"""
    if content_type == "image" and "image" in modal_processors:
        return modal_processors["image"]
    elif content_type == "table" and "table" in modal_processors:
        return modal_processors["table"]
    else:
        return modal_processors.get("generic")


class QueryMixin:
    """Query mixin for PathPocket multimodal queries"""

    async def aquery(
        self, query: str, mode: str = "mix", system_prompt: str | None = None, **kwargs
    ) -> str:
        """
        Execute a text query

        Args:
            query: Query text
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            system_prompt: Optional system prompt
            **kwargs: Other query parameters

        Returns:
            str: Query result
        """
        # Ensure RAG core is initialized
        await self._ensure_rag_core_initialized()

        # Create query parameters
        query_param = QueryParam(mode=mode, **kwargs)

        self.logger.info(f"Executing text query: {query[:100]}...")
        self.logger.info(f"Query mode: {mode}")

        # Use rag_engine.aquery_llm to get complete result with references
        from pathpocket.lightrag_base import QueryResult
        
        try:
            # Try to use rag_engine.aquery_llm for complete results
            if hasattr(self, 'rag_engine') and hasattr(self.rag_engine, 'aquery_llm'):
                complete_result = await self.rag_engine.aquery_llm(
                    query=query,
                    param=query_param,
                    system_prompt=system_prompt
                )
                if complete_result:
                    llm_response = complete_result.get("llm_response", {})
                    if llm_response.get("is_streaming"):
                        self.logger.warning("Streaming response not fully supported in string return")
                        return "Streaming response (use async iterator for full support)"
                    return llm_response.get("content", "")
        except Exception as e:
            self.logger.warning(f"Failed to use aquery_llm, falling back to kg_query: {e}")

        # Fallback to query_operate.kg_query
        from pathpocket.query_operate import kg_query
        
        # Execute query
        result = await kg_query(
            query=query,
            knowledge_graph_inst=self.rag_core.chunk_entity_relation_graph,
            entities_vdb=self.rag_core.entities_vdb,
            relationships_vdb=self.rag_core.relationships_vdb,
            text_chunks_db=self.rag_core.text_chunks,
            query_param=query_param,
            global_config=self.rag_core.__dict__,
            hashing_kv=self.rag_core.llm_response_cache,
            system_prompt=system_prompt,
            chunks_vdb=self.rag_core.chunks_vdb,
        )
        
        if result is None:
            from pathpocket.prompt import PROMPTS
            return PROMPTS.get("fail_response", "No result found")
        
        # Return content (handle streaming if needed)
        if result.is_streaming:
            # For streaming, return iterator as string representation
            self.logger.warning("Streaming response not fully supported in string return")
            return "Streaming response (use async iterator for full support)"
        
        self.logger.info("Text query completed")
        return result.content

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
            multimodal_content: List of multimodal content, each element contains:
                - type: Content type ("image", "table", etc.)
                - Other fields depend on type (e.g., img_path, table_data, latex, etc.)
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters, will be passed to QueryParam

        Returns:
            str: Query result

        Examples:
            # Image query
            result = await rag.aquery_with_multimodal(
                "Analyze the content in this image",
                multimodal_content=[{
                    "type": "image",
                    "img_path": "./image.jpg"
                }]
            )

            # Multiple images query
            result = await rag.aquery_with_multimodal(
                "Compare these images",
                multimodal_content=[
                    {"type": "image", "img_path": "./image1.jpg"},
                    {"type": "image", "img_path": "./image2.jpg"}
                ]
            )
        """
        # Ensure RAG core is initialized
        await self._ensure_rag_core_initialized()

        print(f"[DEBUG] ========== aquery_with_multimodal CALLED ==========")
        print(f"[DEBUG] query: {query[:100]}, multimodal_content: {multimodal_content}")
        self.logger.info(f"Executing multimodal query: {query[:100]}...")
        self.logger.info(f"Query mode: {mode}")

        # If no multimodal content, fallback to pure text query
        if not multimodal_content:
            print(f"[DEBUG] No multimodal content, falling back to text query")
            self.logger.info("No multimodal content provided, executing text query")
            return await self.aquery(query, mode=mode, **kwargs)

        # Check if we have images and vision_model_func is available
        has_images = any(c.get("type") == "image" for c in multimodal_content)
        can_use_vision = hasattr(self, "vision_model_func") and self.vision_model_func
        
        print(f"[DEBUG] ========== Checking vision model availability ==========")
        print(f"[DEBUG] has_images: {has_images}, can_use_vision: {can_use_vision}")
        print(f"[DEBUG] multimodal_content count: {len(multimodal_content)}")
        print(f"[DEBUG] multimodal_content types: {[c.get('type') for c in multimodal_content]}")
        print(f"[DEBUG] multimodal_content: {multimodal_content}")
        print(f"[DEBUG] hasattr vision_model_func: {hasattr(self, 'vision_model_func')}")
        if hasattr(self, "vision_model_func"):
            print(f"[DEBUG] vision_model_func value: {self.vision_model_func}")
            print(f"[DEBUG] vision_model_func type: {type(self.vision_model_func)}")
            print(f"[DEBUG] vision_model_func is None: {self.vision_model_func is None}")
            print(f"[DEBUG] bool(vision_model_func): {bool(self.vision_model_func)}")
        else:
            print(f"[DEBUG] WARNING: vision_model_func not found as attribute")
        print(f"[DEBUG] ========================================================")
        
        if has_images and can_use_vision:
            # Use similar image chunks from VDB_IMAGES table as context (NOT from text vector table)
            print(f"[DEBUG] Using vision model with similar image chunks context from VDB_IMAGES table")
            self.logger.info("Using vision model with similar image chunks from VDB_IMAGES table for image query")
            
            # Check if query already contains context from similar images (from VDB_IMAGES)
            # If so, extract it and use as context instead of querying text vector table
            kg_context = ""
            
            # Check if query contains context from similar images
            if "检索到的相似病理图像" in query or "相似图像关联的文本内容" in query:
                # Query already contains context from VDB_IMAGES, use it directly
                print(f"[DEBUG] Query already contains context from VDB_IMAGES table, using it as context")
                # Extract the context part (everything after the user question)
                query_lines = query.split("\n")
                user_question_end = 0
                for i, line in enumerate(query_lines):
                    if "===" in line and ("检索到的相似病理图像" in line or "相似图像关联的文本内容" in line):
                        user_question_end = i
                        break
                
                if user_question_end > 0:
                    # Extract user question (before context)
                    user_question = "\n".join(query_lines[:user_question_end]).strip()
                    # Extract context (after user question)
                    kg_context = "\n".join(query_lines[user_question_end:]).strip()
                    # Update query to just the user question
                    query = user_question
                    print(f"[DEBUG] Extracted user question: {query[:100]}")
                    print(f"[DEBUG] Extracted context from VDB_IMAGES: {len(kg_context)} chars")
                else:
                    # Context is embedded in query, use query as-is but don't retrieve from text vector table
                    print(f"[DEBUG] Context embedded in query, using query as-is without text vector retrieval")
                    kg_context = query
            else:
                # No context in query, but we should NOT retrieve from text vector table for image queries
                # Only use what's in the query
                print(f"[DEBUG] No VDB_IMAGES context in query, using query as-is (NOT retrieving from text vector table)")
                kg_context = ""
            
            # Get all images - prefer base64-encoded data, fallback to paths
            image_data_list = []
            for c in multimodal_content:
                if c.get("type") == "image":
                    # Prefer base64-encoded image data if available
                    image_base64 = c.get("image_base64")
                    if image_base64:
                        # Already base64-encoded, use directly
                        image_data_list.append({
                            "base64": image_base64,
                            "path": c.get("img_path", ""),
                            "is_similar": c.get("is_similar_image", False),
                            "similarity": c.get("similarity", 0.0),
                            "entity_name": c.get("entity_name", "")
                        })
                        print(f"[DEBUG] Using pre-encoded base64 image (length: {len(image_base64)})")
                    else:
                        # Fallback: read from path and encode
                        image_path = c.get("img_path")
                        if image_path:
                            image_data_list.append({
                                "path": image_path,
                                "needs_encoding": True
                            })
            
            if image_data_list:
                print(f"[DEBUG] Processing {len(image_data_list)} images with vision model")
                try:
                    import base64
                    from pathlib import Path
                    
                    # Process first image (can extend to multiple images later)
                    img_data = image_data_list[0]
                    
                    if "base64" in img_data:
                        # Already encoded, use directly
                        image_base64 = img_data["base64"]
                        image_path = img_data.get("path", "")
                        print(f"[DEBUG] Using pre-encoded base64 image from path: {image_path}")
                    else:
                        # Need to encode from path
                        image_path = img_data["path"]
                        print(f"[DEBUG] Encoding image from path: {image_path}")
                        print(f"[DEBUG] Image path exists: {Path(image_path).exists()}")
                        print(f"[DEBUG] Image path is absolute: {Path(image_path).is_absolute()}")
                        
                        if not Path(image_path).exists():
                            raise FileNotFoundError(f"Image file not found: {image_path}")
                        
                        with open(image_path, "rb") as f:
                            image_bytes = f.read()
                            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                        
                        print(f"[DEBUG] Image encoded to base64, length: {len(image_base64)}")
                        
                        # Clean base64 string (remove whitespace)
                        image_base64 = image_base64.strip().replace("\n", "").replace("\r", "").replace(" ", "")
                        print(f"[DEBUG] Base64 cleaned, new length: {len(image_base64)}")
                    
                    # Build enhanced prompt with KG context
                    if kg_context:
                        enhanced_prompt = f"""User question: {query}

Relevant knowledge from knowledge graph:
{kg_context}

Please analyze the image and provide a comprehensive answer based on:
1. The image content you see
2. The relevant knowledge from the knowledge graph above

Combine both sources of information to give a detailed and accurate answer."""
                    else:
                        enhanced_prompt = query
                    
                    print(f"[DEBUG] ========== Calling vision_model_func ==========")
                    print(f"[DEBUG]   - prompt length: {len(enhanced_prompt)}")
                    print(f"[DEBUG]   - prompt preview: {enhanced_prompt[:200]}")
                    print(f"[DEBUG]   - image_data length: {len(image_base64)}")
                    print(f"[DEBUG]   - image_data type: {type(image_base64)}")
                    print(f"[DEBUG]   - image_data is not empty: {bool(image_base64)}")
                    print(f"[DEBUG]   - vision_model_func exists: {hasattr(self, 'vision_model_func')}")
                    if hasattr(self, "vision_model_func"):
                        print(f"[DEBUG]   - vision_model_func: {self.vision_model_func}")
                    
                    # Call vision model with enhanced prompt
                    vision_answer = await self.vision_model_func(
                        enhanced_prompt,
                        image_data=image_base64,
                        system_prompt="You are a professional medical image analyst. Analyze the image carefully and provide detailed information based on both the image content and the relevant medical knowledge provided."
                    )
                    
                    print(f"[DEBUG] Vision model returned answer, length: {len(vision_answer) if vision_answer else 0}")
                    print(f"[DEBUG] Vision model answer preview: {vision_answer[:200] if vision_answer else 'None'}")
                    return vision_answer
                except Exception as e:
                    print(f"[DEBUG] Exception in vision model call: {e}")
                    import traceback
                    traceback.print_exc()
                    self.logger.warning(f"Failed to use vision model: {e}, falling back to enhanced query")
                    # Fall through to enhanced query method
            else:
                print(f"[DEBUG] WARNING: No image paths found in multimodal_content")
                print(f"[DEBUG] multimodal_content: {multimodal_content}")
        
        # Fallback: Process multimodal content to generate enhanced query text
        print(f"[DEBUG] Calling _process_multimodal_query_content with {len(multimodal_content)} items")
        enhanced_query = await self._process_multimodal_query_content(
            query, multimodal_content
        )
        print(f"[DEBUG] _process_multimodal_query_content returned, length: {len(enhanced_query)}")

        self.logger.info(
            f"Generated enhanced query length: {len(enhanced_query)} characters"
        )

        # Execute enhanced query with regular LLM
        result = await self.aquery(enhanced_query, mode=mode, **kwargs)

        self.logger.info("Multimodal query completed")
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
            print(f"[DEBUG] ========== Processing multimodal content {i+1}/{len(multimodal_content)} ==========")
            print(f"[DEBUG] content_type: {content_type}, content: {content}")

            try:
                # Get appropriate processor
                processor = get_processor_for_type(self.modal_processors, content_type)
                print(f"[DEBUG] get_processor_for_type returned: {processor is not None}, content_type: {content_type}")
                print(f"[DEBUG] modal_processors keys: {list(self.modal_processors.keys()) if hasattr(self, 'modal_processors') else 'N/A'}")

                if processor:
                    print(f"[DEBUG] Processor found, using _generate_query_content_description")
                    # Generate content description
                    description = await self._generate_query_content_description(
                        processor, content, content_type
                    )
                    enhanced_parts.append(
                        f"\nRelated {content_type} content: {description}"
                    )
                else:
                    print(f"[DEBUG] No processor found for {content_type}")
                    # If no appropriate processor, but it's an image and we have vision_model_func, try to describe it
                    if content_type == "image":
                        print(f"[DEBUG] Content type is image, checking vision_model_func")
                        print(f"[DEBUG] hasattr(self, 'vision_model_func'): {hasattr(self, 'vision_model_func')}")
                        if hasattr(self, "vision_model_func"):
                            print(f"[DEBUG] self.vision_model_func: {self.vision_model_func}")
                        if hasattr(self, "vision_model_func") and self.vision_model_func:
                            print(f"[DEBUG] No processor but vision_model_func available, attempting to describe image")
                            description = await self._describe_image_for_query(None, content)
                            enhanced_parts.append(
                                f"\nRelated {content_type} content: {description}"
                            )
                        else:
                            print(f"[DEBUG] vision_model_func not available, using basic description")
                            basic_desc = str(content)[:200]
                            enhanced_parts.append(
                                f"\nRelated {content_type} content: {basic_desc}"
                            )
                    else:
                        # If no appropriate processor, use basic description
                        basic_desc = str(content)[:200]
                        enhanced_parts.append(
                            f"\nRelated {content_type} content: {basic_desc}"
                        )

            except Exception as e:
                print(f"[DEBUG] Exception in _process_multimodal_query_content: {e}")
                import traceback
                traceback.print_exc()
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                # Continue processing other content
                continue

        enhanced_query = "\n".join(enhanced_parts)
        enhanced_query += "\n\nPlease provide a comprehensive answer based on the user query and the provided multimodal content information."

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
        """Generate image description for query using vision model if available"""
        image_path = content.get("img_path")
        captions = content.get("image_caption", content.get("img_caption", []))
        footnotes = content.get("image_footnote", content.get("img_footnote", []))

        # Debug logging
        print(f"[DEBUG] _describe_image_for_query called with image_path: {image_path}")
        print(f"[DEBUG] hasattr(self, 'vision_model_func'): {hasattr(self, 'vision_model_func')}")
        if hasattr(self, "vision_model_func"):
            print(f"[DEBUG] self.vision_model_func: {self.vision_model_func}")
        print(f"[DEBUG] image_path exists: {Path(image_path).exists() if image_path else False}")

        # If vision model is available and image exists, use it
        if (
            hasattr(self, "vision_model_func")
            and self.vision_model_func
            and image_path
            and Path(image_path).exists()
        ):
            try:
                print(f"[DEBUG] Attempting to encode image: {image_path}")
                print(f"[DEBUG] Image path exists: {Path(image_path).exists()}")
                print(f"[DEBUG] Image path is absolute: {Path(image_path).is_absolute()}")
                # Encode image to base64
                image_base64 = self._encode_image_to_base64(image_path)
                print(f"[DEBUG] Image encoded, base64 length: {len(image_base64) if image_base64 else 0}")
                print(f"[DEBUG] image_base64 type: {type(image_base64)}")
                print(f"[DEBUG] image_base64 is not empty: {bool(image_base64)}")
                if image_base64:
                    # Clean base64 string (remove whitespace)
                    image_base64 = image_base64.strip().replace("\n", "").replace("\r", "").replace(" ", "")
                    print(f"[DEBUG] Base64 cleaned, new length: {len(image_base64)}")
                    print(f"[DEBUG] image_base64 first 50 chars: {image_base64[:50]}")
                    print(f"[DEBUG] image_base64 last 50 chars: {image_base64[-50:]}")
                    prompt = "Please briefly describe the main content, key elements, and important information in this image."
                    print(f"[DEBUG] Calling vision_model_func with prompt: {prompt[:100]}")
                    print(f"[DEBUG] Calling vision_model_func with image_data length: {len(image_base64)}")
                    print(f"[DEBUG] vision_model_func exists: {hasattr(self, 'vision_model_func')}")
                    if hasattr(self, "vision_model_func"):
                        print(f"[DEBUG] vision_model_func: {self.vision_model_func}")
                    description = await self.vision_model_func(
                        prompt,
                        image_data=image_base64,
                        system_prompt="You are a professional image analyst who can accurately describe image content.",
                    )
                    print(f"[DEBUG] vision_model_func returned description length: {len(description) if description else 0}")
                    print(f"[DEBUG] vision_model_func returned description: {description[:200] if description else 'None'}")
                    return description
                else:
                    print(f"[DEBUG] image_base64 is empty, falling back to caption")
            except Exception as e:
                print(f"[DEBUG] Exception in _describe_image_for_query: {e}")
                import traceback
                traceback.print_exc()
                self.logger.warning(f"Failed to use vision model for image: {e}")
                # Fall through to caption-based description

        # Fallback: use caption and footnote information
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
        table_data = content.get("table_data", content.get("table_body", ""))
        table_caption = content.get("table_caption", "")

        if isinstance(table_caption, list):
            table_caption = ", ".join(table_caption)

        # Use LLM to analyze table
        prompt = f"""Please analyze the main content, structure, and key information of the following table data:

Table data:
{table_data}

Table caption: {table_caption}

Please briefly summarize the main content, data characteristics, and important findings of the table."""

        description = await processor.modal_caption_func(
            prompt, system_prompt="You are a professional data analyst who can accurately analyze table data."
        )

        return description

    async def _describe_generic_for_query(
        self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """Generate generic content description for query"""
        content_str = str(content)

        prompt = f"""Please analyze the following {content_type} type content and extract its main information and key features:

Content: {content_str}

Please briefly summarize the main characteristics and important information of this content."""

        description = await processor.modal_caption_func(
            prompt,
            system_prompt=f"You are a professional content analyst who can accurately analyze {content_type} type content.",
        )

        return description

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_string
        except Exception as e:
            self.logger.error(f"Failed to encode image {image_path}: {e}")
            return ""
