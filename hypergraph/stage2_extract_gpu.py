"""
Stage 2: Extract entities and relations using LLM (GPU required)
Input: kv_store_text_chunks.json
Output: kv_store_llm_response_cache.json
"""


import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import asyncio
import os
import logging
import httpx
from pathlib import Path
from typing import List, Dict, Any
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pathpocket import (
    PathPocket,
    PathPocketConfig,
    fix_invalid_doc_status,
)
from pathpocket.operate import extract_entities

# Disable httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)


async def main():
    # ========== Configuration ==========
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
    VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "./MODEL/Qwen3-30B-A3B-Instruct-2507-FP8")
    _language = os.getenv("PIPELINE_LANGUAGE", "English")
    max_tokens = int(os.getenv("CHUNK_TOKEN_SIZE", "2400"))
    working_dir = os.getenv("WORKING_DIR", "./pathpocket_storage")

    custom_entity_types = [
        "Disease", "Symptom", "PathologicalFinding", "AnatomicalSite", "CellType",
        "HistologicalPattern", "Gene", "GeneticMutation", "Biomarker", "MolecularPathway",
        "Pathogen", "RiskFactor", "Pathogenesis", "DiagnosticMethod", "StainingMethod",
        "LabTest", "DiagnosticCriteria", "Treatment", "Drug", "MedicalDevice",
        "Prognosis", "Organization", "Location", "Person",
    ]
    
    print(f"\n{'='*60}")
    print("Stage 2: Extract entities and relations using LLM (GPU required)")
    print(f"{'='*60}")
    print(f"vLLM Base URL: {VLLM_BASE_URL}")
    print(f"vLLM Model: {VLLM_MODEL_NAME}")
    print(f"Working directory: {working_dir}")
    print(f"{'='*60}\n")
    
    # Check vLLM server connection
    print("Checking vLLM server connection...")
    max_retries = 10
    retry_delay = 5
    server_ready = False
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                models_url = f"{VLLM_BASE_URL}/models"
                response = await client.get(models_url, follow_redirects=True)
                if response.status_code == 200:
                    print(f"✅ vLLM server is ready at {VLLM_BASE_URL}")
                    server_ready = True
                    break
                else:
                    print(f"⚠️  Attempt {attempt + 1}/{max_retries}: Server responded with status {response.status_code}, retrying...")
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt < max_retries - 1:
                print(f"⏳ Attempt {attempt + 1}/{max_retries}: Server not ready yet, waiting {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                print(f"❌ Cannot connect to vLLM server at {VLLM_BASE_URL} after {max_retries} attempts")
                raise RuntimeError(f"vLLM server not available at {VLLM_BASE_URL}") from e
    
    if not server_ready:
        raise RuntimeError(f"vLLM server not available at {VLLM_BASE_URL}")
    
    # 扩大连接池限制，适配高并发吞吐能力
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
        timeout=300.0
    )

    # Initialize OpenAI-compatible client for vLLM
    vllm_client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        timeout=300.0,
        max_retries=3,
        http_client=http_client
    )
    
    # Define LLM model function using vLLM
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        keyword_extraction = kwargs.pop("keyword_extraction", False)
        max_tokens = kwargs.pop("max_tokens", 8192)
        temperature = kwargs.pop("temperature", 0.0)
        stream = kwargs.pop("stream", False)
        timeout = kwargs.pop("timeout", 300)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        
        request_params = {
            "model": VLLM_MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
        }
        
        if keyword_extraction:
            request_params["response_format"] = {"type": "json_object"}
        
        try:
            if stream:
                response_stream = await vllm_client.chat.completions.create(
                    stream=True,
                    **request_params
                )
                full_response = ""
                async for chunk in response_stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                return full_response
            else:
                response = await vllm_client.chat.completions.create(
                    **request_params
                )
                return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            print(f"❌ vLLM API error: {error_msg}")
            raise RuntimeError(f"vLLM API error: {error_msg}") from e
    
    # Create PathPocket configuration
    config = PathPocketConfig(
        working_dir=working_dir,
        enable_image_processing=True,
        enable_table_processing=True,
    )
    
    # Dummy embedding function (required by RAG engine initialization)
    class DummyEmbeddingFunc:
        """Dummy embedding function that raises error if called"""
        def __init__(self):
            self.embedding_dim = 1024  # Default dimension
            self.max_token_size = 4096
        
        async def __call__(self, texts):
            raise RuntimeError("Embedding should not be called in Stage 2. This is a placeholder function.")
    
    dummy_embedding_func = DummyEmbeddingFunc()
    
    # Initialize PathPocket with LLM but no embedding
    rag = PathPocket(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=dummy_embedding_func, 
        rag_engine_kwargs={
            "graph_storage": "HyperNetXStorage",
            "vector_storage": "NanoVectorDBStorage",
            "addon_params": {
                "entity_types": custom_entity_types,
                "language": _language,
            },
            "default_llm_timeout": 200,
            # 提高并发
            "llm_model_max_async": 256,  
            "chunk_token_size": max_tokens,
            "enable_llm_cache_for_entity_extract": True, 
            "entity_extract_max_gleaning": 1,
        }
    )
    
    # Fix any invalid doc_status records
    await fix_invalid_doc_status(rag)
    
    # Initialize RAG engine
    init_result = await rag._ensure_rag_engine_initialized()
    if not init_result.get("success"):
        raise RuntimeError(f"Failed to initialize RAG engine: {init_result.get('error')}")
    
    # Get all documents that need entity extraction
    from pathpocket.lightrag_base import DocStatus
    preprocessed_docs = await rag.rag_engine.doc_status.get_docs_by_status(DocStatus.PREPROCESSED)
    docs_to_process = list(preprocessed_docs.keys())
    
    if not docs_to_process:
        print("✅ No documents need entity extraction!")
        return

    # ==========================================
    #全局 Chunk 池化 + 动态批处理，几千个 chunk 一起排队喂给 GPU
    # ==========================================
    
    print(f"\n{'='*60}")
    print(f"Gathering ALL chunks from {len(docs_to_process)} documents to maximize GPU usage...")
    print(f"{'='*60}\n")
    
    all_chunk_ids = []
    # 1. 收集所有文档里的全部 chunk_id
    for doc_id in docs_to_process:
        doc_status = await rag.rag_engine.doc_status.get_by_id(doc_id)
        if not doc_status: continue
        chunks_list = doc_status.get("chunks_list", [])
        all_chunk_ids.extend(chunks_list)
        
    # 去重
    all_chunk_ids = list(set(all_chunk_ids))
    total_chunks = len(all_chunk_ids)
    print(f"🎯 Total unique chunks gathered: {total_chunks}")
    
    if total_chunks == 0:
        print("✅ No chunks need to be processed!")
        return

    # 设置每个大批次处理的 Chunk 数量。
    # 设成 2000 能保证 GPU 持续运转很久，然后集中写一次盘。
    CHUNK_BATCH_SIZE = 2000  

    for i in range(0, total_chunks, CHUNK_BATCH_SIZE):
        batch_chunk_ids = all_chunk_ids[i : i + CHUNK_BATCH_SIZE]
        batch_num = (i // CHUNK_BATCH_SIZE) + 1
        total_batches = (total_chunks + CHUNK_BATCH_SIZE - 1) // CHUNK_BATCH_SIZE
        
        print(f"\n{'='*60}")
        print(f"🚀 Processing Batch {batch_num}/{total_batches} (Chunks {i+1} to {min(i+CHUNK_BATCH_SIZE, total_chunks)})...")
        
        try:
            # 2. 从数据库获取这批 Chunk 的实际数据内容
            chunks = {}
            chunk_data_list = await rag.rag_engine.text_chunks.get_by_ids(batch_chunk_ids)
            for idx, chunk_data in enumerate(chunk_data_list):
                if chunk_data:
                    chunk_id = batch_chunk_ids[idx]
                    chunks[chunk_id] = chunk_data
            
            if not chunks:
                continue
                
            print(f"Submitting {len(chunks)} chunks to GPU engine (128 concurrent streams)...")
            
            # 3. 提交海量并发请求
            await extract_entities(
                chunks=chunks,
                global_config={
                    **rag.rag_engine.__dict__,
                    "llm_model_func": llm_model_func,
                    "addon_params": {
                        "entity_types": custom_entity_types,
                        "language": _language,
                    },
                },
                pipeline_status=None,
                pipeline_status_lock=None,
                llm_response_cache=rag.rag_engine.llm_response_cache,
                text_chunks_storage=rag.rag_engine.text_chunks,
            )
            
            # 4. 跑完几千条后，集中写一次盘
            print(f"💾 Batch {batch_num} completed! Saving cache to disk...")
            await rag.rag_engine.llm_response_cache.index_done_callback()
            await rag.rag_engine.text_chunks.index_done_callback()
            print(f"✅ Cache saved successfully for Batch {batch_num}")
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_num}: {e}")
            import traceback
            traceback.print_exc()
            # 出现异常时安全保存进度
            try:
                print("Attempting to emergency save current progress...")
                await rag.rag_engine.llm_response_cache.index_done_callback()
                await rag.rag_engine.text_chunks.index_done_callback()
            except:
                pass
            continue

    print(f"\n✅ Stage 2 completed successfully!")
    print(f"Output files saved in:")
    print(f"  - {working_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
