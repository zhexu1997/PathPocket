from __future__ import annotations

import os
from typing import Any, Dict, Optional

import bootstrap  # noqa: F401

from rag_models import create_embedding_func, create_rerank_bundle, get_virchow2_feature_func
from pathpocket import PathPocket
from pathpocket.config import PathPocketConfig


from qwen_api import chat_multimodal, get_model_name

async def _default_llm_func(prompt, **kwargs):
    messages = [{"role": "user", "content": prompt}]
    return await chat_multimodal(messages, model=get_model_name())

async def initialize_rag(
    *,
    working_dir: Optional[str] = None,
    rag_kwargs: Optional[Dict[str, Any]] = None,
) -> PathPocket:
    """Initialize PathPocket for full inference (with LLM and retrieval)."""
    extra = dict(rag_kwargs or {})
    if working_dir is None:
        working_dir = extra.pop("working_dir", None) or os.getenv("WORKING_DIR", "./pathpocket_storage")

    config = PathPocketConfig(
        working_dir=working_dir,
        enable_image_processing=True,
        enable_table_processing=True,
    )

    embedding_func = create_embedding_func()
    virchow2_feature_func = get_virchow2_feature_func()
    rerank_bundle = create_rerank_bundle()
    rerank_model_func = rerank_bundle.get("rerank_model_func")
    rerank_pairs_batch_func = rerank_bundle.get("rerank_pairs_batch_func")

    graph_storage = extra.pop("graph_storage", None) or os.getenv("GRAPH_STORAGE", "PGHypergraphStorage")
    vector_storage = extra.pop("vector_storage", None) or os.getenv("VECTOR_STORAGE", "PGVectorStorage")

    embedding_func_max_async = int(extra.pop("embedding_func_max_async", os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "16")))

    rag_engine_kwargs: Dict[str, Any] = {
        "graph_storage": graph_storage,
        "vector_storage": vector_storage,
        "addon_params": {},
        "embedding_func_max_async": embedding_func_max_async,
    }

    evidence_catalog_json = os.getenv("EVIDENCE_LEVEL_TITLE_CATEGORY_JSON", "").strip()
    if not evidence_catalog_json:
        _pkg = os.path.dirname(os.path.dirname(__file__))
        _default_catalog = os.path.join(_pkg, "pathpocket", "evidence_level_title_category.slim.json")
        if not os.path.isfile(_default_catalog):
            _default_catalog = os.path.join(_pkg, "pathpocket", "evidence_level_title_category.json")
        if os.path.isfile(_default_catalog):
            evidence_catalog_json = _default_catalog

    if evidence_catalog_json:
        rag_engine_kwargs["evidence_level_title_category_json"] = evidence_catalog_json

    if rerank_model_func:
        rag_engine_kwargs["rerank_model_func"] = rerank_model_func
    if rerank_pairs_batch_func:
        rag_engine_kwargs["rerank_pairs_batch_func"] = rerank_pairs_batch_func

    rag_engine_kwargs.update(extra)

    rag = PathPocket(
        config=config,
        llm_model_func=_default_llm_func,
        vision_model_func=_default_llm_func,
        embedding_func=embedding_func,
        conch_feature_func=virchow2_feature_func,
        rag_engine_kwargs=rag_engine_kwargs,
    )

    init_result = await rag._ensure_rag_engine_initialized()
    if not init_result.get("success"):
        raise RuntimeError(f"Failed to initialize RAG engine: {init_result.get('error')}")

    if hasattr(rag.rag_engine, "pathology_images_vdb") and rag.rag_engine.pathology_images_vdb:
        try:
            if hasattr(rag.rag_engine.pathology_images_vdb, "initialize"):
                await rag.rag_engine.pathology_images_vdb.initialize()
        except Exception:
            pass

    return rag


async def initialize_rag_retrieval_only(
    *,
    working_dir: Optional[str],
    retrieval_rag_kwargs: Optional[Dict[str, Any]] = None,
) -> PathPocket:
    """Initialize PathPocket for retrieval/rerank/image-similarity only (no LLM on GPU)."""
    extra = dict(retrieval_rag_kwargs or {})
    if working_dir is None:
        working_dir = extra.pop("working_dir", None) or os.getenv("WORKING_DIR", "./pathpocket_storage_cache_vl2b")

    config = PathPocketConfig(
        working_dir=working_dir,
        enable_image_processing=True,
        enable_table_processing=True,
    )

    embedding_func = create_embedding_func()
    virchow2_feature_func = get_virchow2_feature_func()
    rerank_bundle = create_rerank_bundle()
    rerank_model_func = rerank_bundle.get("rerank_model_func")
    rerank_pairs_batch_func = rerank_bundle.get("rerank_pairs_batch_func")

    graph_storage = extra.pop("graph_storage", None) or os.getenv("GRAPH_STORAGE", "PGHypergraphStorage")
    vector_storage = extra.pop("vector_storage", None) or os.getenv("VECTOR_STORAGE", "PGVectorStorage")

    if "embedding_func_max_async" in extra:
        embedding_func_max_async = int(extra.pop("embedding_func_max_async"))
    else:
        embedding_func_max_async = int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "16"))

    rag_engine_kwargs: Dict[str, Any] = {
        "graph_storage": graph_storage,
        "vector_storage": vector_storage,
        "addon_params": {},
        "embedding_func_max_async": embedding_func_max_async,
    }

    evidence_catalog_json = os.getenv("EVIDENCE_LEVEL_TITLE_CATEGORY_JSON", "").strip()
    if not evidence_catalog_json:
        _pkg = os.path.dirname(os.path.dirname(__file__))
        _default_catalog = os.path.join(_pkg, "pathpocket", "evidence_level_title_category.slim.json")
        if not os.path.isfile(_default_catalog):
            _default_catalog = os.path.join(_pkg, "pathpocket", "evidence_level_title_category.json")
        if os.path.isfile(_default_catalog):
            evidence_catalog_json = _default_catalog

    if evidence_catalog_json:
        rag_engine_kwargs["evidence_level_title_category_json"] = evidence_catalog_json

    if rerank_model_func:
        rag_engine_kwargs["rerank_model_func"] = rerank_model_func
    if rerank_pairs_batch_func:
        rag_engine_kwargs["rerank_pairs_batch_func"] = rerank_pairs_batch_func

    rag_engine_kwargs.update(extra)

    rag = PathPocket(
        config=config,
        llm_model_func=_noop_llm,
        vision_model_func=_noop_llm,
        embedding_func=embedding_func,
        conch_feature_func=virchow2_feature_func,
        rag_engine_kwargs=rag_engine_kwargs,
    )

    init_result = await rag._ensure_rag_engine_initialized()
    if not init_result.get("success"):
        raise RuntimeError(f"Failed to initialize RAG engine: {init_result.get('error')}")

    if hasattr(rag.rag_engine, "pathology_images_vdb") and rag.rag_engine.pathology_images_vdb:
        try:
            if hasattr(rag.rag_engine.pathology_images_vdb, "initialize"):
                await rag.rag_engine.pathology_images_vdb.initialize()
        except Exception:
            pass

    return rag
