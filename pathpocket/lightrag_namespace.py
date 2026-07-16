from __future__ import annotations

from typing import Any, Iterable


# All namespace should not be changed
class NameSpace:
    KV_STORE_FULL_DOCS = "full_docs"
    KV_STORE_TEXT_CHUNKS = "text_chunks"
    KV_STORE_LLM_RESPONSE_CACHE = "llm_response_cache"
    KV_STORE_FULL_ENTITIES = "full_entities"
    KV_STORE_FULL_RELATIONS = "full_relations"
    KV_STORE_ENTITY_CHUNKS = "entity_chunks"
    KV_STORE_RELATION_CHUNKS = "relation_chunks"

    VECTOR_STORE_ENTITIES = "entities"
    VECTOR_STORE_RELATIONSHIPS = "relationships"
    VECTOR_STORE_CHUNKS = "chunks"
    VECTOR_STORE_PATHOLOGY_IMAGES = "pathology_images"

    GRAPH_STORE_CHUNK_ENTITY_RELATION = "chunk_entity_relation"

    DOC_STATUS = "doc_status"


def is_namespace(namespace: str, base_namespace: str | Iterable[str]):
    if isinstance(base_namespace, str):
        return namespace.endswith(base_namespace)
    return any(is_namespace(namespace, ns) for ns in base_namespace)


def resolve_vector_store_cosine_threshold(
    namespace: str, global_config: dict[str, Any]
) -> float:
    """
    Base cosine from ``vector_db_storage_cls_kwargs``; optional overrides on ``global_config``:
    ``relation_cosine_better_than_threshold`` / ``chunk_cosine_better_than_threshold`` /
    ``pathology_images_cosine_better_than_threshold`` (Virchow2 图像索引).
    """
    kwargs = global_config.get("vector_db_storage_cls_kwargs", {})
    base = kwargs.get("cosine_better_than_threshold")
    if base is None:
        raise ValueError(
            "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
        )
    base_f = float(base)
    if is_namespace(namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
        o = global_config.get("relation_cosine_better_than_threshold")
        return float(o) if o is not None else base_f
    if is_namespace(namespace, NameSpace.VECTOR_STORE_CHUNKS):
        o = global_config.get("chunk_cosine_better_than_threshold")
        return float(o) if o is not None else base_f
    if is_namespace(namespace, NameSpace.VECTOR_STORE_PATHOLOGY_IMAGES):
        o = global_config.get("pathology_images_cosine_better_than_threshold")
        return float(o) if o is not None else base_f
    return base_f
