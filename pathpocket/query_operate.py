"""
Query operations — delegates to ``pathpocket.operate.kg_query`` (structured keywords + hypergraph RAG).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pathpocket.core import logger
from pathpocket.lightrag_base import QueryParam, QueryResult


async def kg_query(
    query: str,
    knowledge_graph_inst: Any,
    entities_vdb: Any,
    relationships_vdb: Any,
    text_chunks_db: Any,
    query_param: QueryParam,
    global_config: Dict[str, Any],
    hashing_kv: Any = None,
    system_prompt: str | None = None,
    chunks_vdb: Any = None,
) -> Optional[QueryResult]:
    if not query:
        from pathpocket.prompt import PROMPTS

        return QueryResult(content=PROMPTS.get("fail_response", "Query is empty"))

    logger.info("Executing query (delegated to operate.kg_query): %s...", query[:100])
    from pathpocket.operate import kg_query as operate_kg_query

    return await operate_kg_query(
        query.strip(),
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        global_config,
        hashing_kv=hashing_kv,
        system_prompt=system_prompt,
        chunks_vdb=chunks_vdb,
    )
