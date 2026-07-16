"""PostgreSQL-based hypergraph storage for multi-entity relationships with high performance.

This implementation uses PostgreSQL to store hyperedges (multi-entity relationships)
with indexes for fast queries, similar to Neo4j performance but with native hyperedge support.
"""

import os
import json
import re
from dataclasses import dataclass
from typing import final, Any, Optional
from pathpocket.lightrag_utils import logger
from pathpocket.lightrag_base import BaseGraphStorage
from pathpocket.shared_storage import get_data_init_lock
from pathpocket.lightrag_kg.postgres_impl import (
    PostgreSQLDB,
    ClientManager,
    PGGraphQueryException,
)
from pathpocket.lightrag_kg.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )
except ImportError:
    # Fallback if tenacity not available
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    stop_after_attempt = lambda x: None
    wait_exponential = lambda **kwargs: None
    retry_if_exception_type = lambda *args: None


@final
@dataclass
class PGHypergraphStorage(BaseGraphStorage):
    """PostgreSQL-based hypergraph storage with native hyperedge support.
    
    Features:
    - Native support for hyperedges (multi-entity relationships)
    - Indexed queries for fast lookups (similar to Neo4j)
    - Incremental updates (no need to load entire graph)
    - Concurrent access support
    - Transaction support
    """
    
    def __post_init__(self):
        self.db: PostgreSQLDB | None = None
        self.table_name: str = ""
    
    def _get_table_name(self) -> str:
        """Generate table name based on workspace and namespace."""
        workspace = self.workspace
        namespace = self.namespace
        
        if workspace and workspace.strip() and workspace.strip().lower() != "default":
            safe_workspace = re.sub(r"[^a-zA-Z0-9_]", "_", workspace.strip())
            safe_namespace = re.sub(r"[^a-zA-Z0-9_]", "_", namespace)
            return f"hypergraph_{safe_workspace}_{safe_namespace}"
        else:
            return f"hypergraph_{re.sub(r'[^a-zA-Z0-9_]', '_', namespace)}"
    
    async def initialize(self):
        """Initialize PostgreSQL hypergraph storage."""
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()
            
            # Set workspace
            if self.db.workspace:
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                pass
            else:
                self.workspace = "default"
            
            self.table_name = self._get_table_name()
            
            logger.info(
                f"[{self.workspace}] PostgreSQL Hypergraph initialized: table='{self.table_name}'"
            )
            
            # Create hyperedge table with indexes
            async with self.db.pool.acquire() as connection:
                # Create hyperedge table
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    edge_id VARCHAR(512) NOT NULL,
                    workspace VARCHAR(255) NOT NULL DEFAULT '{self.workspace}',
                    entities TEXT[] NOT NULL,
                    entity_count INTEGER NOT NULL,
                    edge_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, edge_id)
                );
                """
                
                # Create node attributes table for storing node properties
                node_table_name = f"{self.table_name}_nodes"
                create_node_table = f"""
                CREATE TABLE IF NOT EXISTS {node_table_name} (
                    workspace VARCHAR(255) NOT NULL,
                    node_id VARCHAR(255) NOT NULL,
                    node_data JSONB NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, node_id)
                );
                """
                
                # Create indexes for fast queries
                create_indexes = [
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_edge_id_idx ON {self.table_name} (edge_id);",
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_workspace_idx ON {self.table_name} (workspace);",
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_entities_gin_idx ON {self.table_name} USING GIN (entities);",
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_entity_count_idx ON {self.table_name} (entity_count);",
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_edge_data_gin_idx ON {self.table_name} USING GIN (edge_data);",
                    f"CREATE INDEX IF NOT EXISTS {node_table_name}_workspace_idx ON {node_table_name} (workspace);",
                    f"CREATE INDEX IF NOT EXISTS {node_table_name}_node_id_idx ON {node_table_name} (node_id);",
                    f"CREATE INDEX IF NOT EXISTS {node_table_name}_node_data_gin_idx ON {node_table_name} USING GIN (node_data);",
                ]
                
                # Store node table name for later use
                self._node_table_name = node_table_name
                
                try:
                    await connection.execute(create_table_query)
                    await connection.execute(create_node_table)
                    for idx_query in create_indexes:
                        await connection.execute(idx_query)
                    logger.info(f"[{self.workspace}] Created hypergraph table, node table and indexes: {self.table_name}, {node_table_name}")
                except Exception as e:
                    logger.warning(f"[{self.workspace}] Error creating table/indexes (may already exist): {e}")
                
                # Initialize node table name if not set (for existing databases)
                if not hasattr(self, '_node_table_name'):
                    self._node_table_name = node_table_name
    
    async def finalize(self):
        """Finalize storage."""
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
    
    async def index_done_callback(self) -> None:
        """PG handles persistence automatically."""
        pass
    
    async def drop(self) -> dict[str, str]:
        """Drop all data from storage and clean up resources.
        
        This method will delete all hyperedges and nodes from the current workspace.
        
        Returns:
            dict[str, str]: Operation status and message
        """
        try:
            if self.db is None:
                return {
                    "status": "error",
                    "message": "Database connection not initialized"
                }
            
            async with self.db.pool.acquire() as connection:
                # Delete all hyperedges for the current workspace
                delete_hyperedges_query = f"""
                    DELETE FROM {self.table_name}
                    WHERE workspace = $1
                """
                await connection.execute(delete_hyperedges_query, self.workspace)
                logger.info(
                    f"[{self.workspace}] Dropped all hyperedges from {self.table_name}"
                )
                
                # Also delete all nodes from node table if it exists
                if hasattr(self, '_node_table_name') and self._node_table_name:
                    delete_nodes_query = f"""
                        DELETE FROM {self._node_table_name}
                        WHERE workspace = $1
                    """
                    await connection.execute(delete_nodes_query, self.workspace)
                    logger.info(
                        f"[{self.workspace}] Dropped all nodes from {self._node_table_name}"
                    )
                
                return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping data: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _query(self, query: str, params: list = None, readonly: bool = True) -> list:
        """Execute a query.
        
        Args:
            query: SQL query with $1, $2, etc. placeholders
            params: List of parameters in order (for $1, $2, etc.)
        """
        if self.db is None:
            raise RuntimeError("Storage not initialized")
        
        async with self.db.pool.acquire() as connection:
            try:
                if params:
                    result = await connection.fetch(query, *params)
                else:
                    result = await connection.fetch(query)
                return [dict(row) for row in result]
            except Exception as e:
                logger.error(f"[{self.workspace}] Query error: {e}")
                logger.error(f"Query: {query}")
                if params:
                    logger.error(f"Params: {params}")
                raise
    
    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists (by checking if it's in any hyperedge)."""
        query = f"""
        SELECT 1 FROM {self.table_name}
        WHERE workspace = $1 AND $2 = ANY(entities)
        LIMIT 1
        """
        result = await self._query(query, [self.workspace, node_id], readonly=True)
        return len(result) > 0
    
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if any hyperedge contains both nodes."""
        query = f"""
        SELECT 1 FROM {self.table_name}
        WHERE workspace = $1 
          AND $2 = ANY(entities) 
          AND $3 = ANY(entities)
        LIMIT 1
        """
        result = await self._query(
            query,
            [self.workspace, source_node_id, target_node_id],
            readonly=True
        )
        return len(result) > 0
    
    async def has_hyperedge(self, edge_id: str) -> bool:
        """Check if a hyperedge exists by ID."""
        query = f"""
        SELECT 1 FROM {self.table_name}
        WHERE workspace = $1 AND edge_id = $2
        LIMIT 1
        """
        result = await self._query(
            query,
            [self.workspace, edge_id],
            readonly=True
        )
        return len(result) > 0
    
    async def get_node(self, node_id: str) -> Optional[dict[str, str]]:
        """Get node by ID with its attributes."""
        # First check if node exists
        if not await self.has_node(node_id):
            return None
        
        # Try to get node attributes from node table
        if hasattr(self, '_node_table_name'):
            query = f"""
            SELECT node_data
            FROM {self._node_table_name}
            WHERE workspace = $1 AND node_id = $2
            """
            result = await self._query(
                query,
                [self.workspace, node_id],
                readonly=True
            )
            
            if result:
                node_data = result[0]["node_data"]
                if isinstance(node_data, str):
                    import json
                    node_data = json.loads(node_data)
                # Ensure entity_id is included
                if isinstance(node_data, dict):
                    node_data["entity_id"] = node_id
                    return node_data
        
        # Fallback: return basic node info if no attributes stored
        return {"entity_id": node_id}
    
    async def get_nodes_batch(self, node_ids: list[str], batch_size: int = 1000) -> dict[str, dict]:
        """
        Retrieve multiple nodes in batch with their attributes.
        
        Args:
            node_ids: List of node entity IDs to fetch
            batch_size: Batch size for the query
        
        Returns:
            A dictionary mapping each node_id to its node data (or basic dict if not found)
        """
        if not node_ids:
            return {}
        
        nodes_dict = {}
        
        # Get node attributes from node table if available
        if hasattr(self, '_node_table_name') and self._node_table_name:
            # Process in batches
            for i in range(0, len(node_ids), batch_size):
                batch = node_ids[i:i + batch_size]
                
                query = f"""
                SELECT node_id, node_data
                FROM {self._node_table_name}
                WHERE workspace = $1 AND node_id = ANY($2)
                """
                result = await self._query(
                    query,
                    [self.workspace, batch],
                    readonly=True
                )
                
                for row in result:
                    node_id = row["node_id"]
                    node_data = row["node_data"]
                    if isinstance(node_data, str):
                        import json
                        node_data = json.loads(node_data)
                    if isinstance(node_data, dict):
                        node_data["entity_id"] = node_id
                        nodes_dict[node_id] = node_data
        
        # For nodes not found in node table, return basic info
        for node_id in node_ids:
            if node_id not in nodes_dict:
                # Check if node exists in hyperedges
                if await self.has_node(node_id):
                    nodes_dict[node_id] = {"entity_id": node_id}
                # If node doesn't exist, don't add it (return None equivalent by not adding)
        
        return nodes_dict
    
    async def get_hyperedge(self, edge_id: str) -> Optional[dict[str, Any]]:
        """Get hyperedge by ID."""
        query = f"""
        SELECT edge_id, entities, entity_count, edge_data
        FROM {self.table_name}
        WHERE workspace = $1 AND edge_id = $2
        """
        result = await self._query(
            query,
            [self.workspace, edge_id],
            readonly=True
        )
        
        if result:
            row = result[0]
            edge_data = row["edge_data"]
            if isinstance(edge_data, str):
                edge_data = json.loads(edge_data)
            
            return {
                "edge_id": row["edge_id"],
                "entities": row["entities"],
                "entity_count": row["entity_count"],
                **edge_data,
            }
        return None
    
    async def node_degree(self, node_id: str) -> int:
        """Return number of hyperedges containing this node."""
        query = f"""
        SELECT COUNT(*) as count
        FROM {self.table_name}
        WHERE workspace = $1 AND $2 = ANY(entities)
        """
        result = await self._query(
            query,
            [self.workspace, node_id],
            readonly=True
        )
        return result[0]["count"] if result else 0
    
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Return sum of degrees of two nodes."""
        src_deg = await self.node_degree(src_id)
        tgt_deg = await self.node_degree(tgt_id)
        return src_deg + tgt_deg
    
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Optional[dict[str, str]]:
        """Get edge between two nodes (returns first hyperedge containing both)."""
        query = f"""
        SELECT edge_id, entities, entity_count, edge_data
        FROM {self.table_name}
        WHERE workspace = $1 
          AND $2 = ANY(entities) 
          AND $3 = ANY(entities)
        ORDER BY entity_count ASC
        LIMIT 1
        """
        result = await self._query(
            query,
            [self.workspace, source_node_id, target_node_id],
            readonly=True
        )
        
        if result:
            row = result[0]
            edge_data = row["edge_data"]
            if isinstance(edge_data, str):
                edge_data = json.loads(edge_data)
            
            return {
                "source": source_node_id,
                "target": target_node_id,
                **edge_data,
            }
        return None
    
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]]:
        """Get all edges for a node (returns pairs from hyperedges)."""
        query = f"""
        SELECT entities
        FROM {self.table_name}
        WHERE workspace = $1 AND $2 = ANY(entities)
        """
        result = await self._query(
            query,
            [self.workspace, source_node_id],
            readonly=True
        )
        
        pairs = []
        for row in result:
            entities = row["entities"]
            # Generate all pairs containing source_node_id
            for entity in entities:
                if entity != source_node_id:
                    pairs.append((source_node_id, entity))
        
        return pairs
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PGGraphQueryException,)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert node with its attributes."""
        if not hasattr(self, '_node_table_name'):
            # Node table not initialized, skip
            return
        
        # Ensure entity_id is in node_data
        node_data_with_id = dict(node_data)
        node_data_with_id["entity_id"] = node_id
        
        # Convert node_data to JSONB
        import json
        node_data_json = json.dumps(node_data_with_id, ensure_ascii=False)
        
        query = f"""
        INSERT INTO {self._node_table_name} (workspace, node_id, node_data, updated_at)
        VALUES ($1, $2, $3::jsonb, CURRENT_TIMESTAMP)
        ON CONFLICT (workspace, node_id) 
        DO UPDATE SET
            node_data = EXCLUDED.node_data,
            updated_at = CURRENT_TIMESTAMP
        """
        
        try:
            async with self.db.pool.acquire() as connection:
                await connection.execute(
                    query,
                    self.workspace,
                    node_id,
                    node_data_json,
                )
            logger.debug(f"[{self.workspace}] Upserted node {node_id} with attributes")
        except Exception as e:
            logger.error(f"[{self.workspace}] Error upserting node {node_id}: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PGGraphQueryException,)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Upsert binary edge (creates a hyperedge with 2 entities)."""
        await self.upsert_hyperedge(
            edge_id=f"{source_node_id}_{target_node_id}",
            node_ids=[source_node_id, target_node_id],
            edge_data=edge_data
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PGGraphQueryException,)),
    )
    async def upsert_hyperedge(
        self, edge_id: str, node_ids: list[str], edge_data: dict[str, str]
    ) -> None:
        """Upsert hyperedge (multi-entity relationship).
        
        This is the key method that provides fast hyperedge support.
        Uses PostgreSQL's UPSERT (ON CONFLICT) for atomic updates.
        """
        if not node_ids:
            logger.warning(f"[{self.workspace}] Cannot create hyperedge with no entities: {edge_id}")
            return
        
        # Sort entities for consistency
        sorted_entities = sorted(node_ids)
        entity_count = len(sorted_entities)
        
        # Convert edge_data to JSONB
        edge_data_json = json.dumps(edge_data, ensure_ascii=False)
        
        query = f"""
        INSERT INTO {self.table_name} (edge_id, workspace, entities, entity_count, edge_data, updated_at)
        VALUES ($1, $2, $3, $4, $5::jsonb, CURRENT_TIMESTAMP)
        ON CONFLICT (workspace, edge_id) 
        DO UPDATE SET
            entities = EXCLUDED.entities,
            entity_count = EXCLUDED.entity_count,
            edge_data = EXCLUDED.edge_data,
            updated_at = CURRENT_TIMESTAMP
        """
        
        try:
            async with self.db.pool.acquire() as connection:
                await connection.execute(
                    query,
                    edge_id,
                    self.workspace,
                    sorted_entities,
                    entity_count,
                    edge_data_json,
                )
            logger.debug(
                f"[{self.workspace}] Upserted hyperedge {edge_id} with {entity_count} entities"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error upserting hyperedge {edge_id}: {e}"
            )
            raise
    
    async def delete_node(self, node_id: str) -> None:
        """Delete node (removes from all hyperedges)."""
        # Remove node from all hyperedges and update entity_count
        query = f"""
        UPDATE {self.table_name}
        SET entities = array_remove(entities, $1),
            entity_count = array_length(array_remove(entities, $1), 1),
            updated_at = CURRENT_TIMESTAMP
        WHERE workspace = $2 AND $1 = ANY(entities)
        """
        
        # Delete hyperedges that end up with < 2 entities
        delete_query = f"""
        DELETE FROM {self.table_name}
        WHERE workspace = $1 
          AND entity_count < 2
        """
        
        try:
            async with self.db.pool.acquire() as connection:
                await connection.execute(query, node_id, self.workspace)
                await connection.execute(delete_query, self.workspace)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting node {node_id}: {e}")
            raise
    
    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes with their attributes."""
        # Get all unique node IDs from hyperedges
        query = f"""
        SELECT DISTINCT unnest(entities) as entity_id
        FROM {self.table_name}
        WHERE workspace = $1
        """
        result = await self._query(query, [self.workspace], readonly=True)
        node_ids = [row["entity_id"] for row in result]
        
        # Get node attributes from node table if available
        if hasattr(self, '_node_table_name') and node_ids:
            # Batch fetch node attributes
            query = f"""
            SELECT node_id, node_data
            FROM {self._node_table_name}
            WHERE workspace = $1 AND node_id = ANY($2)
            """
            attrs_result = await self._query(
                query,
                [self.workspace, node_ids],
                readonly=True
            )
            
            # Create a mapping of node_id -> node_data
            attrs_map = {}
            for row in attrs_result:
                node_data = row["node_data"]
                if isinstance(node_data, str):
                    import json
                    node_data = json.loads(node_data)
                if isinstance(node_data, dict):
                    attrs_map[row["node_id"]] = node_data
            
            # Return nodes with attributes
            nodes = []
            for node_id in node_ids:
                if node_id in attrs_map:
                    nodes.append(attrs_map[node_id])
                else:
                    nodes.append({"entity_id": node_id})
            return nodes
        
        # Fallback: return basic node info
        return [{"entity_id": node_id} for node_id in node_ids]
    
    async def get_all_edges(self) -> list[dict]:
        """Get all edges (returns all hyperedges)."""
        query = f"""
        SELECT edge_id, entities, entity_count, edge_data
        FROM {self.table_name}
        WHERE workspace = $1
        """
        result = await self._query(query, [self.workspace], readonly=True)
        
        edges = []
        for row in result:
            edge_data = row["edge_data"]
            if isinstance(edge_data, str):
                edge_data = json.loads(edge_data)
            
            edges.append({
                "edge_id": row["edge_id"],
                "entities": row["entities"],
                "entity_count": row["entity_count"],
                **edge_data,
            })
        
        return edges
    
    async def get_all_hyperedges(self) -> list[dict]:
        """Get all hyperedges."""
        return await self.get_all_edges()
    
    async def remove_nodes(self, nodes: list[str]):
        """Remove multiple nodes."""
        for node in nodes:
            await self.delete_node(node)
    
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Remove edges containing both nodes in each pair."""
        # Find hyperedges containing each pair and delete them
        for source, target in edges:
            query = f"""
            DELETE FROM {self.table_name}
            WHERE workspace = $1 
              AND $2 = ANY(entities) 
              AND $3 = ANY(entities)
            """
            try:
                async with self.db.pool.acquire() as connection:
                    await connection.execute(query, self.workspace, source, target)
            except Exception as e:
                logger.error(f"[{self.workspace}] Error removing edge ({source}, {target}): {e}")
    
    async def get_all_labels(self) -> list[str]:
        """Get all node labels (entity IDs)."""
        nodes = await self.get_all_nodes()
        return sorted([node["entity_id"] for node in nodes])
    
    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels by node degree (most connected entities)."""
        query = f"""
        SELECT unnest(entities) as entity_id, COUNT(*) as degree
        FROM {self.table_name}
        WHERE workspace = $1
        GROUP BY entity_id
        ORDER BY degree DESC
        LIMIT $2
        """
        result = await self._query(query, [self.workspace, limit], readonly=True)
        return [row["entity_id"] for row in result]
    
    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels with fuzzy matching."""
        query_lower = query.lower().strip()
        if not query_lower:
            return []
        
        # Get all nodes and filter
        all_nodes = await self.get_all_nodes()
        matches = []
        for node in all_nodes:
            node_str = node["entity_id"]
            node_lower = node_str.lower()
            if query_lower in node_lower:
                if node_lower == query_lower:
                    score = 1000
                elif node_lower.startswith(query_lower):
                    score = 500
                else:
                    score = 100 - len(node_str)
                matches.append((node_str, score))
        
        matches.sort(key=lambda x: (-x[1], x[0]))
        return [m[0] for m in matches[:limit]]
    
    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """Get subgraph centered on a node."""
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        
        result = KnowledgeGraph()
        
        if node_label == "*":
            # Get all nodes sorted by degree
            popular = await self.get_popular_labels(limit=max_nodes)
            result.is_truncated = len(await self.get_all_labels()) > max_nodes
            selected_nodes = set(popular)
        else:
            # BFS from starting node
            if not await self.has_node(node_label):
                return result
            
            visited = set()
            queue = [(node_label, 0)]
            selected_nodes = set()
            
            while queue and len(selected_nodes) < max_nodes:
                current, depth = queue.pop(0)
                if current in visited or depth > max_depth:
                    continue
                
                visited.add(current)
                selected_nodes.add(current)
                
                # Get hyperedges containing this node
                query = f"""
                SELECT entities
                FROM {self.table_name}
                WHERE workspace = $1 AND $2 = ANY(entities)
                """
                hyperedges = await self._query(query, [self.workspace, current], readonly=True)
                
                for row in hyperedges:
                    for neighbor in row["entities"]:
                        if neighbor not in visited and len(selected_nodes) < max_nodes:
                            queue.append((neighbor, depth + 1))
            
            result.is_truncated = len(selected_nodes) >= max_nodes
        
        # Get nodes and edges for selected nodes
        # Get all hyperedges that contain at least one selected node
        query = f"""
        SELECT edge_id, entities, entity_count, edge_data
        FROM {self.table_name}
        WHERE workspace = $1
        """
        all_hyperedges = await self._query(query, [self.workspace], readonly=True)
        
        # Build result
        seen_edges = set()
        for node_id in selected_nodes:
            # Create node
            result.nodes.append(KnowledgeGraphNode(
                id=node_id,
                labels=[node_id],
                properties={"entity_id": node_id}
            ))
        
        # Add hyperedges (only if all entities are in selected_nodes)
        for row in all_hyperedges:
            edge_id = row["edge_id"]
            entities = row["entities"]
            
            # Only include if all entities are in selected set
            if all(entity in selected_nodes for entity in entities):
                if edge_id not in seen_edges:
                    edge_data = row["edge_data"]
                    if isinstance(edge_data, str):
                        edge_data = json.loads(edge_data)
                    
                    # For hyperedges, use first entity as source, rest as comma-separated
                    result.edges.append(KnowledgeGraphEdge(
                        id=edge_id,
                        type="HYPEREDGE",
                        source=entities[0] if entities else "",
                        target=",".join(entities[1:]) if len(entities) > 1 else "",
                        properties={**edge_data, "node_count": len(entities), "entities": entities}
                    ))
                    seen_edges.add(edge_id)
        
        logger.info(
            f"[{self.workspace}] Hypergraph query: {len(result.nodes)} nodes, {len(result.edges)} hyperedges"
        )
        return result
    
    async def query_hyperedges_with_core_requirements(
        self,
        core_entities: list[str],
        core_relationships: list[str],
        candidate_entities: list[str] = None,
        min_entity_count: int = 1,
        min_core_entities: int = 1,
    ) -> list[dict]:
        """
        Query hyperedges with core requirements:
        - entities must contain at least min_core_entities core entities
        - if core_relationships is non-empty, edge keywords must contain at least one of them (case-insensitive substring)
        - Higher priority for relations matching more keywords
        
        Args:
            core_entities: List of core entities (at least min_core_entities must be in relations' entities)
            core_relationships: List of core relationship keywords (used for scoring; when non-empty, also required in edge keywords)
            candidate_entities: Optional list of candidate entities (for scoring)
            min_entity_count: Minimum number of entities in hyperedge
            min_core_entities: Minimum number of core entities that must be in hyperedge (default: 1)
        
        Returns:
            List of hyperedge dicts with match_score, sorted by score descending
        """
        if not core_entities:
            logger.warning("Core entities are empty, returning empty results")
            return []
        
        # Query hyperedges containing at least one core entity (for index usage)
        # The actual filtering by min_core_entities will be done in application layer
        # Use array overlap (&&) to check if entities array contains any core entities
        query = f"""
        SELECT edge_id, entities, entity_count, edge_data, created_at
        FROM {self.table_name}
        WHERE workspace = $1
          AND entities && $2::text[]  -- Contains at least one core entity (for index usage)
          AND entity_count >= $3
        """
        
        params = [self.workspace, core_entities, min_entity_count]
        param_idx = 4
        
        # Order by: number of matched core entities (desc), then entity_count (asc)
        query += """
        ORDER BY 
          (SELECT COUNT(*) FROM unnest(entities) WHERE unnest = ANY($2::text[])) DESC,
          entity_count ASC
        LIMIT 500
        """
        
        results = await self._query(query, params, readonly=True)
        
        # Calculate match scores in application layer
        # Filter: only include relations with at least min_core_entities core entities
        scored_results = []
        for row in results:
            edge_entities = row["entities"]
            edge_data = row["edge_data"]
            if isinstance(edge_data, str):
                edge_data = json.loads(edge_data)
            edge_keywords = edge_data.get("keywords", "")
            if isinstance(edge_keywords, (list, tuple)):
                edge_keywords = ", ".join(str(x) for x in edge_keywords)
            elif not isinstance(edge_keywords, str):
                edge_keywords = str(edge_keywords) if edge_keywords is not None else ""
            
            # if core_relationships:
            #     kw_lower = edge_keywords.lower()
            #     if not any(
            #         r.strip() and r.lower() in kw_lower for r in core_relationships
            #     ):
            #         continue
            
            # Calculate match score
            matched_core_entities = len([e for e in edge_entities if e in core_entities])
            matched_candidate_entities = 0
            if candidate_entities:
                matched_candidate_entities = len([e for e in edge_entities if e in candidate_entities])
            
            # Filter: must have at least min_core_entities core entities
            if matched_core_entities < min_core_entities:
                continue
            
            matched_core_relationships = len([
                r for r in core_relationships
                if r.lower() in edge_keywords.lower()
            ])
            
            # Score calculation: 
            match_score = matched_core_entities
            
            scored_results.append({
                "hyperedge": {
                    "edge_id": row["edge_id"],
                    "entities": edge_entities,
                    "entity_count": row["entity_count"],
                    "edge_data": edge_data,
                    "created_at": row.get("created_at"),
                },
                "match_score": match_score,
                "matched_core_entities_count": matched_core_entities,
                "matched_core_relationships_count": matched_core_relationships,
                "matched_candidate_entities_count": matched_candidate_entities,
            })
        
        # Sort by match score (descending)
        scored_results.sort(key=lambda x: x["match_score"], reverse=True)
        
        logger.info(
            f"[{self.workspace}] Query with core requirements: {len(scored_results)} hyperedges found"
        )
        
        return scored_results
    
    async def query_entities_from_hyperedges(
        self,
        entity_keywords: list[str],
        limit: int = 50,
    ) -> list[dict]:
        """
        Query entities directly from the node table (hypergraph_*_nodes) using entity keywords.
        
        This method queries the node table to find entities that match the given keywords.
        It searches in:
        1. node_id (exact match or contains)
        2. node_data JSONB fields (description, entity_name, etc.)
        
        Args:
            entity_keywords: List of entity names/keywords to search for
            limit: Maximum number of entities to return
        
        Returns:
            List of entity dicts with entity_id and node_data
        """
        if not entity_keywords or not hasattr(self, '_node_table_name') or not self._node_table_name:
            return []
        
        unique_keywords = list(dict.fromkeys(k for k in entity_keywords if k))
        if not unique_keywords:
            return []
        
        query = f"""
        SELECT node_id, node_data
        FROM {self._node_table_name}
        WHERE workspace = $1
          AND (
            node_id = ANY($2::text[])
            OR node_data->>'entity_name' = ANY($2::text[])
          )
        LIMIT $3
        """
        params = [self.workspace, unique_keywords, limit]
        
        results = await self._query(query, params, readonly=True)
        
        # Format results
        entities_data = []
        seen_entity_ids = set()
        
        for row in results:
            node_id = row["node_id"]
            if node_id in seen_entity_ids:
                continue
            seen_entity_ids.add(node_id)
            
            node_data = row["node_data"]
            if isinstance(node_data, str):
                import json
                node_data = json.loads(node_data)
            
            # Format similar to vector search results
            entity_dict = {
                "entity_name": node_id,
                "entity_id": node_id,
                "description": node_data.get("description", "") if isinstance(node_data, dict) else "",
                "entity_type": node_data.get("entity_type", "") if isinstance(node_data, dict) else "",
                "source_id": node_data.get("source_id", "") if isinstance(node_data, dict) else "",
                "file_path": node_data.get("file_path", "") if isinstance(node_data, dict) else "",
                "created_at": row.get("updated_at"),  # Use updated_at as created_at
            }
            # Add all other node_data fields
            if isinstance(node_data, dict):
                for key, value in node_data.items():
                    if key not in entity_dict:
                        entity_dict[key] = value
            
            entities_data.append(entity_dict)
        
        logger.info(
            f"[{self.workspace}] Query entities from node table: {len(entities_data)} entities found matching {len(entity_keywords)} keywords"
        )
        
        return entities_data

