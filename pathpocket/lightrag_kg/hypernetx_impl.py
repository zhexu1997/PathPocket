"""HyperNetX-based hypergraph storage for multi-entity relationships."""

import os
import json
from dataclasses import dataclass
from typing import final, Any

import hypernetx as hnx

from pathpocket.lightrag_kg.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from pathpocket.lightrag_utils import logger
from pathpocket.lightrag_base import BaseGraphStorage
from pathpocket.shared_storage import (
    get_namespace_lock,
    get_update_flag,
    set_all_update_flags,
)


@final
@dataclass
class HyperNetXStorage(BaseGraphStorage):
    """Storage for hypergraphs using HyperNetX library.
    
    Supports multi-entity relationships (hyperedges) where each edge can connect
    any number of entities, not just pairs.
    """
    
    @staticmethod
    def load_hypergraph(file_name: str) -> tuple[hnx.Hypergraph, dict] | None:
        """Load hypergraph from JSON file."""
        if os.path.exists(file_name):
            with open(file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct hypergraph from saved data
            # Format: {"hyperedges": {edge_id: [node_list]}, "node_attrs": {...}, "edge_attrs": {...}}
            hyperedges = data.get("hyperedges", {})
            node_attrs = data.get("node_attrs", {})
            edge_attrs = data.get("edge_attrs", {})
            
            H = hnx.Hypergraph(hyperedges)
            return H, {"node_attrs": node_attrs, "edge_attrs": edge_attrs}
        return None
    
    @staticmethod
    def save_hypergraph(H: hnx.Hypergraph, attrs: dict, file_name: str, workspace: str = "_"):
        """Save hypergraph to JSON file."""
        # Convert hypergraph to serializable format
        hyperedges = {}
        for edge_id in H.edges:
            hyperedges[str(edge_id)] = list(H.edges[edge_id])
        
        data = {
            "hyperedges": hyperedges,
            "node_attrs": attrs.get("node_attrs", {}),
            "edge_attrs": attrs.get("edge_attrs", {}),
        }
        
        logger.info(
            f"[{workspace}] Writing hypergraph with {len(H.nodes)} nodes, {len(H.edges)} hyperedges"
        )
        
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
            self.workspace = ""
        
        os.makedirs(workspace_dir, exist_ok=True)
        self._hypergraph_file = os.path.join(
            workspace_dir, f"hypergraph_{self.namespace}.json"
        )
        self._storage_lock = None
        self.storage_updated = None
        self._hypergraph = None
        self._attrs = {"node_attrs": {}, "edge_attrs": {}}
        
        # Load initial hypergraph
        loaded = HyperNetXStorage.load_hypergraph(self._hypergraph_file)
        if loaded is not None:
            self._hypergraph, self._attrs = loaded
            logger.info(
                f"[{self.workspace}] Loaded hypergraph from {self._hypergraph_file} with "
                f"{len(self._hypergraph.nodes)} nodes, {len(self._hypergraph.edges)} hyperedges"
            )
        else:
            logger.info(f"[{self.workspace}] Created new empty hypergraph: {self._hypergraph_file}")
            self._hypergraph = hnx.Hypergraph({})
    
    async def initialize(self):
        """Initialize storage data."""
        self.storage_updated = await get_update_flag(self.namespace, workspace=self.workspace)
        self._storage_lock = get_namespace_lock(self.namespace, workspace=self.workspace)
    
    async def _get_hypergraph(self) -> hnx.Hypergraph:
        """Get hypergraph, reloading if updated by another process."""
        async with self._storage_lock:
            if self.storage_updated.value:
                logger.info(f"[{self.workspace}] Reloading hypergraph due to external modifications")
                loaded = HyperNetXStorage.load_hypergraph(self._hypergraph_file)
                if loaded:
                    self._hypergraph, self._attrs = loaded
                else:
                    self._hypergraph = hnx.Hypergraph({})
                    self._attrs = {"node_attrs": {}, "edge_attrs": {}}
                self.storage_updated.value = False
            return self._hypergraph
    
    async def has_node(self, node_id: str) -> bool:
        H = await self._get_hypergraph()
        return node_id in H.nodes
    
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if any hyperedge contains both nodes."""
        H = await self._get_hypergraph()
        for edge_id in H.edges:
            edge_nodes = set(H.edges[edge_id])
            if source_node_id in edge_nodes and target_node_id in edge_nodes:
                return True
        return False
    
    async def has_hyperedge(self, edge_id: str) -> bool:
        """Check if a hyperedge exists by ID."""
        H = await self._get_hypergraph()
        return edge_id in H.edges
    
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        H = await self._get_hypergraph()
        if node_id in H.nodes:
            return self._attrs["node_attrs"].get(node_id, {})
        return None
    
    async def node_degree(self, node_id: str) -> int:
        """Return number of hyperedges containing this node."""
        H = await self._get_hypergraph()
        if node_id in H.nodes:
            return len(list(H.nodes[node_id]))  # H.nodes[n] returns edges containing n
        return 0
    
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Return sum of degrees of two nodes."""
        src_deg = await self.node_degree(src_id)
        tgt_deg = await self.node_degree(tgt_id)
        return src_deg + tgt_deg
    
    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        """Get first hyperedge containing both nodes."""
        H = await self._get_hypergraph()
        for edge_id in H.edges:
            edge_nodes = set(H.edges[edge_id])
            if source_node_id in edge_nodes and target_node_id in edge_nodes:
                return self._attrs["edge_attrs"].get(str(edge_id), {})
        return None
    
    async def get_hyperedge(self, edge_id: str) -> dict[str, Any] | None:
        """Get hyperedge by ID, including its member nodes."""
        H = await self._get_hypergraph()
        if edge_id in H.edges:
            attrs = self._attrs["edge_attrs"].get(edge_id, {})
            return {
                "id": edge_id,
                "nodes": list(H.edges[edge_id]),
                **attrs
            }
        return None
    
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all node pairs from hyperedges containing this node."""
        H = await self._get_hypergraph()
        if source_node_id not in H.nodes:
            return None
        
        pairs = []
        for edge_id in list(H.nodes[source_node_id]):
            edge_nodes = list(H.edges[edge_id])
            for node in edge_nodes:
                if node != source_node_id:
                    pairs.append((source_node_id, node))
        return pairs
    
    async def get_node_hyperedges(self, node_id: str) -> list[str] | None:
        """Get all hyperedge IDs containing this node."""
        H = await self._get_hypergraph()
        if node_id not in H.nodes:
            return None
        return list(H.nodes[node_id])
    
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Add or update a node with its attributes."""
        H = await self._get_hypergraph()
        # HyperNetX adds nodes automatically when they're in hyperedges
        # Store attributes separately
        self._attrs["node_attrs"][node_id] = node_data
    
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """Add a binary hyperedge (for backward compatibility)."""
        await self.upsert_hyperedge(
            edge_id=f"{source_node_id}_{target_node_id}",
            node_ids=[source_node_id, target_node_id],
            edge_data=edge_data
        )
    
    async def upsert_hyperedge(self, edge_id: str, node_ids: list[str], edge_data: dict[str, str]) -> None:
        """Add or update a hyperedge connecting multiple nodes."""
        H = await self._get_hypergraph()
        
        # Add the hyperedge using new HyperNetX 2.x API
        # In HyperNetX 2.x, add_edges_from takes dict {edge_id: [nodes]}
        if edge_id in H.edges:
            H.remove_edges(edge_id)
        
        # Create new hypergraph with updated edges
        current_edges = {str(e): list(H.edges[e]) for e in H.edges}
        current_edges[edge_id] = list(node_ids)
        self._hypergraph = hnx.Hypergraph(current_edges)
        
        # Store edge attributes
        self._attrs["edge_attrs"][edge_id] = edge_data
        
        logger.debug(f"[{self.workspace}] Added hyperedge {edge_id} connecting {len(node_ids)} nodes")
    
    async def delete_node(self, node_id: str) -> None:
        """Remove a node and all hyperedges containing it."""
        H = await self._get_hypergraph()
        if node_id in H.nodes:
            # Get edges to remove
            edges_to_remove = list(H.nodes[node_id])
            if edges_to_remove:
                H.remove_edges(edges_to_remove)
                for edge_id in edges_to_remove:
                    self._attrs["edge_attrs"].pop(str(edge_id), None)
            
            self._attrs["node_attrs"].pop(node_id, None)
            logger.debug(f"[{self.workspace}] Removed node {node_id} and {len(edges_to_remove)} hyperedges")
    
    async def remove_nodes(self, nodes: list[str]):
        """Remove multiple nodes."""
        for node in nodes:
            await self.delete_node(node)
    
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Remove edges containing both nodes in each pair."""
        H = await self._get_hypergraph()
        all_edges_to_remove = []
        for source, target in edges:
            for edge_id in H.edges:
                edge_nodes = set(H.edges[edge_id])
                if source in edge_nodes and target in edge_nodes:
                    all_edges_to_remove.append(edge_id)
        if all_edges_to_remove:
            H.remove_edges(all_edges_to_remove)
            for edge_id in all_edges_to_remove:
                self._attrs["edge_attrs"].pop(str(edge_id), None)
    
    async def remove_hyperedge(self, edge_id: str):
        """Remove a specific hyperedge by ID."""
        H = await self._get_hypergraph()
        if edge_id in H.edges:
            H.remove_edges([edge_id])
            self._attrs["edge_attrs"].pop(edge_id, None)
    
    async def get_all_labels(self) -> list[str]:
        H = await self._get_hypergraph()
        return sorted([str(node) for node in H.nodes])
    
    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        H = await self._get_hypergraph()
        # Sort by number of hyperedge memberships
        node_degrees = []
        for node in H.nodes:
            degree = len(list(H.nodes[node]))
            node_degrees.append((str(node), degree))
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in node_degrees[:limit]]
    
    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        H = await self._get_hypergraph()
        query_lower = query.lower().strip()
        if not query_lower:
            return []
        
        matches = []
        for node in H.nodes:
            node_str = str(node)
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
        
        H = await self._get_hypergraph()
        result = KnowledgeGraph()
        
        if node_label == "*":
            # Get all nodes sorted by degree
            node_degrees = [(str(n), len(list(H.nodes[n]))) for n in H.nodes]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = set(n for n, _ in node_degrees[:max_nodes])
            result.is_truncated = len(node_degrees) > max_nodes
        else:
            if node_label not in H.nodes:
                return result
            
            # BFS from starting node
            visited = set()
            queue = [(node_label, 0)]
            while queue and len(visited) < max_nodes:
                current, depth = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                if depth < max_depth:
                    # Find neighbors through hyperedges
                    for edge_id in list(H.nodes[current]):
                        for neighbor in H.edges[edge_id]:
                            if neighbor not in visited:
                                queue.append((neighbor, depth + 1))
            
            selected_nodes = visited
            result.is_truncated = len(queue) > 0
        
        # Build result
        seen_edges = set()
        for node in selected_nodes:
            node_attrs = self._attrs["node_attrs"].get(str(node), {})
            result.nodes.append(KnowledgeGraphNode(
                id=str(node),
                labels=[str(node)],
                properties=node_attrs
            ))
            
            # Add hyperedges
            if str(node) in [str(n) for n in H.nodes]:
                for edge_id in list(H.nodes[str(node)]):
                    if str(edge_id) in seen_edges:
                        continue
                    
                    edge_nodes = list(H.edges[edge_id])
                    # Only include if all nodes are in selected set
                    if all(str(n) in [str(s) for s in selected_nodes] for n in edge_nodes):
                        edge_attrs = self._attrs["edge_attrs"].get(str(edge_id), {})
                        # For hyperedges, we use a special format
                        result.edges.append(KnowledgeGraphEdge(
                            id=str(edge_id),
                            type="HYPEREDGE",
                            source=",".join(str(n) for n in edge_nodes),  # All nodes as comma-separated
                            target="",  # Empty for hyperedges
                            properties={**edge_attrs, "node_count": len(edge_nodes)}
                        ))
                        seen_edges.add(str(edge_id))
        
        logger.info(
            f"[{self.workspace}] Hypergraph query: {len(result.nodes)} nodes, {len(result.edges)} hyperedges"
        )
        return result
    
    async def get_all_nodes(self) -> list[dict]:
        H = await self._get_hypergraph()
        nodes = []
        for node in H.nodes:
            node_data = self._attrs["node_attrs"].get(str(node), {}).copy()
            node_data["id"] = str(node)
            nodes.append(node_data)
        return nodes
    
    async def get_all_edges(self) -> list[dict]:
        """Get all hyperedges."""
        H = await self._get_hypergraph()
        edges = []
        for edge_id in H.edges:
            edge_data = self._attrs["edge_attrs"].get(str(edge_id), {}).copy()
            edge_data["id"] = str(edge_id)
            edge_data["nodes"] = list(H.edges[edge_id])
            edges.append(edge_data)
        return edges
    
    async def index_done_callback(self) -> bool:
        """Save hypergraph to disk."""
        async with self._storage_lock:
            if self.storage_updated.value:
                logger.info(f"[{self.workspace}] Hypergraph updated externally, reloading...")
                loaded = HyperNetXStorage.load_hypergraph(self._hypergraph_file)
                if loaded:
                    self._hypergraph, self._attrs = loaded
                self.storage_updated.value = False
                return False
        
        async with self._storage_lock:
            try:
                HyperNetXStorage.save_hypergraph(
                    self._hypergraph, self._attrs, self._hypergraph_file, self.workspace
                )
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                self.storage_updated.value = False
                return True
            except Exception as e:
                logger.error(f"[{self.workspace}] Error saving hypergraph: {e}")
                return False
    
    async def drop(self) -> dict[str, str]:
        """Drop all data."""
        try:
            async with self._storage_lock:
                if os.path.exists(self._hypergraph_file):
                    os.remove(self._hypergraph_file)
                self._hypergraph = hnx.Hypergraph({})
                self._attrs = {"node_attrs": {}, "edge_attrs": {}}
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                self.storage_updated.value = False
                logger.info(f"[{self.workspace}] Dropped hypergraph: {self._hypergraph_file}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping hypergraph: {e}")
            return {"status": "error", "message": str(e)}
