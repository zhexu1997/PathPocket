STORAGE_IMPLEMENTATIONS = {
    "KV_STORAGE": {
        "implementations": [
            "JsonKVStorage",
            "RedisKVStorage",
            "PGKVStorage",
            "MongoKVStorage",
        ],
        "required_methods": ["get_by_id", "upsert"],
    },
    "GRAPH_STORAGE": {
        "implementations": [
            "NetworkXStorage",
            "HyperNetXStorage",
            "Neo4JStorage",
            "PGGraphStorage",
            "PGHypergraphStorage",  # New: Fast hypergraph storage with PostgreSQL
            "MongoGraphStorage",
            "MemgraphStorage",
        ],
        "required_methods": ["upsert_node", "upsert_edge"],
    },
    "VECTOR_STORAGE": {
        "implementations": [
            "NanoVectorDBStorage",
            "MilvusVectorDBStorage",
            "PGVectorStorage",
            "FaissVectorDBStorage",
            "QdrantVectorDBStorage",
            "MongoVectorDBStorage",
            # "ChromaVectorDBStorage",
        ],
        "required_methods": ["query", "upsert"],
    },
    "DOC_STATUS_STORAGE": {
        "implementations": [
            "JsonDocStatusStorage",
            "RedisDocStatusStorage",
            "PGDocStatusStorage",
            "MongoDocStatusStorage",
        ],
        "required_methods": ["get_docs_by_status"],
    },
}

# Storage implementation environment variable without default value
STORAGE_ENV_REQUIREMENTS: dict[str, list[str]] = {
    # KV Storage Implementations
    "JsonKVStorage": [],
    "MongoKVStorage": [
        "MONGO_URI",
        "MONGO_DATABASE",
    ],
    "RedisKVStorage": ["REDIS_URI"],
    "PGKVStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    # Graph Storage Implementations
    "NetworkXStorage": [],
    "HyperNetXStorage": [],  # Hypergraph storage for multi-entity relationships
    "Neo4JStorage": ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"],
    "MongoGraphStorage": [
        "MONGO_URI",
        "MONGO_DATABASE",
    ],
    "MemgraphStorage": ["MEMGRAPH_URI"],
    "AGEStorage": [
        "AGE_POSTGRES_DB",
        "AGE_POSTGRES_USER",
        "AGE_POSTGRES_PASSWORD",
    ],
    "PGGraphStorage": [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DATABASE",
    ],
    "PGHypergraphStorage": [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DATABASE",
    ],
    # Vector Storage Implementations
    "NanoVectorDBStorage": [],
    "MilvusVectorDBStorage": [
        "MILVUS_URI",
        "MILVUS_DB_NAME",
    ],
    # "ChromaVectorDBStorage": [],
    "PGVectorStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    "FaissVectorDBStorage": [],
    "QdrantVectorDBStorage": ["QDRANT_URL"],  # QDRANT_API_KEY has default value None
    "MongoVectorDBStorage": [
        "MONGO_URI",
        "MONGO_DATABASE",
    ],
    # Document Status Storage Implementations
    "JsonDocStatusStorage": [],
    "RedisDocStatusStorage": ["REDIS_URI"],
    "PGDocStatusStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    "MongoDocStatusStorage": [
        "MONGO_URI",
        "MONGO_DATABASE",
    ],
}

# Storage implementation module mapping
STORAGES = {
    "NetworkXStorage": "pathpocket.lightrag_kg.networkx_impl",
    "HyperNetXStorage": "pathpocket.lightrag_kg.hypernetx_impl",
    "JsonKVStorage": "pathpocket.lightrag_kg.json_kv_impl",
    "NanoVectorDBStorage": "pathpocket.lightrag_kg.nano_vector_db_impl",
    "JsonDocStatusStorage": "pathpocket.lightrag_kg.json_doc_status_impl",
    "Neo4JStorage": "pathpocket.lightrag_kg.neo4j_impl",
    "MilvusVectorDBStorage": "pathpocket.lightrag_kg.milvus_impl",
    "MongoKVStorage": "pathpocket.lightrag_kg.mongo_impl",
    "MongoDocStatusStorage": "pathpocket.lightrag_kg.mongo_impl",
    "MongoGraphStorage": "pathpocket.lightrag_kg.mongo_impl",
    "MongoVectorDBStorage": "pathpocket.lightrag_kg.mongo_impl",
    "RedisKVStorage": "pathpocket.lightrag_kg.redis_impl",
    "RedisDocStatusStorage": "pathpocket.lightrag_kg.redis_impl",
    "ChromaVectorDBStorage": "pathpocket.lightrag_kg.deprecated.chroma_impl",
    "PGKVStorage": "pathpocket.lightrag_kg.postgres_impl",
    "PGVectorStorage": "pathpocket.lightrag_kg.postgres_impl",
    "AGEStorage": "pathpocket.lightrag_kg.age_impl",
    "PGGraphStorage": "pathpocket.lightrag_kg.postgres_impl",
    "PGHypergraphStorage": "pathpocket.lightrag_kg.pg_hypergraph_impl",
    "PGDocStatusStorage": "pathpocket.lightrag_kg.postgres_impl",
    "FaissVectorDBStorage": "pathpocket.lightrag_kg.faiss_impl",
    "QdrantVectorDBStorage": "pathpocket.lightrag_kg.qdrant_impl",
    "MemgraphStorage": "pathpocket.lightrag_kg.memgraph_impl",
}


def verify_storage_implementation(storage_type: str, storage_name: str) -> None:
    """Verify if storage implementation is compatible with specified storage type

    Args:
        storage_type: Storage type (KV_STORAGE, GRAPH_STORAGE etc.)
        storage_name: Storage implementation name

    Raises:
        ValueError: If storage implementation is incompatible or missing required methods
    """
    if storage_type not in STORAGE_IMPLEMENTATIONS:
        raise ValueError(f"Unknown storage type: {storage_type}")

    storage_info = STORAGE_IMPLEMENTATIONS[storage_type]
    if storage_name not in storage_info["implementations"]:
        raise ValueError(
            f"Storage implementation '{storage_name}' is not compatible with {storage_type}. "
            f"Compatible implementations are: {', '.join(storage_info['implementations'])}"
        )
