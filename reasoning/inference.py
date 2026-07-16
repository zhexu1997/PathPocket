#!/usr/bin/env python3
"""
Single-question inference entry point for PathPocket reasoning.
Demonstrates how to initialize the RAG engine, perform retrieval, and generate an answer.
"""

import argparse
import asyncio
import os
import sys

import bootstrap  # Ensures correct import paths
from pathpocket.lightrag_base import QueryParam
from rag_init import initialize_rag

async def main() -> None:
    parser = argparse.ArgumentParser(description="PathPocket Single Question Inference")
    parser.add_argument("--query", "-q", type=str, required=True, help="The question to ask")
    parser.add_argument("--image", "-i", type=str, action="append", default=[], help="Optional image paths (repeatable)")
    parser.add_argument("--mode", "-m", type=str, default="mix", help="Retrieval mode (mix, global, local)")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "60")), help="Top entities/relations to retrieve")
    parser.add_argument("--chunk-top-k", type=int, default=int(os.getenv("CHUNK_TOP_K", "30")), help="Top chunks to retrieve")
    args = parser.parse_args()

    print("=" * 60)
    print("PathPocket Single Question Inference")
    print("=" * 60)
    print(f"Query: {args.query}")
    if args.image:
        print(f"Images: {args.image}")
    print(f"Mode: {args.mode}")

    print("\n[1/3] Initializing RAG components (Database, Embeddings, Virchow2, LLM)...")
    rag = await initialize_rag()

    query_param = QueryParam(
        mode=args.mode,
        top_k=args.top_k,
        chunk_top_k=args.chunk_top_k,
        # Default thresholds
        cosine_better_than_threshold=float(os.getenv("COSINE_BETTER_THAN_THRESHOLD", "0.35")),
        min_rerank_score=float(os.getenv("MIN_RERANK_SCORE", "0.25")),
    )

    print("\n[2/3] Retrieving evidence and generating answer...")
    if args.image:
        # Multimodal query
        multimodal_content = []
        for img_path in args.image:
            multimodal_content.append({"type": "image", "image_path": img_path})
        
        answer = await rag.aquery_with_multimodal(
            args.query,
            multimodal_content=multimodal_content,
            param=query_param
        )
    else:
        # Text-only query
        answer = await rag.aquery(
            args.query,
            param=query_param
        )

    print("\n[3/3] Answer generated successfully:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
