#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# PathPocket Hypergraph Construction Pipeline
# Automates stages 1-4: Parsing, Extraction, Merging, and Embedding
# ==============================================================================

# Ensure we are in the hypergraph directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "      PathPocket Hypergraph Construction Pipeline         "
echo "=========================================================="

# ------------------------------------------------------------------------------
# 1. Environment Configuration
# ------------------------------------------------------------------------------
# These defaults can be overridden by exporting the variables before running
export PDF_INPUT_DIR="${PDF_INPUT_DIR:-./pdfs}"
export MINERU_OUTPUT_DIR="${MINERU_OUTPUT_DIR:-./mineru_output}"
export WORKING_DIR="${WORKING_DIR:-./pathpocket_storage}"
export PIPELINE_LANGUAGE="${PIPELINE_LANGUAGE:-English}"
export CHUNK_TOKEN_SIZE="${CHUNK_TOKEN_SIZE:-2400}"

# LLM setup for extraction (Stage 2)
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-./MODEL/Qwen3-30B-A3B-Instruct-2507-FP8}"

# Embedding setup (Stage 4)
export EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-./models/bge-m3}"
export VIRCHOW2_MODEL_PATH="${VIRCHOW2_MODEL_PATH:-./models/Virchow2}"

# Display configuration
echo "Configuration:"
echo "  PDF_INPUT_DIR      : $PDF_INPUT_DIR"
echo "  MINERU_OUTPUT_DIR  : $MINERU_OUTPUT_DIR"
echo "  WORKING_DIR        : $WORKING_DIR"
echo "  PIPELINE_LANGUAGE  : $PIPELINE_LANGUAGE"
echo "  VLLM_BASE_URL      : $VLLM_BASE_URL"
echo "  EMBEDDING_MODEL    : $EMBEDDING_MODEL_PATH"
echo "----------------------------------------------------------"

# Helper functions
step_start() {
    echo -e "\n\033[1;34m>>> Starting Stage: $1...\033[0m"
}

step_done() {
    echo -e "\033[1;32m<<< Completed Stage: $1\033[0m\n"
}

# ------------------------------------------------------------------------------
# 2. Execution Pipeline
# ------------------------------------------------------------------------------

# --- Stage 1 ---
step_start "1. MinerU Parse + Semantic Chunking"
# Pass any CLI arguments (e.g. --skip-mineru) directly to stage 1
python3 stage1_parse_cpu.py "$@"
step_done "1. MinerU Parse + Semantic Chunking"

# --- Stage 2 ---
step_start "2. Entity & Relation Extraction"
echo "Note: This step requires vLLM to be running at $VLLM_BASE_URL"
python3 stage2_extract_gpu.py
step_done "2. Entity & Relation Extraction"

# --- Stage 3 ---
step_start "3a. Merge Entities and Relations"
python3 stage3_merge_cpu.py
step_done "3a. Merge Entities and Relations"

step_start "3b. Insert Chunks into DB"
DOC_CHUNKS_JSON="${WORKING_DIR}/kv_store_text_chunks.json"
if [ -f "$DOC_CHUNKS_JSON" ]; then
    python3 stage3_insert_chunk.py --json "$DOC_CHUNKS_JSON"
else
    echo "Warning: $DOC_CHUNKS_JSON not found, skipping chunk insertion."
fi
step_done "3b. Insert Chunks into DB"

# --- Stage 4 ---
step_start "4a. Embed Chunks (Text & Images)"
python3 stage4_embed_chunks_gpu.py
step_done "4a. Embed Chunks"

step_start "4b. Embed Entities"
python3 stage4_embed_entities_streaming.py
step_done "4b. Embed Entities"

step_start "4c. Embed Relations"
python3 stage4_embed_relations_streaming.py
step_done "4c. Embed Relations"

echo "=========================================================="
echo "🎉 Hypergraph Construction Pipeline completed successfully!"
echo "=========================================================="
