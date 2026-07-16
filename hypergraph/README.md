# Hypergraph construction (stages 1–4)

Convert PDFs into a pathology hypergraph knowledge base.

| Stage | Script | Resource | Output |
|-------|--------|----------|--------|
| 1 | `stage1_parse_cpu.py` | GPU/CPU (MinerU) + CPU | `mineru_output/…` + `kv_store_*.json` |
| 2 | `stage2_extract_gpu.py` | GPU + vLLM | `kv_store_llm_response_cache.json` |
| 3 | `stage3_merge_cpu.py` | CPU (+ optional Postgres) | Merged entities/relations / hypergraph |
| 3b | `stage3_insert_chunk.py` | Postgres | Import chunks into `DOC_CHUNKS` |
| 4 | `stage4_embed_*.py` | GPU | Chunk / entity / relation (+ pathology image) embeddings |

All scripts add the PathPocket root to `sys.path` and import the shared `pathpocket` package.

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `PDF_INPUT_DIR` | `./pdfs` | Directory of PDF files |
| `MINERU_OUTPUT_DIR` | `./mineru_output` | MinerU parse output (also chunking input) |
| `MINERU_BACKEND` | `pipeline` | MinerU backend (`pipeline`, `vlm-…`, …) |
| `MINERU_MODEL_SOURCE` | — | e.g. `local`, `modelscope`, `huggingface` |
| `MINERU_LANG` | — | OCR language hint for MinerU |
| `MINERU_BIN` | `mineru` on PATH | Override MinerU executable |
| `WORKING_DIR` | `./pathpocket_storage` | KV / workspace directory |
| `PIPELINE_LANGUAGE` | `English` | Prompt language |
| `CHUNK_TOKEN_SIZE` | `2400` | Max tokens per text chunk |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | Stage 2 LLM endpoint |
| `VLLM_MODEL_NAME` | `./MODEL/Qwen3-30B-A3B-Instruct-2507-FP8` | Served model id/path |
| `EMBEDDING_MODEL_PATH` | `./models/bge-m3` | Stage 4 text embedding |
| `VIRCHOW2_MODEL_PATH` | `./models/Virchow2` | Stage 4 image embedding |
| `IMAGE_PATH_PREFIX_FROM` / `TO` | — | Optional cross-machine path remap |
| `DOC_CHUNKS_JSON` | `./pathpocket_storage/kv_store_text_chunks.json` | Stage 3b input |
| `POSTGRES_*` | — | Required for Postgres-backed merge / embed |

## Usage Pipeline

You can run the entire pipeline (Stages 1 through 4) automatically using the provided master script:

```bash
./run_pipeline.sh
```

You can pass arguments to control MinerU behavior directly to the shell script:
```bash
./run_pipeline.sh --skip-mineru    # Skip PDF parsing, just run chunking and the rest
./run_pipeline.sh --enhance-only   # Enhance existing MinerU outputs, then run the rest
```

### Stage 1 — MinerU parse + semantic chunking

Install [MinerU](https://github.com/opendatalab/MinerU) for PDF parsing:

```bash
pip install -U "mineru[core]"
```

Full pipeline (MinerU → enhance → chunk):

```bash
export PDF_INPUT_DIR=./pdfs
export MINERU_OUTPUT_DIR=./mineru_output
export WORKING_DIR=./pathpocket_storage
python stage1_parse_cpu.py
```

Partial runs:

```bash
# Chunk only (mineru_output already exists):
python stage1_parse_cpu.py --skip-mineru

# MinerU + enhance only:
python stage1_parse_cpu.py --mineru-only

# Enhance existing MinerU outputs only:
python stage1_parse_cpu.py --enhance-only

# Force re-parse / re-enhance:
python stage1_parse_cpu.py --force
```

Expected MinerU layout:

```text
mineru_output/
  <pdf_stem>/
    auto/   # or ocr / txt / vlm
      <pdf_stem>_content_list.json
      <pdf_stem>_content_list_enhanced.json
      images/
```

Chunking rules: text grouped by semantic paragraphs (max ~2400 tokens); each image and each table is its own chunk.

## Stage 2 — entity & relation extraction

Install and serve [vLLM](https://docs.vllm.ai/) with a long-context instruct model, e.g. Qwen3-30B-A3B-Instruct-FP8:

```bash
vllm serve ./MODEL/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --gpu-memory-utilization 0.95 \
  --max_model_len 32768 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --port 8000 \
  --host 0.0.0.0

export WORKING_DIR=./pathpocket_storage
python stage2_extract_gpu.py
```

## Stage 3 — merge

```bash
export WORKING_DIR=./pathpocket_storage
python stage3_merge_cpu.py
# Optional chunk import to Postgres (DOC_CHUNKS):
python stage3_insert_chunk.py --json ./pathpocket_storage/kv_store_text_chunks.json
```

## Stage 4 — embeddings

```bash
export WORKING_DIR=./pathpocket_storage
export EMBEDDING_MODEL_PATH=./models/bge-m3
export VIRCHOW2_MODEL_PATH=./models/Virchow2
python stage4_embed_chunks_gpu.py
python stage4_embed_entities_streaming.py
python stage4_embed_relations_streaming.py
```
