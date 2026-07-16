# Pathology reasoning pipeline

Self-contained single-question inference over a PathPocket hypergraph store.

```
reasoning/
├── pathpocket is ../pathpocket (shared engine)
├── inference.py            # CLI entry point for a single question
├── qwen_api.py             # Simple OpenAI/Qwen wrapper for LLM generation
├── rag_init.py             # Initialize RAG components
├── rag_models.py           # Model loaders (Embedding, Virchow2, Reranker)
├── bootstrap.py            # Path resolution
├── .env.example
└── requirements-*.txt
```

## Setup

```bash
cd reasoning
cp .env.example .env   # fill OPENAI_API_KEY, Postgres config, local model paths

# Install required packages
pip install -r requirements-api.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt
```

## Usage

Ask a single question and retrieve evidence from your local `WORKING_DIR` hypergraph:

```bash
# Text-only query
python inference.py --query "什么是胃癌的高危因素？"

# Multimodal query (with an image)
python inference.py --query "这张病理图像提示了什么特征？" --image "/path/to/image1.jpg"

# Change retrieval mode (mix, global, local)
python inference.py --query "总结一下结直肠癌的治疗指南" --mode global
```

## Environment Variables
- `OPENAI_API_KEY`: Required for LLM generation.
- `WORKING_DIR`: Path to KV storage (default: `./pathpocket_storage`).
- `GRAPH_STORAGE`, `VECTOR_STORAGE`: e.g., `PGHypergraphStorage`, `PGVectorStorage`.
- `POSTGRES_*`: DB credentials.
- `EMBEDDING_MODEL_PATH`: Local path to text embedding model (e.g., `bge-m3`).
- `VIRCHOW2_MODEL_PATH`: Local path to Virchow2 image embedding model.
- `RERANK_MODEL_PATH`: Local path to reranker model.
