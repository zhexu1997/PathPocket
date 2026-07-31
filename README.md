<div align="center">
  <img src="assets/logo_lockup.png" alt="PathPocket Logo" width="400"/>
</div>

# PathPocket: A Multi-modal Agentic Co-pilot for Evidence Grounded Computational Pathology

> **Abstract:** Pathology is the cornerstone of modern medicine, where accurate decision-making relies heavily on evidence-based practices. While artificial intelligence (AI) shows high potential in pathology, most existing models generate diagnostic outputs without providing verifiable clinical evidence, which limits clinician trust and hinders real-world translation. To overcome these limitations, we present PathPocket, a multimodal agentic co-pilot designed specifically for evidence grounded pathology. We construct the most comprehensive pathology evidence corpus to date, encompassing approximately 110,472 public and authorized documents structured across a rigorous hierarchy of evidence from clinical guideline to expert opinion. From this meticulously graded foundation, we build a large-scale multimodal pathology hypergraph containing over 4.55 million entities and 7.10 million relations. Serving as a robust knowledge engine, this hypergraph provides traceable evidence for a collaborative multi-agent reasoning framework integrating case understanding, evidence retrieval, evidence filtering, and response generation. This enables PathPocket to seamlessly resolve a wide spectrum of clinical tasks, ranging from text-only queries to complex multimodal diagnosis involving region-of-interest (ROI) and gigapixel whole-slide image (WSI). We rigorously evaluate the system on a multidimensional benchmark of over 21,000 real-world cases, where it significantly outperforms existing state-of-the-art methods. Crucially, a randomized, two-stage crossover reader study involving both senior and junior pathologists demonstrates that PathPocket substantially enhances clinical workflows, yielding a +6.8\% relative improvement in diagnostic accuracy, a +11.6\% boost in clinician confidence, and a +32.6\% relative reduction in the senior-junior performance gap. By grounding pathology reasoning directly in verifiable literature, PathPocket bridges the gap between algorithmic predictions and clinical practice, demonstrating substantial clinical and translational value for evidence-grounded computational pathology.

---

## 📂 Codebase Structure

Copy this folder to any machine — it is completely self-contained and does not depend on scripts outside `PathPocket/`.

```text
PathPocket/
├── pathpocket/     # Shared RAG engine & multimodal pathology hypergraph utilities
├── hypergraph/     # Construction pipeline (MinerU parsing → extraction → merge → embed)
├── reasoning/      # Multi-modal pathology reasoning & inference pipeline
└── README.md       # This file
```

## 🚀 Quick Start

### 1) Hypergraph Construction

The `hypergraph` module builds a structured knowledge base from raw clinical PDFs. See detailed instructions in [hypergraph/README.md](hypergraph/README.md).

```bash
cd PathPocket/hypergraph

# We provide a master script to automate Stages 1-4:
./run_pipeline.sh

# Or you can run them individually:
export PDF_INPUT_DIR=./pdfs
export MINERU_OUTPUT_DIR=./mineru_output
export WORKING_DIR=./pathpocket_storage

python stage1_parse_cpu.py   # MinerU parse + semantic chunk
# Make sure vLLM is running, then:
python stage2_extract_gpu.py # Entity & relation extraction
python stage3_merge_cpu.py   # Knowledge merge
python stage3_insert_chunk.py --json ./pathpocket_storage/kv_store_text_chunks.json # Insert chunks to DB
python stage4_embed_chunks_gpu.py           # Embed chunks
python stage4_embed_entities_streaming.py   # Embed entities
python stage4_embed_relations_streaming.py  # Embed relations
```

### 2) Pathology Reasoning

The `reasoning` module processes queries using the generated hypergraph, bridging visual features and textual evidence. See [reasoning/README.md](reasoning/README.md).

```bash
cd PathPocket/reasoning
cp .env.example .env   # Configure your API key, Postgres, and local model paths

# Ask a text-only clinical question:
python inference.py --query "What are the high-risk factors for gastric cancer?"

# Ask a complex query grounded with a pathological image:
python inference.py --query "Diagnose this ROI patch based on morphological patterns." --image "/path/to/roi.png"
```

## ⚙️ Configuration

- We highly recommend using **environment variables** or `.env` files for managing local paths and API secrets.
- **Model Weights**: Please place all downloaded model weights (e.g., `bge-m3`, `Virchow2`, `Qwen3-Reranker`, and your preferred LLMs) together in a single directory (e.g., `./models/`) and update your environment variables in `.env` to point to these local paths. They are not bundled in this repository.

## 📝 Citation

If you find PathPocket, the pathology hypergraph, or our code useful in your research, please consider citing our work:

```bibtex
@article{xu2026multi,
  title={A Multi-modal Agentic Co-pilot for Evidence Grounded Computational Pathology},
  author={Xu, Zhe and Zhang, Zhengyu and Cai, Zhiyuan and Xu, Jiahao and Lin, Yijie and Liu, Ziyi and Hou, Junlin and Wang, Hongyi and Nie, Yuxiang and Liang, Ling and others},
  journal={arXiv preprint arXiv:2606.08093},
  year={2026}
}
```

## 🙏 Acknowledgement

We would like to express our gratitude to the developers and maintainers of the open-source projects that made this work possible. Our implementation heavily relies on and draws inspiration from:
- [MinerU](https://github.com/opendatalab/MinerU) for robust PDF parsing and structured data extraction.
- [LightRAG](https://github.com/HKUDS/LightRAG) for the foundational architecture of retrieval-augmented generation.
- [vLLM](https://github.com/vllm-project/vllm) for high-throughput LLM serving.
- The creators of foundational models including [Qwen](https://github.com/QwenLM/Qwen), [Virchow2](https://huggingface.co/paige-ai/Virchow2), and [BGE-M3](https://huggingface.co/BAAI/bge-m3).
