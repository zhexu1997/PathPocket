#!/usr/bin/env python3
"""
Stage 1: MinerU PDF parse → enhance → semantic chunking (CPU only).

Input:  PDF files under PDF_INPUT_DIR (optional if mineru_output already exists)
Output: MINERU_OUTPUT_DIR/<stem>/{auto,…}/*_content_list(_enhanced).json
        WORKING_DIR/kv_store_doc_status.json
        WORKING_DIR/kv_store_full_docs.json
        WORKING_DIR/kv_store_text_chunks.json

Requires MinerU CLI on PATH for PDF parsing:
  pip install -U "mineru[core]"   # see https://github.com/opendatalab/MinerU

Examples:
  export PDF_INPUT_DIR=./pdfs
  export MINERU_OUTPUT_DIR=./mineru_output
  export WORKING_DIR=./pathpocket_storage
  python stage1_parse_cpu.py

  # Chunk only (mineru_output already present):
  python stage1_parse_cpu.py --skip-mineru

  # MinerU + enhance only:
  python stage1_parse_cpu.py --mineru-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time as time_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pathpocket import (
    PathPocket,
    PathPocketConfig,
    find_all_enhanced_parsed_files,
    fix_invalid_doc_status,
    merge_small_chunks,
    merge_titles_with_content,
)
from pathpocket.core import compute_mdhash_id
from pathpocket.operate import chunking_by_token_size
from pathpocket.utils import TiktokenTokenizer, separate_content

logging.getLogger("httpx").setLevel(logging.WARNING)

PARSE_METHODS = ("auto", "ocr", "txt", "vlm")


# ---------------------------------------------------------------------------
# MinerU (former stage 0)
# ---------------------------------------------------------------------------

def _find_mineru_bin() -> str:
    env_bin = os.getenv("MINERU_BIN", "").strip()
    if env_bin:
        return env_bin
    which = shutil.which("mineru")
    if which:
        return which
    raise FileNotFoundError(
        "MinerU CLI not found. Install with: pip install -U 'mineru[core]' "
        "or set MINERU_BIN to the executable path."
    )


def iter_pdfs(input_dir: Path, recursive: bool) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"PDF input directory not found: {input_dir}")
    if recursive:
        return sorted(
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == ".pdf"
        )
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )


def content_list_paths(out_root: Path, stem: str) -> List[Path]:
    paths: List[Path] = []
    stem_dir = out_root / stem
    if not stem_dir.is_dir():
        return paths
    for method in PARSE_METHODS:
        method_dir = stem_dir / method
        if not method_dir.is_dir():
            continue
        candidate = method_dir / f"{stem}_content_list.json"
        if candidate.is_file():
            paths.append(candidate)
    return paths


def is_already_parsed(out_root: Path, stem: str) -> bool:
    return bool(content_list_paths(out_root, stem))


def run_mineru(
    pdf: Path,
    output_dir: Path,
    *,
    mineru_bin: str,
    backend: str,
    source: str,
    lang: str,
    extra_args: Sequence[str],
    timeout_sec: int,
) -> None:
    cmd = [mineru_bin, "-p", str(pdf), "-o", str(output_dir), "-b", backend]
    if source:
        cmd.extend(["--source", source])
    if lang:
        cmd.extend(["-l", lang])
    cmd.extend(list(extra_args))

    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec if timeout_sec > 0 else None,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"MinerU failed for {pdf.name} (exit {result.returncode}):\n{err[-4000:]}"
        )
    if result.stdout.strip():
        tail = "\n".join(result.stdout.strip().splitlines()[-20:])
        print(tail)


def enhance_one_content_list(
    json_file: Path,
    method_dir: Path,
    stem: str,
    *,
    force: bool,
) -> Optional[Path]:
    from pathpocket.utils import enhance_content_list, read_content_list_from_json

    enhanced_file = method_dir / f"{stem}_content_list_enhanced.json"
    if enhanced_file.is_file() and not force:
        print(f"  skip enhance (exists): {enhanced_file.name}")
        return enhanced_file

    content_list = read_content_list_from_json(json_file, method_dir)
    if not content_list:
        print(f"  warn: empty content_list: {json_file}")
        return None

    enhanced = enhance_content_list(content_list, file_name=stem)
    with open(enhanced_file, "w", encoding="utf-8") as f:
        json.dump(enhanced, f, ensure_ascii=False, indent=2)
    print(
        f"  enhanced: {enhanced_file.name} "
        f"({len(content_list)} → {len(enhanced)} items)"
    )
    return enhanced_file


def enhance_output_tree(
    out_root: Path, stems: Iterable[str], *, force: bool
) -> Tuple[int, int]:
    ok, fail = 0, 0
    for stem in stems:
        for json_file in content_list_paths(out_root, stem):
            method_dir = json_file.parent
            try:
                if enhance_one_content_list(json_file, method_dir, stem, force=force):
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1
                print(f"  enhance error ({stem}): {e}")
    return ok, fail


def run_mineru_stage(args: argparse.Namespace) -> Tuple[int, List[str]]:
    """Run MinerU parse (+ optional enhance). Returns (exit_code, stems)."""
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.mineru_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step A: MinerU PDF parse")
    print("=" * 60)
    print(f"Input PDFs:  {input_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Backend:     {args.backend}")
    print(f"Enhance:     {args.enhance}")
    print(f"Force:       {args.force}")
    print("=" * 60)

    if args.enhance_only:
        stems = [
            d.name
            for d in output_dir.iterdir()
            if d.is_dir() and d.name not in PARSE_METHODS
        ]
        if not stems:
            print(f"No parsed folders under {output_dir}")
            return 1, []
        ok, fail = enhance_output_tree(output_dir, sorted(stems), force=args.force)
        print(f"Enhance done: ok={ok} fail={fail}")
        return (0 if fail == 0 else 1), sorted(stems)

    pdfs = iter_pdfs(input_dir, recursive=args.recursive)
    if not pdfs:
        print(f"No PDF files found under {input_dir}")
        return 1, []

    print(f"Found {len(pdfs)} PDF(s)")
    mineru_bin = _find_mineru_bin()
    print(f"MinerU binary: {mineru_bin}")

    success, skipped, failed = 0, 0, 0
    done_stems: List[str] = []

    for i, pdf in enumerate(pdfs, 1):
        stem = pdf.stem
        print(f"\n[{i}/{len(pdfs)}] {pdf.name}")
        if not args.force and is_already_parsed(output_dir, stem):
            print("  skip (already parsed)")
            skipped += 1
            done_stems.append(stem)
            continue
        try:
            run_mineru(
                pdf,
                output_dir,
                mineru_bin=mineru_bin,
                backend=args.backend,
                source=args.source,
                lang=args.lang,
                extra_args=args.extra_arg,
                timeout_sec=args.timeout,
            )
            if not is_already_parsed(output_dir, stem):
                raise RuntimeError(
                    f"MinerU finished but no *_content_list.json under "
                    f"{output_dir / stem}/{{auto,ocr,txt,vlm}}/"
                )
            success += 1
            done_stems.append(stem)
            print(f"  ok: {stem}")
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"  timeout after {args.timeout}s")
        except Exception as e:
            failed += 1
            print(f"  failed: {e}")

    if args.enhance and done_stems:
        print("\n--- Enhance content lists ---")
        e_ok, e_fail = enhance_output_tree(output_dir, done_stems, force=args.force)
        print(f"Enhance: ok={e_ok} fail={e_fail}")

    print(
        f"\nMinerU done: parsed={success} skipped={skipped} failed={failed} "
        f"total={len(pdfs)}"
    )
    return (0 if failed == 0 else 1), done_stems


# ---------------------------------------------------------------------------
# Semantic chunking (former stage 1)
# ---------------------------------------------------------------------------

async def process_text_content_stage1(
    rag,
    text: str,
    file_path: str,
    doc_id: str,
    split_by_character: str = "\n\n",
    split_by_character_only: bool = False,
):
    chunk_token_size = getattr(rag.rag_core, "chunk_token_size", 1200)
    chunk_overlap_token_size = getattr(rag.rag_core, "chunk_overlap_token_size", 100)

    chunks_list = chunking_by_token_size(
        tokenizer=rag.rag_engine.tokenizer,
        content=text,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        chunk_overlap_token_size=chunk_overlap_token_size,
        chunk_token_size=chunk_token_size,
    )

    chunks = {}
    chunk_ids = []
    for chunk_data in chunks_list:
        chunk_id = compute_mdhash_id(chunk_data["content"], prefix="chunk-")
        chunk_ids.append(chunk_id)
        chunks[chunk_id] = {
            **chunk_data,
            "full_doc_id": doc_id,
            "file_path": os.path.basename(file_path),
            "llm_cache_list": [],
        }

    await rag.rag_engine.text_chunks.upsert(chunks)

    if rag.rag_engine.full_docs:
        await rag.rag_engine.full_docs.upsert({
            doc_id: {
                "content": text,
                "file_path": file_path,
                "chunks_count": len(chunks),
                "update_time": int(time_module.time()),
                "_id": doc_id,
            }
        })

    return chunk_ids


async def process_multimodal_content_stage1(
    rag,
    multimodal_items: List[Dict[str, Any]],
    file_path: str,
    doc_id: str,
):
    if not multimodal_items:
        return []

    existing_doc_status = await rag.rag_engine.doc_status.get_by_id(doc_id)
    existing_chunks_count = (
        existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
    )

    multimodal_data_list = []
    for index, item in enumerate(multimodal_items):
        content_type = item.get("type", "unknown")
        processor = rag._get_processor_for_type(content_type)
        if not processor:
            continue

        item_info = {
            "page_idx": item.get("page_idx", 0),
            "index": index,
            "type": content_type,
        }
        description, entity_info = await processor.generate_description_only(
            modal_content=item,
            content_type=content_type,
            item_info=item_info,
            entity_name=None,
        )
        multimodal_data_list.append({
            "index": index,
            "content_type": content_type,
            "description": description,
            "entity_info": entity_info,
            "original_item": item,
            "item_info": item_info,
            "chunk_order_index": existing_chunks_count + index,
            "file_path": file_path,
        })

    if not multimodal_data_list:
        return []

    rag_chunks = rag._convert_to_rag_chunks(multimodal_data_list, file_path, doc_id)
    await rag.rag_engine.text_chunks.upsert(rag_chunks)
    return list(rag_chunks.keys())


async def update_doc_status_stage1(
    rag,
    doc_id: str,
    chunk_ids: List[str],
    file_path: str,
    text_content: str = "",
    processing_start_time: int = None,
):
    import datetime
    import hashlib
    from datetime import timezone

    from pathpocket.lightrag_utils import get_content_summary

    existing_doc_status = await rag.rag_engine.doc_status.get_by_id(doc_id)
    content_summary = get_content_summary(text_content, max_length=100) if text_content else ""
    content_length = len(text_content) if text_content else 0

    if existing_doc_status and existing_doc_status.get("track_id"):
        track_id = existing_doc_status.get("track_id")
    else:
        timestamp = datetime.datetime.now(timezone.utc)
        time_str = timestamp.strftime("%Y%m%d_%H%M%S")
        hash_str = hashlib.md5(doc_id.encode()).hexdigest()[:8]
        track_id = f"insert_{time_str}_{hash_str}"

    current_time = int(time_module.time())
    if processing_start_time is None:
        processing_start_time = current_time
    processing_end_time = current_time
    now_iso = datetime.datetime.now(timezone.utc).isoformat()

    if existing_doc_status:
        doc_status_data = {
            "status": "preprocessed",
            "multimodal_processed": False,
            "chunks_count": len(chunk_ids),
            "chunks_list": chunk_ids,
            "content_summary": content_summary or existing_doc_status.get("content_summary", ""),
            "content_length": content_length or existing_doc_status.get("content_length", 0),
            "file_path": file_path or existing_doc_status.get("file_path", ""),
            "track_id": track_id,
            "created_at": existing_doc_status.get("created_at", now_iso),
            "updated_at": now_iso,
            "update_time": current_time,
            "metadata": {
                **existing_doc_status.get("metadata", {}),
                "processing_start_time": processing_start_time,
                "processing_end_time": processing_end_time,
            },
        }
    else:
        doc_status_data = {
            "content_summary": content_summary,
            "content_length": content_length,
            "file_path": file_path,
            "status": "preprocessed",
            "multimodal_processed": False,
            "chunks_count": len(chunk_ids),
            "chunks_list": chunk_ids,
            "track_id": track_id,
            "created_at": now_iso,
            "updated_at": now_iso,
            "update_time": current_time,
            "metadata": {
                "processing_start_time": processing_start_time,
                "processing_end_time": processing_end_time,
            },
        }

    await rag.rag_engine.doc_status.upsert({doc_id: doc_status_data})


async def run_chunking_stage(args: argparse.Namespace) -> int:
    """Semantic chunking from MINERU_OUTPUT_DIR into WORKING_DIR."""
    output_dir = args.mineru_output_dir
    _language = args.language
    max_tokens = args.chunk_token_size
    working_dir = args.working_dir

    print(f"\n{'='*60}")
    print("Step B: Semantic chunking (CPU)")
    print(f"{'='*60}")
    print(f"MinerU output: {output_dir}")
    print(f"Working dir:   {working_dir}")
    print(f"{'='*60}\n")

    parsed_files = find_all_enhanced_parsed_files(Path(output_dir))
    if not parsed_files:
        print(f"Error: no parsed content under {output_dir}")
        return 1

    print(f"Found {len(parsed_files)} parsed file(s)\n")

    config = PathPocketConfig(
        working_dir=working_dir,
        enable_image_processing=True,
        enable_table_processing=True,
    )

    async def dummy_llm_func(*_a, **_kw):
        raise RuntimeError("LLM should not be called in Stage 1")

    class DummyEmbeddingFunc:
        def __init__(self):
            self.embedding_dim = 1024
            self.max_token_size = 4096

        async def __call__(self, texts):
            raise RuntimeError("Embedding should not be called in Stage 1")

    rag = PathPocket(
        config=config,
        llm_model_func=dummy_llm_func,
        embedding_func=DummyEmbeddingFunc(),
        rag_engine_kwargs={
            "graph_storage": "HyperNetXStorage",
            "vector_storage": "NanoVectorDBStorage",
            "addon_params": {"language": _language},
            "chunk_token_size": max_tokens,
            "enable_llm_cache_for_entity_extract": False,
        },
    )

    await fix_invalid_doc_status(rag)
    init_result = await rag._ensure_rag_engine_initialized()
    if not init_result.get("success"):
        raise RuntimeError(f"Failed to initialize RAG engine: {init_result.get('error')}")

    tokenizer = TiktokenTokenizer(model_name="gpt-4o-mini")
    total_files = len(parsed_files)

    for idx, (content_list, file_stem, method, json_file) in enumerate(parsed_files, 1):
        print(f"\n{'='*60}")
        print(f"Chunking {idx}/{total_files}: {file_stem} ({method})")
        print(f"{'='*60}")

        merged_content_list = merge_titles_with_content(content_list)
        final_content_list = merge_small_chunks(
            merged_content_list,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
        )
        doc_id = compute_mdhash_id(
            "\n".join([str(item) for item in final_content_list]), prefix="doc-"
        )

        doc_status_data = await rag.rag_engine.doc_status.get_by_id(doc_id)
        if doc_status_data and doc_status_data.get("status") == "processed":
            print(f"⏭️  Skipping {file_stem}: already processed")
            continue

        text_content_str, multimodal_items = separate_content(final_content_list)
        file_name = f"{file_stem}.pdf" if file_stem else "parsed_document.pdf"

        try:
            processing_start_time = int(time_module.time())
            all_chunk_ids: List[str] = []

            if text_content_str and text_content_str.strip():
                text_chunk_ids = await process_text_content_stage1(
                    rag=rag,
                    text=text_content_str,
                    file_path=file_name,
                    doc_id=doc_id,
                )
                all_chunk_ids.extend(text_chunk_ids)

            if multimodal_items:
                multimodal_chunk_ids = await process_multimodal_content_stage1(
                    rag=rag,
                    multimodal_items=multimodal_items,
                    file_path=file_name,
                    doc_id=doc_id,
                )
                all_chunk_ids.extend(multimodal_chunk_ids)

            await update_doc_status_stage1(
                rag=rag,
                doc_id=doc_id,
                chunk_ids=all_chunk_ids,
                file_path=file_name,
                text_content=text_content_str,
                processing_start_time=processing_start_time,
            )
            print(f"✅ {file_stem} chunking completed!")

        except Exception as e:
            print(f"❌ Error processing {file_stem}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("Saving storages...")
    print(f"{'='*60}")
    await rag.rag_engine.text_chunks.index_done_callback()
    if rag.rag_engine.full_docs:
        await rag.rag_engine.full_docs.index_done_callback()
    await rag.rag_engine.doc_status.index_done_callback()

    print(f"\n✅ Stage 1 completed!")
    print(f"  - {working_dir}/kv_store_doc_status.json")
    print(f"  - {working_dir}/kv_store_full_docs.json")
    print(f"  - {working_dir}/kv_store_text_chunks.json")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1: MinerU PDF parse + enhance + semantic chunking"
    )

    # Paths
    p.add_argument(
        "--input-dir",
        default=os.getenv("PDF_INPUT_DIR", "./pdfs"),
        help="PDF input directory (env: PDF_INPUT_DIR)",
    )
    p.add_argument(
        "--mineru-output-dir",
        default=os.getenv("MINERU_OUTPUT_DIR", "./mineru_output"),
        help="MinerU output directory (env: MINERU_OUTPUT_DIR)",
    )
    p.add_argument(
        "--working-dir",
        default=os.getenv("WORKING_DIR", "./pathpocket_storage"),
        help="KV storage directory (env: WORKING_DIR)",
    )

    # Mode switches
    p.add_argument(
        "--skip-mineru",
        action="store_true",
        help="Skip MinerU; chunk from existing mineru_output only",
    )
    p.add_argument(
        "--mineru-only",
        action="store_true",
        help="Run MinerU (+ enhance) only; skip chunking",
    )
    p.add_argument(
        "--enhance-only",
        action="store_true",
        help="Only enhance existing MinerU outputs (no mineru, no chunking)",
    )

    # MinerU options
    p.add_argument("--recursive", action="store_true", help="Recursively find PDFs")
    p.add_argument("--force", "-f", action="store_true", help="Re-parse / re-enhance")
    p.add_argument(
        "--backend", "-b",
        default=os.getenv("MINERU_BACKEND", "pipeline"),
        help="MinerU backend (env: MINERU_BACKEND)",
    )
    p.add_argument(
        "--source",
        default=os.getenv("MINERU_MODEL_SOURCE", ""),
        help="MinerU model source (env: MINERU_MODEL_SOURCE)",
    )
    p.add_argument(
        "--lang", "-l",
        default=os.getenv("MINERU_LANG", ""),
        help="MinerU OCR language (env: MINERU_LANG)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("MINERU_TIMEOUT_SEC", "3600")),
        help="Per-PDF timeout seconds (env: MINERU_TIMEOUT_SEC)",
    )
    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra arg for mineru CLI (repeatable)",
    )
    p.add_argument(
        "--enhance",
        dest="enhance",
        action="store_true",
        default=True,
        help="Write *_content_list_enhanced.json (default: on)",
    )
    p.add_argument(
        "--no-enhance",
        dest="enhance",
        action="store_false",
        help="Skip writing *_content_list_enhanced.json",
    )

    # Chunking options
    p.add_argument(
        "--language",
        default=os.getenv("PIPELINE_LANGUAGE", "English"),
        help="Prompt language (env: PIPELINE_LANGUAGE)",
    )
    p.add_argument(
        "--chunk-token-size",
        type=int,
        default=int(os.getenv("CHUNK_TOKEN_SIZE", "2400")),
        help="Max tokens per text chunk (env: CHUNK_TOKEN_SIZE)",
    )
    return p.parse_args()


async def async_main() -> int:
    args = parse_args()

    if args.enhance_only:
        args.skip_mineru = False
        args.mineru_only = True

    mineru_rc = 0
    if not args.skip_mineru:
        mineru_rc, _ = run_mineru_stage(args)
        if mineru_rc != 0 and not args.mineru_only:
            print("MinerU stage had failures; continuing to chunking if output exists...")
    elif args.enhance and not args.mineru_only:
        out = Path(args.mineru_output_dir).expanduser().resolve()
        stems = [
            d.name for d in out.iterdir()
            if d.is_dir() and d.name not in PARSE_METHODS
        ]
        if stems:
            print("--- Enhance existing MinerU outputs ---")
            enhance_output_tree(out, sorted(stems), force=args.force)

    if args.mineru_only or args.enhance_only:
        return mineru_rc

    chunk_rc = await run_chunking_stage(args)
    return chunk_rc if mineru_rc == 0 else max(mineru_rc, chunk_rc)


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
