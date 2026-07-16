"""
Import kv_store_text_chunks.json into PostgreSQL DOC_CHUNKS.

Rows already present for (workspace, id) are skipped (INSERT ... ON CONFLICT DO NOTHING).

Same column mapping as stage3_merge_cpu.save_text_chunks_to_postgres. Connection env
vars match stage4_embed_chunks_gpu.py (POSTGRES_*).

Memory: loading the whole file with json.load() uses roughly the file size (often 2–3×
after parsing) in RAM. For large kv_store_text_chunks.json, use streaming (ijson): install
with `pip install ijson`, or pass --stream always; auto mode streams when file size
exceeds --stream-threshold-mb (default 256).
"""

from __future__ import annotations

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Tuple

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_JSON = os.getenv(
    "DOC_CHUNKS_JSON",
    "./pathpocket_storage/kv_store_text_chunks.json",
)

DEFAULT_STREAM_THRESHOLD_MB = float(
    os.getenv("DOC_CHUNKS_IMPORT_STREAM_MB", "256")
)


def _iter_chunks_from_json(
    path: Path, *, stream: bool
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield (chunk_id, chunk_data). If stream=True, use ijson (one chunk in memory at a time)."""
    if not stream:
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Dict[str, Any]] = json.load(f)
        yield from data.items()
        return

    try:
        import ijson  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "Streaming mode requires the `ijson` package. Install: pip install ijson"
        ) from e

    with open(path, "rb") as f:
        for chunk_id, chunk_data in ijson.kvitems(f, ""):
            if not isinstance(chunk_data, dict):
                chunk_data = {}
            yield str(chunk_id), chunk_data


def _resolve_stream_mode(
    path: Path, mode: str, threshold_mb: float
) -> Tuple[bool, str]:
    """
    Returns (use_stream, reason_for_log).
    mode: auto | always | never
    """
    size_mb = path.stat().st_size / (1024 * 1024)
    if mode == "always":
        return True, "always"
    if mode == "never":
        return False, "never"
    # auto
    if size_mb >= threshold_mb:
        return True, f"auto (file {size_mb:.1f} MB >= {threshold_mb} MB)"
    return False, f"auto (file {size_mb:.1f} MB < {threshold_mb} MB)"


async def import_text_chunks_to_doc_chunks(
    text_chunks_file: Path,
    workspace: str,
    batch_progress_every: int = 100,
    stream_mode: str = "auto",
    stream_threshold_mb: float = DEFAULT_STREAM_THRESHOLD_MB,
) -> None:
    import asyncpg

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "zhexu")
    password = os.getenv("POSTGRES_PASSWORD", "")
    database = os.getenv("POSTGRES_DATABASE", "pathrag")

    if not text_chunks_file.is_file():
        raise FileNotFoundError(f"Text chunks file not found: {text_chunks_file}")

    use_stream, stream_reason = _resolve_stream_mode(
        text_chunks_file, stream_mode, stream_threshold_mb
    )
    file_mb = text_chunks_file.stat().st_size / (1024 * 1024)

    print(f"\n{'='*60}")
    print("Import kv_store_text_chunks.json → PostgreSQL DOC_CHUNKS")
    print(f"{'='*60}")
    print(f"File: {text_chunks_file}")
    print(f"JSON on disk: {file_mb:.1f} MB")
    print(f"Workspace: {workspace}")
    print(
        f"Load mode: {'streaming (ijson)' if use_stream else 'full json.load()'} — {stream_reason}"
    )
    if not use_stream and file_mb >= 128:
        print(
            "⚠️  Large file loaded fully into memory; use --stream auto (default) with "
            "`pip install ijson`, or --stream always, to reduce peak RAM."
        )
    print(f"PostgreSQL: {user}@{host}:{port}/{database}")
    print(f"{'='*60}\n")

    if use_stream:
        try:
            import ijson  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Streaming mode requires `ijson`. Install: pip install ijson\n"
                "Or use --stream never to load the whole file (high RAM for large JSON)."
            ) from e

    chunk_iter = _iter_chunks_from_json(text_chunks_file, stream=use_stream)

    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )

    insert_skip_existing_sql = """
        INSERT INTO DOC_CHUNKS
        (workspace, id, full_doc_id, chunk_order_index, tokens, content, file_path,
         llm_cache_list, create_time, update_time)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10)
        ON CONFLICT (workspace, id) DO NOTHING
    """

    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS DOC_CHUNKS (
                id VARCHAR(255),
                workspace VARCHAR(255),
                full_doc_id VARCHAR(256),
                chunk_order_index INTEGER,
                tokens INTEGER,
                content TEXT,
                file_path TEXT NULL,
                llm_cache_list JSONB NULL DEFAULT '[]'::jsonb,
                create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT DOC_CHUNKS_PK PRIMARY KEY (workspace, id)
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_chunks_workspace ON DOC_CHUNKS (workspace);"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_chunks_full_doc_id ON DOC_CHUNKS (full_doc_id);"
        )

        current_time = datetime.now(timezone.utc).replace(tzinfo=None)
        inserted_count = 0
        skipped_count = 0
        error_count = 0
        idx = 0

        for chunk_id, chunk_data in chunk_iter:
            idx += 1
            try:
                create_time = current_time
                update_time = current_time
                if "create_time" in chunk_data:
                    try:
                        create_time = datetime.fromtimestamp(
                            chunk_data["create_time"], tz=timezone.utc
                        ).replace(tzinfo=None)
                    except (ValueError, TypeError, OSError):
                        pass
                if "update_time" in chunk_data:
                    try:
                        update_time = datetime.fromtimestamp(
                            chunk_data["update_time"], tz=timezone.utc
                        ).replace(tzinfo=None)
                    except (ValueError, TypeError, OSError):
                        pass

                llm_cache_list = chunk_data.get("llm_cache_list", [])
                if not isinstance(llm_cache_list, list):
                    llm_cache_list = []

                status = await conn.execute(
                    insert_skip_existing_sql,
                    workspace,
                    chunk_id,
                    chunk_data.get("full_doc_id", ""),
                    chunk_data.get("chunk_order_index", 0),
                    chunk_data.get("tokens", 0),
                    chunk_data.get("content", ""),
                    chunk_data.get("file_path", ""),
                    json.dumps(llm_cache_list),
                    create_time,
                    update_time,
                )
                # asyncpg returns e.g. "INSERT 0 1" / "INSERT 0 0" — last number = rows inserted
                try:
                    n_inserted = int(str(status).split()[-1])
                except (ValueError, IndexError):
                    n_inserted = 0
                if n_inserted > 0:
                    inserted_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    logger.warning("Chunk %s: %s", chunk_id, e)

            if idx % batch_progress_every == 0:
                print(f"  Processed {idx} ...")

        total = idx
        if total == 0:
            logger.warning("No text chunks in JSON")
            return

        print(f"\n✅ DOC_CHUNKS import finished")
        print(f"   Total in JSON: {total}")
        print(f"   Inserted (new): {inserted_count}")
        print(f"   Skipped (already in DB): {skipped_count}")
        print(f"   Errors: {error_count}")
    finally:
        await conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load kv_store_text_chunks.json into PostgreSQL DOC_CHUNKS."
    )
    p.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=Path(DEFAULT_JSON),
        help=f"Path to kv_store_text_chunks.json (default: {DEFAULT_JSON})",
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=os.getenv("PATHPOCKET_WORKSPACE", "default"),
        help="RAG workspace column (default: env PATHPOCKET_WORKSPACE or 'default')",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N rows (default: 100)",
    )
    p.add_argument(
        "--stream",
        dest="stream_mode",
        choices=("auto", "always", "never"),
        default="auto",
        help="auto: stream with ijson if file >= --stream-threshold-mb; always: always stream; never: json.load whole file",
    )
    p.add_argument(
        "--stream-threshold-mb",
        type=float,
        default=DEFAULT_STREAM_THRESHOLD_MB,
        help="In auto mode, stream if file size >= this many MB (default: env DOC_CHUNKS_IMPORT_STREAM_MB or 256)",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    await import_text_chunks_to_doc_chunks(
        text_chunks_file=args.json_path,
        workspace=args.workspace,
        batch_progress_every=max(1, args.progress_every),
        stream_mode=args.stream_mode,
        stream_threshold_mb=max(0.0, args.stream_threshold_mb),
    )


if __name__ == "__main__":
    asyncio.run(main())
