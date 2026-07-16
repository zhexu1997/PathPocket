"""
Map retrieved evidence ``file_path`` to catalog ``evidence_level`` / category using
``evidence_level_title_category.json`` (see ``build_evidence_level_mapping.py``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from evidence_level_labels import label_for_level
except ImportError:  # pragma: no cover - package layout
    _FALLBACK_LABELS = [
        "Guideline",
        "Meta-Analysis & Systematic Review",
        "RCT",
        "Observational & Cohort Study",
        "Case Report",
        "Consensus",
        "Textbook",
        "Expert Opinion",
    ]

    def label_for_level(n: int) -> str:
        try:
            i = int(n) - 1
        except (TypeError, ValueError):
            return f"Level {n}"
        if 0 <= i < len(_FALLBACK_LABELS):
            return _FALLBACK_LABELS[i]
        return f"Level {n}"


_GRAPH_FIELD_SEP = "<SEP>"
_TECHNICAL_DIR_NAMES = frozenset(
    {"auto", "web", "mineru_output", "images", "tables", "output"}
)
_TECHNICAL_CHUNK_BASENAMES = frozenset(
    {"raw.nxml", "raw.xml", "content_list.json"}
)


def _normalize_lookup_key(s: str) -> str:
    return str(s or "").strip().casefold()


def _split_path_segments(file_path: str) -> list[str]:
    raw = str(file_path or "").strip()
    if not raw:
        return []
    parts: list[str] = []
    for piece in raw.split(_GRAPH_FIELD_SEP):
        piece = piece.strip()
        if piece:
            parts.append(piece)
    return parts or [raw]


def _is_technical_chunk_basename(base: str) -> bool:
    low = str(base or "").strip().lower()
    if not low:
        return True
    if low in _TECHNICAL_CHUNK_BASENAMES:
        return True
    return low.endswith(".nxml") or low.endswith("_content_list.json")


def _mineru_source_basename(file_path: str) -> Optional[str]:
    """MinerU chunks often use ``.../Book.pdf/auto/raw.nxml``; catalog keys are ``Book.pdf``."""
    segs = file_path.replace("\\", "/").split("/")
    for i, seg in enumerate(segs):
        if seg.lower() not in ("auto", "web") or i == 0:
            continue
        parent = segs[i - 1].strip()
        if not parent or parent in _TECHNICAL_DIR_NAMES:
            continue
        if parent.lower().endswith(".pdf"):
            return parent
        return f"{parent}.pdf"
    return None


def _lookup_path_candidates(file_path: Any) -> list[str]:
    """Return lookup keys highest-priority first (MinerU PDF folder, then PDF names)."""
    priority: list[str] = []
    secondary: list[str] = []
    seen: set[str] = set()

    def add(bucket: list[str], value: str) -> None:
        v = str(value or "").strip()
        if not v or v in seen:
            return
        seen.add(v)
        bucket.append(v)

    for part in _split_path_segments(str(file_path or "")):
        norm_part = part.replace("\\", "/")
        is_mineru = "mineru_output" in norm_part.lower() or "/auto/" in norm_part.lower()

        mineru_base = _mineru_source_basename(part)
        if mineru_base:
            add(priority, mineru_base)

        segs = norm_part.split("/")
        for seg in reversed(segs):
            seg = seg.strip()
            if not seg or seg in _TECHNICAL_DIR_NAMES:
                continue
            if seg.lower().endswith(".pdf"):
                add(priority, seg)

        base = os.path.basename(norm_part)
        if _is_technical_chunk_basename(base) and len(segs) >= 2:
            parent = segs[-2].strip()
            if parent and parent not in _TECHNICAL_DIR_NAMES:
                add(priority, f"{parent}/{base}")
                add(priority, parent)

        if base and not _is_technical_chunk_basename(base):
            add(secondary, base)

        if not is_mineru:
            add(secondary, part)
            add(secondary, os.path.normpath(part))
            try:
                add(secondary, os.path.realpath(part))
            except OSError:
                pass

    return priority + secondary


@dataclass
class EvidenceLevelCatalog:
    by_path: dict[str, dict[str, Any]]
    by_basename: dict[str, dict[str, Any]]
    by_norm_basename: dict[str, dict[str, Any]]
    by_stem: dict[str, dict[str, Any]]

    def _lookup_single(self, key: str) -> Optional[dict[str, Any]]:
        if key in self.by_path:
            return self.by_path[key]
        base = os.path.basename(key.replace("\\", "/"))
        if _is_technical_chunk_basename(base):
            return None
        if base in self.by_basename:
            return self.by_basename[base]
        norm_base = _normalize_lookup_key(base)
        if norm_base in self.by_norm_basename:
            return self.by_norm_basename[norm_base]
        stem = os.path.splitext(base)[0]
        norm_stem = _normalize_lookup_key(stem)
        if norm_stem in self.by_stem:
            return self.by_stem[norm_stem]
        return None

    def lookup(self, file_path: Any) -> Optional[dict[str, Any]]:
        s = str(file_path or "").strip()
        if not s:
            return None
        low = s.lower()
        if low in ("unknown", "unknown_source"):
            return None
        for candidate in _lookup_path_candidates(s):
            row = self._lookup_single(candidate)
            if row is not None:
                return row
        return None


_CATALOG_BY_KEY: dict[tuple[str, float], EvidenceLevelCatalog] = {}

_EVIDENCE_LLM_FIELD_KEYS = (
    "evidence_catalog_title",
    "evidence_level",
    "evidence_level_label",
)


def _default_catalog_path() -> Optional[str]:
    env = os.getenv("EVIDENCE_LEVEL_TITLE_CATEGORY_JSON", "").strip()
    if env and os.path.isfile(env):
        return env
    pkg_root = os.path.dirname(os.path.abspath(__file__))
    for name in (
        "evidence_level_title_category.slim.json",
        "evidence_level_title_category.json",
    ):
        candidate = os.path.join(pkg_root, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _catalog_path_from_config(global_config: dict) -> Optional[str]:
    v = global_config.get("evidence_level_title_category_json")
    if v is not None and str(v).strip():
        return str(v).strip()
    return _default_catalog_path()


def evidence_fields_for_llm_context(item: dict[str, Any]) -> dict[str, Any]:
    """Pick evidence source / tier / rerank fields for LLM retrieval context."""
    if not isinstance(item, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _EVIDENCE_LLM_FIELD_KEYS:
        val = item.get(key)
        if val is not None and val != "":
            out[key] = val
    rerank_score = item.get("rerank_score")
    if rerank_score is not None:
        try:
            out["rerank_score"] = round(float(rerank_score), 4)
        except (TypeError, ValueError):
            out["rerank_score"] = rerank_score
    return out


def enrich_reference_list_from_catalog(
    reference_list: list[dict], global_config: dict
) -> None:
    """In-place: attach catalog evidence fields to reference_list entries."""
    if not reference_list:
        return
    path = _catalog_path_from_config(global_config)
    cat = get_evidence_level_catalog(path) if path else None
    for ref in reference_list:
        if not isinstance(ref, dict):
            continue
        meta = cat.lookup(ref.get("file_path")) if cat is not None else None
        if not meta:
            continue
        lv = meta["evidence_level"]
        ref["evidence_level"] = lv
        ref["evidence_level_label"] = label_for_level(lv)
        if meta.get("evidence_category") is not None:
            ref["evidence_category"] = meta["evidence_category"]
        if meta.get("evidence_catalog_title") is not None:
            ref["evidence_catalog_title"] = meta["evidence_catalog_title"]


def reference_display_title(ref: dict[str, Any]) -> str:
    title = str(
        ref.get("evidence_catalog_title") or ref.get("title") or ""
    ).strip()
    if title and not title.lower().endswith(".nxml"):
        return title
    fp = str(ref.get("file_path") or "").strip()
    base = os.path.basename(fp) if fp else ""
    for suffix in (".nxml", "_content_list.json", ".json", ".pdf", ".txt"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base.strip() or title


# Backward-compatible alias
_reference_display_title = reference_display_title


def format_reference_list_line(ref: dict[str, Any]) -> str:
    rid = ref.get("reference_id")
    if not rid:
        return ""
    src = reference_display_title(ref)
    line = f"[{rid}] {src}"
    lv = ref.get("evidence_level")
    lbl = ref.get("evidence_level_label")
    if lv is not None:
        line += f" (Level {lv}: {lbl})" if lbl else f" (Level {lv})"
    return line


def get_evidence_level_catalog(path: Optional[str]) -> Optional[EvidenceLevelCatalog]:
    if not path or not str(path).strip():
        return None
    rp = os.path.realpath(str(path).strip())
    try:
        mtime = os.path.getmtime(rp)
    except OSError:
        logger.debug("Evidence level catalog not readable: %s", rp)
        return None
    ck = (rp, mtime)
    cached = _CATALOG_BY_KEY.get(ck)
    if cached is not None:
        return cached
    try:
        with open(rp, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to load evidence level catalog %s: %s", rp, e)
        return None
    records = data.get("records")
    if not isinstance(records, list):
        logger.warning("Evidence level catalog %s: missing records list", rp)
        return None
    by_path: dict[str, dict[str, Any]] = {}
    by_basename: dict[str, dict[str, Any]] = {}
    by_norm_basename: dict[str, dict[str, Any]] = {}
    by_stem: dict[str, dict[str, Any]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        fp = str(rec.get("file_path") or "").strip()
        if not fp:
            continue
        try:
            level = int(rec["evidence_level"])
        except (KeyError, TypeError, ValueError):
            continue
        row: dict[str, Any] = {
            "evidence_level": level,
            "evidence_category": rec.get("category"),
            "evidence_catalog_title": rec.get("title"),
        }
        by_path[fp] = row
        base = os.path.basename(fp.replace("\\", "/"))
        by_basename.setdefault(base, row)
        by_norm_basename.setdefault(_normalize_lookup_key(base), row)
        stem = os.path.splitext(base)[0]
        by_stem.setdefault(_normalize_lookup_key(stem), row)
    cat = EvidenceLevelCatalog(
        by_path=by_path,
        by_basename=by_basename,
        by_norm_basename=by_norm_basename,
        by_stem=by_stem,
    )
    _CATALOG_BY_KEY[ck] = cat
    logger.info(
        "Loaded evidence level catalog %s (%d path keys, %d basename keys)",
        rp,
        len(by_path),
        len(by_basename),
    )
    return cat


def enrich_evidence_items_from_catalog(
    items: list[dict], global_config: dict
) -> None:
    """In-place: set evidence_level / evidence_level_label / evidence_category / evidence_catalog_title when lookup hits."""
    if not items:
        return
    path = _catalog_path_from_config(global_config)
    cat = get_evidence_level_catalog(path)
    if cat is None:
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        meta = cat.lookup(item.get("file_path"))
        if not meta:
            continue
        lv = meta["evidence_level"]
        item["evidence_level"] = lv
        item["evidence_level_label"] = label_for_level(lv)
        if meta.get("evidence_category") is not None:
            item["evidence_category"] = meta["evidence_category"]
        if meta.get("evidence_catalog_title") is not None:
            item["evidence_catalog_title"] = meta["evidence_catalog_title"]

from pathlib import Path

def _default_level_map_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "reasoning" / "data" / "evidence_level_map.json",
        here.parents[1] / "data" / "evidence_level_map.json",
        here.parents[1] / "evidence_level_map.json",
    ]
    env = os.getenv("EVIDENCE_LEVEL_MAP_PATH", "").strip()
    if env:
        return Path(env)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


_LEVEL_MAP_CACHE: Optional[dict[str, int]] = None


def load_evidence_level_map(path: Optional[str | Path] = None) -> dict[str, int]:
    """
    Loads file_path -> level mapping.
    Level is an integer (0..7). Smaller means higher importance.
    """
    global _LEVEL_MAP_CACHE
    if _LEVEL_MAP_CACHE is not None and path is None:
        return _LEVEL_MAP_CACHE

    p = Path(path) if path is not None else _default_level_map_path()
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    # Ensure ints
    level_map = {}
    for k, v in data.items():
        try:
            level_map[str(k)] = int(v)
        except Exception:
            continue

    if path is None:
        _LEVEL_MAP_CACHE = level_map
    return level_map
