"""GPU embedding / Virchow2 / rerank model loaders (copied from batch_inference_llm)."""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pathpocket.lightrag_utils import EmbeddingFunc

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    CrossEncoder = None
    torch = None
    np = None

try:
    import ollama as ollama_lib
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama_lib = None


def create_embedding_func():
    """Create embedding function based on available method"""
    EMBEDDING_METHOD = os.getenv("EMBEDDING_METHOD", "direct").lower()
    OLLAMA_EMBEDDING_HOST = os.getenv("OLLAMA_EMBEDDING_HOST", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest")
    EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "./models/bge-m3")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    
    if EMBEDDING_METHOD == "direct" and SENTENCE_TRANSFORMERS_AVAILABLE:
        print("🚀 Using direct model loading (sentence-transformers)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if os.path.exists(EMBEDDING_MODEL_PATH):
            model_name = EMBEDDING_MODEL_PATH
            print(f"  Loading from local path: {model_name}")
        else:
            model_name = "BAAI/bge-m3"
            print(f"  Loading from HuggingFace: {model_name}")
        
        embedding_model = SentenceTransformer(model_name, device=device)
        print(f"  ✅ Model loaded on {device}")
        
        async def _direct_embed(texts):
            def _work():
                embeddings = embedding_model.encode(
                    texts,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                return np.array(embeddings)

            return await asyncio.to_thread(lambda: run_gpu_locked(_work))
        
        return EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=_direct_embed
        )
    else:
        if EMBEDDING_METHOD == "direct":
            print("⚠️  sentence-transformers not available, falling back to Ollama")
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=2, max=10),
            retry=retry_if_exception_type((Exception,)),
        )
        async def _ollama_embed(texts):
            if not OLLAMA_AVAILABLE:
                raise RuntimeError("Ollama is not available")
            
            import numpy as np
            embeddings = []
            for text in texts:
                try:
                    client = ollama_lib.AsyncClient(host=OLLAMA_EMBEDDING_HOST)
                    try:
                        response = await client.embed(
                            model=OLLAMA_EMBEDDING_MODEL,
                            input=text
                        )
                        if response and "embeddings" in response:
                            embeddings.append(response["embeddings"][0])
                        else:
                            raise ValueError("Empty embedding response from Ollama")
                    finally:
                        try:
                            await client._client.aclose()
                        except:
                            pass
                except Exception as e:
                    response = ollama_lib.embed(
                        model=OLLAMA_EMBEDDING_MODEL,
                        input=text
                    )
                    if response and "embeddings" in response:
                        embeddings.append(response["embeddings"][0])
                    else:
                        raise ValueError(f"Empty embedding response from Ollama: {e}")
            
            return np.array(embeddings)
        
        return EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=_ollama_embed
        )


# ========== Virchow2 Feature Function ==========
def get_virchow2_feature_func():
    """Get Virchow2 feature extractor if available"""
    try:
        from pathpocket.virchow2_feature_extractor import (
            Virchow2FeatureExtractor,
            Virchow2FeatureExtractorWrapper
        )
        
        VIRCHOW2_MODEL_PATH = os.getenv("VIRCHOW2_MODEL_PATH", "./models/Virchow2")
        
        if not torch:
            print("⚠️  torch not available, skipping Virchow2")
            return None
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        virchow2_extractor = Virchow2FeatureExtractor(
            model_path=VIRCHOW2_MODEL_PATH,
            device=device,
            batch_size=8
        )
        return Virchow2FeatureExtractorWrapper(virchow2_extractor)
    except Exception as e:
        print(f"⚠️  Virchow2 not available: {e}")
        return None


# bge-m3 embedding + CrossEncoder/Qwen3 rerank share one GPU; serialize all CUDA inference.
_gpu_inference_lock = threading.Lock()
_T = TypeVar("_T")


def recover_gpu_context() -> None:
    """Best-effort CUDA sync + cache clear after a kernel error."""
    try:
        if torch and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_gpu_locked(work: Callable[[], _T]) -> _T:
    with _gpu_inference_lock:
        try:
            result = work()
            if torch and torch.cuda.is_available():
                torch.cuda.synchronize()
            return result
        except Exception as e:
            err = str(e).lower()
            if "cuda" in err or "illegal memory" in err:
                recover_gpu_context()
                result = work()
                if torch and torch.cuda.is_available():
                    torch.cuda.synchronize()
                return result
            raise


def _rerank_batch_size() -> int:
    try:
        return int(os.getenv("RERANK_BATCH_SIZE", "32"))
    except Exception:
        return 32


def _rerank_pair_chunk_size() -> int:
    """Max query-doc pairs per GPU submit (avoids poisoning CUDA on large mix reranks)."""
    try:
        return max(1, int(os.getenv("RERANK_PAIR_CHUNK_SIZE", "16")))
    except Exception:
        return 16


def _truncate_rerank_text(text: str, *, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars > 0 and len(t) > max_chars:
        return t[:max_chars]
    return t


def _rerank_max_doc_chars() -> int:
    try:
        return max(0, int(os.getenv("RERANK_MAX_DOC_CHARS", "4000")))
    except Exception:
        return 4000


def _attach_rerank_pairs_batch(
    bundle: Dict[str, Optional[Callable]],
) -> Dict[str, Optional[Callable]]:
    """Expose batch scorer on the single-query func for global_config fallbacks."""
    single = bundle.get("rerank_model_func")
    batch = bundle.get("rerank_pairs_batch_func")
    if single is not None and batch is not None:
        single.pairs_batch = batch  # type: ignore[attr-defined]
    return bundle


def create_rerank_func():
    """Create single-query rerank function (backward compatible)."""
    return create_rerank_bundle().get("rerank_model_func")


def create_rerank_bundle() -> Dict[str, Optional[Callable]]:
    """Create rerank callables: single-query and optional multi-query pair batching."""
    RERANK_METHOD = os.getenv("RERANK_METHOD", "local").lower()
    RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH", "./models/Qwen3-Reranker-8B")

    # Try to import CrossEncoder for local rerank (BGE-style)
    try:
        from sentence_transformers import CrossEncoder
        CROSS_ENCODER_AVAILABLE = True
    except ImportError:
        CROSS_ENCODER_AVAILABLE = False
        CrossEncoder = None
    
    # Try to import LightRAG rerank functions for API services
    try:
        from lightrag.rerank import cohere_rerank, jina_rerank, generic_rerank_api
        RERANK_API_AVAILABLE = True
    except ImportError:
        RERANK_API_AVAILABLE = False
        cohere_rerank = None
        jina_rerank = None
        generic_rerank_api = None
    
    # Detect Qwen3-Reranker usage by model path/name
    is_qwen3_reranker = False
    if RERANK_MODEL_PATH:
        base_name = os.path.basename(RERANK_MODEL_PATH)
        if "Qwen3-Reranker-0.6B" in RERANK_MODEL_PATH or "Qwen3-Reranker-0.6B" in base_name:
            is_qwen3_reranker = True
    
    if RERANK_METHOD == "local" and is_qwen3_reranker:
        print("🚀 Using Qwen3-Reranker-0.6B local rerank")
        try:
            import torch as _torch  # use local alias to avoid shadowing global
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            print(f"⚠️  transformers or torch not available for Qwen3-Reranker: {e}")
            return {"rerank_model_func": None, "rerank_pairs_batch_func": None}
        
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        model_name_or_path = RERANK_MODEL_PATH if os.path.exists(RERANK_MODEL_PATH) else "Qwen/Qwen3-Reranker-0.6B"
        print(f"  Loading Qwen3-Reranker from: {model_name_or_path} on {device}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
            # model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device).eval()

            model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=_torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            ).eval()
        except Exception as e:
            print(f"  ⚠️  Failed to load Qwen3-Reranker model: {e}")
            return {"rerank_model_func": None, "rerank_pairs_batch_func": None}
        
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        max_length = 256
        
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

        def _format_instruction(instruction, query, doc):
            if instruction is None:
                instruction = "Given a web search query, retrieve relevant passages that answer the query"
            return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc
            )

        def _process_inputs(pairs):
            inputs = tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
            )
            for i, ele in enumerate(inputs["input_ids"]):
                inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
            inputs = tokenizer.pad(
                inputs,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            )
            for key in inputs:
                inputs[key] = inputs[key].to(model.device)
            return inputs

        @_torch.no_grad()
        def _compute_logits(inputs, **kwargs):
            batch_scores = model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = _torch.stack([false_vector, true_vector], dim=1)
            batch_scores = _torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            return scores

        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        batch_size = _rerank_batch_size()

        def _score_formatted_pairs(formatted_pairs: List[str]) -> List[float]:
            pair_chunk = _rerank_pair_chunk_size()

            def _work() -> List[float]:
                all_scores: List[float] = []
                for pc_start in range(0, len(formatted_pairs), pair_chunk):
                    pc_batch = formatted_pairs[pc_start : pc_start + pair_chunk]
                    for start in range(0, len(pc_batch), batch_size):
                        batch = pc_batch[start : start + batch_size]
                        inputs = _process_inputs(batch)
                        all_scores.extend(_compute_logits(inputs))
                    recover_gpu_context()
                return all_scores

            return run_gpu_locked(_work)

        async def _qwen3_rerank_pairs_batch(
            pairs: List[Tuple[str, str]],
        ) -> List[float]:
            if not pairs:
                return []
            max_doc = _rerank_max_doc_chars()
            formatted = [
                _format_instruction(
                    instruction,
                    query,
                    _truncate_rerank_text(doc, max_chars=max_doc),
                )
                for query, doc in pairs
            ]
            return await asyncio.to_thread(_score_formatted_pairs, formatted)

        async def _qwen3_rerank_func(query: str, documents: List[str], top_n: int = None):
            """Local rerank function using Qwen3-Reranker-0.6B (batched to control memory)"""
            if not documents:
                return []

            formatted = [
                _format_instruction(instruction, query, doc) for doc in documents
            ]
            all_scores = await asyncio.to_thread(_score_formatted_pairs, formatted)

            ranked_indices = sorted(
                range(len(all_scores)),
                key=lambda i: all_scores[i],
                reverse=True,
            )

            if top_n is not None:
                ranked_indices = ranked_indices[:top_n]

            return [
                {"index": idx, "relevance_score": float(all_scores[idx])}
                for idx in ranked_indices
            ]

        return _attach_rerank_pairs_batch(
            {
                "rerank_model_func": _qwen3_rerank_func,
                "rerank_pairs_batch_func": _qwen3_rerank_pairs_batch,
            }
        )

    if RERANK_METHOD == "local" and CROSS_ENCODER_AVAILABLE and not is_qwen3_reranker:
        print("🚀 Using local rerank model (CrossEncoder)")
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        
        # Try to load from local path or HuggingFace
        if os.path.exists(RERANK_MODEL_PATH):
            model_name = RERANK_MODEL_PATH
            print(f"  Loading from local path: {model_name}")
        else:
            # Try common BGE reranker models
            model_name = "BAAI/bge-reranker-base"
            print(f"  Loading from HuggingFace: {model_name}")
        
        try:
            rerank_model = CrossEncoder(model_name, device=device)
            print(f"  ✅ Rerank model loaded on {device}")
        except Exception as e:
            print(f"  ⚠️  Failed to load rerank model: {e}")
            return {"rerank_model_func": None, "rerank_pairs_batch_func": None}
        
        predict_batch_size = _rerank_batch_size()
        pair_chunk = _rerank_pair_chunk_size()
        max_doc = _rerank_max_doc_chars()
        print(
            f"  Rerank batch_size={predict_batch_size} (RERANK_BATCH_SIZE), "
            f"pair_chunk={pair_chunk} (RERANK_PAIR_CHUNK_SIZE)"
        )

        def _normalize_cross_pairs(pairs: List[List[str]]) -> List[List[str]]:
            out: List[List[str]] = []
            for pair in pairs:
                if len(pair) < 2:
                    continue
                out.append([pair[0], _truncate_rerank_text(pair[1], max_chars=max_doc)])
            return out

        def _cross_encoder_predict(pairs: List[List[str]]) -> List[float]:
            norm_pairs = _normalize_cross_pairs(pairs)

            def _work() -> List[float]:
                all_scores: List[float] = []
                for start in range(0, len(norm_pairs), pair_chunk):
                    sub = norm_pairs[start : start + pair_chunk]
                    scores = rerank_model.predict(sub, batch_size=predict_batch_size)
                    all_scores.extend(float(s) for s in scores)
                    recover_gpu_context()
                return all_scores

            return run_gpu_locked(_work)

        async def _local_rerank_pairs_batch(
            pairs: List[Tuple[str, str]],
        ) -> List[float]:
            if not pairs:
                return []
            cross_pairs = [
                [query, _truncate_rerank_text(doc, max_chars=max_doc)]
                for query, doc in pairs
            ]
            return await asyncio.to_thread(_cross_encoder_predict, cross_pairs)

        async def _local_rerank_func(query: str, documents: List[str], top_n: int = None):
            """Local rerank function using CrossEncoder"""
            if not documents:
                return []

            pairs = [[query, doc] for doc in documents]
            scores = await asyncio.to_thread(_cross_encoder_predict, pairs)

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )

            if top_n is not None:
                ranked_indices = ranked_indices[:top_n]

            return [
                {"index": idx, "relevance_score": scores[idx]}
                for idx in ranked_indices
            ]

        return _attach_rerank_pairs_batch(
            {
                "rerank_model_func": _local_rerank_func,
                "rerank_pairs_batch_func": _local_rerank_pairs_batch,
            }
        )
    
    elif RERANK_METHOD == "cohere" and RERANK_API_AVAILABLE:
        print("🚀 Using Cohere rerank API")
        from functools import partial
        
        RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-v3.5")
        RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")
        RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "https://api.cohere.com/v2/rerank")
        RERANK_ENABLE_CHUNKING = os.getenv("RERANK_ENABLE_CHUNKING", "false").lower() == "true"
        RERANK_MAX_TOKENS_PER_DOC = int(os.getenv("RERANK_MAX_TOKENS_PER_DOC", "4096"))
        
        if not RERANK_API_KEY:
            print("  ⚠️  RERANK_API_KEY not set, skipping rerank")
            return {"rerank_model_func": None, "rerank_pairs_batch_func": None}
        
        rerank_func = partial(
            cohere_rerank,
            model=RERANK_MODEL,
            api_key=RERANK_API_KEY,
            base_url=RERANK_BASE_URL,
            enable_chunking=RERANK_ENABLE_CHUNKING,
            max_tokens_per_doc=RERANK_MAX_TOKENS_PER_DOC,
        )
        
        print(f"  ✅ Cohere rerank configured: {RERANK_MODEL}")
        return {"rerank_model_func": rerank_func, "rerank_pairs_batch_func": None}
    
    elif RERANK_METHOD == "jina" and RERANK_API_AVAILABLE:
        print("🚀 Using Jina rerank API")
        from functools import partial
        
        RERANK_MODEL = os.getenv("RERANK_MODEL", "jina-reranker-v2-base-multilingual")
        RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")
        RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "https://api.jina.ai/v1/rerank")
        
        if not RERANK_API_KEY:
            print("  ⚠️  RERANK_API_KEY not set, skipping rerank")
            return {"rerank_model_func": None, "rerank_pairs_batch_func": None}
        
        rerank_func = partial(
            jina_rerank,
            model=RERANK_MODEL,
            api_key=RERANK_API_KEY,
            base_url=RERANK_BASE_URL,
        )
        
        print(f"  ✅ Jina rerank configured: {RERANK_MODEL}")
        return {"rerank_model_func": rerank_func, "rerank_pairs_batch_func": None}
    
    else:
        if RERANK_METHOD == "local":
            print("⚠️  CrossEncoder not available, rerank disabled")
        elif RERANK_METHOD in ["cohere", "jina"]:
            print(f"⚠️  Rerank API functions not available, rerank disabled")
        else:
            print(f"⚠️  Unknown rerank method: {RERANK_METHOD}, rerank disabled")
        return {"rerank_model_func": None, "rerank_pairs_batch_func": None}
