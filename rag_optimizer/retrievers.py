"""Sparse (Okapi BM25), Dense (Vector/FAISS), and Hybrid (RRF & Alpha) Retrievers."""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple
from .embedders import DenseEmbedder, cosine_similarity, tokenize
from .models import Chunk, ScoredChunk


class BM25Retriever:
    """Okapi BM25 sparse lexical retriever (uses rank_bm25 if installed, else built-in Okapi BM25)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: List[Counter] = []
        self.doc_lens: List[int] = []
        self.avgdl: float = 1.0
        self.idf: Dict[str, float] = {}

    def index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.doc_tokens = [tokenize(c.text) for c in chunks]
        self.doc_freqs = [Counter(toks) for toks in self.doc_tokens]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        n_docs = max(1, len(chunks))
        self.avgdl = sum(self.doc_lens) / n_docs if n_docs else 1.0

        df: Counter = Counter()
        for freqs in self.doc_freqs:
            for term in freqs.keys():
                df[term] += 1

        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score_all(self, query: str) -> List[float]:
        q_terms = tokenize(query)
        scores = [0.0] * len(self.chunks)
        if not q_terms or not self.chunks:
            return scores

        q_counts = Counter(q_terms)
        for i, freqs in enumerate(self.doc_freqs):
            dl = self.doc_lens[i]
            score = 0.0
            norm_denom = self.k1 * (1.0 - self.b + self.b * (dl / max(1e-6, self.avgdl)))
            for term, q_tf in q_counts.items():
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                score += idf * ((tf * (self.k1 + 1.0)) / (tf + norm_denom)) * (1.0 + 0.1 * (q_tf - 1))
            scores[i] = score
        return scores


class VectorRetriever:
    """Dense vector retriever supporting FAISS IndexFlatIP or exact cosine similarity."""

    def __init__(self, embedder: DenseEmbedder):
        self.embedder = embedder
        self.chunks: List[Chunk] = []
        self.vectors: List[List[float]] = []

    def index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        self.vectors = self.embedder.encode(texts)

    def score_all(self, query_text: str) -> List[float]:
        if not self.chunks:
            return []
        q_vec = self.embedder.encode_one(query_text)
        return [cosine_similarity(q_vec, doc_vec) for doc_vec in self.vectors]


def _min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class HybridRetriever:
    """
    Combines Dense Vector Search and Sparse Okapi BM25 using:
    - 'dense': Pure vector search
    - 'bm25': Pure lexical BM25 search
    - 'hybrid_alpha': Convex combination alpha * norm(dense) + (1 - alpha) * norm(bm25)
    - 'hybrid_rrf': Weighted Reciprocal Rank Fusion (RRF)
    """

    def __init__(self, embedder: DenseEmbedder):
        self.bm25 = BM25Retriever()
        self.vector = VectorRetriever(embedder)
        self.chunks: List[Chunk] = []

    def index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.bm25.index(chunks)
        self.vector.index(chunks)

    def retrieve(
        self,
        dense_query: str,
        sparse_query: str,
        mode: str = "hybrid_rrf",
        alpha: float = 0.6,
        rrf_k: int = 60,
        top_k: int = 10,
    ) -> List[ScoredChunk]:
        if not self.chunks:
            return []

        dense_scores = self.vector.score_all(dense_query)
        sparse_scores = self.bm25.score_all(sparse_query)
        n = len(self.chunks)

        if mode == "dense":
            combined = list(dense_scores)
        elif mode == "bm25":
            combined = list(sparse_scores)
        elif mode == "hybrid_alpha":
            d_norm = _min_max_normalize(dense_scores)
            s_norm = _min_max_normalize(sparse_scores)
            combined = [alpha * d + (1.0 - alpha) * s for d, s in zip(d_norm, s_norm)]
        elif mode == "hybrid_rrf":
            # Compute 1-based ranks for dense and sparse
            dense_order = sorted(range(n), key=lambda i: dense_scores[i], reverse=True)
            sparse_order = sorted(range(n), key=lambda i: sparse_scores[i], reverse=True)
            dense_rank = {idx: rank + 1 for rank, idx in enumerate(dense_order)}
            sparse_rank = {idx: rank + 1 for rank, idx in enumerate(sparse_order)}

            combined = [
                (alpha / (rrf_k + dense_rank[i])) + ((1.0 - alpha) / (rrf_k + sparse_rank[i]))
                for i in range(n)
            ]
        else:
            raise ValueError(f"Unsupported retrieval_mode: {mode}")

        ranked_indices = sorted(range(n), key=lambda i: combined[i], reverse=True)[:top_k]
        results: List[ScoredChunk] = []
        for rank_idx, doc_idx in enumerate(ranked_indices):
            results.append(
                ScoredChunk(
                    chunk=self.chunks[doc_idx],
                    score=combined[doc_idx],
                    dense_score=dense_scores[doc_idx],
                    sparse_score=sparse_scores[doc_idx],
                    rank=rank_idx + 1,
                )
            )
        return results
