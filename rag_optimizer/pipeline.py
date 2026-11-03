"""End-to-End Configurable RAG Optimization Pipeline."""

import time
from dataclasses import dataclass
from typing import List, Optional
from .chunkers import build_chunker, split_into_sentences
from .compressor import ContextCompressor
from .embedders import DenseEmbedder, tokenize
from .models import Chunk, Document, PipelineConfig, ScoredChunk
from .query_transform import QueryOptimizer
from .rerankers import CrossEncoderReranker, LostInTheMiddleReorderer, MMRDiversifier
from .retrievers import HybridRetriever


@dataclass
class QueryResponse:
    """Detailed execution trace of a single RAG query."""
    query: str
    expanded_sparse_query: str
    hyde_dense_query: str
    ranked_chunks: List[ScoredChunk]
    prompt_ordered_chunks: List[ScoredChunk]
    compressed_contexts: List[str]
    synthesized_answer: str
    raw_token_count: int
    compressed_token_count: int
    latency_ms: float


class OptimizedRAGPipeline:
    """Modular RAG pipeline supporting full ablation and hyperparameter optimization."""

    def __init__(
        self,
        config: PipelineConfig,
        embedder: Optional[DenseEmbedder] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.config = config
        self.embedder = embedder or DenseEmbedder()
        self.reranker = reranker or CrossEncoderReranker()
        self.query_optimizer = QueryOptimizer()
        self.retriever = HybridRetriever(self.embedder)
        self.mmr = MMRDiversifier(self.embedder, lambda_mult=config.mmr_lambda)
        self.compressor = ContextCompressor(target_ratio=config.compression_ratio)
        self.chunks: List[Chunk] = []

    def index_documents(self, documents: List[Document]) -> int:
        """Chunks and indexes documents according to the active PipelineConfig."""
        chunker = build_chunker(
            strategy=self.config.chunk_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            window_size=self.config.window_size,
        )
        self.chunks = chunker.chunk_documents(documents)
        self.retriever.index(self.chunks)
        return len(self.chunks)

    def run_query(self, query: str) -> QueryResponse:
        """Executes the full RAG optimization pipeline with latency and token telemetry."""
        t0 = time.perf_counter()

        # Stage 1: Query Transformation (Sparse Expansion + HyDE)
        if self.config.use_query_expansion:
            sparse_q = self.query_optimizer.expand_for_sparse(query)
            dense_q = self.query_optimizer.generate_hyde_passage(query)
        else:
            sparse_q = query
            dense_q = query

        # Stage 2: Candidate Retrieval (Dense, BM25, Hybrid-Alpha, or Hybrid-RRF)
        initial_k = max(self.config.final_top_k, self.config.initial_top_k) if (self.config.use_reranker or self.config.use_mmr) else self.config.final_top_k
        candidates = self.retriever.retrieve(
            dense_query=dense_q,
            sparse_query=sparse_q,
            mode=self.config.retrieval_mode,
            alpha=self.config.alpha,
            rrf_k=self.config.rrf_k,
            top_k=initial_k,
        )

        # Stage 3: Cross-Encoder Reranking
        if self.config.use_reranker and len(candidates) > 1:
            candidates = self.reranker.rerank(raw_query=query, expanded_query=sparse_q, candidates=candidates)

        # Stage 4: Maximal Marginal Relevance (MMR) Diversification
        if self.config.use_mmr and len(candidates) > self.config.final_top_k:
            ranked_top = self.mmr.diversify(candidates, top_k=self.config.final_top_k)
        else:
            ranked_top = candidates[: self.config.final_top_k]
            for idx, c in enumerate(ranked_top):
                c.rank = idx + 1

        # Stage 5: Lost-in-the-Middle Context Reordering (for prompt construction)
        if self.config.use_lost_in_middle_reorder:
            prompt_ordered = LostInTheMiddleReorderer.reorder(ranked_top)
        else:
            prompt_ordered = list(ranked_top)

        # Stage 6: Contextual Compression
        raw_contexts = [c.chunk.effective_context for c in prompt_ordered]
        raw_tokens = sum(len(ctx.split()) for ctx in raw_contexts)

        if self.config.use_context_compression:
            compressed_contexts = self.compressor.compress(prompt_ordered, raw_query=query, expanded_query=sparse_q)
        else:
            compressed_contexts = raw_contexts

        compressed_tokens = sum(len(ctx.split()) for ctx in compressed_contexts)

        # Stage 7: Extractive Grounded Answer Synthesis with Citations
        answer = self._synthesize_answer(query, sparse_q, ranked_top)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return QueryResponse(
            query=query,
            expanded_sparse_query=sparse_q,
            hyde_dense_query=dense_q,
            ranked_chunks=ranked_top,
            prompt_ordered_chunks=prompt_ordered,
            compressed_contexts=compressed_contexts,
            synthesized_answer=answer,
            raw_token_count=raw_tokens,
            compressed_token_count=compressed_tokens,
            latency_ms=latency_ms,
        )

    def _synthesize_answer(self, query: str, expanded_query: str, top_chunks: List[ScoredChunk]) -> str:
        """Synthesizes a concise, citation-backed answer from top retrieved chunks."""
        if not top_chunks:
            return "No relevant context found."

        q_tokens = set(tokenize(query))
        exp_tokens = set(tokenize(expanded_query))
        scored_sentences = []

        for rank_idx, sc in enumerate(top_chunks):
            doc_id = sc.chunk.doc_id
            rank_weight = 1.0 / (1.0 + 0.35 * rank_idx)
            for sent in split_into_sentences(sc.chunk.effective_context):
                clean_sent = sent.replace(f"[{sc.chunk.metadata.get('title', '')}]", "").strip()
                s_tokens = set(tokenize(clean_sent))
                if not s_tokens:
                    continue
                overlap = (2.2 * len(s_tokens & q_tokens) + 0.8 * len(s_tokens & exp_tokens)) * rank_weight
                scored_sentences.append((overlap, clean_sent, doc_id))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        chosen: List[str] = []
        seen_prefixes = set()
        for _, sent, doc_id in scored_sentences:
            prefix = sent[:45].lower()
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                chosen.append(f"{sent} [{doc_id}]")
            if len(chosen) >= 2:
                break

        return " ".join(chosen) if chosen else f"{top_chunks[0].chunk.effective_context} [{top_chunks[0].chunk.doc_id}]"
