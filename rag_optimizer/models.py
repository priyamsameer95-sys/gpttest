"""Core data models and configuration for the RAG Optimization Pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Document:
    """Represents a raw source document before chunking."""
    doc_id: str
    title: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    """Represents an indexed text chunk with optional parent-window context."""
    chunk_id: str
    doc_id: str
    text: str
    parent_text: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def effective_context(self) -> str:
        """Returns expanded parent context if using Sentence-Window chunking, else chunk text."""
        return self.parent_text if self.parent_text else self.text


@dataclass
class ScoredChunk:
    """A retrieved chunk paired with its relevance score and score breakdown."""
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: Optional[float] = None
    rank: int = 0


@dataclass
class PipelineConfig:
    """Hyperparameters defining a complete RAG pipeline configuration."""
    name: str = "Optimized-RAG"
    chunk_strategy: str = "sentence_window"  # "fixed", "recursive", "sentence_window"
    chunk_size: int = 320                    # target characters per chunk
    chunk_overlap: int = 64                  # overlap characters (for fixed/recursive)
    window_size: int = 1                     # neighbor sentences on each side (for sentence_window)

    retrieval_mode: str = "hybrid_rrf"       # "dense", "bm25", "hybrid_alpha", "hybrid_rrf"
    alpha: float = 0.6                       # weight for dense score (1 - alpha for BM25)
    rrf_k: int = 60                          # Reciprocal Rank Fusion constant
    initial_top_k: int = 10                  # candidates fetched before reranking
    final_top_k: int = 3                     # final chunks passed to context

    use_query_expansion: bool = True         # Multi-query + HyDE expansion
    use_reranker: bool = True                # Cross-encoder reranking
    use_mmr: bool = True                     # Maximal Marginal Relevance deduplication
    mmr_lambda: float = 0.75                 # 1.0 = pure relevance, 0.0 = pure diversity
    use_lost_in_middle_reorder: bool = True  # Put top chunks at start and end of prompt
    use_context_compression: bool = True     # Prune irrelevant sentences inside chunks
    compression_ratio: float = 0.75          # Target fraction of sentences to keep


@dataclass
class EvalSample:
    """Evaluation query with ground-truth relevant document IDs and key facts."""
    query: str
    expected_doc_ids: List[str]
    expected_keywords: List[str]


@dataclass
class EvalMetrics:
    """Quantitative retrieval & context quality metrics for a pipeline configuration."""
    config_name: str
    hit_rate_at_k: float
    mrr: float
    ndcg_at_k: float
    keyword_recall: float
    context_precision: float
    avg_context_tokens: float
    avg_latency_ms: float
    composite_score: float
