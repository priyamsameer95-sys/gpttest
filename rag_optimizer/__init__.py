"""RAG Optimization Pipeline package."""

from .chunkers import (
    FixedWindowChunker,
    RecursiveSemanticChunker,
    SentenceWindowChunker,
    build_chunker,
)
from .compressor import ContextCompressor
from .embedders import DenseEmbedder
from .evaluator import RAGEvaluator
from .models import (
    Chunk,
    Document,
    EvalMetrics,
    EvalSample,
    PipelineConfig,
    ScoredChunk,
)
from .pipeline import OptimizedRAGPipeline, QueryResponse
from .query_transform import QueryOptimizer
from .rerankers import (
    CrossEncoderReranker,
    LostInTheMiddleReorderer,
    MMRDiversifier,
)
from .retrievers import BM25Retriever, HybridRetriever, VectorRetriever
from .sample_corpus import get_eval_benchmark, get_sample_corpus
from .diagnostics import (
    AbstentionAndFaithfulnessGuard,
    ChallengeDiagnosis,
    CorpusHealthAudit,
    RAGDoctor,
    RBACTenantSimulator,
)

__all__ = [
    "Chunk",
    "Document",
    "EvalMetrics",
    "EvalSample",
    "PipelineConfig",
    "ScoredChunk",
    "FixedWindowChunker",
    "RecursiveSemanticChunker",
    "SentenceWindowChunker",
    "build_chunker",
    "DenseEmbedder",
    "BM25Retriever",
    "VectorRetriever",
    "HybridRetriever",
    "QueryOptimizer",
    "CrossEncoderReranker",
    "MMRDiversifier",
    "LostInTheMiddleReorderer",
    "ContextCompressor",
    "OptimizedRAGPipeline",
    "QueryResponse",
    "RAGEvaluator",
    "get_sample_corpus",
    "get_eval_benchmark",
    "AbstentionAndFaithfulnessGuard",
    "ChallengeDiagnosis",
    "CorpusHealthAudit",
    "RAGDoctor",
    "RBACTenantSimulator",
]
