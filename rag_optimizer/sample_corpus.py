"""Sample technical corpus and benchmark evaluation dataset for testing RAG Optimization."""

from typing import List
from .models import Document, EvalSample


def get_sample_corpus() -> List[Document]:
    """Returns a realistic 10-document systems engineering corpus with hard lexical and semantic details."""
    return [
        Document(
            doc_id="DOC-01",
            title="HNSW & Vector Index Memory Quantization",
            content=(
                "Hierarchical Navigable Small World (HNSW) graphs deliver sub-10ms approximate nearest neighbor (ANN) "
                "search at scale, but store full 32-bit floating point vectors in RAM. A corpus of 10 million 1536-dimensional "
                "vectors consumes roughly 61.4 GB of raw memory plus 20-35% graph edge overhead. "
                "To reduce RAM consumption without catastrophic recall loss, production systems apply Scalar Quantization (SQ8 / int8), "
                "which compresses float32 values to 8-bit integers for a 4x memory reduction while retaining 98.5% recall@10. "
                "For extreme compression up to 16x-32x, Product Quantization (PQ) decomposes the vector space into M orthogonal "
                "sub-vectors and quantizes each subspace via k-means centroids, though asymmetric distance computation (ADC) "
                "requires careful rescoring with raw vectors to recover top-k precision."
            ),
            metadata={"domain": "indexing", "tier": "storage"},
        ),
        Document(
            doc_id="DOC-02",
            title="Hybrid Search: Combining Dense Vectors with Okapi BM25 via RRF",
            content=(
                "Pure dense bi-encoder embeddings frequently fail on exact alphanumeric identifiers, error codes like ERR_SSL_9042, "
                "part SKUs such as SKU-8841-X9, and rare proper nouns due to subword tokenizer fragmentation. "
                "Conversely, sparse Okapi BM25 excels at exact lexical token matching (using saturation parameter k1=1.5 and "
                "document length normalization b=0.75) but misses semantic paraphrases. "
                "Hybrid retrieval fuses both signals using Reciprocal Rank Fusion (RRF), defined as "
                "RRF_score(d) = sum(w_m / (k + rank_m(d))) with smoothing constant k=60, or via min-max normalized "
                "convex alpha combination where alpha=0.6 weights dense cosine similarity and 0.4 weights sparse BM25."
            ),
            metadata={"domain": "retrieval", "tier": "search"},
        ),
        Document(
            doc_id="DOC-03",
            title="Small-to-Big Sentence-Window Chunking Architecture",
            content=(
                "Naive fixed-size character chunking splits sentences mid-clause and dilutes dense vector embeddings when a single "
                "500-token chunk discusses multiple unrelated topics. Small-to-Big Sentence-Window chunking decouples the "
                "retrieval unit from the synthesis unit. During indexing, each individual sentence or compact 2-sentence micro-unit "
                "is embedded separately so its vector representation remains sharp and focused. "
                "At retrieval time, once a micro-chunk is matched, the pipeline swaps in its parent window of +/- 2 surrounding "
                "sentences before sending context to the LLM, restoring full antecedent pronouns and surrounding causal context."
            ),
            metadata={"domain": "chunking", "tier": "ingestion"},
        ),
        Document(
            doc_id="DOC-04",
            title="Cross-Encoder Reranking & Bi-Encoder Bottlenecks",
            content=(
                "Bi-encoders encode queries and documents independently into single vectors, compressing an entire passage into "
                "one fixed-size bottleneck and losing fine-grained token-to-token cross-attention. "
                "To maximize Precision@3, a two-stage pipeline first retrieves top-20 candidates via fast bi-encoder + BM25 hybrid search, "
                "then passes concatenated (Query [SEP] Passage) pairs through a Cross-Encoder such as ms-marco-MiniLM-L-6-v2. "
                "Because the Cross-Encoder performs full self-attention across every query token and document token simultaneously, "
                "it increases NDCG@3 by 12-18% while adding only 15-35ms of GPU/CPU inference latency on a 20-chunk candidate pool."
            ),
            metadata={"domain": "reranking", "tier": "ranking"},
        ),
        Document(
            doc_id="DOC-05",
            title="Maximal Marginal Relevance (MMR) for Context Diversity",
            content=(
                "Top-k similarity search often returns near-duplicate chunks originating from adjacent paragraphs or mirrored "
                "documentation pages, wasting precious prompt tokens on repetitive facts. "
                "Maximal Marginal Relevance (MMR) iteratively selects chunks by maximizing "
                "lambda * Sim(Query, d_i) - (1 - lambda) * max(Sim(d_i, d_j)) over already selected chunks d_j. "
                "Setting lambda=0.75 balances high query relevance with a 25% redundancy penalty, ensuring multi-hop queries "
                "receive complementary evidence across distinct documents."
            ),
            metadata={"domain": "reranking", "tier": "diversity"},
        ),
        Document(
            doc_id="DOC-06",
            title="Mitigating Lost-in-the-Middle Attention Degradation",
            content=(
                "Empirical studies by Liu et al. demonstrate that transformer LLMs exhibit a U-shaped retrieval recall curve "
                "across long context windows: models reliably attend to facts placed at the very beginning (primacy bias) and "
                "very end (recency bias) of the prompt, while accuracy drops by up to 20% when critical evidence is buried in "
                "the middle positions. Lost-in-the-Middle context reordering sorts retrieved chunks so Rank 1 is placed first, "
                "Rank 2 is placed at the very end of the context block, and lower-ranked chunks occupy the interior."
            ),
            metadata={"domain": "prompting", "tier": "synthesis"},
        ),
        Document(
            doc_id="DOC-07",
            title="Query Rewriting, Multi-Query Expansion, and HyDE",
            content=(
                "Users frequently submit terse, ambiguous, or poorly phrased queries that share minimal lexical or structural "
                "similarity with authoritative technical documentation. Hypothetical Document Embeddings (HyDE) resolves this "
                "distribution mismatch by generating a hypothetical declarative answer passage from the query and embedding "
                "that passage for dense vector lookup. Combined with sparse synonym expansion and multi-query decomposition "
                "for compound questions, query transformation boosts Recall@10 on zero-shot domain queries by 15-24%."
            ),
            metadata={"domain": "query", "tier": "transformation"},
        ),
        Document(
            doc_id="DOC-08",
            title="Multi-Tenant RBAC Security & Metadata Pre-Filtering",
            content=(
                "In enterprise RAG deployments, applying access-control list (ACL) security filters after vector ANN search "
                "(post-filtering) causes severe k-starvation: if a user only has permission to view 2% of the corpus, top-10 "
                "ANN search may return zero authorized chunks. Production vector engines enforce RBAC tenant isolation via "
                "single-stage pre-filtering using bitmap roaring masks integrated directly into HNSW graph traversal, "
                "guaranteeing zero cross-tenant data leakage and full top-k recall."
            ),
            metadata={"domain": "security", "tier": "governance"},
        ),
        Document(
            doc_id="DOC-09",
            title="Semantic Caching & Contextual Compression for Cost Reduction",
            content=(
                "Repeated or semantically equivalent user queries account for 30-45% of production RAG traffic. A Redis-backed "
                "Semantic Cache indexes historical query embeddings and serves cached responses within 4ms whenever cosine "
                "similarity exceeds a strict 0.94 threshold, governed by document-update TTL invalidation hooks. "
                "For cache misses, sentence-level Contextual Compression prunes tangential sentences from retrieved chunks, "
                "cutting LLM input token costs by 25-40% and reducing time-to-first-token (TTFT)."
            ),
            metadata={"domain": "caching", "tier": "efficiency"},
        ),
        Document(
            doc_id="DOC-10",
            title="RAG Evaluation Metrics: MRR, NDCG@K, and Faithfulness",
            content=(
                "Optimizing a RAG system requires decoupled retrieval and generation metrics. Retrieval quality is measured "
                "via Hit Rate@K, Mean Reciprocal Rank (MRR, which rewards placing the first relevant chunk at rank 1), and "
                "Normalized Discounted Cumulative Gain (NDCG@K, which evaluates the full graded ranking order). "
                "Generation quality is audited through Context Precision, Answer Relevance, and Grounded Faithfulness "
                "(verifying every claim in the answer is entailed by retrieved citations without hallucination)."
            ),
            metadata={"domain": "evaluation", "tier": "metrics"},
        ),
    ]


def get_eval_benchmark() -> List[EvalSample]:
    """Returns 8 challenging benchmark queries testing lexical exactness, colloquial phrasing, and multi-concept retrieval."""
    return [
        EvalSample(
            query="How do we shrink vector RAM footprint from 61GB using int8 or sub-vector centroids?",
            expected_doc_ids=["DOC-01"],
            expected_keywords=["SQ8", "int8", "4x", "Product Quantization", "98.5%"],
        ),
        EvalSample(
            query="Why do dense embeddings fail on exact codes like ERR_SSL_9042 or SKU-8841-X9 and how does k=60 fusion fix it?",
            expected_doc_ids=["DOC-02"],
            expected_keywords=["ERR_SSL_9042", "SKU-8841-X9", "BM25", "Reciprocal Rank Fusion", "60"],
        ),
        EvalSample(
            query="How do we avoid splitting sentences mid-clause while keeping embeddings sharp and preserving surrounding context?",
            expected_doc_ids=["DOC-03"],
            expected_keywords=["Sentence-Window", "micro-unit", "parent window", "surrounding"],
        ),
        EvalSample(
            query="Why is a bi-encoder bottlenecked and how much latency does pairwise Cross-Encoder reranking add?",
            expected_doc_ids=["DOC-04"],
            expected_keywords=["Cross-Encoder", "ms-marco-MiniLM-L-6-v2", "15-35ms", "NDCG@3"],
        ),
        EvalSample(
            query="How do we stop duplicate redundant chunks from crowding the prompt and fix LLMs ignoring facts in the middle?",
            expected_doc_ids=["DOC-05", "DOC-06"],
            expected_keywords=["Maximal Marginal Relevance", "lambda", "U-shaped", "beginning", "end"],
        ),
        EvalSample(
            query="What technique helps when users type short vague queries that don't match technical docs?",
            expected_doc_ids=["DOC-07"],
            expected_keywords=["Hypothetical Document Embeddings", "HyDE", "Recall@10", "synonym"],
        ),
        EvalSample(
            query="Why does post-filtering break multi-tenant RBAC permissions in ANN search?",
            expected_doc_ids=["DOC-08"],
            expected_keywords=["k-starvation", "pre-filtering", "roaring", "HNSW"],
        ),
        EvalSample(
            query="How can we lower LLM token costs and latency using similarity threshold caching and sentence pruning?",
            expected_doc_ids=["DOC-09"],
            expected_keywords=["0.94", "4ms", "Contextual Compression", "25-40%"],
        ),
    ]
