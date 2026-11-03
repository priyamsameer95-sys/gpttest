"""Query Optimization: Multi-Query Expansion, Synonym Injection, and Local HyDE."""

import re
from typing import Dict, List
from .embedders import tokenize


# Domain-aware query expansion dictionary for RAG & Distributed Systems corpora
DOMAIN_EXPANSIONS: Dict[str, List[str]] = {
    "slow": ["latency", "p99", "throughput", "millisecond", "bottleneck"],
    "fast": ["low latency", "throughput", "sub-10ms", "hnsw", "caching"],
    "memory": ["ram", "footprint", "quantization", "scalar int8", "product quantization pq"],
    "ram": ["memory", "quantization", "int8", "product quantization", "footprint"],
    "compress": ["quantization", "int8", "product quantization", "context compression"],
    "exact": ["bm25", "sparse lexical", "keyword match", "sku", "identifier"],
    "keyword": ["bm25", "okapi", "sparse retrieval", "inverted index", "reciprocal rank fusion"],
    "combine": ["hybrid search", "reciprocal rank fusion rrf", "alpha convex combination"],
    "fusion": ["reciprocal rank fusion rrf", "hybrid search", "dense and sparse"],
    "duplicate": ["maximal marginal relevance mmr", "diversity", "redundant chunks"],
    "redundant": ["maximal marginal relevance mmr", "deduplication", "diversity penalty"],
    "middle": ["lost in the middle", "u-shaped attention curve", "reorder context boundaries"],
    "ignore": ["lost in the middle", "attention degradation", "context reordering"],
    "cut": ["chunk boundary", "sentence window chunking", "recursive splitting"],
    "split": ["chunking strategy", "recursive semantic chunker", "sentence window parent context"],
    "short": ["query expansion", "hyde hypothetical document embeddings", "multi-query rewriting"],
    "vague": ["hyde hypothetical document embeddings", "query decomposition", "synonym expansion"],
    "rerank": ["cross-encoder", "bi-encoder vs cross-encoder", "precision at k"],
    "accurate": ["cross-encoder reranking", "hybrid rrf", "sentence window chunking"],
    "permission": ["rbac", "acl", "pre-filtering", "multi-tenant metadata isolation"],
    "security": ["rbac", "acl pre-filtering", "tenant isolation", "metadata filter"],
    "cache": ["semantic caching", "cosine threshold", "ttl invalidation", "redis vector"],
    "cost": ["semantic caching", "context compression", "token reduction"],
}


class QueryOptimizer:
    """
    Transforms a raw user query into:
    1. Expanded lexical query (for BM25 sparse retrieval)
    2. HyDE (Hypothetical Document Embedding) passage (for Dense Vector retrieval)
    3. Multi-query sub-questions (for complex compound queries)
    """

    def expand_for_sparse(self, query: str) -> str:
        """Enriches the query with domain synonyms while preserving original query weight."""
        q_tokens = set(re.findall(r"[a-zA-Z0-9_-]+", query.lower()))
        expansions: List[str] = []
        for tok in q_tokens:
            for trigger, terms in DOMAIN_EXPANSIONS.items():
                if tok == trigger or tok.startswith(trigger):
                    expansions.extend(terms[:2])
        if not expansions:
            return query
        # Repeat original query twice so original terms dominate BM25 TF while expansions boost recall
        unique_exp = " ".join(dict.fromkeys(expansions))
        return f"{query} {query} {unique_exp}"

    def generate_hyde_passage(self, query: str) -> str:
        """
        Generates a local Hypothetical Document Embedding (HyDE) passage
        that bridges the style gap between short user questions and declarative corpus chunks.
        """
        q_lower = query.lower().strip()
        q_tokens = tokenize(query)
        matched_concepts: List[str] = []

        for tok in q_tokens:
            for trigger, terms in DOMAIN_EXPANSIONS.items():
                if tok == trigger or trigger in q_lower:
                    matched_concepts.extend(terms)

        unique_concepts = list(dict.fromkeys(matched_concepts))[:6]
        concept_str = ", ".join(unique_concepts) if unique_concepts else "retrieval architecture, indexing parameters, and empirical latency trade-offs"

        hyde_template = (
            f"Technical Documentation & Engineering Specification: {query} "
            f"In production systems, addressing {query.rstrip('?')} requires configuring "
            f"{concept_str}. Best practices include optimizing metric thresholds, balancing "
            f"recall and precision, and validating system performance."
        )
        return hyde_template

    def decompose_multi_query(self, query: str) -> List[str]:
        """Splits compound questions ('and', 'vs', 'as well as') into atomic sub-queries."""
        variants = [query]
        parts = re.split(r"\b(?:and also|as well as|versus|vs\.?)\b|(?:\?\s+)", query, flags=re.IGNORECASE)
        clean_parts = [p.strip() for p in parts if len(p.strip()) > 12]
        if len(clean_parts) > 1:
            variants.extend(clean_parts)
        return variants
