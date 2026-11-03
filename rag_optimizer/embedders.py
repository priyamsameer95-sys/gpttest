"""Dense embedding engines: Sentence-Transformers with zero-dependency local semantic fallback."""

import hashlib
import math
import re
from collections import Counter
from typing import Dict, List, Optional


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "once", "here", "there", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "of", "what", "which", "who", "whom", "this",
    "that", "these", "those", "it", "its"
}

# Semantic concept clusters so related technical terms map to shared latent dimensions
SEMANTIC_CLUSTERS = {
    "CLUSTER_LATENCY": {"latency", "p99", "p95", "slow", "fast", "millisecond", "ms", "throughput", "timeout", "speed", "performance"},
    "CLUSTER_VECTOR": {"vector", "embedding", "dense", "cosine", "faiss", "hnsw", "ivf", "ann", "similarity", "semantic", "index"},
    "CLUSTER_LEXICAL": {"bm25", "sparse", "keyword", "lexical", "tfidf", "inverted", "token", "exact", "term", "frequency"},
    "CLUSTER_HYBRID": {"hybrid", "rrf", "reciprocal", "fusion", "alpha", "convex", "combine", "blend"},
    "CLUSTER_RERANK": {"rerank", "reranker", "cross-encoder", "cross", "encoder", "mmr", "diversity", "marginal", "precision"},
    "CLUSTER_CHUNK": {"chunk", "chunking", "window", "overlap", "sentence", "recursive", "split", "boundary", "parent", "context"},
    "CLUSTER_QUERY": {"query", "hyde", "hypothetical", "rewrite", "expansion", "decomposition", "multi-query", "reformulate"},
    "CLUSTER_MEMORY": {"memory", "ram", "quantization", "int8", "pq", "product", "scalar", "compression", "footprint", "gb"},
    "CLUSTER_HALLUCINATION": {"hallucination", "faithfulness", "grounded", "lost-in-the-middle", "middle", "attention", "recall"},
    "CLUSTER_CACHE": {"cache", "redis", "semantic-cache", "ttl", "hit", "invalidation", "cost"},
    "CLUSTER_SECURITY": {"rbac", "acl", "permission", "tenant", "isolation", "filter", "metadata", "security", "pii"},
}


def normalize_token(token: str) -> str:
    """Lightweight Porter-style suffix stripper for morphological alignment."""
    t = token.lower().strip()
    for suffix in ("ization", "ational", "iveness", "ingly", "edly", "ments", "ings", "ions", "tion", "ness", "able", "ible", "ally", "ing", "ers", "ies", "ed", "es", "er", "ly", "s"):
        if len(t) > len(suffix) + 3 and t.endswith(suffix):
            if suffix == "ies":
                return t[:-3] + "y"
            return t[:-len(suffix)]
    return t


def tokenize(text: str) -> List[str]:
    """Tokenizes text into normalized content words."""
    raw_tokens = re.findall(r"[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*", text.lower())
    tokens: List[str] = []
    for tok in raw_tokens:
        if tok in STOPWORDS or len(tok) < 2:
            continue
        tokens.append(tok)
        stem = normalize_token(tok)
        if stem != tok:
            tokens.append(stem)
    return tokens


class DenseEmbedder:
    """
    Unified dense embedder:
    Uses `sentence_transformers` (e.g., 'all-MiniLM-L6-v2') if available,
    otherwise falls back to an offline semantic-cluster + subword n-gram IDF projection engine.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384, force_local: bool = False):
        self.model_name = model_name
        self.dim = dim
        self._st_model = None
        self.idf: Dict[str, float] = {}
        self.backend_name = "LocalSemanticProjection-384d"

        if not force_local:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._st_model = SentenceTransformer(model_name)
                self.backend_name = f"SentenceTransformer({model_name})"
            except Exception:
                self._st_model = None

    def fit(self, corpus_texts: List[str]) -> None:
        """Fits corpus IDF weights for the local dense projection fallback."""
        if self._st_model is not None:
            return
        n_docs = max(1, len(corpus_texts))
        df: Counter = Counter()
        for text in corpus_texts:
            unique_feats = set(self._extract_features(text).keys())
            for feat in unique_feats:
                df[feat] += 1
        self.idf = {
            feat: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for feat, freq in df.items()
        }

    def _extract_features(self, text: str) -> Counter:
        """Extracts lexical stems, semantic cluster triggers, and character 4-grams."""
        feats: Counter = Counter()
        raw_words = set(re.findall(r"[a-zA-Z0-9_-]+", text.lower()))
        tokens = tokenize(text)

        for tok in tokens:
            feats[f"w:{tok}"] += 1.5

        # Bigrams of stems
        stems = [normalize_token(w) for w in raw_words if w not in STOPWORDS and len(w) > 2]
        for i in range(len(tokens) - 1):
            feats[f"bg:{tokens[i]}_{tokens[i+1]}"] += 1.0

        # Semantic concept expansion
        for cluster_id, vocab in SEMANTIC_CLUSTERS.items():
            overlap = raw_words.intersection(vocab) or set(stems).intersection({normalize_token(v) for v in vocab})
            if overlap:
                feats[f"sem:{cluster_id}"] += 2.5 * len(overlap)

        # Character 4-grams for typo/morphology resilience
        clean = re.sub(r"\s+", " ", text.lower())
        for tok in raw_words:
            if len(tok) >= 4 and tok not in STOPWORDS:
                for i in range(len(tok) - 3):
                    feats[f"c4:{tok[i:i+4]}"] += 0.35

        return feats

    def _project_vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        feats = self._extract_features(text)
        for feat, tf in feats.items():
            weight = (1.0 + math.log(tf)) * self.idf.get(feat, 1.2)
            # Signed feature hashing (Weinberger et al.) across 2 buckets to minimize collisions
            h = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:16], 16)
            idx1 = h % self.dim
            sign1 = 1.0 if ((h >> 16) & 1) == 0 else -1.0
            idx2 = (h >> 20) % self.dim
            sign2 = 1.0 if ((h >> 36) & 1) == 0 else -1.0
            vec[idx1] += sign1 * weight
            vec[idx2] += sign2 * (weight * 0.5)

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-9:
            vec = [v / norm for v in vec]
        return vec

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encodes a batch of texts into unit-normalized dense vectors."""
        if self._st_model is not None:
            embeddings = self._st_model.encode(texts, normalize_embeddings=True)
            return [list(map(float, emb)) for emb in embeddings]
        return [self._project_vector(t) for t in texts]

    def encode_one(self, text: str) -> List[float]:
        """Encodes a single string into a unit-normalized dense vector."""
        return self.encode([text])[0]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two unit-normalized vectors."""
    return sum(a * b for a, b in zip(v1, v2))
