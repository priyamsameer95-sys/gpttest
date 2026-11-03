"""Post-Retrieval Optimization: Cross-Encoder Reranking, MMR Diversification, and Lost-in-the-Middle Reordering."""

import re
from typing import List
from .embedders import DenseEmbedder, cosine_similarity, tokenize
from .models import ScoredChunk


class CrossEncoderReranker:
    """
    Pairwise (Query, Passage) Cross-Encoder Reranker.
    Uses `sentence_transformers.CrossEncoder` if installed, else a joint lexical-proximity
    and semantic-coverage interaction scorer.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", force_local: bool = False):
        self.model_name = model_name
        self._ce_model = None
        self.backend_name = "LocalCrossInteractionReranker"

        if not force_local:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
                self._ce_model = CrossEncoder(model_name)
                self.backend_name = f"CrossEncoder({model_name})"
            except Exception:
                self._ce_model = None

    def _interaction_score(self, raw_query: str, expanded_query: str, candidate: ScoredChunk) -> float:
        """Computes fine-grained query-document cross-interaction score."""
        passage = candidate.chunk.effective_context.lower()
        q_tokens = tokenize(raw_query)
        exp_tokens = tokenize(expanded_query)
        p_tokens = tokenize(passage)

        if not q_tokens or not p_tokens:
            return candidate.score

        p_set = set(p_tokens)
        # 1. Exact query term coverage ratio
        matched_q = [t for t in q_tokens if t in p_set]
        coverage = len(set(matched_q)) / max(1, len(set(q_tokens)))

        # 2. Expanded concept coverage
        matched_exp = [t for t in exp_tokens if t in p_set]
        exp_coverage = len(set(matched_exp)) / max(1, len(set(exp_tokens)))

        # 3. Ordered bigram / phrase bonus
        q_words = re.findall(r"[a-z0-9]+", raw_query.lower())
        phrase_bonus = 0.0
        for i in range(len(q_words) - 1):
            bg = f"{q_words[i]} {q_words[i+1]}"
            if len(bg) > 5 and bg in passage:
                phrase_bonus += 0.25

        # 4. Term proximity / density bonus (how close query terms appear in the passage)
        positions = [idx for idx, tok in enumerate(p_tokens) if tok in set(q_tokens)]
        proximity_bonus = 0.0
        if len(positions) >= 2:
            min_gap = min(positions[i+1] - positions[i] for i in range(len(positions) - 1))
            proximity_bonus = 1.0 / (1.0 + 0.25 * max(0, min_gap - 1))

        # Combine interaction signals with dense + sparse anchors
        interaction = (
            0.40 * coverage
            + 0.20 * exp_coverage
            + 0.15 * proximity_bonus
            + min(0.25, phrase_bonus)
            + 0.25 * candidate.dense_score
            + 0.05 * min(1.0, candidate.sparse_score / 10.0)
        )
        return interaction

    def rerank(self, raw_query: str, expanded_query: str, candidates: List[ScoredChunk]) -> List[ScoredChunk]:
        if not candidates:
            return []

        if self._ce_model is not None:
            pairs = [(raw_query, c.chunk.effective_context) for c in candidates]
            scores = self._ce_model.predict(pairs)
            for cand, sc in zip(candidates, scores):
                cand.rerank_score = float(sc)
                cand.score = float(sc)
        else:
            for cand in candidates:
                rr_score = self._interaction_score(raw_query, expanded_query, cand)
                cand.rerank_score = rr_score
                cand.score = rr_score

        reranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        for idx, cand in enumerate(reranked):
            cand.rank = idx + 1
        return reranked


class MMRDiversifier:
    """
    Maximal Marginal Relevance (MMR) selector.
    Balances query relevance against inter-chunk redundancy so multiple chunks from the
    exact same paragraph don't crowd out complementary context.
    """

    def __init__(self, embedder: DenseEmbedder, lambda_mult: float = 0.75):
        self.embedder = embedder
        self.lambda_mult = lambda_mult

    def diversify(self, candidates: List[ScoredChunk], top_k: int = 3) -> List[ScoredChunk]:
        if len(candidates) <= top_k:
            return candidates

        texts = [c.chunk.effective_context for c in candidates]
        vecs = self.embedder.encode(texts)

        # Normalize candidate scores into [0, 1]
        scores = [c.score for c in candidates]
        lo, hi = min(scores), max(scores)
        norm_scores = [(s - lo) / (hi - lo) if hi - lo > 1e-9 else 1.0 for s in scores]

        selected_indices: List[int] = [0]  # Always pick top-1 highest relevance first
        remaining = list(range(1, len(candidates)))

        while len(selected_indices) < top_k and remaining:
            best_idx = -1
            best_mmr = -1e9
            for idx in remaining:
                rel = norm_scores[idx]
                max_sim = max(cosine_similarity(vecs[idx], vecs[sel]) for sel in selected_indices)
                # Penalize selecting an identical parent window twice
                same_parent_penalty = 0.0
                for sel in selected_indices:
                    if candidates[idx].chunk.effective_context == candidates[sel].chunk.effective_context:
                        same_parent_penalty = 0.55
                mmr_score = self.lambda_mult * rel - (1.0 - self.lambda_mult) * max_sim - same_parent_penalty
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        selected: List[ScoredChunk] = []
        for rank_idx, idx in enumerate(selected_indices):
            cand = candidates[idx]
            cand.rank = rank_idx + 1
            selected.append(cand)
        return selected


class LostInTheMiddleReorderer:
    """
    Reorders top-k chunks into a U-shaped attention pattern (Liu et al., 2023):
    Places highest-relevance chunks at the very beginning and very end of the prompt context.
    Example for 4 chunks [1, 2, 3, 4] -> [1, 3, 4, 2] (1st at start, 2nd at end).
    """

    @staticmethod
    def reorder(chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        if len(chunks) <= 2:
            return chunks
        left: List[ScoredChunk] = []
        right: List[ScoredChunk] = []
        for idx, chunk in enumerate(chunks):
            if idx % 2 == 0:
                left.append(chunk)
            else:
                right.insert(0, chunk)
        return left + right
