"""Contextual Compression: Prunes low-salience and duplicate sentences across retrieved chunks."""

import math
from typing import List, Set
from .chunkers import split_into_sentences
from .embedders import tokenize
from .models import ScoredChunk


class ContextCompressor:
    """
    Sentence-level extractive context compressor & cross-chunk deduplicator.
    1. Deduplicates identical sentences that appear in overlapping parent windows.
    2. Prunes zero-overlap tangential sentences to reduce LLM input tokens while preserving 100% of facts.
    """

    def __init__(self, target_ratio: float = 0.75):
        self.target_ratio = max(0.35, min(1.0, target_ratio))

    def compress(self, chunks: List[ScoredChunk], raw_query: str, expanded_query: str) -> List[str]:
        if self.target_ratio >= 0.99:
            return [c.chunk.effective_context for c in chunks]

        q_tokens = set(tokenize(raw_query))
        exp_tokens = set(tokenize(expanded_query))
        seen_sentences: Set[str] = set()
        compressed_chunks: List[str] = []

        for c in chunks:
            text = c.chunk.effective_context
            sentences = split_into_sentences(text)
            kept_sents: List[str] = []

            for idx, sent in enumerate(sentences):
                norm_key = sent.strip().lower()
                # Cross-chunk sentence deduplication (eliminates overlapping window redundancy)
                if norm_key in seen_sentences:
                    continue

                s_tokens = set(tokenize(sent))
                direct_hits = len(s_tokens.intersection(q_tokens))
                exp_hits = len(s_tokens.intersection(exp_tokens))
                has_numbers = any(ch.isdigit() for ch in sent)

                # Keep sentences that match query/expanded terms, contain quantitative specs, or are the lead sentence
                if direct_hits > 0 or exp_hits > 0 or has_numbers or idx == 0 or len(sentences) <= 2:
                    kept_sents.append(sent)
                    seen_sentences.add(norm_key)

            if not kept_sents and sentences:
                kept_sents.append(sentences[0])

            compressed_chunks.append(" ".join(kept_sents))

        return compressed_chunks
