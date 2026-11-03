"""Chunking strategies for RAG optimization: Fixed, Recursive, and Sentence-Window."""

import re
from typing import List
from .models import Chunk, Document


def split_into_sentences(text: str) -> List[str]:
    """Splits text into clean sentences while preserving punctuation."""
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'`])|\n+', text.strip())
    return [s.strip() for s in raw if s and s.strip()]


class BaseChunker:
    """Base interface for document chunkers."""
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        raise NotImplementedError


class FixedWindowChunker(BaseChunker):
    """Baseline sliding-window character chunker."""

    def __init__(self, chunk_size: int = 320, chunk_overlap: int = 64):
        self.chunk_size = max(64, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        for doc in documents:
            text = doc.content.strip()
            idx = 0
            chunk_num = 0
            while idx < len(text):
                end = min(len(text), idx + self.chunk_size)
                sub = text[idx:end].strip()
                if sub:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}::fixed_{chunk_num}",
                            doc_id=doc.doc_id,
                            text=sub,
                            parent_text=None,
                            start_char=idx,
                            end_char=end,
                            metadata={**doc.metadata, "title": doc.title, "strategy": "fixed"},
                        )
                    )
                    chunk_num += 1
                if end == len(text):
                    break
                idx += step
        return chunks


class RecursiveSemanticChunker(BaseChunker):
    """Splits text hierarchically by paragraph, newline, and sentence boundaries."""

    def __init__(self, chunk_size: int = 320, chunk_overlap: int = 64):
        self.chunk_size = max(80, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            sentences = split_into_sentences(doc.content)
            current_group: List[str] = []
            current_len = 0
            chunk_num = 0

            for sent in sentences:
                sent_len = len(sent)
                if current_len + sent_len + 1 > self.chunk_size and current_group:
                    chunk_text = f"{doc.title}: " + " ".join(current_group)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}::rec_{chunk_num}",
                            doc_id=doc.doc_id,
                            text=chunk_text,
                            parent_text=None,
                            metadata={**doc.metadata, "title": doc.title, "strategy": "recursive"},
                        )
                    )
                    chunk_num += 1
                    # Keep overlap sentences from tail
                    overlap_group: List[str] = []
                    overlap_len = 0
                    for prev_sent in reversed(current_group):
                        if overlap_len + len(prev_sent) <= self.chunk_overlap:
                            overlap_group.insert(0, prev_sent)
                            overlap_len += len(prev_sent) + 1
                        else:
                            break
                    current_group = overlap_group
                    current_len = overlap_len

                current_group.append(sent)
                current_len += sent_len + 1

            if current_group:
                chunk_text = f"{doc.title}: " + " ".join(current_group)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::rec_{chunk_num}",
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        parent_text=None,
                        metadata={**doc.metadata, "title": doc.title, "strategy": "recursive"},
                    )
                )
        return chunks


class SentenceWindowChunker(BaseChunker):
    """
    Small-to-Big (Parent-Child) Chunking:
    Indexes tight sentence units for high-specificity vector retrieval,
    but expands to a surrounding window of +/- `window_size` sentences at synthesis time.
    """

    def __init__(self, chunk_size: int = 240, window_size: int = 1):
        self.chunk_size = max(80, chunk_size)
        self.window_size = max(1, window_size)

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            sentences = split_into_sentences(doc.content)
            if not sentences:
                continue

            # Group sentences into compact micro-units (~chunk_size/2)
            micro_units: List[List[int]] = []
            curr_indices: List[int] = []
            curr_len = 0
            target_micro = max(90, self.chunk_size // 2)

            for i, sent in enumerate(sentences):
                if curr_len + len(sent) > target_micro and curr_indices:
                    micro_units.append(curr_indices)
                    curr_indices = []
                    curr_len = 0
                curr_indices.append(i)
                curr_len += len(sent)
            if curr_indices:
                micro_units.append(curr_indices)

            for unit_idx, sent_indices in enumerate(micro_units):
                start_i = sent_indices[0]
                end_i = sent_indices[-1]
                core_text = " ".join(sentences[i] for i in sent_indices)

                win_start = max(0, start_i - self.window_size)
                win_end = min(len(sentences) - 1, end_i + self.window_size)
                expanded_text = " ".join(sentences[win_start : win_end + 1])

                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::sw_{unit_idx}",
                        doc_id=doc.doc_id,
                        text=f"{doc.title} — {core_text}",
                        parent_text=f"[{doc.title}] {expanded_text}",
                        metadata={
                            **doc.metadata,
                            "title": doc.title,
                            "strategy": "sentence_window",
                        },
                    )
                )
        return chunks


def build_chunker(strategy: str, chunk_size: int = 320, chunk_overlap: int = 64, window_size: int = 1) -> BaseChunker:
    """Factory function for chunking strategies."""
    if strategy == "fixed":
        return FixedWindowChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy == "recursive":
        return RecursiveSemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy == "sentence_window":
        return SentenceWindowChunker(chunk_size=chunk_size, window_size=window_size)
    raise ValueError(f"Unknown chunk_strategy: {strategy}")
