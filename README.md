# RAG Optimization Pipeline (`rag_optimizer`)

A modular, self-contained **Retrieval-Augmented Generation (RAG) Optimization & Benchmarking Suite** in Python. It runs out-of-the-box offline with zero mandatory dependencies and automatically upgrades to `sentence-transformers` (`all-MiniLM-L6-v2` + `cross-encoder/ms-marco-MiniLM-L-6-v2`) and `faiss-cpu` when installed.

## Architecture & Optimization Stages

1. **Chunking Strategies (`rag_optimizer/chunkers.py`)**
   - `FixedWindowChunker`: Baseline character sliding window.
   - `RecursiveSemanticChunker`: Hierarchy-aware splitting by paragraph and sentence boundaries.
   - `SentenceWindowChunker`: Small-to-Big (Parent-Child) indexing — embeds compact micro-sentences for high vector specificity and expands to surrounding $\pm k$ sentences at synthesis time.
2. **Query Transformation (`rag_optimizer/query_transform.py`)**
   - Sparse domain synonym & technical acronym injection.
   - Local **HyDE (Hypothetical Document Embeddings)** generator to align short user questions with declarative corpus passages.
   - Multi-query decomposition for compound questions.
3. **Hybrid Retrieval (`rag_optimizer/retrievers.py`)**
   - Combines **Dense Vector Search** and **Sparse Okapi BM25** via **Reciprocal Rank Fusion (RRF, $k=60$)** or **Convex $\alpha$-Fusion**.
4. **Post-Retrieval Reranking & Ordering (`rag_optimizer/rerankers.py`)**
   - **Cross-Encoder Reranker**: Joint query-passage interaction scoring.
   - **Maximal Marginal Relevance (MMR)**: Eliminates redundant chunks while preserving multi-document coverage.
   - **Lost-in-the-Middle Reordering**: Places top-ranked chunks at the very beginning and very end of the prompt window (U-shaped attention curve optimization).
5. **Contextual Compression (`rag_optimizer/compressor.py`)**
   - Cross-chunk sentence deduplication and relevance pruning (reduces prompt tokens by ~25% without sacrificing recall).
6. **Automated Evaluation & Grid-Search Tuner (`rag_optimizer/evaluator.py`)**
   - Computes **Hit Rate@K**, **MRR**, **NDCG@K**, **Fact/Keyword Recall**, **Context Precision**, **Token Efficiency**, and **Latency (ms)**.

## Quick Start

```bash
# Run the 6-stage ablation benchmark + side-by-side query inspection
python3 main.py

# Run automated hyperparameter grid search across chunkers, retrieval modes, and alpha weights
python3 main.py --tune

# Test a custom query side-by-side (Naive Baseline vs. Full Optimized RAG)
python3 main.py --query "Why do dense embeddings fail on exact codes like ERR_SSL_9042?"
```
