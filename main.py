#!/usr/bin/env python3
"""
RAG Optimization Pipeline — CLI Benchmark, Hyperparameter Tuner & Side-by-Side Inspector.

Usage:
    python3 main.py                           # Run full ablation benchmark + side-by-side query demo
    python3 main.py --query "your question"   # Run a custom query comparing Baseline vs Optimized RAG
    python3 main.py --tune                    # Run automated hyperparameter grid search
"""

import argparse
import json
from typing import List
from rag_optimizer import (
    Document,
    EvalMetrics,
    OptimizedRAGPipeline,
    PipelineConfig,
    RAGEvaluator,
    get_eval_benchmark,
    get_sample_corpus,
)


def get_ablation_configs() -> List[PipelineConfig]:
    """Returns 6 progressive RAG configurations from Naive Baseline to Full Optimized RAG."""
    return [
        PipelineConfig(
            name="1. Naive-Baseline (Fixed+Dense)",
            chunk_strategy="fixed",
            chunk_size=180,
            chunk_overlap=20,
            retrieval_mode="dense",
            initial_top_k=3,
            final_top_k=3,
            use_query_expansion=False,
            use_reranker=False,
            use_mmr=False,
            use_lost_in_middle_reorder=False,
            use_context_compression=False,
        ),
        PipelineConfig(
            name="2. Lexical-BM25 (Recursive)",
            chunk_strategy="recursive",
            chunk_size=260,
            chunk_overlap=40,
            retrieval_mode="bm25",
            initial_top_k=3,
            final_top_k=3,
            use_query_expansion=False,
            use_reranker=False,
            use_mmr=False,
            use_lost_in_middle_reorder=False,
            use_context_compression=False,
        ),
        PipelineConfig(
            name="3. Hybrid-Alpha (Rec+Convex)",
            chunk_strategy="recursive",
            chunk_size=280,
            chunk_overlap=50,
            retrieval_mode="hybrid_alpha",
            alpha=0.55,
            initial_top_k=3,
            final_top_k=3,
            use_query_expansion=False,
            use_reranker=False,
            use_mmr=False,
            use_lost_in_middle_reorder=False,
            use_context_compression=False,
        ),
        PipelineConfig(
            name="4. Hybrid-RRF + SentenceWindow",
            chunk_strategy="sentence_window",
            chunk_size=220,
            window_size=1,
            retrieval_mode="hybrid_rrf",
            alpha=0.55,
            rrf_k=60,
            initial_top_k=3,
            final_top_k=3,
            use_query_expansion=False,
            use_reranker=False,
            use_mmr=False,
            use_lost_in_middle_reorder=False,
            use_context_compression=False,
        ),
        PipelineConfig(
            name="5. RRF + HyDE + CrossReranker",
            chunk_strategy="sentence_window",
            chunk_size=180,
            window_size=1,
            retrieval_mode="hybrid_rrf",
            alpha=0.55,
            rrf_k=60,
            initial_top_k=10,
            final_top_k=3,
            use_query_expansion=True,
            use_reranker=True,
            use_mmr=False,
            use_lost_in_middle_reorder=False,
            use_context_compression=False,
        ),
        PipelineConfig(
            name="6. Full-Optimized-RAG (All Stages)",
            chunk_strategy="sentence_window",
            chunk_size=180,
            window_size=1,
            retrieval_mode="hybrid_rrf",
            alpha=0.55,
            rrf_k=60,
            initial_top_k=12,
            final_top_k=3,
            use_query_expansion=True,
            use_reranker=True,
            use_mmr=True,
            mmr_lambda=0.85,
            use_lost_in_middle_reorder=True,
            use_context_compression=True,
            compression_ratio=0.75,
        ),
    ]


def print_metrics_table(metrics_list: List[EvalMetrics]) -> None:
    """Prints a formatted ASCII benchmark table of RAG optimization stages."""
    sep = "=" * 116
    print("\n" + sep)
    print(" RAG OPTIMIZATION ABLATION BENCHMARK (Evaluated across 8 Ground-Truth Queries & 10 Technical Docs)")
    print(sep)
    header = (
        f"{'Configuration Stage':<35} | {'Hit@3':>6} | {'MRR':>6} | {'NDCG@3':>7} | "
        f"{'FactRecall':>10} | {'CtxPrec':>7} | {'AvgToks':>7} | {'Latency':>8} | {'Score':>6}"
    )
    print(header)
    print("-" * 116)
    for m in metrics_list:
        print(
            f"{m.config_name:<35} | "
            f"{m.hit_rate_at_k * 100:>5.1f}% | "
            f"{m.mrr:>6.3f} | "
            f"{m.ndcg_at_k:>7.3f} | "
            f"{m.keyword_recall * 100:>9.1f}% | "
            f"{m.context_precision * 100:>6.1f}% | "
            f"{m.avg_context_tokens:>7.1f} | "
            f"{m.avg_latency_ms:>6.2f}ms | "
            f"{m.composite_score:>6.3f}"
        )
    print(sep)


def compare_query_side_by_side(documents: List[Document], query: str) -> None:
    """Runs a single query through Naive Baseline vs. Full Optimized RAG and displays the trace."""
    configs = get_ablation_configs()
    baseline_cfg = configs[0]
    optimized_cfg = configs[-1]

    baseline_pipe = OptimizedRAGPipeline(baseline_cfg)
    optimized_pipe = OptimizedRAGPipeline(
        optimized_cfg,
        embedder=baseline_pipe.embedder,
        reranker=baseline_pipe.reranker,
    )

    n_base_chunks = baseline_pipe.index_documents(documents)
    n_opt_chunks = optimized_pipe.index_documents(documents)

    base_resp = baseline_pipe.run_query(query)
    opt_resp = optimized_pipe.run_query(query)

    print("\n" + "=" * 116)
    print(f" SIDE-BY-SIDE QUERY INSPECTION")
    print(f" User Query: \"{query}\"")
    print("=" * 116)

    print(f"\n[A] {baseline_cfg.name} (Indexed {n_base_chunks} fixed chunks)")
    print(f"    Retrieved Docs (Top-3): {[c.chunk.doc_id for c in base_resp.ranked_chunks]}")
    for c in base_resp.ranked_chunks:
        snippet = c.chunk.effective_context.replace("\n", " ")[:110]
        print(f"      Rank #{c.rank}: [{c.chunk.doc_id}] (score={c.score:.3f}) -> \"{snippet}...\"")
    print(f"    Context Tokens: {base_resp.raw_token_count} tokens | Latency: {base_resp.latency_ms:.2f} ms")
    print(f"    Synthesized Answer:\n      {base_resp.synthesized_answer}")

    print("\n" + "-" * 116)
    print(f"[B] {optimized_cfg.name} (Indexed {n_opt_chunks} sentence-window micro-chunks)")
    print(f"    1. Sparse Query Expansion : \"{opt_resp.expanded_sparse_query}\"")
    print(f"    2. Local HyDE Passage     : \"{opt_resp.hyde_dense_query[:115]}...\"")
    print(f"    3. Cross-Reranked + MMR Diversified Top-3:")
    for c in opt_resp.ranked_chunks:
        snippet = c.chunk.effective_context.replace("\n", " ")[:110]
        print(
            f"      Rank #{c.rank}: [{c.chunk.doc_id}] "
            f"(rerank={c.score:.3f}, dense={c.dense_score:.3f}, bm25={c.sparse_score:.2f}) -> \"{snippet}...\""
        )
    prompt_order_ids = [f"#{c.rank}:{c.chunk.doc_id}" for c in opt_resp.prompt_ordered_chunks]
    savings_pct = (
        100.0 * (opt_resp.raw_token_count - opt_resp.compressed_token_count) / max(1, opt_resp.raw_token_count)
    )
    print(f"    4. Lost-in-the-Middle Prompt Order : {prompt_order_ids} (Top ranks placed at start & end)")
    print(
        f"    5. Contextual Token Compression    : {opt_resp.raw_token_count} -> {opt_resp.compressed_token_count} tokens "
        f"({savings_pct:.1f}% reduction) | Latency: {opt_resp.latency_ms:.2f} ms"
    )
    print(f"    6. Grounded Synthesized Answer:\n      {opt_resp.synthesized_answer}")
    print("=" * 116 + "\n")


def run_hyperparameter_grid_search(evaluator: RAGEvaluator) -> None:
    """Runs an automated grid search over chunking strategies, retrieval modes, and alpha weights."""
    grid_configs: List[PipelineConfig] = []
    for strategy in ["fixed", "recursive", "sentence_window"]:
        for mode in ["dense", "bm25", "hybrid_alpha", "hybrid_rrf"]:
            for alpha in [0.4, 0.55, 0.7]:
                if mode in ("dense", "bm25") and alpha != 0.55:
                    continue
                name = f"Grid({strategy[:6]}|{mode}|a={alpha})"
                grid_configs.append(
                    PipelineConfig(
                        name=name,
                        chunk_strategy=strategy,
                        chunk_size=180,
                        window_size=1,
                        retrieval_mode=mode,
                        alpha=alpha,
                        rrf_k=60,
                        initial_top_k=10,
                        final_top_k=3,
                        use_query_expansion=True,
                        use_reranker=True,
                        use_mmr=True,
                        mmr_lambda=0.85,
                        use_lost_in_middle_reorder=True,
                        use_context_compression=True,
                    )
                )

    best_cfg, all_metrics = evaluator.run_ablation_and_tuning(grid_configs)
    all_metrics.sort(key=lambda m: m.composite_score, reverse=True)
    print("\n>>> TOP 5 HYPERPARAMETER GRID-SEARCH CONFIGURATIONS:")
    print_metrics_table(all_metrics[:5])
    print(f"Optimal Configuration Selected: {best_cfg.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Optimization Pipeline & Benchmark Suite")
    parser.add_argument("--query", type=str, default=None, help="Run a custom query side-by-side")
    parser.add_argument("--tune", action="store_true", help="Run automated hyperparameter grid search")
    parser.add_argument("--export-json", type=str, default=None, help="Export benchmark metrics to JSON file")
    args = parser.parse_args()

    documents = get_sample_corpus()
    eval_samples = get_eval_benchmark()
    evaluator = RAGEvaluator(documents, eval_samples)

    print(f"Embedder Backend : {evaluator.shared_embedder.backend_name}")
    print(f"Reranker Backend : {evaluator.shared_reranker.backend_name}")

    # 1. Run Progressive Ablation Benchmark
    ablation_configs = get_ablation_configs()
    best_cfg, metrics_list = evaluator.run_ablation_and_tuning(ablation_configs)
    print_metrics_table(metrics_list)

    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump([m.__dict__ for m in metrics_list], f, indent=2)
        print(f"Exported benchmark report to {args.export_json}")

    # 2. Optional Hyperparameter Grid Search
    if args.tune:
        run_hyperparameter_grid_search(evaluator)

    # 3. Side-by-Side Query Inspection
    demo_query = (
        args.query
        if args.query
        else "How do we stop duplicate redundant chunks from crowding the prompt and fix LLMs ignoring facts in the middle?"
    )
    compare_query_side_by_side(documents, demo_query)


if __name__ == "__main__":
    main()
