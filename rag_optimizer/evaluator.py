"""Evaluation suite and automated Hyperparameter Grid-Search Optimizer for RAG pipelines."""

import math
from typing import List, Tuple
from .embedders import DenseEmbedder
from .models import Document, EvalMetrics, EvalSample, PipelineConfig
from .pipeline import OptimizedRAGPipeline
from .rerankers import CrossEncoderReranker


class RAGEvaluator:
    """Evaluates and benchmarks RAG pipeline configurations against ground-truth QA datasets."""

    def __init__(self, documents: List[Document], eval_samples: List[EvalSample]):
        self.documents = documents
        self.eval_samples = eval_samples
        self.shared_embedder = DenseEmbedder()
        self.shared_reranker = CrossEncoderReranker()

    def evaluate_config(self, config: PipelineConfig) -> EvalMetrics:
        pipeline = OptimizedRAGPipeline(
            config=config,
            embedder=self.shared_embedder,
            reranker=self.shared_reranker,
        )
        pipeline.index_documents(self.documents)

        hits = 0.0
        rr_sum = 0.0
        ndcg_sum = 0.0
        kw_recall_sum = 0.0
        precision_sum = 0.0
        total_tokens = 0.0
        total_latency = 0.0
        n = max(1, len(self.eval_samples))

        for sample in self.eval_samples:
            resp = pipeline.run_query(sample.query)
            expected_docs = set(sample.expected_doc_ids)
            retrieved_docs = [c.chunk.doc_id for c in resp.ranked_chunks]

            # 1. Hit Rate@K & MRR
            first_rank = None
            for idx, doc_id in enumerate(retrieved_docs):
                if doc_id in expected_docs:
                    first_rank = idx + 1
                    break
            if first_rank is not None:
                hits += 1.0
                rr_sum += 1.0 / first_rank

            # 2. NDCG@K (binary relevance per unique expected document)
            dcg = 0.0
            seen_relevant = set()
            for idx, doc_id in enumerate(retrieved_docs):
                if doc_id in expected_docs and doc_id not in seen_relevant:
                    dcg += 1.0 / math.log2(idx + 2)
                    seen_relevant.add(doc_id)
            ideal_hits = min(len(expected_docs), config.final_top_k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
            ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0

            # 3. Context Precision (fraction of retrieved chunks belonging to relevant docs)
            rel_chunks = sum(1 for doc_id in retrieved_docs if doc_id in expected_docs)
            precision_sum += rel_chunks / max(1, len(retrieved_docs))

            # 4. Ground-Truth Keyword / Fact Recall in Final Compressed Context
            joined_context = " ".join(resp.compressed_contexts).lower()
            if sample.expected_keywords:
                kw_hits = sum(1 for kw in sample.expected_keywords if kw.lower() in joined_context)
                kw_recall_sum += kw_hits / len(sample.expected_keywords)
            else:
                kw_recall_sum += 1.0

            total_tokens += resp.compressed_token_count
            total_latency += resp.latency_ms

        hit_rate = hits / n
        mrr = rr_sum / n
        ndcg = ndcg_sum / n
        kw_recall = kw_recall_sum / n
        ctx_precision = precision_sum / n
        avg_tokens = total_tokens / n
        avg_latency = total_latency / n

        # Composite objective combining ranking accuracy, fact recall, and context precision
        composite = (
            0.25 * mrr
            + 0.25 * ndcg
            + 0.30 * kw_recall
            + 0.15 * ctx_precision
            + 0.05 * hit_rate
        )

        return EvalMetrics(
            config_name=config.name,
            hit_rate_at_k=hit_rate,
            mrr=mrr,
            ndcg_at_k=ndcg,
            keyword_recall=kw_recall,
            context_precision=ctx_precision,
            avg_context_tokens=avg_tokens,
            avg_latency_ms=avg_latency,
            composite_score=composite,
        )

    def run_ablation_and_tuning(self, candidate_configs: List[PipelineConfig]) -> Tuple[PipelineConfig, List[EvalMetrics]]:
        """Benchmarks a suite of RAG configurations and returns the best PipelineConfig."""
        all_metrics: List[EvalMetrics] = []
        best_config = candidate_configs[0]
        best_score = -1.0

        for cfg in candidate_configs:
            metrics = self.evaluate_config(cfg)
            all_metrics.append(metrics)
            if metrics.composite_score > best_score:
                best_score = metrics.composite_score
                best_config = cfg

        return best_config, all_metrics
