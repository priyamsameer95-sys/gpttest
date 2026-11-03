"""Comprehensive TDD test suite for the RAG Optimization & Diagnostic Pipeline."""

import unittest
from rag_optimizer import (
    AbstentionAndFaithfulnessGuard,
    BM25Retriever,
    ContextCompressor,
    CrossEncoderReranker,
    DenseEmbedder,
    Document,
    FixedWindowChunker,
    HybridRetriever,
    LostInTheMiddleReorderer,
    MMRDiversifier,
    OptimizedRAGPipeline,
    PipelineConfig,
    QueryOptimizer,
    RAGDoctor,
    RAGEvaluator,
    RBACTenantSimulator,
    RecursiveSemanticChunker,
    SentenceWindowChunker,
    get_eval_benchmark,
    get_sample_corpus,
)
from rag_doctor import diagnose_custom_pipeline, inspect_chunk_boundaries


class TestRAGOptimizerSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = get_sample_corpus()
        cls.benchmarks = get_eval_benchmark()
        cls.doctor = RAGDoctor(cls.corpus)

    def test_chunking_strategies_preserve_or_expand_context(self):
        fixed = FixedWindowChunker(chunk_size=180, chunk_overlap=20).chunk_documents(self.corpus)
        recursive = RecursiveSemanticChunker(chunk_size=240, chunk_overlap=40).chunk_documents(self.corpus)
        sw = SentenceWindowChunker(chunk_size=180, window_size=1).chunk_documents(self.corpus)

        self.assertGreater(len(fixed), 10)
        self.assertGreater(len(recursive), 10)
        self.assertGreater(len(sw), 10)
        # SentenceWindow chunks must have expanded parent_text larger than or equal to core text
        for c in sw:
            self.assertIsNotNone(c.parent_text)
            self.assertGreaterEqual(len(c.effective_context), len(c.text) - 10)

    def test_hybrid_rrf_outperforms_naive_on_exact_sku(self):
        q = "Lookup exact error code ERR_SSL_9042 and part identifier SKU-8841-X9"
        res = self.doctor.opt_pipe.run_query(q)
        self.assertEqual(res.ranked_chunks[0].chunk.doc_id, "DOC-02")
        self.assertIn("SKU-8841-X9", res.ranked_chunks[0].chunk.effective_context)

    def test_lost_in_the_middle_reordering(self):
        q = "How do we stop duplicate redundant chunks from crowding the prompt and fix LLMs ignoring facts in the middle?"
        res = self.doctor.opt_pipe.run_query(q)
        ranks = [c.rank for c in res.prompt_ordered_chunks]
        self.assertEqual(ranks, [1, 3, 2])

    def test_all_eight_challenges_resolved(self):
        challenges = self.doctor.run_all_challenge_simulations()
        self.assertEqual(len(challenges), 8)
        for ch in challenges:
            self.assertTrue(ch.passed, f"Challenge {ch.challenge_id} failed to resolve")
            self.assertGreater(ch.remediated_score, ch.naive_score)

    def test_custom_pipeline_slider_parameters(self):
        """Tests that diagnose_custom_pipeline honors custom chunk_size, alpha, and top_k sliders."""
        res = diagnose_custom_pipeline(
            self.doctor,
            query="How do we shrink vector RAM footprint from 61GB using int8?",
            chunk_strategy="sentence_window",
            chunk_size=200,
            retrieval_mode="hybrid_rrf",
            alpha=0.65,
            top_k=3,
            use_query_expansion=True,
            use_reranker=True,
            use_mmr=True,
            use_lost_in_middle=True,
            use_compression=True,
            use_guardrail=True,
        )
        self.assertIn("active_config", res)
        self.assertEqual(res["active_config"]["chunk_size"], 200)
        self.assertAlmostEqual(res["active_config"]["alpha"], 0.65)
        self.assertEqual(res["optimized"]["docs"][0], "DOC-01")

    def test_chunk_boundary_visualizer_inspection(self):
        """Tests the interactive chunk boundary visualizer for a selected document."""
        viz = inspect_chunk_boundaries(self.doctor, doc_id="DOC-01", chunk_size=180, window_size=1)
        self.assertEqual(viz["doc_id"], "DOC-01")
        self.assertGreater(len(viz["fixed_chunks"]), 1)
        self.assertGreater(len(viz["sentence_window_chunks"]), 1)
        self.assertIn("broken_boundary", viz["fixed_chunks"][0])
        self.assertIn("core_text", viz["sentence_window_chunks"][0])
        self.assertIn("expanded_window", viz["sentence_window_chunks"][0])


if __name__ == "__main__":
    unittest.main()
