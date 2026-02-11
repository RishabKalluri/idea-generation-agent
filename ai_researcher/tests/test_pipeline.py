#!/usr/bin/env python3
"""
Pipeline Component Tests

Comprehensive tests for validating each component of the AI research
idea generation pipeline.

Run with:
    cd ai_researcher
    python -m pytest tests/test_pipeline.py -v
    
Or without pytest:
    python tests/test_pipeline.py
"""

import os
import sys
import json
import importlib.util
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# MODULE LOADING (avoid anthropic dependency)
# ============================================================================

def _load_module(name: str, path: str):
    """Load a module from file path."""
    full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    spec = importlib.util.spec_from_file_location(name, full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules
paper_retrieval = _load_module("paper_retrieval", "modules/paper_retrieval.py")
idea_generation = _load_module("idea_generation", "modules/idea_generation.py")
deduplication = _load_module("deduplication", "modules/deduplication.py")
idea_filtering = _load_module("idea_filtering", "modules/idea_filtering.py")
idea_ranking = _load_module("idea_ranking", "modules/idea_ranking.py")
style_normalization = _load_module("style_normalization", "modules/style_normalization.py")
semantic_scholar = _load_module("semantic_scholar", "utils/semantic_scholar.py")

# Import classes
Paper = semantic_scholar.Paper
SeedIdea = idea_generation.SeedIdea
FullProposal = idea_generation.FullProposal


# ============================================================================
# TEST FIXTURES - Sample Data
# ============================================================================

def create_sample_papers(n: int = 5) -> List[Paper]:
    """Create sample Paper objects for testing."""
    papers = [
        Paper(
            paper_id="paper1",
            title="Chain-of-Thought Prompting for Reasoning",
            abstract="We show that generating intermediate reasoning steps improves LLM performance.",
            year=2022,
            citation_count=1500,
            authors=["Wei, J.", "Wang, X."]
        ),
        Paper(
            paper_id="paper2",
            title="Self-Consistency Improves Chain of Thought Reasoning",
            abstract="We propose self-consistency, sampling multiple reasoning paths and selecting the most consistent answer.",
            year=2023,
            citation_count=800,
            authors=["Wang, X.", "Wei, J."]
        ),
        Paper(
            paper_id="paper3",
            title="Constitutional AI: Harmlessness from AI Feedback",
            abstract="We train harmless AI assistants using AI feedback guided by constitutional principles.",
            year=2022,
            citation_count=500,
            authors=["Bai, Y.", "Kadavath, S."]
        ),
        Paper(
            paper_id="paper4",
            title="Tree of Thoughts: Deliberate Problem Solving",
            abstract="We propose Tree of Thoughts which extends chain-of-thought with tree search.",
            year=2023,
            citation_count=400,
            authors=["Yao, S.", "Yu, D."]
        ),
        Paper(
            paper_id="paper5",
            title="ReAct: Synergizing Reasoning and Acting",
            abstract="We propose ReAct, a paradigm combining reasoning traces and actions.",
            year=2023,
            citation_count=600,
            authors=["Yao, S.", "Zhao, J."]
        ),
    ]
    return papers[:n]


def create_sample_seed_ideas(n: int = 5) -> List[SeedIdea]:
    """Create sample SeedIdea objects for testing."""
    ideas = [
        SeedIdea(
            title="Iterative Self-Verification for Factual Consistency",
            problem="LLMs frequently generate plausible but incorrect facts",
            existing_methods="Chain-of-thought prompting, self-consistency",
            motivation="Self-verification could catch errors before final output",
            proposed_method="Generate, extract claims, verify each claim iteratively",
            experiment_plan="Test on TruthfulQA, FEVER, and NQ datasets",
            rag_used=True
        ),
        SeedIdea(
            title="Multi-Agent Debate for Bias Reduction",
            problem="LLMs exhibit social biases and stereotypes",
            existing_methods="Debiasing prompts, RLHF",
            motivation="Multiple perspectives could cancel out individual biases",
            proposed_method="Have multiple LLM agents debate from different viewpoints",
            experiment_plan="Test on BBQ, StereoSet, and WinoBias datasets",
            rag_used=False
        ),
        SeedIdea(
            title="Iterative Self-Checking for Factual Accuracy",  # Similar to #1
            problem="LLMs generate incorrect facts frequently",
            existing_methods="Chain-of-thought, self-consistency voting",
            motivation="Self-checking loops could identify errors",
            proposed_method="Generate response, check facts, iterate until consistent",
            experiment_plan="Evaluate on TruthfulQA and FEVER benchmarks",
            rag_used=True
        ),
        SeedIdea(
            title="Confidence-Weighted Response Generation",
            problem="LLMs don't know what they don't know",
            existing_methods="Temperature scaling, verbalized confidence",
            motivation="Weighting by confidence could improve reliability",
            proposed_method="Generate confidence scores and weight final answers accordingly",
            experiment_plan="Test on calibration benchmarks and QA tasks",
            rag_used=False
        ),
        SeedIdea(
            title="Code-Augmented Mathematical Reasoning",
            problem="LLMs make arithmetic and symbolic errors",
            existing_methods="Program-aided language models",
            motivation="Executing code provides ground truth verification",
            proposed_method="Generate code for computations, execute, verify results",
            experiment_plan="Test on GSM8K, MATH, and SVAMP datasets",
            rag_used=True
        ),
    ]
    return ideas[:n]


def create_sample_proposals(n: int = 3) -> List[FullProposal]:
    """Create sample FullProposal objects for testing."""
    proposals = [
        FullProposal(
            title="Iterative Self-Verification for Factual Consistency",
            problem_statement="Large language models frequently generate plausible but factually incorrect information, leading to potential misinformation spread.",
            motivation="By incorporating explicit verification steps, we can catch errors before they propagate to final outputs.",
            proposed_method="We propose ISV: (1) Generate initial response, (2) Extract factual claims, (3) Self-verify each claim against known facts, (4) Regenerate if inconsistencies found.",
            experiment_plan="Datasets: TruthfulQA, FEVER, Natural Questions. Baselines: Standard prompting, CoT, Self-Consistency. Metrics: Accuracy, F1, hallucination rate.",
            test_case_examples="Q: When was the Eiffel Tower built? Initial: 1889. Verification: Eiffel Tower construction 1887-1889. Final: The Eiffel Tower was built between 1887 and 1889.",
            fallback_plan="If full verification proves too slow, implement lightweight heuristic checks focusing on numerical claims and named entities.",
            seed_idea=create_sample_seed_ideas(1)[0]
        ),
        FullProposal(
            title="Multi-Agent Debate for Reducing Social Biases",
            problem_statement="LLMs exhibit harmful social biases and stereotypes that can perpetuate discrimination.",
            motivation="Multiple agents arguing from diverse perspectives can surface and neutralize individual biases.",
            proposed_method="Deploy 3-5 LLM agents with different 'personas' to debate controversial topics, then synthesize a balanced response.",
            experiment_plan="Datasets: BBQ, StereoSet, WinoBias. Baselines: Single model, debiasing prompts. Metrics: Bias score, stereotype score, task accuracy.",
            test_case_examples="Topic: Career advice. Agent A (conservative view), Agent B (progressive view), Agent C (neutral mediator). Final: Balanced advice considering multiple perspectives.",
            fallback_plan="If computational cost is prohibitive, reduce to 2 agents or use lighter models for secondary agents.",
            seed_idea=create_sample_seed_ideas(2)[1]
        ),
        FullProposal(
            title="Confidence-Calibrated Response Generation",
            problem_statement="LLMs often express unwarranted confidence in incorrect answers, making it hard to trust their outputs.",
            motivation="Properly calibrated confidence would help users know when to trust or verify model outputs.",
            proposed_method="(1) Generate multiple candidate responses, (2) Compute confidence via consistency and verbalized uncertainty, (3) Weight final answer by calibrated confidence.",
            experiment_plan="Datasets: TriviaQA, MMLU, calibration benchmarks. Baselines: Temperature scaling, verbalized confidence. Metrics: ECE, Brier score, selective accuracy.",
            test_case_examples="Q: Capital of Burkina Faso? Candidates: Ouagadougou (0.7), Lagos (0.2), Unknown (0.1). Calibrated response with uncertainty indicator.",
            fallback_plan="If calibration is poor, fall back to presenting top-k answers with raw confidence scores.",
            seed_idea=create_sample_seed_ideas(4)[3]
        ),
    ]
    return proposals[:n]


def create_mock_openai_client():
    """Create a mock OpenAI client for testing."""
    mock_client = Mock()
    
    def mock_completion(*args, **kwargs):
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = "Mock LLM response"
        return response
    
    mock_client.chat.completions.create = mock_completion
    return mock_client


# ============================================================================
# UNIT TESTS - Paper Retrieval
# ============================================================================

class TestPaperRetrieval:
    """Tests for paper retrieval module."""
    
    def test_paper_dataclass(self):
        """Test that Paper dataclass has all required fields."""
        paper = Paper(
            paper_id="test123",
            title="Test Paper",
            abstract="Test abstract",
            year=2023,
            citation_count=100,
            authors=["Author A"]
        )
        
        assert paper.paper_id == "test123"
        assert paper.title == "Test Paper"
        assert paper.abstract == "Test abstract"
        assert paper.year == 2023
        assert paper.citation_count == 100
        assert "Author A" in paper.authors
        
        print("✓ Paper dataclass has all required fields")
    
    def test_paper_scoring_prompt_exists(self):
        """Test that PAPER_SCORING_PROMPT is defined."""
        assert hasattr(paper_retrieval, 'PAPER_SCORING_PROMPT')
        prompt = paper_retrieval.PAPER_SCORING_PROMPT
        assert "{title}" in prompt
        assert "{abstract}" in prompt
        assert "{topic}" in prompt
        
        print("✓ PAPER_SCORING_PROMPT is properly defined")
    
    def test_retrieval_agent_prompt_exists(self):
        """Test that RETRIEVAL_AGENT_PROMPT is defined."""
        assert hasattr(paper_retrieval, 'RETRIEVAL_AGENT_PROMPT')
        prompt = paper_retrieval.RETRIEVAL_AGENT_PROMPT
        assert "{topic}" in prompt
        assert "{num_papers}" in prompt
        
        print("✓ RETRIEVAL_AGENT_PROMPT is properly defined")
    
    def test_retrieve_papers_function_exists(self):
        """Test that retrieve_papers function is defined."""
        assert hasattr(paper_retrieval, 'retrieve_papers')
        assert callable(paper_retrieval.retrieve_papers)
        
        print("✓ retrieve_papers function exists")


# ============================================================================
# UNIT TESTS - Idea Generation
# ============================================================================

class TestIdeaGeneration:
    """Tests for idea generation module."""
    
    def test_seed_idea_dataclass(self):
        """Test that SeedIdea has all required fields."""
        idea = SeedIdea(
            title="Test Idea",
            problem="Test problem",
            existing_methods="Test methods",
            motivation="Test motivation",
            proposed_method="Test proposed method",
            experiment_plan="Test plan",
            rag_used=True
        )
        
        assert idea.title == "Test Idea"
        assert idea.problem == "Test problem"
        assert idea.rag_used == True
        
        print("✓ SeedIdea dataclass has all required fields")
    
    def test_full_proposal_dataclass(self):
        """Test that FullProposal has all required fields."""
        proposal = FullProposal(
            title="Test Proposal",
            problem_statement="Problem",
            motivation="Motivation",
            proposed_method="Method",
            experiment_plan="Plan",
            test_case_examples="Examples",
            fallback_plan="Fallback"
        )
        
        assert proposal.title == "Test Proposal"
        assert proposal.problem_statement == "Problem"
        assert proposal.fallback_plan == "Fallback"
        
        print("✓ FullProposal dataclass has all required fields")
    
    def test_seed_idea_format_exists(self):
        """Test that SEED_IDEA_FORMAT is defined."""
        assert hasattr(idea_generation, 'SEED_IDEA_FORMAT')
        fmt = idea_generation.SEED_IDEA_FORMAT
        assert "Title" in fmt
        assert "Problem" in fmt
        assert "Proposed Method" in fmt
        
        print("✓ SEED_IDEA_FORMAT is properly defined")
    
    def test_load_demo_examples(self):
        """Test that demo examples can be loaded."""
        try:
            examples = idea_generation.load_demo_examples()
            assert isinstance(examples, list)
            print(f"✓ Loaded {len(examples)} demo examples")
        except FileNotFoundError:
            print("⚠ Demo examples not found (expected if not created yet)")
    
    def test_format_papers_for_rag(self):
        """Test RAG paper formatting."""
        papers = create_sample_papers(3)
        formatted = idea_generation.format_papers_for_rag(papers)
        
        assert isinstance(formatted, str)
        assert "Chain-of-Thought" in formatted
        assert len(formatted) > 100
        
        print("✓ format_papers_for_rag works correctly")
    
    def test_parse_seed_idea(self):
        """Test parsing seed idea from text."""
        text = """Title: Test Idea Title

Problem: This is the problem statement.

Existing Methods: Current approaches include X and Y.

Motivation: The key insight is Z.

Proposed Method: We propose to do A, B, C.

Experiment Plan: Test on datasets D and E."""

        idea = idea_generation.parse_seed_idea(text)
        
        assert idea.title == "Test Idea Title"
        assert "problem statement" in idea.problem.lower()
        assert "X and Y" in idea.existing_methods
        
        print("✓ parse_seed_idea works correctly")


# ============================================================================
# UNIT TESTS - Deduplication
# ============================================================================

def _check_sentence_transformers():
    """Check if sentence_transformers is available."""
    try:
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError:
        return False

HAS_SENTENCE_TRANSFORMERS = _check_sentence_transformers()


class TestDeduplication:
    """Tests for deduplication module."""
    
    def test_embed_ideas(self):
        """Test that embedding function works."""
        if not HAS_SENTENCE_TRANSFORMERS:
            print("⚠ Skipping test (sentence_transformers not installed)")
            return
        
        ideas = create_sample_seed_ideas(3)
        embeddings = deduplication.embed_ideas(ideas)
        
        assert len(embeddings) == 3
        assert len(embeddings[0]) > 100  # Embedding dimension
        
        print(f"✓ embed_ideas returns {len(embeddings[0])}-dim embeddings")
    
    def test_embedding_consistency(self):
        """Test that embeddings are consistent across runs."""
        if not HAS_SENTENCE_TRANSFORMERS:
            print("⚠ Skipping test (sentence_transformers not installed)")
            return
        
        ideas = create_sample_seed_ideas(2)
        
        emb1 = deduplication.embed_ideas(ideas)
        emb2 = deduplication.embed_ideas(ideas)
        
        import numpy as np
        
        # Embeddings should be identical for same input
        for i in range(len(ideas)):
            similarity = np.dot(emb1[i], emb2[i]) / (
                np.linalg.norm(emb1[i]) * np.linalg.norm(emb2[i])
            )
            assert similarity > 0.999, f"Embedding {i} not consistent"
        
        print("✓ Embeddings are consistent across runs")
    
    def test_deduplicate_similar_ideas(self):
        """Test that deduplication correctly identifies similar ideas."""
        if not HAS_SENTENCE_TRANSFORMERS:
            print("⚠ Skipping test (sentence_transformers not installed)")
            return
        
        ideas = create_sample_seed_ideas(5)
        
        # Ideas 0 and 2 are intentionally similar (both about self-verification)
        unique_ideas = deduplication.deduplicate_ideas(ideas, threshold=0.85)
        
        # Should remove at least one duplicate
        assert len(unique_ideas) < len(ideas)
        
        print(f"✓ Deduplication reduced {len(ideas)} → {len(unique_ideas)} ideas")
    
    def test_deduplicate_preserves_different_ideas(self):
        """Test that very different ideas are preserved."""
        if not HAS_SENTENCE_TRANSFORMERS:
            print("⚠ Skipping test (sentence_transformers not installed)")
            return
        
        ideas = [
            SeedIdea(
                title="Quantum Computing for NLP",
                problem="Classical computing limits NLP scalability",
                existing_methods="Classical transformers",
                motivation="Quantum speedup",
                proposed_method="Quantum attention mechanisms",
                experiment_plan="Simulate on quantum hardware"
            ),
            SeedIdea(
                title="Emotion Detection in Images",
                problem="Facial expression analysis is inaccurate",
                existing_methods="CNN-based approaches",
                motivation="Better emotion understanding",
                proposed_method="Multimodal fusion",
                experiment_plan="Test on FER dataset"
            ),
        ]
        
        unique_ideas = deduplication.deduplicate_ideas(ideas, threshold=0.85)
        
        # Both very different ideas should be kept
        assert len(unique_ideas) == 2
        
        print("✓ Deduplication preserves different ideas")
    
    def test_compute_pairwise_similarities(self):
        """Test pairwise similarity computation."""
        if not HAS_SENTENCE_TRANSFORMERS:
            print("⚠ Skipping test (sentence_transformers not installed)")
            return
        
        ideas = create_sample_seed_ideas(4)
        # First embed the ideas, then compute similarities
        embeddings = deduplication.embed_ideas(ideas)
        sim_matrix = deduplication.compute_pairwise_similarities(embeddings)
        
        assert sim_matrix.shape == (4, 4)
        
        # Diagonal should be 1.0 (self-similarity)
        for i in range(4):
            assert abs(sim_matrix[i, i] - 1.0) < 0.01
        
        # Matrix should be symmetric
        import numpy as np
        assert np.allclose(sim_matrix, sim_matrix.T)
        
        print("✓ Pairwise similarities computed correctly")
    
    def test_analyze_diversity(self):
        """Test diversity analysis."""
        if not HAS_SENTENCE_TRANSFORMERS:
            print("⚠ Skipping test (sentence_transformers not installed)")
            return
        
        ideas = create_sample_seed_ideas(5)
        stats = deduplication.analyze_diversity(ideas, batch_size=2)
        
        # Check expected keys from analyze_diversity function
        assert "batch_non_duplicate_rates" in stats
        assert "cumulative_unique_counts" in stats
        assert "overall_duplicate_rate" in stats
        # overall_duplicate_rate is a percentage (0-100)
        assert 0 <= stats["overall_duplicate_rate"] <= 100
        
        print(f"✓ Diversity stats: duplicate_rate={stats['overall_duplicate_rate']:.1f}%")
    
    def test_deduplication_functions_exist(self):
        """Test that deduplication functions are defined."""
        assert hasattr(deduplication, 'embed_ideas')
        assert hasattr(deduplication, 'deduplicate_ideas')
        assert hasattr(deduplication, 'analyze_diversity')
        assert callable(deduplication.embed_ideas)
        
        print("✓ All deduplication functions exist")


# ============================================================================
# UNIT TESTS - Idea Filtering
# ============================================================================

class TestIdeaFiltering:
    """Tests for idea filtering module."""
    
    def test_novelty_check_prompt_exists(self):
        """Test that NOVELTY_CHECK_PROMPT is defined."""
        assert hasattr(idea_filtering, 'NOVELTY_CHECK_PROMPT')
        prompt = idea_filtering.NOVELTY_CHECK_PROMPT
        assert "{idea_title}" in prompt
        assert "{paper_title}" in prompt
        
        print("✓ NOVELTY_CHECK_PROMPT is defined")
    
    def test_feasibility_check_prompt_exists(self):
        """Test that FEASIBILITY_CHECK_PROMPT is defined."""
        assert hasattr(idea_filtering, 'FEASIBILITY_CHECK_PROMPT')
        prompt = idea_filtering.FEASIBILITY_CHECK_PROMPT
        assert "feasib" in prompt.lower()
        
        print("✓ FEASIBILITY_CHECK_PROMPT is defined")
    
    def test_check_novelty_function_exists(self):
        """Test that check_novelty function is defined."""
        assert hasattr(idea_filtering, 'check_novelty')
        assert callable(idea_filtering.check_novelty)
        
        print("✓ check_novelty function exists")
    
    def test_check_feasibility_function_exists(self):
        """Test that check_feasibility function is defined."""
        assert hasattr(idea_filtering, 'check_feasibility')
        assert callable(idea_filtering.check_feasibility)
        
        print("✓ check_feasibility function exists")
    
    def test_filter_proposals_function_exists(self):
        """Test that filter_proposals function is defined."""
        assert hasattr(idea_filtering, 'filter_proposals')
        assert callable(idea_filtering.filter_proposals)
        
        print("✓ filter_proposals function exists")


# ============================================================================
# UNIT TESTS - Idea Ranking
# ============================================================================

class TestIdeaRanking:
    """Tests for idea ranking module."""
    
    def test_pairwise_comparison_prompt_exists(self):
        """Test that PAIRWISE_COMPARISON_PROMPT is defined."""
        assert hasattr(idea_ranking, 'PAIRWISE_COMPARISON_PROMPT')
        prompt = idea_ranking.PAIRWISE_COMPARISON_PROMPT
        assert "idea" in prompt.lower() or "proposal" in prompt.lower()
        
        print("✓ PAIRWISE_COMPARISON_PROMPT is defined")
    
    def test_pairwise_compare_function_exists(self):
        """Test that pairwise_compare function is defined."""
        assert hasattr(idea_ranking, 'pairwise_compare')
        assert callable(idea_ranking.pairwise_compare)
        
        print("✓ pairwise_compare function exists")
    
    def test_swiss_tournament_round_function_exists(self):
        """Test that swiss_tournament_round function is defined."""
        assert hasattr(idea_ranking, 'swiss_tournament_round')
        assert callable(idea_ranking.swiss_tournament_round)
        
        print("✓ swiss_tournament_round function exists")
    
    def test_rank_proposals_function_exists(self):
        """Test that rank_proposals function is defined."""
        assert hasattr(idea_ranking, 'rank_proposals')
        assert callable(idea_ranking.rank_proposals)
        
        print("✓ rank_proposals function exists")
    
    def test_create_pairings(self):
        """Test pairing creation for Swiss tournament."""
        # Create 4 sample proposals (need to modify titles to be unique)
        proposals = [
            FullProposal(title="Proposal A", problem_statement="PA", motivation="MA",
                         proposed_method="Method A", experiment_plan="Exp A",
                         test_case_examples="Ex A", fallback_plan="Fallback A"),
            FullProposal(title="Proposal B", problem_statement="PB", motivation="MB",
                         proposed_method="Method B", experiment_plan="Exp B",
                         test_case_examples="Ex B", fallback_plan="Fallback B"),
            FullProposal(title="Proposal C", problem_statement="PC", motivation="MC",
                         proposed_method="Method C", experiment_plan="Exp C",
                         test_case_examples="Ex C", fallback_plan="Fallback C"),
            FullProposal(title="Proposal D", problem_statement="PD", motivation="MD",
                         proposed_method="Method D", experiment_plan="Exp D",
                         test_case_examples="Ex D", fallback_plan="Fallback D"),
        ]
        
        # Scores keyed by title
        scores = {
            "Proposal A": 3,
            "Proposal B": 2,
            "Proposal C": 2,
            "Proposal D": 1
        }
        
        pairings = idea_ranking.create_pairings(proposals, scores)
        
        # Should have 2 pairings for 4 items
        assert len(pairings) == 2
        
        # Each pairing should be a tuple of 2 proposals
        for pair in pairings:
            assert len(pair) == 2
            assert hasattr(pair[0], 'title')
            assert hasattr(pair[1], 'title')
        
        # All proposals should be paired (check titles)
        paired_titles = set()
        for p1, p2 in pairings:
            paired_titles.add(p1.title)
            paired_titles.add(p2.title)
        assert paired_titles == {"Proposal A", "Proposal B", "Proposal C", "Proposal D"}
        
        print("✓ create_pairings works correctly")


# ============================================================================
# UNIT TESTS - Style Normalization
# ============================================================================

class TestStyleNormalization:
    """Tests for style normalization module."""
    
    def test_style_normalization_prompt_exists(self):
        """Test that STYLE_NORMALIZATION_PROMPT is defined."""
        assert hasattr(style_normalization, 'STYLE_NORMALIZATION_PROMPT')
        prompt = style_normalization.STYLE_NORMALIZATION_PROMPT
        assert "template" in prompt.lower()
        assert "format" in prompt.lower()
        
        print("✓ STYLE_NORMALIZATION_PROMPT is defined")
    
    def test_format_proposal_for_normalization(self):
        """Test proposal formatting."""
        proposal = create_sample_proposals(1)[0]
        formatted = style_normalization.format_proposal_for_normalization(proposal)
        
        assert "Title:" in formatted
        assert "1. Problem Statement" in formatted
        assert "6. Fallback Plan" in formatted
        
        print("✓ format_proposal_for_normalization works")
    
    def test_parse_normalized_text(self):
        """Test parsing normalized text."""
        original = create_sample_proposals(1)[0]
        
        normalized_text = """Title: Test Normalized Title

1. Problem Statement: This is the normalized problem.

2. Motivation: This is the motivation section.

3. Proposed Method: This is the method.

4. Step-by-Step Experiment Plan: This is the plan.

5. Test Case Examples: These are examples.

6. Fallback Plan: This is the fallback."""

        parsed = style_normalization.parse_normalized_text(normalized_text, original)
        
        assert parsed.title == "Test Normalized Title"
        assert "normalized problem" in parsed.problem_statement.lower()
        
        print("✓ parse_normalized_text works")
    
    def test_preprocess_citations(self):
        """Test citation preprocessing."""
        text = "This is supported [1] by research [2,3,4] and (Smith et al., 2023)."
        processed = style_normalization.preprocess_citations(text)
        
        assert "[1]" not in processed
        assert "[2,3,4]" not in processed
        assert "(Smith et al., 2023)" in processed
        
        print("✓ preprocess_citations works")
    
    def test_update_model_names(self):
        """Test model name updates."""
        text = "We use Claude-2 and LLaMA-7B and Llama-2-70B."
        updated = style_normalization.update_model_names(text)
        
        assert "Claude-3.5" in updated
        assert "LLaMA-3" in updated
        
        print("✓ update_model_names works")
    
    def test_get_default_template(self):
        """Test default template."""
        template = style_normalization.get_default_template()
        
        assert "Title:" in template
        assert "1. Problem Statement" in template
        assert "6. Fallback Plan" in template
        
        print("✓ get_default_template works")


# ============================================================================
# INTEGRATION TESTS (require API key)
# ============================================================================

class TestIntegration:
    """Integration tests that require API access."""
    
    @staticmethod
    def _get_client():
        """Get OpenAI client if API key is available."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    
    def test_live_paper_retrieval(self):
        """Test live paper retrieval from Semantic Scholar."""
        client = self._get_client()
        if not client:
            print("⚠ Skipping live test (no OPENAI_API_KEY)")
            return
        
        papers = paper_retrieval.retrieve_papers(
            topic="chain of thought prompting",
            client=client,
            model_name="gpt-5-mini",
            target_papers=10
        )
        
        assert len(papers) > 0
        # Check papers have expected attributes (Paper-like objects)
        for p in papers:
            assert hasattr(p, 'title'), "Paper missing 'title'"
            assert hasattr(p, 'abstract'), "Paper missing 'abstract'"
            assert hasattr(p, 'paper_id'), "Paper missing 'paper_id'"
        
        print(f"✓ Live paper retrieval returned {len(papers)} papers")
    
    def test_live_idea_generation(self):
        """Test live seed idea generation."""
        client = self._get_client()
        if not client:
            print("⚠ Skipping live test (no OPENAI_API_KEY)")
            return
        
        papers = create_sample_papers(3)
        
        ideas = idea_generation.generate_seed_ideas(
            topic="improving factuality in LLMs",
            papers=papers,
            client=client,
            model_name="gpt-5-mini",
            num_ideas=2,
            rag_rate=0.5,
            num_demo_examples=2
        )
        
        assert len(ideas) > 0
        # Check ideas have expected attributes
        for idea in ideas:
            assert hasattr(idea, 'title'), "Idea missing 'title'"
            assert hasattr(idea, 'proposed_method'), "Idea missing 'proposed_method'"
        
        print(f"✓ Live idea generation created {len(ideas)} ideas")
    
    def test_live_proposal_expansion(self):
        """Test live proposal expansion."""
        client = self._get_client()
        if not client:
            print("⚠ Skipping live test (no OPENAI_API_KEY)")
            return
        
        idea = create_sample_seed_ideas(1)[0]
        proposal = idea_generation.expand_to_full_proposal(
            idea, client, "gpt-5-mini"
        )
        
        # Check proposal has expected attributes
        assert hasattr(proposal, 'title'), "Proposal missing 'title'"
        assert hasattr(proposal, 'proposed_method'), "Proposal missing 'proposed_method'"
        
        # Check that we got some content (either parsed or raw)
        has_content = (
            (proposal.title and len(proposal.title) > 0) or
            (proposal.proposed_method and len(proposal.proposed_method) > 0) or
            (hasattr(proposal, 'raw_text') and proposal.raw_text and len(proposal.raw_text) > 0)
        )
        assert has_content, "Proposal has no content"
        
        title_preview = proposal.title[:40] if proposal.title else "(parsed from raw)"
        print(f"✓ Live proposal expansion: '{title_preview}...'")
    
    def test_live_novelty_check(self):
        """Test live novelty checking."""
        client = self._get_client()
        if not client:
            print("⚠ Skipping live test (no OPENAI_API_KEY)")
            return
        
        proposal = create_sample_proposals(1)[0]
        papers = create_sample_papers(3)
        
        is_novel, reason = idea_filtering.check_novelty(
            proposal, papers, client, "gpt-5-mini"
        )
        
        assert isinstance(is_novel, bool)
        assert isinstance(reason, str)
        
        print(f"✓ Live novelty check: novel={is_novel}")
    
    def test_live_feasibility_check(self):
        """Test live feasibility checking."""
        client = self._get_client()
        if not client:
            print("⚠ Skipping live test (no OPENAI_API_KEY)")
            return
        
        proposal = create_sample_proposals(1)[0]
        
        is_feasible, reason = idea_filtering.check_feasibility(
            proposal, client, "gpt-5-mini"
        )
        
        assert isinstance(is_feasible, bool)
        assert isinstance(reason, str)
        
        print(f"✓ Live feasibility check: feasible={is_feasible}")
    
    def test_live_pairwise_comparison(self):
        """Test live pairwise comparison."""
        client = self._get_client()
        if not client:
            print("⚠ Skipping live test (no OPENAI_API_KEY)")
            return
        
        proposals = create_sample_proposals(2)
        
        # pairwise_compare returns (winner, raw_response) where winner is 'A' or 'B'
        result = idea_ranking.pairwise_compare(
            proposals[0], proposals[1], client, "gpt-5-mini"
        )
        
        # Handle both tuple return and single value return
        if isinstance(result, tuple):
            winner, response = result
        else:
            winner = result
        
        assert winner in ['A', 'B'], f"Expected 'A' or 'B', got {winner}"
        
        print(f"✓ Live pairwise comparison: winner={winner}")
    
    def test_live_style_normalization(self):
        """Test live style normalization."""
        client = self._get_client()
        if not client:
            print("⚠ Skipping live test (no OPENAI_API_KEY)")
            return
        
        proposal = create_sample_proposals(1)[0]
        template = style_normalization.get_default_template()
        
        normalized = style_normalization.normalize_style(
            proposal, template, client, "gpt-5-mini"
        )
        
        # Check normalized has expected attributes (FullProposal-like)
        assert hasattr(normalized, 'title'), "Normalized missing 'title'"
        assert hasattr(normalized, 'proposed_method'), "Normalized missing 'proposed_method'"
        assert len(normalized.title) > 0
        
        print(f"✓ Live style normalization complete")


# ============================================================================
# FULL PIPELINE TEST
# ============================================================================

def test_full_pipeline_small():
    """Run full pipeline with tiny parameters to validate integration."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠ Skipping full pipeline test (no OPENAI_API_KEY)")
        return
    
    if not HAS_SENTENCE_TRANSFORMERS:
        print("⚠ Skipping full pipeline test (sentence_transformers not installed)")
        return
    
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    model = "gpt-5-mini"
    
    print("\n" + "="*60)
    print("FULL PIPELINE INTEGRATION TEST (Small)")
    print("="*60)
    
    # Step 1: Use sample papers (skip retrieval for speed)
    print("\n[1/6] Using sample papers...")
    papers = create_sample_papers(5)
    print(f"  ✓ {len(papers)} papers")
    
    # Step 2: Generate 3 ideas
    print("\n[2/6] Generating ideas...")
    ideas = idea_generation.generate_seed_ideas(
        topic="improving factuality in LLMs",
        papers=papers,
        client=client,
        model_name=model,
        num_ideas=3,
        rag_rate=0.5,
        num_demo_examples=2
    )
    print(f"  ✓ {len(ideas)} ideas generated")
    
    # Step 3: Deduplicate
    print("\n[3/6] Deduplicating...")
    unique_ideas = deduplication.deduplicate_ideas(ideas, threshold=0.9)
    print(f"  ✓ {len(unique_ideas)} unique ideas")
    
    # Step 4: Expand first 2
    print("\n[4/6] Expanding proposals...")
    proposals = []
    for idea in unique_ideas[:2]:
        proposal = idea_generation.expand_to_full_proposal(idea, client, model)
        proposals.append(proposal)
    print(f"  ✓ {len(proposals)} proposals expanded")
    
    # Step 5: Filter (skip for speed, just validate structure)
    print("\n[5/6] Validating proposals...")
    for p in proposals:
        assert p.title and len(p.title) > 0
        assert p.problem_statement and len(p.problem_statement) > 0
        assert p.proposed_method and len(p.proposed_method) > 0
    print(f"  ✓ All proposals valid")
    
    # Step 6: Simple ranking (just one comparison)
    print("\n[6/6] Testing ranking...")
    if len(proposals) >= 2:
        result = idea_ranking.pairwise_compare(
            proposals[0], proposals[1], client, model
        )
        # pairwise_compare returns (winner, response) where winner is 'A' or 'B'
        winner = result[0] if isinstance(result, tuple) else result
        print(f"  ✓ Comparison complete (winner: proposal {winner})")
    
    print("\n" + "="*60)
    print("FULL PIPELINE TEST PASSED!")
    print("="*60)
    
    return proposals


# ============================================================================
# MAIN
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("PIPELINE COMPONENT TESTS")
    print("="*60)
    
    # Unit tests (no API needed)
    print("\n--- Paper Retrieval Tests ---")
    pr_tests = TestPaperRetrieval()
    pr_tests.test_paper_dataclass()
    pr_tests.test_paper_scoring_prompt_exists()
    pr_tests.test_retrieval_agent_prompt_exists()
    pr_tests.test_retrieve_papers_function_exists()
    
    print("\n--- Idea Generation Tests ---")
    ig_tests = TestIdeaGeneration()
    ig_tests.test_seed_idea_dataclass()
    ig_tests.test_full_proposal_dataclass()
    ig_tests.test_seed_idea_format_exists()
    ig_tests.test_load_demo_examples()
    ig_tests.test_format_papers_for_rag()
    ig_tests.test_parse_seed_idea()
    
    print("\n--- Deduplication Tests ---")
    dd_tests = TestDeduplication()
    dd_tests.test_deduplication_functions_exist()
    dd_tests.test_embed_ideas()
    dd_tests.test_embedding_consistency()
    dd_tests.test_deduplicate_similar_ideas()
    dd_tests.test_deduplicate_preserves_different_ideas()
    dd_tests.test_compute_pairwise_similarities()
    dd_tests.test_analyze_diversity()
    
    print("\n--- Filtering Tests ---")
    if_tests = TestIdeaFiltering()
    if_tests.test_novelty_check_prompt_exists()
    if_tests.test_feasibility_check_prompt_exists()
    if_tests.test_check_novelty_function_exists()
    if_tests.test_check_feasibility_function_exists()
    if_tests.test_filter_proposals_function_exists()
    
    print("\n--- Ranking Tests ---")
    ir_tests = TestIdeaRanking()
    ir_tests.test_pairwise_comparison_prompt_exists()
    ir_tests.test_pairwise_compare_function_exists()
    ir_tests.test_swiss_tournament_round_function_exists()
    ir_tests.test_rank_proposals_function_exists()
    ir_tests.test_create_pairings()
    
    print("\n--- Style Normalization Tests ---")
    sn_tests = TestStyleNormalization()
    sn_tests.test_style_normalization_prompt_exists()
    sn_tests.test_format_proposal_for_normalization()
    sn_tests.test_parse_normalized_text()
    sn_tests.test_preprocess_citations()
    sn_tests.test_update_model_names()
    sn_tests.test_get_default_template()
    
    # Integration tests (API needed)
    print("\n--- Integration Tests (require OPENAI_API_KEY) ---")
    int_tests = TestIntegration()
    int_tests.test_live_paper_retrieval()
    int_tests.test_live_idea_generation()
    int_tests.test_live_proposal_expansion()
    int_tests.test_live_novelty_check()
    int_tests.test_live_feasibility_check()
    int_tests.test_live_pairwise_comparison()
    int_tests.test_live_style_normalization()
    
    # Full pipeline test
    test_full_pipeline_small()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
