#!/usr/bin/env python3
"""Test script for the idea ranking module."""

import os
import sys

def test_module_loads():
    """Test that the module can be loaded without anthropic."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        'ranking', 
        'modules/idea_ranking.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    print("✓ Module loaded successfully")
    print(f"  pairwise_compare: {module.pairwise_compare}")
    print(f"  swiss_tournament_round: {module.swiss_tournament_round}")
    print(f"  rank_proposals: {module.rank_proposals}")
    print(f"  create_pairings: {module.create_pairings}")
    print(f"  FullProposal: {module.FullProposal}")
    print(f"  MatchResult: {module.MatchResult}")
    print(f"  TournamentState: {module.TournamentState}")
    
    return module


def test_create_pairings(module):
    """Test the pairing algorithm."""
    # Create mock proposals
    FullProposal = module.FullProposal
    
    proposals = [
        FullProposal(
            title=f"Proposal {i}",
            problem_statement=f"Problem {i}",
            motivation=f"Motivation {i}",
            proposed_method=f"Method {i}",
            experiment_plan=f"Experiment {i}",
            test_case_examples="Examples",
            fallback_plan="Fallback"
        )
        for i in range(6)
    ]
    
    # Test with equal scores
    scores = {f"Proposal {i}": 0 for i in range(6)}
    pairs = module.create_pairings(proposals, scores)
    
    print(f"\n✓ Created {len(pairs)} pairs from {len(proposals)} proposals")
    for i, (a, b) in enumerate(pairs):
        print(f"  Pair {i+1}: {a.title} vs {b.title}")
    
    # Test with different scores
    scores = {"Proposal 0": 3, "Proposal 1": 3, "Proposal 2": 2, 
              "Proposal 3": 1, "Proposal 4": 0, "Proposal 5": 0}
    pairs = module.create_pairings(proposals, scores)
    
    print(f"\n✓ Created pairs with varied scores")
    for i, (a, b) in enumerate(pairs):
        print(f"  Pair {i+1}: {a.title}(score={scores[a.title]}) vs {b.title}(score={scores[b.title]})")
    
    # Test odd number
    odd_proposals = proposals[:5]
    scores = {p.title: 0 for p in odd_proposals}
    pairs = module.create_pairings(odd_proposals, scores)
    
    print(f"\n✓ Created {len(pairs)} pairs from {len(odd_proposals)} proposals (odd number)")
    

def test_tournament_state(module):
    """Test the TournamentState class."""
    FullProposal = module.FullProposal
    TournamentState = module.TournamentState
    MatchResult = module.MatchResult
    
    proposals = [
        FullProposal(
            title=f"Test Proposal {i}",
            problem_statement="Problem",
            motivation="Motivation",
            proposed_method="Method",
            experiment_plan="Experiment",
            test_case_examples="Examples",
            fallback_plan="Fallback"
        )
        for i in range(3)
    ]
    
    state = TournamentState(
        proposals=proposals,
        scores={p.title: 0 for p in proposals}
    )
    
    # Test get_proposal_by_id
    found = state.get_proposal_by_id("Test Proposal 1")
    assert found is not None, "Should find proposal"
    assert found.title == "Test Proposal 1", "Should match title"
    
    not_found = state.get_proposal_by_id("Nonexistent")
    assert not_found is None, "Should not find nonexistent proposal"
    
    print("\n✓ TournamentState works correctly")


def test_format_proposal(module):
    """Test proposal formatting for comparison."""
    FullProposal = module.FullProposal
    
    proposal = FullProposal(
        title="Test Title",
        problem_statement="A very important problem that needs solving",
        motivation="Strong motivation",
        proposed_method="Novel approach using transformers" * 50,  # Long method
        experiment_plan="Test on multiple benchmarks",
        test_case_examples="Example 1, Example 2",
        fallback_plan="Simplify approach"
    )
    
    formatted = module.format_proposal_for_comparison(proposal, max_length=500)
    
    print(f"\n✓ Formatted proposal ({len(formatted)} chars):")
    print(f"  {formatted[:100]}...")
    
    assert len(formatted) <= 503, "Should be within max_length (with ...)"


def test_full_ranking_mock():
    """Test the full ranking pipeline with mock LLM."""
    print("\n" + "="*50)
    print("MOCK RANKING TEST (no API calls)")
    print("="*50)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location('ranking', 'modules/idea_ranking.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    FullProposal = module.FullProposal
    
    # Create test proposals
    proposals = [
        FullProposal(
            title="Self-Correcting Language Models",
            problem_statement="LLMs often make errors without self-reflection",
            motivation="Models that verify their own outputs could be more reliable",
            proposed_method="Add a verification step after each generation",
            experiment_plan="Test on QA and math reasoning tasks",
            test_case_examples="Chain-of-thought examples",
            fallback_plan="Use simpler verification"
        ),
        FullProposal(
            title="Multi-Agent Debate for Reasoning",
            problem_statement="Single models have blind spots in reasoning",
            motivation="Multiple perspectives can improve accuracy",
            proposed_method="Have multiple LLMs debate and reach consensus",
            experiment_plan="Evaluate on complex reasoning benchmarks",
            test_case_examples="Mathematical proofs",
            fallback_plan="Reduce number of agents"
        ),
        FullProposal(
            title="Retrieval-Augmented Generation 2.0",
            problem_statement="Standard RAG has limited context understanding",
            motivation="Better retrieval leads to better generation",
            proposed_method="Use semantic chunking and re-ranking",
            experiment_plan="Test on knowledge-intensive tasks",
            test_case_examples="Open-domain QA",
            fallback_plan="Use simpler retrieval"
        ),
    ]
    
    # Mock comparison function (deterministic for testing)
    def mock_compare(a, b, client, model_name):
        # Return winner based on alphabetical title order (for reproducibility)
        if a.title < b.title:
            return 'A', "Mock: A wins alphabetically"
        else:
            return 'B', "Mock: B wins alphabetically"
    
    # Patch the pairwise_compare function
    original_compare = module.pairwise_compare
    module.pairwise_compare = mock_compare
    
    try:
        # Run tournament with mock
        scores = {p.title: 0 for p in proposals}
        
        for round_num in range(3):
            scores, matches = module.swiss_tournament_round(
                proposals=proposals,
                scores=scores,
                client=None,
                model_name="mock",
                round_num=round_num,
                show_progress=False
            )
            print(f"\nAfter round {round_num + 1}:")
            for title, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {title}: {score}")
        
        print("\n✓ Mock tournament completed successfully")
        
    finally:
        # Restore original function
        module.pairwise_compare = original_compare


def test_with_openai():
    """Test actual ranking with OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠ OPENAI_API_KEY not set, skipping live test")
        return
    
    print("\n" + "="*50)
    print("LIVE RANKING TEST (with OpenAI)")
    print("="*50)
    
    from openai import OpenAI
    client = OpenAI()
    model_name = "gpt-3.5-turbo"  # Use cheaper model for testing
    
    import importlib.util
    spec = importlib.util.spec_from_file_location('ranking', 'modules/idea_ranking.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    FullProposal = module.FullProposal
    
    proposals = [
        FullProposal(
            title="Self-Correcting Language Models",
            problem_statement="LLMs often generate errors without detecting them",
            motivation="Self-verification could improve reliability",
            proposed_method="Add verification step after generation",
            experiment_plan="Test on QA and math tasks",
            test_case_examples="Arithmetic verification",
            fallback_plan="Use simpler checks"
        ),
        FullProposal(
            title="Multi-Agent Debate for Reasoning",
            problem_statement="Single models have reasoning blind spots",
            motivation="Multiple perspectives improve accuracy",
            proposed_method="LLMs debate and reach consensus",
            experiment_plan="Evaluate on reasoning benchmarks",
            test_case_examples="Logic puzzles",
            fallback_plan="Fewer agents"
        ),
    ]
    
    # Test single comparison
    print("\nTesting single pairwise comparison...")
    winner, response = module.pairwise_compare(
        proposals[0], proposals[1], client, model_name
    )
    print(f"  Winner: {winner}")
    print(f"  Response: {response[:100]}...")
    
    print("\n✓ Live pairwise comparison works!")


if __name__ == "__main__":
    print("="*60)
    print("IDEA RANKING MODULE TESTS")
    print("="*60)
    
    module = test_module_loads()
    test_create_pairings(module)
    test_tournament_state(module)
    test_format_proposal(module)
    test_full_ranking_mock()
    test_with_openai()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
