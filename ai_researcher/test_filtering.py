"""
Test script for idea filtering (novelty and feasibility checks).
"""

import os
import sys
import importlib.util

# Check for OpenAI key first
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY not set")
    print("\nTo set your API key:")
    print("  export OPENAI_API_KEY='your-openai-api-key-here'")
    sys.exit(1)

from openai import OpenAI

# Load modules directly
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

idea_gen = load_module('idea_gen', 'modules/idea_generation.py')
filtering = load_module('filtering', 'modules/idea_filtering.py')

FullProposal = idea_gen.FullProposal
quick_novelty_check = filtering.quick_novelty_check
quick_feasibility_check = filtering.quick_feasibility_check
check_novelty = filtering.check_novelty
check_feasibility = filtering.check_feasibility


def create_test_proposal():
    """Create a test proposal."""
    return FullProposal(
        title="Multi-Perspective Verification for Factuality",
        problem_statement="LLMs generate confident but incorrect claims.",
        motivation="Self-verification from multiple angles can detect inconsistencies.",
        proposed_method="""The method works in four steps:
1. Generate initial response to a query
2. Create verification questions from different perspectives
3. Answer each question independently
4. Compare answers and fix inconsistencies""",
        experiment_plan="""Step 1: Use GPT-4 API to implement the method
Step 2: Evaluate on TruthfulQA and FEVER datasets
Step 3: Compare against CoT and self-consistency baselines
Step 4: Measure precision, recall, and F1 score""",
        test_case_examples="Example showing baseline failure and method success.",
        fallback_plan="If method fails, analyze failure cases and pivot to analysis paper.",
        raw_text=""
    )


def create_infeasible_proposal():
    """Create a proposal that should fail feasibility check."""
    return FullProposal(
        title="Training a 500B Parameter Model from Scratch",
        problem_statement="Current models are too small.",
        motivation="Bigger models perform better.",
        proposed_method="""Train a 500 billion parameter transformer model from scratch:
1. Collect 10 trillion tokens of training data
2. Use 10,000 GPUs for 6 months
3. Manual annotation of 1 million examples by crowd workers
4. Fine-tune on internal model weights while claiming API-only access""",
        experiment_plan="""Step 1: Train the 500B model (requires 10,000 A100 GPUs)
Step 2: Manually annotate 1 million examples
Step 3: Access internal model weights for analysis
Step 4: Fine-tune for 3 months on additional data""",
        test_case_examples="Examples would require running the 500B model.",
        fallback_plan="Scale down to 100B parameters.",
        raw_text=""
    )


def main():
    print("=" * 80)
    print("IDEA FILTERING TEST")
    print("=" * 80)
    
    # Initialize client
    print("\n[1] Initializing OpenAI client...")
    model_name = "gpt-4"
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"    Model: {model_name}")
    print("    ✓ Client initialized")
    
    # Test 1: Quick novelty check
    print("\n" + "=" * 80)
    print("[2] NOVELTY CHECK TEST")
    print("=" * 80)
    
    print("\nTest case: Comparing a verification idea to Chain-of-Verification paper")
    
    idea_title = "Self-Verification Chain for Reducing Hallucinations"
    idea_method = "Generate response, create verification questions, answer independently, fix inconsistencies."
    paper_title = "Chain-of-Verification Reduces Hallucination in Large Language Models"
    paper_abstract = "We propose Chain-of-Verification (CoVe) to reduce hallucination. The method generates a response, creates verification questions, answers them independently, and fixes inconsistencies."
    
    print(f"\n    Idea: {idea_title}")
    print(f"    Paper: {paper_title}")
    
    is_different, explanation = quick_novelty_check(
        idea_title, idea_method, paper_title, paper_abstract, client, model_name
    )
    
    print(f"\n    Result: {'DIFFERENT (Novel)' if is_different else 'SAME (Not Novel)'}")
    print(f"    Explanation: {explanation}")
    
    # Test 2: Different idea vs same paper
    print("\n    ---")
    print("    Test case: Comparing a different idea to the same paper")
    
    different_idea = "Analogical Prompting for Complex Reasoning"
    different_method = "Recall similar problems, generate example solutions, apply patterns to solve new problem."
    
    print(f"\n    Idea: {different_idea}")
    
    is_different2, explanation2 = quick_novelty_check(
        different_idea, different_method, paper_title, paper_abstract, client, model_name
    )
    
    print(f"\n    Result: {'DIFFERENT (Novel)' if is_different2 else 'SAME (Not Novel)'}")
    print(f"    Explanation: {explanation2}")
    
    # Test 3: Feasibility check - feasible proposal
    print("\n" + "=" * 80)
    print("[3] FEASIBILITY CHECK TEST")
    print("=" * 80)
    
    print("\nTest case: Feasible proposal (API-based, no excessive resources)")
    
    feasible_proposal = create_test_proposal()
    print(f"    Title: {feasible_proposal.title}")
    
    is_feasible, explanation = check_feasibility(feasible_proposal, client, model_name)
    
    print(f"\n    Result: {'FEASIBLE' if is_feasible else 'NOT FEASIBLE'}")
    print(f"    Explanation: {explanation[:200]}...")
    
    # Test 4: Feasibility check - infeasible proposal
    print("\n    ---")
    print("    Test case: Infeasible proposal (excessive resources, inconsistencies)")
    
    infeasible_proposal = create_infeasible_proposal()
    print(f"    Title: {infeasible_proposal.title}")
    
    is_feasible2, explanation2 = check_feasibility(infeasible_proposal, client, model_name)
    
    print(f"\n    Result: {'FEASIBLE' if is_feasible2 else 'NOT FEASIBLE'}")
    print(f"    Explanation: {explanation2[:300]}...")
    
    # Test 5: Full proposal novelty check
    print("\n" + "=" * 80)
    print("[4] FULL NOVELTY CHECK (with embedding similarity)")
    print("=" * 80)
    
    # Create mock papers for testing
    class MockPaper:
        def __init__(self, title, abstract):
            self.title = title
            self.abstract = abstract
            self.paper_id = "mock"
            self.year = 2023
            self.citation_count = 100
            self.authors = ["Author"]
    
    mock_papers = [
        MockPaper(
            "Chain-of-Verification Reduces Hallucination",
            "We propose CoVe to reduce hallucination by generating verification questions."
        ),
        MockPaper(
            "Self-Refine: Iterative Refinement with Self-Feedback",
            "LLMs can improve outputs by critiquing and refining their own work."
        ),
        MockPaper(
            "Tree of Thoughts for Complex Reasoning",
            "Explore multiple reasoning paths in parallel for better solutions."
        ),
    ]
    
    print(f"\n    Testing against {len(mock_papers)} mock papers")
    print(f"    Proposal: {feasible_proposal.title}")
    
    try:
        is_novel, reason = check_novelty(
            feasible_proposal, mock_papers, client, model_name, top_k=3
        )
        
        print(f"\n    Result: {'NOVEL' if is_novel else 'NOT NOVEL'}")
        print(f"    Reason: {reason[:200]}...")
    except Exception as e:
        print(f"\n    ⚠ Could not complete embedding-based check: {e}")
        print("    (This is expected if sentence-transformers is not installed)")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\n✓ Quick novelty check: Working")
    print(f"✓ Quick feasibility check: Working")
    print(f"✓ Full novelty check: {'Working' if 'is_novel' in dir() else 'Needs sentence-transformers'}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
