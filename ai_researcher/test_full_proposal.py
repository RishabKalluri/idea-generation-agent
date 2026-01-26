"""
Test script for full proposal expansion.
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

# Load idea generation module directly
spec = importlib.util.spec_from_file_location('idea_gen', 'modules/idea_generation.py')
idea_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(idea_gen)

SeedIdea = idea_gen.SeedIdea
FullProposal = idea_gen.FullProposal
expand_to_full_proposal = idea_gen.expand_to_full_proposal
load_full_proposal_demo = idea_gen.load_full_proposal_demo
save_proposals_to_file = idea_gen.save_proposals_to_file


def main():
    print("=" * 80)
    print("FULL PROPOSAL EXPANSION TEST")
    print("=" * 80)
    
    # Test loading demo example
    print("\n[1] Testing demo example loading...")
    demo = load_full_proposal_demo()
    if demo:
        print(f"    ✓ Loaded demo example ({len(demo)} characters)")
        print(f"    First 200 chars: {demo[:200]}...")
    else:
        print("    ⚠ No demo example found (will work without it)")
    
    # Create a test seed idea
    print("\n[2] Creating test seed idea...")
    seed_idea = SeedIdea(
        title="Multi-Perspective Verification for Reducing Hallucinations",
        problem="LLMs generate confident but incorrect factual claims without self-awareness of their errors.",
        existing_methods="Retrieval augmentation helps but requires external databases. Fine-tuning is expensive and doesn't generalize.",
        motivation="By querying the same fact from multiple angles and comparing answers, we can detect internal inconsistencies that reveal hallucinations.",
        proposed_method="Generate initial response, create diverse verification questions targeting the same facts from different angles, answer each independently, flag inconsistencies, regenerate with corrections.",
        experiment_plan="Evaluate on TruthfulQA, FEVER, and biography generation tasks. Compare against CoT and RAG baselines."
    )
    
    print(f"    Title: {seed_idea.title}")
    print(f"    Problem: {seed_idea.problem[:100]}...")
    
    # Initialize client
    print("\n[3] Initializing OpenAI client...")
    model_name = "gpt-4"
    print(f"    Model: {model_name}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("    ✓ Client initialized")
    
    # Expand to full proposal
    print("\n[4] Expanding seed idea to full proposal...")
    print("    (This may take 30-60 seconds)")
    print("-" * 80)
    
    try:
        proposal = expand_to_full_proposal(
            seed_idea=seed_idea,
            client=client,
            model_name=model_name,
            use_demo=True
        )
        
        if proposal:
            print("\n" + "=" * 80)
            print("GENERATED FULL PROPOSAL")
            print("=" * 80)
            
            print(f"\n1. TITLE:\n{proposal.title}")
            
            print(f"\n2. PROBLEM STATEMENT:\n{proposal.problem_statement[:500]}...")
            
            print(f"\n3. MOTIVATION:\n{proposal.motivation[:500]}...")
            
            print(f"\n4. PROPOSED METHOD:\n{proposal.proposed_method[:500]}...")
            
            print(f"\n5. EXPERIMENT PLAN:\n{proposal.experiment_plan[:500]}...")
            
            print(f"\n6. TEST CASE EXAMPLES:\n{proposal.test_case_examples[:500]}...")
            
            print(f"\n7. FALLBACK PLAN:\n{proposal.fallback_plan[:500]}...")
            
            # Save to file
            output_file = "test_proposal_output.txt"
            save_proposals_to_file([proposal], output_file)
            print(f"\n✓ Full proposal saved to {output_file}")
            
            # Statistics
            print("\n" + "=" * 80)
            print("STATISTICS")
            print("=" * 80)
            print(f"Total length: {len(proposal.raw_text)} characters")
            print(f"Problem statement: {len(proposal.problem_statement)} chars")
            print(f"Proposed method: {len(proposal.proposed_method)} chars")
            print(f"Experiment plan: {len(proposal.experiment_plan)} chars")
            print(f"Test cases: {len(proposal.test_case_examples)} chars")
            
        else:
            print("\n✗ Failed to generate proposal")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
