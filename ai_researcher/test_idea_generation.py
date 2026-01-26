"""
Test script for seed idea generation.
Uses direct imports to avoid anthropic dependency.
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

# Load idea_generation module directly to avoid anthropic dependency
spec = importlib.util.spec_from_file_location('idea_generation', 'modules/idea_generation.py')
idea_gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(idea_gen_module)

generate_seed_ideas = idea_gen_module.generate_seed_ideas
load_demo_examples = idea_gen_module.load_demo_examples
save_ideas_to_file = idea_gen_module.save_ideas_to_file
SeedIdea = idea_gen_module.SeedIdea


def main():
    """Test seed idea generation."""
    
    print("=" * 80)
    print("Seed Idea Generation Test")
    print("=" * 80)
    
    # Initialize client
    print("\n[Initialization]")
    model_name = "gpt-4"  # or "gpt-3.5-turbo" for cheaper testing
    print(f"Model: {model_name}")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✓ Client initialized")
    
    # Load and show demo examples
    print("\n[Demo Examples]")
    examples = load_demo_examples(num_examples=3)
    print(f"Loaded {len(examples)} demo examples")
    
    # Show first example preview
    if examples:
        first_example = examples[0][:200] + "..." if len(examples[0]) > 200 else examples[0]
        print(f"\nExample 1 preview:\n{first_example}")
    
    # Define research topic
    topic = ("novel prompting methods that can improve factuality and "
             "reduce hallucination of large language models")
    
    print(f"\n[Research Topic]")
    print(f"{topic}")
    
    # Generate a small number of ideas for testing
    num_test_ideas = 5  # Small number for testing
    
    print(f"\n[Generating {num_test_ideas} Test Ideas]")
    print("(In production, you'd generate 4000 ideas)")
    print("-" * 80)
    
    try:
        # For testing, we'll use empty papers list (no RAG)
        # In production, you'd pass papers from retrieve_papers()
        ideas = generate_seed_ideas(
            topic=topic,
            papers=[],  # No RAG for quick test
            client=client,
            model_name=model_name,
            num_ideas=num_test_ideas,
            rag_rate=0.0,  # No RAG for this test
            num_demo_examples=3,
            show_progress=True
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("GENERATED IDEAS")
        print("=" * 80)
        
        if ideas:
            for i, idea in enumerate(ideas, 1):
                print(f"\n{'─'*80}")
                print(f"IDEA {i}: {idea.title}")
                print(f"{'─'*80}")
                print(f"\nProblem: {idea.problem[:200]}..." if len(idea.problem) > 200 else f"\nProblem: {idea.problem}")
                print(f"\nProposed Method: {idea.proposed_method[:300]}..." if len(idea.proposed_method) > 300 else f"\nProposed Method: {idea.proposed_method}")
                print(f"\nExperiment: {idea.experiment_plan[:200]}..." if len(idea.experiment_plan) > 200 else f"\nExperiment: {idea.experiment_plan}")
            
            # Save to file
            output_file = "test_ideas_output.txt"
            save_ideas_to_file(ideas, output_file)
            print(f"\n✓ Ideas saved to {output_file}")
        else:
            print("\n⚠ No ideas generated")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Cost estimation
    print("\n" + "=" * 80)
    print("COST ESTIMATION")
    print("=" * 80)
    print(f"Ideas generated: {num_test_ideas}")
    print(f"Estimated cost: ~${num_test_ideas * 0.01:.2f} (GPT-4)")
    print(f"\nFor 4000 ideas: ~$40 (GPT-4) or ~$4 (GPT-3.5-turbo)")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
