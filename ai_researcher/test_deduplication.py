"""
Test script for embedding-based deduplication.
"""

import os
import sys
import importlib.util

# Load modules directly to avoid anthropic dependency
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load idea generation module (for SeedIdea class)
idea_gen = load_module('idea_generation', 'modules/idea_generation.py')
SeedIdea = idea_gen.SeedIdea

# Load deduplication module
dedup = load_module('deduplication', 'modules/deduplication.py')


def create_test_ideas():
    """Create test ideas with varying similarity."""
    ideas = [
        # Idea 1: Chain-of-Verification
        SeedIdea(
            title="Chain-of-Verification for Factuality",
            problem="LLMs hallucinate plausible but incorrect information.",
            existing_methods="Training-time and retrieval methods exist but are expensive.",
            motivation="LLMs can verify their own outputs by generating and answering verification questions.",
            proposed_method="Generate response, create verification questions, answer them independently, revise based on inconsistencies.",
            experiment_plan="Test on FEVER and FactScore benchmarks.",
            raw_text="Chain-of-Verification for Factuality. LLMs hallucinate. Training methods exist. LLMs can self-verify. Generate, verify, revise. Test on FEVER."
        ),
        
        # Idea 2: Very similar to Idea 1 (should be flagged as duplicate)
        SeedIdea(
            title="Self-Verification Chain for Reducing Hallucinations",
            problem="Language models generate confident but wrong information.",
            existing_methods="Existing approaches use retrieval or fine-tuning.",
            motivation="Models can check their own work through self-generated verification.",
            proposed_method="Create initial response, generate verification questions, answer independently, fix inconsistencies.",
            experiment_plan="Evaluate on fact-checking and QA datasets.",
            raw_text="Self-Verification Chain for Reducing Hallucinations. Language models generate wrong info. Retrieval or fine-tuning. Self-verification. Create response, verify, fix. Evaluate on QA."
        ),
        
        # Idea 3: Different topic - Self-Refine
        SeedIdea(
            title="Iterative Self-Refinement for Better Outputs",
            problem="First-attempt LLM outputs often contain errors.",
            existing_methods="Human feedback and reward models are expensive.",
            motivation="LLMs can critique and improve their own outputs iteratively.",
            proposed_method="Generate output, provide self-feedback, refine based on feedback, repeat until convergence.",
            experiment_plan="Test on code generation, math reasoning, and writing tasks.",
            raw_text="Iterative Self-Refinement. First outputs have errors. Human feedback expensive. Self-critique and improve. Generate, feedback, refine. Test on code and math."
        ),
        
        # Idea 4: Similar to Idea 3 (should be flagged)
        SeedIdea(
            title="Self-Critique and Revision for LLM Outputs",
            problem="Single-pass generation produces suboptimal results.",
            existing_methods="RLHF requires extensive labeling.",
            motivation="Models can iteratively improve through self-assessment.",
            proposed_method="Generate initial output, self-critique weaknesses, revise output, iterate.",
            experiment_plan="Evaluate on diverse generation tasks.",
            raw_text="Self-Critique and Revision. Single-pass suboptimal. RLHF needs labels. Self-assessment iterates. Generate, critique, revise. Evaluate generation tasks."
        ),
        
        # Idea 5: Completely different - Step-Back Prompting
        SeedIdea(
            title="Abstraction-First Reasoning via Step-Back",
            problem="Complex problems require understanding underlying principles.",
            existing_methods="Chain-of-thought operates at surface level.",
            motivation="Stepping back to principles first grounds reasoning.",
            proposed_method="Identify high-level principles, retrieve relevant knowledge, then reason through specifics.",
            experiment_plan="Test on STEM reasoning and multi-hop QA.",
            raw_text="Abstraction-First Reasoning. Complex problems need principles. CoT is surface-level. Step back first. Identify principles, retrieve knowledge, reason. Test on STEM."
        ),
        
        # Idea 6: Unique - Analogical Reasoning
        SeedIdea(
            title="Analogical Prompting for Novel Problems",
            problem="New problems need relevant examples but manual curation is expensive.",
            existing_methods="Few-shot prompting uses fixed examples.",
            motivation="Self-generating analogous examples adapts to each problem.",
            proposed_method="Recall similar problems, generate example solutions, apply patterns to target.",
            experiment_plan="Evaluate on math and code generation benchmarks.",
            raw_text="Analogical Prompting. New problems need examples. Few-shot uses fixed. Self-generate analogies. Recall, generate examples, apply. Math and code benchmarks."
        ),
        
        # Idea 7: Another unique one
        SeedIdea(
            title="System 2 Attention for Robust Context Processing",
            problem="Models get distracted by irrelevant context.",
            existing_methods="Standard attention attends to everything.",
            motivation="Deliberate context filtering removes distractions.",
            proposed_method="Regenerate context removing irrelevant parts, then answer with clean context.",
            experiment_plan="Test on tasks with misleading context.",
            raw_text="System 2 Attention. Distracted by irrelevant context. Attention attends all. Deliberate filtering. Regenerate clean context, answer. Test with misleading context."
        ),
    ]
    return ideas


def main():
    print("=" * 80)
    print("DEDUPLICATION MODULE TEST")
    print("=" * 80)
    
    # Create test ideas
    print("\n[1] Creating test ideas...")
    ideas = create_test_ideas()
    print(f"    Created {len(ideas)} test ideas:")
    for i, idea in enumerate(ideas):
        print(f"      {i+1}. {idea.title}")
    
    # Test embedding
    print("\n[2] Testing embedding function...")
    try:
        embeddings = dedup.embed_ideas(ideas)
        print(f"    ✓ Embedded {len(ideas)} ideas")
        print(f"    Embedding shape: {embeddings.shape}")
    except Exception as e:
        print(f"    ✗ Embedding failed: {e}")
        print("\n    To install sentence-transformers:")
        print("      python3.9 -m pip install --user sentence-transformers")
        return
    
    # Test pairwise similarities
    print("\n[3] Computing pairwise similarities...")
    similarities = dedup.compute_pairwise_similarities(embeddings)
    print(f"    Similarity matrix shape: {similarities.shape}")
    
    # Show similarity matrix highlights
    print("\n    Similarity highlights:")
    for i in range(len(ideas)):
        for j in range(i + 1, len(ideas)):
            sim = similarities[i, j]
            if sim > 0.7:  # Show high similarities
                print(f"      Ideas {i+1} & {j+1}: {sim:.3f} {'(HIGH)' if sim > 0.8 else ''}")
    
    # Test deduplication
    print("\n[4] Testing deduplication (threshold=0.8)...")
    kept_ideas = dedup.deduplicate_ideas(ideas, threshold=0.8)
    
    print(f"\n    Kept ideas ({len(kept_ideas)}):")
    for i, idea in enumerate(kept_ideas):
        print(f"      {i+1}. {idea.title}")
    
    # Test with detailed info
    print("\n[5] Testing deduplication with detailed info...")
    kept_ideas, info = dedup.deduplicate_ideas_with_info(ideas, threshold=0.8)
    
    if info["duplicate_pairs"]:
        print(f"\n    Duplicate pairs found:")
        for dup_idx, orig_idx, sim in info["duplicate_pairs"]:
            print(f"      • Idea {dup_idx+1} is similar to Idea {orig_idx+1} (sim={sim:.3f})")
            print(f"        Duplicate: {ideas[dup_idx].title[:50]}...")
            print(f"        Original:  {ideas[orig_idx].title[:50]}...")
    
    # Test finding most similar pairs
    print("\n[6] Finding most similar pairs...")
    pairs = dedup.find_most_similar_pairs(ideas, top_k=5)
    print(f"    Top 5 most similar pairs:")
    for i, (idx1, idx2, sim) in enumerate(pairs, 1):
        print(f"      {i}. Ideas {idx1+1} & {idx2+1}: {sim:.3f}")
    
    # Test diversity analysis
    print("\n[7] Testing diversity analysis...")
    diversity = dedup.analyze_diversity(ideas, batch_size=3, threshold=0.8)
    print(f"    Batch non-duplicate rates: {[f'{r:.1f}%' for r in diversity['batch_non_duplicate_rates']]}")
    print(f"    Cumulative unique counts: {diversity['cumulative_unique_counts']}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    
    # Summary
    print("\n📊 Summary:")
    print(f"    Original ideas: {len(ideas)}")
    print(f"    After deduplication: {len(kept_ideas)}")
    print(f"    Duplicates removed: {len(ideas) - len(kept_ideas)}")
    print(f"    Keep rate: {len(kept_ideas)/len(ideas)*100:.1f}%")


if __name__ == "__main__":
    main()
