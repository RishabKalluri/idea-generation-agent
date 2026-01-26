"""
Full Pipeline: Paper Retrieval → Seed Idea Generation

This script runs the complete AI research idea generation pipeline:
1. Retrieve relevant papers using LLM-guided search
2. Generate seed ideas using retrieved papers as RAG context
"""

import os
import sys
import time
import importlib.util
from datetime import datetime

# Check for API key first
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY not set")
    print("\nTo set your API key:")
    print("  export OPENAI_API_KEY='your-openai-api-key-here'")
    sys.exit(1)

from openai import OpenAI

# Load modules directly to avoid anthropic dependency
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load paper retrieval module
paper_retrieval = load_module('paper_retrieval', 'modules/paper_retrieval_openai.py')
retrieve_papers = paper_retrieval.retrieve_papers
build_rag_context = paper_retrieval.build_rag_context

# Load idea generation module
idea_generation = load_module('idea_generation', 'modules/idea_generation.py')
generate_seed_ideas = idea_generation.generate_seed_ideas
save_ideas_to_file = idea_generation.save_ideas_to_file


def run_pipeline(
    topic: str,
    num_papers: int = 50,
    num_ideas: int = 10,
    model_name: str = "gpt-4",
    rag_rate: float = 0.5,
    min_paper_score: int = 7
):
    """
    Run the full pipeline: Paper Retrieval → Idea Generation.
    
    Args:
        topic: Research topic
        num_papers: Target number of papers to retrieve
        num_ideas: Number of ideas to generate
        model_name: OpenAI model to use
        rag_rate: Fraction of ideas using RAG (0.0-1.0)
        min_paper_score: Minimum relevance score for papers
    """
    
    start_time = time.time()
    
    print("=" * 80)
    print("AI RESEARCH IDEA GENERATION PIPELINE")
    print("=" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Topic: {topic}")
    print(f"  Model: {model_name}")
    print(f"  Target papers: {num_papers}")
    print(f"  Target ideas: {num_ideas}")
    print(f"  RAG rate: {rag_rate*100:.0f}%")
    
    # Initialize client
    print("\n" + "=" * 80)
    print("STEP 0: INITIALIZATION")
    print("=" * 80)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✓ OpenAI client initialized")
    
    # Check Semantic Scholar API
    ss_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if ss_key:
        print("✓ Semantic Scholar API key found")
    else:
        print("⚠ No Semantic Scholar API key (may hit rate limits)")
    
    # =========================================================================
    # STEP 1: PAPER RETRIEVAL
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: PAPER RETRIEVAL")
    print("=" * 80)
    print(f"\nUsing LLM to search for relevant papers on:")
    print(f"  '{topic}'")
    print(f"\nThis will:")
    print(f"  • Generate smart search queries using {model_name}")
    print(f"  • Retrieve papers from Semantic Scholar")
    print(f"  • Score papers for relevance")
    print(f"  • Keep papers with score >= {min_paper_score}")
    print("-" * 80)
    
    retrieval_start = time.time()
    
    try:
        papers = retrieve_papers(
            topic=topic,
            client=client,
            model_name=model_name,
            target_papers=num_papers,
            min_score=min_paper_score
        )
        
        retrieval_time = time.time() - retrieval_start
        
        print(f"\n✓ Paper retrieval complete!")
        print(f"  Retrieved: {len(papers)} relevant papers")
        print(f"  Time: {retrieval_time/60:.1f} minutes")
        
        if papers:
            # Show top 5 papers
            print(f"\n  Top 5 papers:")
            for i, paper in enumerate(papers[:5], 1):
                title = paper.title[:60] + "..." if len(paper.title) > 60 else paper.title
                print(f"    {i}. {title} ({paper.year})")
        
    except Exception as e:
        print(f"\n⚠ Paper retrieval failed: {e}")
        print("  Continuing without RAG context...")
        papers = []
        retrieval_time = time.time() - retrieval_start
    
    # =========================================================================
    # STEP 2: SEED IDEA GENERATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: SEED IDEA GENERATION")
    print("=" * 80)
    print(f"\nGenerating {num_ideas} research ideas")
    print(f"  • Using {len(papers)} papers for RAG context")
    print(f"  • RAG applied to {rag_rate*100:.0f}% of ideas")
    print(f"  • 6 demo examples for few-shot learning")
    print("-" * 80)
    
    generation_start = time.time()
    
    try:
        ideas = generate_seed_ideas(
            topic=topic,
            papers=papers,
            client=client,
            model_name=model_name,
            num_ideas=num_ideas,
            rag_rate=rag_rate if papers else 0.0,
            num_demo_examples=6,
            papers_per_rag=10,
            show_progress=True
        )
        
        generation_time = time.time() - generation_start
        
        print(f"\n✓ Idea generation complete!")
        print(f"  Generated: {len(ideas)} ideas")
        print(f"  Time: {generation_time/60:.1f} minutes")
        
    except Exception as e:
        print(f"\n✗ Idea generation failed: {e}")
        import traceback
        traceback.print_exc()
        ideas = []
        generation_time = time.time() - generation_start
    
    # =========================================================================
    # STEP 3: SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: SAVE RESULTS")
    print("=" * 80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if ideas:
        # Save ideas
        ideas_file = f"pipeline_ideas_{timestamp}.txt"
        save_ideas_to_file(ideas, ideas_file)
        print(f"✓ Saved {len(ideas)} ideas to {ideas_file}")
        
        # Save summary
        summary_file = f"pipeline_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("AI RESEARCH IDEA GENERATION - PIPELINE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Topic: {topic}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Papers Retrieved: {len(papers)}\n")
            f.write(f"Ideas Generated: {len(ideas)}\n")
            f.write(f"RAG Rate: {rag_rate*100:.0f}%\n\n")
            f.write(f"Retrieval Time: {retrieval_time/60:.1f} minutes\n")
            f.write(f"Generation Time: {generation_time/60:.1f} minutes\n")
            f.write(f"Total Time: {(time.time()-start_time)/60:.1f} minutes\n\n")
            f.write("Generated Idea Titles:\n")
            f.write("-" * 40 + "\n")
            for i, idea in enumerate(ideas, 1):
                f.write(f"{i}. {idea.title}\n")
        print(f"✓ Saved summary to {summary_file}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    
    print(f"\n📊 Results:")
    print(f"  • Papers retrieved: {len(papers)}")
    print(f"  • Ideas generated: {len(ideas)}")
    
    print(f"\n⏱️ Timing:")
    print(f"  • Paper retrieval: {retrieval_time/60:.1f} minutes")
    print(f"  • Idea generation: {generation_time/60:.1f} minutes")
    print(f"  • Total time: {total_time/60:.1f} minutes")
    
    # Estimate cost
    # Paper retrieval: ~15 LLM calls + ~num_papers scoring calls
    # Idea generation: ~num_ideas LLM calls
    retrieval_calls = 15 + len(papers) if papers else 15
    generation_calls = num_ideas
    total_calls = retrieval_calls + generation_calls
    
    if model_name == "gpt-4":
        est_cost = total_calls * 0.01
    elif model_name == "gpt-4-turbo-preview":
        est_cost = total_calls * 0.005
    else:  # gpt-3.5-turbo
        est_cost = total_calls * 0.001
    
    print(f"\n💰 Estimated cost: ${est_cost:.2f}")
    print(f"  (~{total_calls} API calls × ${0.01 if 'gpt-4' in model_name else 0.001}/call)")
    
    if ideas:
        print(f"\n📁 Output files:")
        print(f"  • {ideas_file}")
        print(f"  • {summary_file}")
        
        print(f"\n🎯 Sample ideas generated:")
        for i, idea in enumerate(ideas[:3], 1):
            title = idea.title[:70] + "..." if len(idea.title) > 70 else idea.title
            print(f"  {i}. {title}")
        if len(ideas) > 3:
            print(f"  ... and {len(ideas) - 3} more")
    
    print("\n" + "=" * 80)
    
    return papers, ideas


def main():
    """Run the pipeline with default settings."""
    
    # Configuration - adjust these as needed
    TOPIC = ("novel prompting methods that can improve factuality and "
             "reduce hallucination of large language models")
    
    # For testing: small numbers
    # For production: num_papers=120, num_ideas=4000
    NUM_PAPERS = 30      # Papers to retrieve (production: 120)
    NUM_IDEAS = 10       # Ideas to generate (production: 4000)
    MODEL = "gpt-4"      # or "gpt-3.5-turbo" for cheaper
    RAG_RATE = 0.5       # 50% of ideas use paper context
    
    print("\n" + "🚀" * 40)
    print("\n  Starting AI Research Idea Generation Pipeline")
    print("\n" + "🚀" * 40 + "\n")
    
    papers, ideas = run_pipeline(
        topic=TOPIC,
        num_papers=NUM_PAPERS,
        num_ideas=NUM_IDEAS,
        model_name=MODEL,
        rag_rate=RAG_RATE
    )
    
    return papers, ideas


if __name__ == "__main__":
    main()
