"""
Complete usage example showing how to use the paper retrieval module.

This is a working example that demonstrates the full workflow.
"""

import os
from anthropic import Anthropic

# Example 1: Basic Paper Retrieval
def example_basic_retrieval():
    """Most basic usage - just retrieve papers on a topic."""
    from modules.paper_retrieval import retrieve_papers
    from config import ANTHROPIC_API_KEY, MODEL_NAME
    
    # Initialize client
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Define topic
    topic = "chain of thought prompting for improved reasoning in language models"
    
    # Retrieve papers (LLM will intelligently search)
    papers = retrieve_papers(
        topic=topic,
        client=client,
        model_name=MODEL_NAME
    )
    
    print(f"Found {len(papers)} relevant papers")
    return papers


# Example 2: Custom Parameters
def example_custom_parameters():
    """Retrieve more papers with stricter filtering."""
    from modules.paper_retrieval import retrieve_papers
    from config import ANTHROPIC_API_KEY, MODEL_NAME
    from anthropic import Anthropic
    
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    topic = "multi-modal learning combining vision and language"
    
    # Get 200 papers, keep only those with score >= 8
    papers = retrieve_papers(
        topic=topic,
        client=client,
        model_name=MODEL_NAME,
        target_papers=200,    # Retrieve more papers
        min_score=8           # Higher quality threshold
    )
    
    return papers


# Example 3: Building RAG Context
def example_rag_context():
    """Retrieve papers and build context for LLM prompting."""
    from modules.paper_retrieval import retrieve_papers, build_rag_context
    from config import ANTHROPIC_API_KEY, MODEL_NAME
    from anthropic import Anthropic
    
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    topic = "efficient fine-tuning methods for large language models"
    
    # Retrieve papers
    papers = retrieve_papers(topic, client, MODEL_NAME)
    
    # Build RAG context with top 10 papers
    context = build_rag_context(papers, max_papers=10)
    
    # Now use this context in a prompt
    prompt = f"""
{context}

Based on these papers about efficient fine-tuning methods, 
propose a novel research idea that combines or extends these approaches.
"""
    
    # Generate idea using the context
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    
    idea = response.content[0].text
    print("Generated idea:")
    print(idea)
    
    return papers, idea


# Example 4: Multiple Topics with Deduplication
def example_multiple_topics():
    """Retrieve papers from multiple related topics."""
    from modules.paper_retrieval import retrieve_papers, deduplicate_papers
    from config import ANTHROPIC_API_KEY, MODEL_NAME
    from anthropic import Anthropic
    
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Related topics
    topics = [
        "retrieval augmented generation for factuality",
        "knowledge grounding in language models",
        "external memory for neural networks"
    ]
    
    all_papers = []
    
    # Retrieve papers for each topic
    for topic in topics:
        print(f"\nRetrieving papers for: {topic}")
        papers = retrieve_papers(
            topic=topic,
            client=client,
            model_name=MODEL_NAME,
            target_papers=50,  # Fewer per topic
            min_score=7
        )
        all_papers.extend(papers)
    
    # Remove duplicates
    unique_papers = deduplicate_papers(all_papers)
    
    print(f"\nTotal papers: {len(all_papers)}")
    print(f"Unique papers: {len(unique_papers)}")
    
    return unique_papers


# Example 5: Complete Pipeline Integration
def example_full_pipeline():
    """
    Complete example showing how paper retrieval fits into
    the full idea generation pipeline.
    """
    from modules.paper_retrieval import retrieve_papers, build_rag_context
    from config import ANTHROPIC_API_KEY, MODEL_NAME
    from anthropic import Anthropic
    
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Step 1: Start with a seed idea
    seed_idea = "Methods to reduce hallucination in LLM code generation"
    
    print("=" * 80)
    print("FULL PIPELINE EXAMPLE")
    print("=" * 80)
    print(f"\nSeed Idea: {seed_idea}")
    
    # Step 2: Retrieve relevant papers
    print("\n[1/3] Retrieving relevant papers...")
    papers = retrieve_papers(
        topic=seed_idea,
        client=client,
        model_name=MODEL_NAME,
        target_papers=120,
        min_score=7
    )
    print(f"✓ Retrieved {len(papers)} relevant papers")
    
    # Step 3: Build RAG context
    print("\n[2/3] Building RAG context...")
    context = build_rag_context(papers, max_papers=10)
    print(f"✓ Built context from top 10 papers")
    
    # Step 4: Generate full proposal with RAG
    print("\n[3/3] Generating full research proposal...")
    
    proposal_prompt = f"""
You are a research scientist. Based on the seed idea and relevant literature,
generate a detailed research proposal.

Seed Idea: {seed_idea}

{context}

Generate a detailed research proposal that includes:
1. Background and motivation
2. Proposed method
3. Expected contributions
4. Experimental plan

Make it novel by combining insights from the papers in a new way.
"""
    
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4096,
        messages=[{"role": "user", "content": proposal_prompt}]
    )
    
    proposal = response.content[0].text
    
    print("✓ Generated full proposal")
    print("\n" + "=" * 80)
    print("GENERATED PROPOSAL")
    print("=" * 80)
    print(proposal)
    
    return proposal


# Example 6: Analyzing Retrieved Papers
def example_paper_analysis():
    """Retrieve papers and analyze what was found."""
    from modules.paper_retrieval import retrieve_papers
    from config import ANTHROPIC_API_KEY, MODEL_NAME
    from anthropic import Anthropic
    from collections import Counter
    
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    topic = "transformer architectures for computer vision"
    
    # Retrieve papers
    papers = retrieve_papers(topic, client, MODEL_NAME)
    
    # Analysis
    print("\n" + "=" * 80)
    print("PAPER ANALYSIS")
    print("=" * 80)
    
    # Year distribution
    print("\n[Year Distribution]")
    year_counts = Counter(p.year for p in papers if p.year > 0)
    for year in sorted(year_counts.keys(), reverse=True)[:5]:
        count = year_counts[year]
        bar = "█" * (count * 2)
        print(f"{year}: {bar} ({count})")
    
    # Citation statistics
    print("\n[Citation Statistics]")
    citations = [p.citation_count for p in papers]
    print(f"Average: {sum(citations)/len(citations):.1f}")
    print(f"Median: {sorted(citations)[len(citations)//2]}")
    print(f"Max: {max(citations)}")
    print(f"Highly cited (>100): {sum(1 for c in citations if c > 100)}")
    
    # Top authors
    print("\n[Frequent Authors]")
    all_authors = []
    for paper in papers:
        all_authors.extend(paper.authors)
    author_counts = Counter(all_authors).most_common(10)
    for author, count in author_counts:
        if count > 1:  # Only show authors with multiple papers
            print(f"  {author}: {count} papers")
    
    return papers


# Main function to run examples
def main():
    """Run one of the examples."""
    import sys
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key'")
        return
    
    print("Available examples:")
    print("1. Basic retrieval")
    print("2. Custom parameters")
    print("3. RAG context building")
    print("4. Multiple topics")
    print("5. Full pipeline")
    print("6. Paper analysis")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nSelect example (1-6): ")
    
    examples = {
        '1': example_basic_retrieval,
        '2': example_custom_parameters,
        '3': example_rag_context,
        '4': example_multiple_topics,
        '5': example_full_pipeline,
        '6': example_paper_analysis
    }
    
    if choice in examples:
        print(f"\nRunning example {choice}...")
        examples[choice]()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
