"""
Test script for LLM-guided paper retrieval using OpenAI API.

This uses GPT-4 or GPT-3.5-turbo instead of Claude.
"""

import os
from openai import OpenAI
from modules.paper_retrieval_openai import retrieve_papers


def main():
    """Test the LLM-guided paper retrieval system with OpenAI."""
    
    print("=" * 80)
    print("LLM-Guided Paper Retrieval Test (OpenAI)")
    print("=" * 80)
    
    # Check for API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("\n✗ Error: OPENAI_API_KEY not set")
        print("  Set it with: export OPENAI_API_KEY='your-key'")
        return
    
    semantic_scholar_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if not semantic_scholar_key:
        print("\n⚠ Warning: SEMANTIC_SCHOLAR_API_KEY not set")
        print("  You may hit rate limits. Get a key at:")
        print("  https://www.semanticscholar.org/product/api")
        print()
    
    # Initialize OpenAI client
    print("\n[Initialization]")
    
    # Choose model
    # GPT-4 is better quality but more expensive
    # GPT-3.5-turbo is faster and cheaper but less reliable
    model_name = "gpt-4"  # or "gpt-3.5-turbo"
    
    print(f"Model: {model_name}")
    client = OpenAI(api_key=openai_api_key)
    print("✓ Client initialized")
    
    # Define research topic
    topic = ("novel prompting methods that can improve factuality and "
             "reduce hallucination of large language models")
    
    print(f"\n[Research Topic]")
    print(f"{topic}")
    
    # Retrieve papers
    print(f"\n[Starting Retrieval]")
    print(f"This will use GPT to generate search queries and retrieve papers...")
    print(f"Target: 120 papers (may take a few minutes)")
    print("-" * 80)
    
    try:
        papers = retrieve_papers(
            topic=topic,
            client=client,
            model_name=model_name,
            target_papers=120,  # Target number of papers
            min_score=7  # Minimum relevance score
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        if papers:
            print(f"\n✓ Retrieved {len(papers)} relevant papers (score >= 7)")
            
            # Show top 10
            print(f"\nTop 10 Papers:")
            print("-" * 80)
            
            for i, paper in enumerate(papers[:10], 1):
                print(f"\n{i}. {paper.title}")
                print(f"   Year: {paper.year}, Citations: {paper.citation_count}")
                
                if paper.authors:
                    authors_str = ", ".join(paper.authors[:3])
                    if len(paper.authors) > 3:
                        authors_str += f" et al. (+{len(paper.authors) - 3})"
                    print(f"   Authors: {authors_str}")
                
                if paper.abstract:
                    abstract_preview = paper.abstract[:150]
                    if len(paper.abstract) > 150:
                        abstract_preview += "..."
                    print(f"   Abstract: {abstract_preview}")
            
            if len(papers) > 10:
                print(f"\n... and {len(papers) - 10} more papers")
            
            # Statistics
            print("\n" + "=" * 80)
            print("STATISTICS")
            print("=" * 80)
            
            # Year distribution
            years = [p.year for p in papers if p.year > 0]
            if years:
                print(f"Year range: {min(years)} - {max(years)}")
                
                recent_count = sum(1 for y in years if y >= 2020)
                print(f"Recent papers (2020+): {recent_count} ({recent_count*100/len(years):.1f}%)")
            
            # Citation distribution
            citations = [p.citation_count for p in papers]
            if citations:
                avg_citations = sum(citations) / len(citations)
                print(f"Average citations: {avg_citations:.1f}")
                highly_cited = sum(1 for c in citations if c >= 100)
                print(f"Highly cited (100+): {highly_cited} ({highly_cited*100/len(citations):.1f}%)")
            
            # Abstract availability
            with_abstract = sum(1 for p in papers if p.abstract)
            print(f"Papers with abstracts: {with_abstract}/{len(papers)} ({with_abstract*100/len(papers):.1f}%)")
            
            # Cost estimation
            print("\n" + "=" * 80)
            print("COST ESTIMATION")
            print("=" * 80)
            num_calls = len(action_history) + len(papers) if 'action_history' in locals() else 150
            if model_name == "gpt-4":
                est_cost = num_calls * 0.01  # Rough estimate
                print(f"Estimated cost: ${est_cost:.2f}")
                print(f"(~{num_calls} API calls × ~$0.01 per call)")
            else:
                est_cost = num_calls * 0.002
                print(f"Estimated cost: ${est_cost:.2f}")
                print(f"(~{num_calls} API calls × ~$0.002 per call)")
            
        else:
            print("\n⚠ No papers found matching the criteria")
        
    except Exception as e:
        print(f"\n✗ Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
