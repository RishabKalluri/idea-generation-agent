"""
Standalone test for OpenAI paper retrieval - no anthropic dependencies.
"""

import os
import sys

# Check for OpenAI key first
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY not set")
    print("\nTo set your API key:")
    print("  export OPENAI_API_KEY='your-openai-api-key-here'")
    print("\nGet your key at: https://platform.openai.com/api-keys")
    sys.exit(1)

# Import directly to avoid anthropic dependency
import importlib.util

# Load Semantic Scholar module directly
ss_spec = importlib.util.spec_from_file_location(
    "semantic_scholar",
    "utils/semantic_scholar.py"
)
ss_module = importlib.util.module_from_spec(ss_spec)
sys.modules['semantic_scholar'] = ss_module
ss_spec.loader.exec_module(ss_module)

SemanticScholarClient = ss_module.SemanticScholarClient
Paper = ss_module.Paper

# Load paper retrieval OpenAI module
pr_spec = importlib.util.spec_from_file_location(
    "paper_retrieval_openai",
    "modules/paper_retrieval_openai.py"
)
pr_module = importlib.util.module_from_spec(pr_spec)

# Inject the semantic scholar classes into the module's namespace
pr_module.SemanticScholarClient = SemanticScholarClient
pr_module.Paper = Paper

sys.modules['paper_retrieval_openai'] = pr_module
pr_spec.loader.exec_module(pr_module)

retrieve_papers = pr_module.retrieve_papers

# Now import OpenAI
from openai import OpenAI


def main():
    """Test the OpenAI paper retrieval system."""
    
    print("=" * 80)
    print("OpenAI Paper Retrieval Test (Standalone)")
    print("=" * 80)
    
    # Check for Semantic Scholar key
    ss_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if ss_key:
        print("\n✓ SEMANTIC_SCHOLAR_API_KEY is set")
    else:
        print("\n⚠ SEMANTIC_SCHOLAR_API_KEY not set (using lower rate limits)")
        print("  Get a free key at: https://www.semanticscholar.org/product/api")
    
    # Initialize OpenAI client
    print("\n[Initialization]")
    model_name = "gpt-4"  # or "gpt-3.5-turbo" for cheaper
    print(f"Model: {model_name}")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✓ Client initialized")
    
    # Define research topic
    topic = ("novel prompting methods that can improve factuality and "
             "reduce hallucination of large language models")
    
    print(f"\n[Research Topic]")
    print(f"{topic}")
    
    # Retrieve papers
    print(f"\n[Starting Retrieval]")
    print(f"Using GPT to intelligently search for papers...")
    print(f"Target: 120 papers (this may take 2-5 minutes)")
    print("-" * 80)
    
    try:
        papers = retrieve_papers(
            topic=topic,
            client=client,
            model_name=model_name,
            target_papers=120,
            min_score=7
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
                    preview = paper.abstract[:150]
                    if len(paper.abstract) > 150:
                        preview += "..."
                    print(f"   Abstract: {preview}")
            
            if len(papers) > 10:
                print(f"\n... and {len(papers) - 10} more papers")
            
            # Statistics
            print("\n" + "=" * 80)
            print("STATISTICS")
            print("=" * 80)
            
            years = [p.year for p in papers if p.year > 0]
            if years:
                print(f"Year range: {min(years)} - {max(years)}")
                recent = sum(1 for y in years if y >= 2020)
                print(f"Recent papers (2020+): {recent} ({recent*100/len(years):.1f}%)")
            
            citations = [p.citation_count for p in papers]
            if citations:
                avg_cit = sum(citations) / len(citations)
                print(f"Average citations: {avg_cit:.1f}")
                highly_cited = sum(1 for c in citations if c >= 100)
                print(f"Highly cited (100+): {highly_cited} ({highly_cited*100/len(citations):.1f}%)")
            
            with_abstract = sum(1 for p in papers if p.abstract)
            print(f"Papers with abstracts: {with_abstract}/{len(papers)} ({with_abstract*100/len(papers):.1f}%)")
            
        else:
            print("\n⚠ No papers found")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✓ Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
