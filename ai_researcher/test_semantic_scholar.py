"""
Example script demonstrating the Semantic Scholar API wrapper usage.

This script shows how to use the three main functions:
1. KeywordQuery - Search papers by keywords
2. PaperQuery - Get detailed paper information
3. GetReferences - Get references of a paper
"""

from utils.semantic_scholar import SemanticScholarClient


def main():
    """Demonstrate Semantic Scholar API wrapper functionality."""
    
    # Initialize the client
    client = SemanticScholarClient()
    
    print("=" * 80)
    print("Semantic Scholar API Wrapper Demo")
    print("=" * 80)
    
    # Example 1: Keyword Search
    print("\n[1] KeywordQuery: Searching for 'transformers attention mechanism'")
    print("-" * 80)
    papers = client.KeywordQuery("transformers attention mechanism", limit=5)
    
    if papers:
        print(f"Found {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper.title}")
            print(f"   Year: {paper.year}, Citations: {paper.citation_count}")
            print(f"   Authors: {', '.join(paper.authors[:3])}")
            if len(paper.authors) > 3:
                print(f"            ... and {len(paper.authors) - 3} more")
            print(f"   Paper ID: {paper.paper_id}")
            if paper.abstract:
                abstract_preview = paper.abstract[:150] + "..." if len(paper.abstract) > 150 else paper.abstract
                print(f"   Abstract: {abstract_preview}")
            else:
                print(f"   Abstract: [Not available]")
        
        # Save first paper ID for next examples
        first_paper_id = papers[0].paper_id
        
        # Example 2: Paper Query
        print("\n\n[2] PaperQuery: Getting detailed info for first paper")
        print("-" * 80)
        paper = client.PaperQuery(first_paper_id)
        if paper:
            print(f"Title: {paper.title}")
            print(f"Year: {paper.year}")
            print(f"Citations: {paper.citation_count}")
            print(f"Authors: {', '.join(paper.authors)}")
            if paper.abstract:
                print(f"Abstract: {paper.abstract[:200]}...")
        
        # Example 3: Get References
        print("\n\n[3] GetReferences: Getting references for first paper")
        print("-" * 80)
        references = client.GetReferences(first_paper_id, limit=5)
        if references:
            print(f"Found {len(references)} references:")
            for i, ref in enumerate(references, 1):
                print(f"\n{i}. {ref.title}")
                print(f"   Year: {ref.year}, Citations: {ref.citation_count}")
                print(f"   Authors: {', '.join(ref.authors[:2])}")
                if len(ref.authors) > 2:
                    print(f"            ... and {len(ref.authors) - 2} more")
        else:
            print("No references found.")
    else:
        print("No papers found.")
    
    # Show cache statistics
    print("\n\n[Cache Statistics]")
    print("-" * 80)
    print(f"Cached requests: {client.get_cache_size()}")
    print(f"Total API calls made: {len(client.request_times)}")
    
    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
