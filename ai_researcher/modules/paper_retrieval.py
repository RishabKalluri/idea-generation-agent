"""
Paper Retrieval Module using Semantic Scholar RAG.

This module handles retrieving relevant papers from Semantic Scholar
to provide context for idea generation.
"""

from typing import List
from utils import SemanticScholarClient, Paper


def retrieve_papers(query: str, num_papers: int = 20) -> List[Paper]:
    """
    Retrieve relevant papers from Semantic Scholar based on query.
    
    Args:
        query: Search query string
        num_papers: Number of papers to retrieve
        
    Returns:
        List of Paper objects
    """
    client = SemanticScholarClient()
    papers = client.KeywordQuery(query, limit=num_papers)
    return papers


def build_rag_context(papers: List[Paper], max_papers: int = 10) -> str:
    """
    Build RAG context from retrieved papers for LLM prompting.
    
    Args:
        papers: List of Paper objects
        max_papers: Maximum number of papers to include in context
        
    Returns:
        Formatted context string for LLM
    """
    if not papers:
        return ""
    
    # Limit to max_papers
    selected_papers = papers[:max_papers]
    
    context_parts = ["Retrieved relevant research papers:\n"]
    
    for i, paper in enumerate(selected_papers, 1):
        context_parts.append(f"\n[Paper {i}]")
        context_parts.append(f"Title: {paper.title}")
        context_parts.append(f"Year: {paper.year}")
        context_parts.append(f"Citations: {paper.citation_count}")
        
        if paper.authors:
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += f" et al."
            context_parts.append(f"Authors: {authors_str}")
        
        if paper.abstract:
            # Truncate very long abstracts
            abstract = paper.abstract[:500] + "..." if len(paper.abstract) > 500 else paper.abstract
            context_parts.append(f"Abstract: {abstract}")
        
        context_parts.append("")  # Empty line between papers
    
    return "\n".join(context_parts)


def retrieve_diverse_papers(queries: List[str], papers_per_query: int = 20) -> List[Paper]:
    """
    Retrieve papers from multiple queries to get diverse coverage.
    
    Args:
        queries: List of search query strings
        papers_per_query: Number of papers to retrieve per query
        
    Returns:
        Combined list of Paper objects (may contain duplicates)
    """
    client = SemanticScholarClient()
    all_papers = []
    
    for query in queries:
        papers = client.KeywordQuery(query, limit=papers_per_query)
        all_papers.extend(papers)
    
    return all_papers


def deduplicate_papers(papers: List[Paper]) -> List[Paper]:
    """
    Remove duplicate papers based on paper_id.
    
    Args:
        papers: List of Paper objects
        
    Returns:
        Deduplicated list of Paper objects
    """
    seen_ids = set()
    unique_papers = []
    
    for paper in papers:
        if paper.paper_id not in seen_ids:
            seen_ids.add(paper.paper_id)
            unique_papers.append(paper)
    
    return unique_papers
