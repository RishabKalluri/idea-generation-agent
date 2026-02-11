"""
Paper Retrieval Module using Tavily + arXiv.

This module retrieves relevant papers by:
1. Using LLM to generate diverse search queries
2. Searching arXiv via Tavily API
3. Fetching full metadata via arxiv Python package
4. Scoring papers for relevance
5. Generating focused summaries for highly relevant papers
"""

import os
import re
import time
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# ============================================================================
# COMMON INTERFACE
# ============================================================================

@dataclass
class RetrievedPaper:
    """Common format for retrieved papers across all retrieval methods."""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    summary: str  # Either abstract or generated summary
    relevance_score: float  # 0-1 normalized
    year: int = 0
    url: str = ""
    

# ============================================================================
# PROMPTS
# ============================================================================

QUERY_GENERATION_PROMPT = """Generate diverse search queries to find academic papers on arXiv about this research topic.

Research Topic: {topic}

Generate 8-12 different search queries that cover:
1. Direct problem statements (e.g., "reducing LLM hallucination", "factual accuracy in language models")
2. Techniques and methods (e.g., "chain of thought prompting", "self-consistency decoding")
3. Benchmarks and evaluation (e.g., "TruthfulQA evaluation", "factuality benchmarks")
4. Baseline approaches (e.g., "retrieval augmented generation", "knowledge grounding")
5. Related theoretical concepts

Return ONLY the search queries, one per line. No numbering, bullets, or explanations."""

PAPER_SCORING_PROMPT = """Score this paper's relevance to a research topic.

Research Topic: {topic}

Paper Title: {title}
Paper Abstract: {abstract}

Score from 1-10 based on:
1. Direct relevance to the research topic
2. Whether it's an empirical/experimental paper (not just a survey or position paper)
3. Potential to inspire new research ideas in this area

Respond with ONLY a single integer from 1-10."""

SUMMARY_GENERATION_PROMPT = """Summarize this academic paper with focus on aspects relevant to the research topic.

Research Topic: {topic}

Paper Title: {title}

Paper Content:
{content}

Provide a focused summary (200-300 words) covering:
1. The main problem addressed
2. Key methodology and techniques
3. Main results and findings
4. Relevance to the research topic

Summary:"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _call_llm(client, model_name: str, prompt: str, max_tokens: int = 1024) -> str:
    """Call the LLM and return response text. Injects human feedback if available."""
    messages = []
    feedback = _get_human_feedback()
    if feedback:
        messages.append({"role": "system", "content": feedback})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model_name,
        max_completion_tokens=max_tokens,
        messages=messages
    )
    content = response.choices[0].message.content
    return content if content is not None else ""


def _get_human_feedback() -> str:
    """Load formatted human feedback if available."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fb_path = os.path.join(current_dir, "..", "utils", "feedback.py")
        import importlib.util
        spec = importlib.util.spec_from_file_location("feedback", fb_path)
        fb_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fb_module)
        return fb_module.get_formatted_feedback()
    except Exception:
        return ""


def _extract_arxiv_id(url: str) -> Optional[str]:
    """Extract arXiv ID from URL.
    
    Examples:
        https://arxiv.org/abs/2301.12345 -> 2301.12345
        https://arxiv.org/pdf/2301.12345.pdf -> 2301.12345
    """
    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+)',
        r'arxiv\.org/pdf/(\d+\.\d+)',
        r'(\d{4}\.\d{4,5})',  # Just the ID pattern
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _extract_score(response: str) -> int:
    """Extract integer score from LLM response."""
    response = response.strip()
    
    # Try exact match first
    if response.isdigit():
        score = int(response)
        if 1 <= score <= 10:
            return score
    
    # Look for patterns
    patterns = [
        r'(\d+)\s*/\s*10',
        r'[Ss]core[:\s]+(\d+)',
        r'^(\d+)\b',
        r'\b([1-9]|10)\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
    
    return 0


# ============================================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================================

def retrieve_papers(
    topic: str,
    client,
    model_name: str,
    max_papers: int = 20,
    min_relevance_score: int = 5,
    summarize_threshold: int = 8
) -> List[RetrievedPaper]:
    """
    Retrieve relevant papers using Tavily search + arXiv metadata.
    
    Args:
        topic: Research topic description
        client: OpenAI client for LLM calls
        model_name: Model name for LLM calls
        max_papers: Maximum number of papers to return
        min_relevance_score: Minimum score (1-10) to keep papers
        summarize_threshold: Score threshold for PDF summarization (papers >= this get full summaries)
    
    Returns:
        List of RetrievedPaper objects sorted by relevance
    """
    try:
        from tavily import TavilyClient
        import arxiv
    except ImportError as e:
        raise ImportError(
            f"Required packages not installed: {e}\n"
            "Install with: pip install tavily-python arxiv pymupdf4llm"
        )
    
    # Load API key
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not set in environment")
    
    tavily_client = TavilyClient(api_key=tavily_api_key)
    arxiv_client = arxiv.Client()
    
    print(f"\n[Tavily+arXiv Retrieval] Starting retrieval for topic:")
    print(f"  {topic}")
    
    # -------------------------------------------------------------------------
    # Step 1: Generate search queries
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Generating search queries...")
    
    prompt = QUERY_GENERATION_PROMPT.format(topic=topic)
    response = _call_llm(client, model_name, prompt, max_tokens=512)
    
    queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
    queries = [re.sub(r'^[\d\.\-\*\)]+\s*', '', q) for q in queries]  # Remove numbering
    
    print(f"  Generated {len(queries)} queries:")
    for q in queries[:5]:
        print(f"    - {q[:60]}...")
    if len(queries) > 5:
        print(f"    ... and {len(queries) - 5} more")
    
    # -------------------------------------------------------------------------
    # Step 2: Search via Tavily (restricted to arxiv.org)
    # -------------------------------------------------------------------------
    print(f"\n[2/5] Searching arXiv via Tavily...")
    
    all_results = {}  # arxiv_id -> search result
    
    for i, query in enumerate(queries):
        try:
            # Search with Tavily, restricted to arxiv.org
            search_results = tavily_client.search(
                query=query,
                search_depth="advanced",
                include_domains=["arxiv.org"],
                max_results=10
            )
            
            results = search_results.get("results", [])
            new_count = 0
            
            for result in results:
                url = result.get("url", "")
                arxiv_id = _extract_arxiv_id(url)
                
                if arxiv_id and arxiv_id not in all_results:
                    all_results[arxiv_id] = {
                        "arxiv_id": arxiv_id,
                        "url": url,
                        "title": result.get("title", ""),
                        "snippet": result.get("content", "")
                    }
                    new_count += 1
            
            print(f"  [{i+1}/{len(queries)}] '{query[:35]}...' -> {len(results)} results, {new_count} new")
            
            # Small delay to be nice to the API
            time.sleep(0.2)
            
        except Exception as e:
            print(f"  [{i+1}/{len(queries)}] Error: {e}")
    
    print(f"  Found {len(all_results)} unique arXiv papers")
    
    if not all_results:
        print("  Warning: No papers found!")
        return []
    
    # -------------------------------------------------------------------------
    # Step 3: Fetch full metadata from arXiv
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Fetching metadata from arXiv...")
    
    arxiv_ids = list(all_results.keys())
    papers_with_metadata = []
    
    # Batch fetch metadata
    try:
        search = arxiv.Search(id_list=arxiv_ids)
        results = list(arxiv_client.results(search))
        
        for result in results:
            arxiv_id = result.entry_id.split("/")[-1]
            # Remove version suffix if present (e.g., 2301.12345v1 -> 2301.12345)
            arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
            
            papers_with_metadata.append({
                "arxiv_id": arxiv_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "published": result.published,
                "pdf_url": result.pdf_url,
                "url": result.entry_id
            })
        
        print(f"  Retrieved metadata for {len(papers_with_metadata)} papers")
        
    except Exception as e:
        print(f"  Error fetching metadata: {e}")
        # Fall back to using Tavily snippets
        for arxiv_id, info in all_results.items():
            papers_with_metadata.append({
                "arxiv_id": arxiv_id,
                "title": info["title"],
                "authors": [],
                "abstract": info["snippet"],
                "published": None,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "url": info["url"]
            })
    
    # -------------------------------------------------------------------------
    # Step 4: Score papers for relevance
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Scoring papers for relevance...")
    
    scored_papers = []
    
    for i, paper in enumerate(papers_with_metadata):
        if (i + 1) % 10 == 0:
            print(f"  Scored {i+1}/{len(papers_with_metadata)} papers...")
        
        abstract = paper.get("abstract", "")
        if not abstract or len(abstract.strip()) < 50:
            continue
        
        prompt = PAPER_SCORING_PROMPT.format(
            topic=topic,
            title=paper["title"],
            abstract=abstract[:1500]
        )
        
        try:
            response = _call_llm(client, model_name, prompt, max_tokens=64)
            score = _extract_score(response)
            
            if score >= min_relevance_score:
                scored_papers.append((paper, score))
                
            if len(scored_papers) <= 3:
                print(f"    Sample: '{paper['title'][:40]}...' -> {score}")
                
        except Exception as e:
            print(f"    Error scoring: {e}")
    
    print(f"  Kept {len(scored_papers)} papers with score >= {min_relevance_score}")
    
    # Sort by score and limit
    scored_papers.sort(key=lambda x: x[1], reverse=True)
    scored_papers = scored_papers[:max_papers]
    
    # -------------------------------------------------------------------------
    # Step 5: Generate summaries for highly relevant papers
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Generating summaries for top papers...")
    
    retrieved_papers = []
    
    for paper, score in scored_papers:
        # Determine summary
        if score >= summarize_threshold:
            # Try to get full PDF summary
            summary = _get_pdf_summary(
                paper["pdf_url"],
                paper["title"],
                topic,
                client,
                model_name
            )
            if not summary:
                summary = paper.get("abstract", "")[:500]
        else:
            # Use abstract as summary
            summary = paper.get("abstract", "")[:500]
        
        # Extract year from published date
        year = 0
        if paper.get("published"):
            year = paper["published"].year
        
        retrieved_paper = RetrievedPaper(
            paper_id=paper["arxiv_id"],
            title=paper["title"],
            authors=paper.get("authors", []),
            abstract=paper.get("abstract", ""),
            summary=summary,
            relevance_score=score / 10.0,  # Normalize to 0-1
            year=year,
            url=paper.get("url", "")
        )
        retrieved_papers.append(retrieved_paper)
    
    print(f"\n✓ Retrieved {len(retrieved_papers)} relevant papers")
    
    return retrieved_papers


def _get_pdf_summary(
    pdf_url: str,
    title: str,
    topic: str,
    client,
    model_name: str
) -> Optional[str]:
    """
    Download PDF and generate a focused summary.
    
    Returns None if PDF processing fails.
    """
    try:
        import pymupdf4llm
        import requests
    except ImportError:
        return None
    
    try:
        # Download PDF to temp file
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        try:
            # Convert to markdown
            md_content = pymupdf4llm.to_markdown(tmp_path)
            
            # Limit content length for LLM
            if len(md_content) > 15000:
                # Take intro and methods sections (first ~40%)
                md_content = md_content[:15000] + "\n\n[Content truncated...]"
            
            # Generate summary
            prompt = SUMMARY_GENERATION_PROMPT.format(
                topic=topic,
                title=title,
                content=md_content
            )
            
            summary = _call_llm(client, model_name, prompt, max_tokens=512)
            return summary.strip()
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"    Warning: Could not summarize PDF for '{title[:40]}...': {e}")
        return None


# ============================================================================
# ADAPTER FOR EXISTING PIPELINE
# ============================================================================

def retrieve_papers_as_legacy(
    topic: str,
    client,
    model_name: str,
    target_papers: int = 20
):
    """
    Adapter that returns papers in the legacy Paper format for compatibility
    with the existing pipeline.
    
    This allows using Tavily retrieval as a drop-in replacement.
    """
    # Import the legacy Paper class
    import importlib.util
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ss_path = os.path.join(current_dir, "..", "utils", "semantic_scholar.py")
    
    spec = importlib.util.spec_from_file_location("semantic_scholar", ss_path)
    ss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss_module)
    Paper = ss_module.Paper
    
    # Get papers using new interface
    retrieved = retrieve_papers(
        topic=topic,
        client=client,
        model_name=model_name,
        max_papers=target_papers
    )
    
    # Convert to legacy format
    legacy_papers = []
    for rp in retrieved:
        legacy_papers.append(Paper(
            paper_id=rp.paper_id,
            title=rp.title,
            abstract=rp.summary,  # Use summary instead of abstract for richer context
            year=rp.year,
            citation_count=0,  # Not available from arXiv
            authors=rp.authors
        ))
    
    return legacy_papers
