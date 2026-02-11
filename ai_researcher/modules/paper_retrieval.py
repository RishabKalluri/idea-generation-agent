"""
Paper Retrieval Module using Semantic Scholar RAG.

This module handles retrieving relevant papers from Semantic Scholar
to provide context for idea generation. Supports multiple retrieval
strategies that can be configured in settings.py.

Available strategies:
- llm_guided: LLM agent iteratively searches and follows references
- keyword: Simple keyword-based search
- custom: User-defined retrieval functions
"""

import os
import re
import importlib.util
from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod

# Load SemanticScholar directly to avoid anthropic dependency
def _load_semantic_scholar():
    """Load SemanticScholar module without triggering anthropic import."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ss_path = os.path.join(current_dir, "..", "utils", "semantic_scholar.py")
    
    spec = importlib.util.spec_from_file_location("semantic_scholar_direct", ss_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SemanticScholarClient, module.Paper

SemanticScholarClient, Paper = _load_semantic_scholar()


# ============================================================================
# RETRIEVAL STRATEGY REGISTRY
# ============================================================================

# Registry for custom retrieval methods
_custom_retrieval_methods: Dict[str, Callable] = {}


def register_retrieval_method(name: str, func: Callable):
    """
    Register a custom paper retrieval method.
    
    Args:
        name: Name to identify this retrieval method
        func: Callable with signature:
              func(topic: str, client, model_name: str, target_papers: int) -> List[Paper]
    
    Example:
        def my_retrieval(topic, client, model_name, target_papers):
            # Your custom logic here
            return [Paper(...), ...]
        
        register_retrieval_method("my_method", my_retrieval)
    """
    _custom_retrieval_methods[name] = func
    print(f"[Paper Retrieval] Registered custom method: {name}")


def get_available_methods() -> List[str]:
    """Get list of all available retrieval methods."""
    built_in = ["llm_guided", "keyword", "tavily"]
    custom = list(_custom_retrieval_methods.keys())
    return built_in + custom


def retrieve_papers(
    topic: str, 
    client, 
    model_name: str, 
    target_papers: int = 120,
    method: str = None
) -> List[Paper]:
    """
    Main entry point for paper retrieval. Dispatches to the configured method.
    
    Args:
        topic: Research topic description
        client: OpenAI client for LLM calls
        model_name: Model to use for LLM calls
        target_papers: Target number of papers to retrieve
        method: Override retrieval method (uses settings.RETRIEVAL_METHOD if None)
    
    Returns:
        List of relevant Paper objects
    """
    # Load settings to get configured method
    current_dir = os.path.dirname(os.path.abspath(__file__))
    settings_path = os.path.join(current_dir, "..", "config", "settings.py")
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    
    # Determine which method to use
    retrieval_method = method or getattr(settings, 'RETRIEVAL_METHOD', 'llm_guided')
    min_score = getattr(settings, 'MIN_PAPER_RELEVANCE_SCORE', 7)
    
    print(f"\n[Paper Retrieval] Using method: {retrieval_method}")
    
    if retrieval_method == "llm_guided":
        return retrieve_papers_llm_guided(
            topic=topic,
            client=client,
            model_name=model_name,
            target_papers=target_papers,
            min_score=min_score
        )
    elif retrieval_method == "keyword":
        return retrieve_papers_keyword(
            topic=topic,
            client=client,
            model_name=model_name,
            target_papers=target_papers,
            min_score=min_score
        )
    elif retrieval_method == "tavily":
        return retrieve_papers_tavily(
            topic=topic,
            client=client,
            model_name=model_name,
            target_papers=target_papers,
            min_score=min_score,
            settings=settings
        )
    elif retrieval_method in _custom_retrieval_methods:
        return _custom_retrieval_methods[retrieval_method](
            topic=topic,
            client=client,
            model_name=model_name,
            target_papers=target_papers
        )
    else:
        available = get_available_methods()
        raise ValueError(
            f"Unknown retrieval method: '{retrieval_method}'. "
            f"Available methods: {available}"
        )


# ============================================================================
# TAVILY + ARXIV RETRIEVAL
# ============================================================================

def retrieve_papers_tavily(
    topic: str,
    client,
    model_name: str,
    target_papers: int = 20,
    min_score: int = 5,
    settings = None
) -> List[Paper]:
    """
    Retrieve papers using Tavily search + arXiv, returning legacy Paper format.
    
    This is a wrapper that uses the paper_retrieval_tavily module and converts
    the results to the legacy Paper format for compatibility with the pipeline.
    
    Args:
        topic: Research topic description
        client: OpenAI client for LLM calls
        model_name: Model name
        target_papers: Maximum number of papers to return
        min_score: Minimum relevance score (1-10)
        settings: Settings module (optional, for additional config)
    
    Returns:
        List of Paper objects
    """
    # Import the tavily module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tavily_path = os.path.join(current_dir, "paper_retrieval_tavily.py")
    
    spec = importlib.util.spec_from_file_location("paper_retrieval_tavily", tavily_path)
    tavily_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tavily_module)
    
    # Get additional settings
    summarize_threshold = 8
    if settings:
        summarize_threshold = getattr(settings, 'TAVILY_SUMMARIZE_THRESHOLD', 8)
        target_papers = getattr(settings, 'TAVILY_MAX_PAPERS', target_papers)
    
    # Retrieve papers using Tavily
    retrieved_papers = tavily_module.retrieve_papers(
        topic=topic,
        client=client,
        model_name=model_name,
        max_papers=target_papers,
        min_relevance_score=min_score,
        summarize_threshold=summarize_threshold
    )
    
    # Convert to legacy Paper format
    legacy_papers = []
    for rp in retrieved_papers:
        legacy_papers.append(Paper(
            paper_id=rp.paper_id,
            title=rp.title,
            abstract=rp.summary,  # Use summary for richer context
            year=rp.year,
            citation_count=0,  # Not available from arXiv
            authors=rp.authors
        ))
    
    return legacy_papers


# ============================================================================
# PROMPTS
# ============================================================================

RETRIEVAL_AGENT_PROMPT = """You are a research assistant tasked with finding relevant papers on a topic.
You have access to the Semantic Scholar API through these functions:

1. KeywordQuery(keywords: str) - Search for papers by keywords. Returns up to 20 papers.
2. PaperQuery(paper_id: str) - Get details about a specific paper.
3. GetReferences(paper_id: str) - Get papers referenced by a given paper. Returns up to 20 papers.

Research Topic: {topic}

Your goal is to find diverse, relevant papers on this topic. Use a combination of:
- Keyword searches with different phrasings and sub-aspects
- Following references of highly relevant papers you find
- Exploring different angles and related areas

After each action, you'll see the results. Continue until you've found approximately 120 unique papers.

Progress: {num_papers} unique papers found so far (target: 120)

Previous actions:
{history}

What is your next action? Respond with ONLY a single function call in this exact format:
KeywordQuery("your search terms here")
or
GetReferences("paper_id_here")
or
PaperQuery("paper_id_here")

If you've found enough papers (120+), respond with:
DONE"""

PAPER_SCORING_PROMPT = """You are evaluating papers for relevance to a research topic.

Research Topic: {topic}

Paper Title: {title}
Paper Abstract: {abstract}

Score this paper from 1-10 based on these criteria:
1. The paper should be directly relevant to the specified topic
2. The paper should be an empirical paper involving computational experiments 
   (NOT a position paper, survey, or pure analysis paper)
3. The paper is interesting and could inspire new research projects

Provide your score as a single integer from 1-10, and nothing else."""

KEYWORD_GENERATION_PROMPT = """Generate diverse search queries to find papers about this research topic.

Research Topic: {topic}

Generate 8-10 different keyword search queries that would help find relevant papers.
Cover different aspects, methods, applications, and related areas.

Return ONLY the queries, one per line, no numbering or explanations."""


# ============================================================================
# LLM-GUIDED RETRIEVAL (Default Method)
# ============================================================================

def retrieve_papers_llm_guided(topic: str, client, model_name: str, 
                               target_papers: int = 120, min_score: int = 7) -> List[Paper]:
    """
    Given a research topic, use an LLM to generate Semantic Scholar API calls
    and retrieve relevant papers.
    
    Args:
        topic: Research topic description
        client: Anthropic client for LLM calls
        model_name: Model to use for generating API calls
        target_papers: Target number of papers to retrieve (default: 120)
        min_score: Minimum relevance score to keep papers (default: 7)
    
    Returns:
        List of top-ranked relevant papers (score >= min_score)
    """
    ss_client = SemanticScholarClient()
    
    # Track papers and history
    all_papers = {}  # paper_id -> Paper
    action_history = []
    max_iterations = 50  # Safety limit
    
    print(f"\n[Paper Retrieval] Starting LLM-guided retrieval for topic:")
    print(f"  {topic}")
    print(f"  Target: {target_papers} papers")
    
    # Step 1: LLM-guided retrieval
    for iteration in range(max_iterations):
        num_papers = len(all_papers)
        
        if num_papers >= target_papers:
            print(f"\n✓ Target reached: {num_papers} papers found")
            break
        
        # Build history string
        history_str = _build_history_string(action_history)
        
        # Get next action from LLM
        prompt = RETRIEVAL_AGENT_PROMPT.format(
            topic=topic,
            num_papers=num_papers,
            history=history_str
        )
        
        try:
            response = _call_llm(client, model_name, prompt)
            action = response.strip()
            
            if action == "DONE":
                print(f"\n✓ LLM indicated completion: {num_papers} papers found")
                break
            
            # Parse and execute action
            papers_found = _execute_action(action, ss_client)
            
            # Add new papers
            new_count = 0
            for paper in papers_found:
                if paper.paper_id not in all_papers:
                    all_papers[paper.paper_id] = paper
                    new_count += 1
            
            # Record action
            action_history.append({
                'action': action,
                'papers_found': len(papers_found),
                'new_papers': new_count
            })
            
            print(f"  [{iteration+1}] {action[:50]}... -> {len(papers_found)} results, {new_count} new (total: {len(all_papers)})")
            
        except Exception as e:
            print(f"  Error on iteration {iteration+1}: {e}")
            # Continue with next iteration
            continue
    
    papers_list = list(all_papers.values())
    print(f"\n[Paper Retrieval] Retrieved {len(papers_list)} unique papers")
    
    # Step 2: Score papers
    print(f"\n[Paper Scoring] Scoring papers for relevance...")
    scored_papers = _score_papers(papers_list, topic, client, model_name)
    
    # Step 3: Filter and sort
    filtered_papers = [(paper, score) for paper, score in scored_papers if score >= min_score]
    filtered_papers.sort(key=lambda x: x[1], reverse=True)
    
    result_papers = [paper for paper, score in filtered_papers]
    
    print(f"[Paper Scoring] Kept {len(result_papers)} papers with score >= {min_score}")
    if result_papers:
        avg_score = sum(score for _, score in filtered_papers) / len(filtered_papers)
        print(f"                Average score: {avg_score:.2f}")
    
    return result_papers


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
        spec = importlib.util.spec_from_file_location("feedback", fb_path)
        fb_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fb_module)
        return fb_module.get_formatted_feedback()
    except Exception:
        return ""


def _parse_action(action_str: str) -> Optional[Tuple[str, str]]:
    """
    Parse action string into (function_name, argument).
    
    Examples:
        KeywordQuery("deep learning") -> ("KeywordQuery", "deep learning")
        GetReferences("abc123") -> ("GetReferences", "abc123")
    """
    # Match function_name("argument")
    match = re.match(r'(\w+)\(["\']([^"\']+)["\']\)', action_str.strip())
    if match:
        return match.group(1), match.group(2)
    return None


def _execute_action(action_str: str, ss_client: SemanticScholarClient) -> List[Paper]:
    """Execute a Semantic Scholar API action and return papers."""
    parsed = _parse_action(action_str)
    
    if not parsed:
        print(f"    Warning: Could not parse action: {action_str}")
        return []
    
    function_name, argument = parsed
    
    try:
        if function_name == "KeywordQuery":
            return ss_client.KeywordQuery(argument, limit=20)
        elif function_name == "PaperQuery":
            paper = ss_client.PaperQuery(argument)
            return [paper] if paper else []
        elif function_name == "GetReferences":
            return ss_client.GetReferences(argument, limit=20)
        else:
            print(f"    Warning: Unknown function: {function_name}")
            return []
    except Exception as e:
        print(f"    Error executing {function_name}: {e}")
        return []


def _build_history_string(action_history: List[Dict], max_actions: int = 5) -> str:
    """Build a summary of recent actions for the LLM."""
    if not action_history:
        return "No actions yet."
    
    # Show only recent actions
    recent = action_history[-max_actions:]
    lines = []
    
    for i, entry in enumerate(recent, start=len(action_history)-len(recent)+1):
        action = entry['action']
        # Truncate long actions
        if len(action) > 60:
            action = action[:57] + "..."
        lines.append(f"{i}. {action} -> {entry['papers_found']} found, {entry['new_papers']} new")
    
    return "\n".join(lines)


def _score_papers(papers: List[Paper], topic: str, client, model_name: str) -> List[Tuple[Paper, int]]:
    """
    Score papers for relevance using LLM.
    
    Returns:
        List of (Paper, score) tuples
    """
    scored_papers = []
    
    for i, paper in enumerate(papers):
        if (i + 1) % 10 == 0:
            print(f"  Scored {i+1}/{len(papers)} papers...")
        
        # Skip papers without abstracts
        if not paper.abstract or len(paper.abstract.strip()) < 50:
            scored_papers.append((paper, 0))
            continue
        
        # Score the paper
        prompt = PAPER_SCORING_PROMPT.format(
            topic=topic,
            title=paper.title,
            abstract=paper.abstract[:1000]  # Limit abstract length
        )
        
        try:
            # Use higher token limit for reasoning models (includes reasoning + output tokens)
            response = _call_llm(client, model_name, prompt, max_tokens=512)
            score = _extract_score(response)
            scored_papers.append((paper, score))
            # Debug: show first few scores
            if len(scored_papers) <= 3:
                print(f"    Sample score: '{response.strip()[:30]}' → {score}")
        except Exception as e:
            print(f"    Error scoring paper '{paper.title[:50]}...': {e}")
            scored_papers.append((paper, 0))
    
    print(f"  Scored {len(papers)}/{len(papers)} papers ✓")
    return scored_papers


def _extract_score(response: str) -> int:
    """Extract integer score from LLM response."""
    response = response.strip()
    
    # Try exact match first (just a number)
    if response.isdigit():
        score = int(response)
        if 1 <= score <= 10:
            return score
    
    # Look for patterns like "8/10", "Score: 8", "8 out of 10"
    patterns = [
        r'(\d+)\s*/\s*10',      # 8/10
        r'[Ss]core[:\s]+(\d+)', # Score: 8
        r'^(\d+)\b',            # 8 at start
        r'\b([1-9]|10)\b',      # Any 1-10
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
    
    return 0


# ============================================================================
# KEYWORD-BASED RETRIEVAL
# ============================================================================

def retrieve_papers_keyword(topic: str, client, model_name: str,
                            target_papers: int = 120, min_score: int = 7) -> List[Paper]:
    """
    Retrieve papers using LLM-generated keyword queries (simpler than llm_guided).
    
    This method:
    1. Uses LLM to generate diverse keyword queries from the topic
    2. Executes each query against Semantic Scholar
    3. Deduplicates results
    4. Optionally scores and filters papers
    
    Args:
        topic: Research topic description
        client: OpenAI client for LLM calls
        model_name: Model to use
        target_papers: Target number of papers
        min_score: Minimum relevance score (set to 0 to skip scoring)
    
    Returns:
        List of relevant Paper objects
    """
    ss_client = SemanticScholarClient()
    
    print(f"\n[Keyword Retrieval] Generating search queries for topic:")
    print(f"  {topic}")
    
    # Step 1: Generate keyword queries using LLM
    prompt = KEYWORD_GENERATION_PROMPT.format(topic=topic)
    response = _call_llm(client, model_name, prompt, max_tokens=512)
    
    # Parse queries (one per line)
    queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
    # Remove any numbering or bullet points
    queries = [re.sub(r'^[\d\.\-\*\)]+\s*', '', q) for q in queries]
    
    print(f"  Generated {len(queries)} search queries:")
    for q in queries[:5]:
        print(f"    - {q[:60]}...")
    if len(queries) > 5:
        print(f"    ... and {len(queries) - 5} more")
    
    # Step 2: Execute queries and collect papers
    all_papers = {}  # paper_id -> Paper
    papers_per_query = max(10, target_papers // len(queries)) if queries else 20
    
    for i, query in enumerate(queries):
        try:
            papers = ss_client.KeywordQuery(query, limit=papers_per_query)
            new_count = 0
            for paper in papers:
                if paper.paper_id not in all_papers:
                    all_papers[paper.paper_id] = paper
                    new_count += 1
            print(f"  [{i+1}/{len(queries)}] '{query[:40]}...' -> {len(papers)} results, {new_count} new")
        except Exception as e:
            print(f"  [{i+1}/{len(queries)}] Error: {e}")
    
    papers_list = list(all_papers.values())
    print(f"\n[Keyword Retrieval] Retrieved {len(papers_list)} unique papers")
    
    # Step 3: Score and filter if min_score > 0
    if min_score > 0:
        print(f"\n[Paper Scoring] Scoring papers for relevance...")
        scored_papers = _score_papers(papers_list, topic, client, model_name)
        
        filtered_papers = [(paper, score) for paper, score in scored_papers if score >= min_score]
        filtered_papers.sort(key=lambda x: x[1], reverse=True)
        
        result_papers = [paper for paper, score in filtered_papers]
        
        print(f"[Paper Scoring] Kept {len(result_papers)} papers with score >= {min_score}")
        if result_papers:
            avg_score = sum(score for _, score in filtered_papers) / len(filtered_papers)
            print(f"                Average score: {avg_score:.2f}")
        
        return result_papers
    else:
        # Return all papers without scoring (faster but less precise)
        return papers_list


# ============================================================================
# UTILITY FUNCTIONS (keeping existing ones)
# ============================================================================

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
