"""
Paper Retrieval Module using OpenAI API (GPT-4).

This is an OpenAI-compatible version of paper_retrieval.py
"""

import re
from typing import List, Dict, Tuple, Optional
from utils import SemanticScholarClient, Paper


# ============================================================================
# PROMPTS (same as paper_retrieval.py)
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


# ============================================================================
# MAIN RETRIEVAL FUNCTION (OpenAI version)
# ============================================================================

def retrieve_papers(topic: str, client, model_name: str = "gpt-4", 
                   target_papers: int = 120, min_score: int = 7) -> List[Paper]:
    """
    Given a research topic, use an LLM to generate Semantic Scholar API calls
    and retrieve relevant papers. OpenAI-compatible version.
    
    Args:
        topic: Research topic description
        client: OpenAI client for LLM calls
        model_name: Model to use (default: "gpt-4", also works with "gpt-3.5-turbo")
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
    print(f"  Model: {model_name}")
    
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
            response = _call_llm_openai(client, model_name, prompt)
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
            continue
    
    papers_list = list(all_papers.values())
    print(f"\n[Paper Retrieval] Retrieved {len(papers_list)} unique papers")
    
    # Step 2: Score papers
    print(f"\n[Paper Scoring] Scoring papers for relevance...")
    scored_papers = _score_papers_openai(papers_list, topic, client, model_name)
    
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
# HELPER FUNCTIONS (OpenAI-specific)
# ============================================================================

def _call_llm_openai(client, model_name: str, prompt: str, max_tokens: int = 1024) -> str:
    """Call OpenAI LLM and return response text."""
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def _parse_action(action_str: str) -> Optional[Tuple[str, str]]:
    """Parse action string into (function_name, argument)."""
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
    
    recent = action_history[-max_actions:]
    lines = []
    
    for i, entry in enumerate(recent, start=len(action_history)-len(recent)+1):
        action = entry['action']
        if len(action) > 60:
            action = action[:57] + "..."
        lines.append(f"{i}. {action} -> {entry['papers_found']} found, {entry['new_papers']} new")
    
    return "\n".join(lines)


def _score_papers_openai(papers: List[Paper], topic: str, client, model_name: str) -> List[Tuple[Paper, int]]:
    """Score papers for relevance using OpenAI LLM."""
    scored_papers = []
    
    for i, paper in enumerate(papers):
        if (i + 1) % 10 == 0:
            print(f"  Scored {i+1}/{len(papers)} papers...")
        
        if not paper.abstract or len(paper.abstract.strip()) < 50:
            scored_papers.append((paper, 0))
            continue
        
        prompt = PAPER_SCORING_PROMPT.format(
            topic=topic,
            title=paper.title,
            abstract=paper.abstract[:1000]
        )
        
        try:
            response = _call_llm_openai(client, model_name, prompt, max_tokens=10)
            score = _extract_score(response)
            scored_papers.append((paper, score))
        except Exception as e:
            print(f"    Error scoring paper '{paper.title[:50]}...': {e}")
            scored_papers.append((paper, 0))
    
    print(f"  Scored {len(papers)}/{len(papers)} papers ✓")
    return scored_papers


def _extract_score(response: str) -> int:
    """Extract integer score from LLM response."""
    match = re.search(r'\b([1-9]|10)\b', response.strip())
    if match:
        return int(match.group(1))
    return 0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def build_rag_context(papers: List[Paper], max_papers: int = 10) -> str:
    """Build RAG context from retrieved papers for LLM prompting."""
    if not papers:
        return ""
    
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
            abstract = paper.abstract[:500] + "..." if len(paper.abstract) > 500 else paper.abstract
            context_parts.append(f"Abstract: {abstract}")
        
        context_parts.append("")
    
    return "\n".join(context_parts)
