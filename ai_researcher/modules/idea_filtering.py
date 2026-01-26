"""
Idea Filtering Module.

Performs novelty and feasibility checks on generated proposals.
Filters out proposals that are:
- Not novel (essentially the same as existing papers)
- Not feasible (require resources beyond typical academic labs)
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

# Load classes directly to avoid anthropic dependency
import importlib.util

def _load_classes():
    """Load FullProposal and Paper classes without triggering anthropic import."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load FullProposal from idea_generation
    idea_gen_path = os.path.join(current_dir, "idea_generation.py")
    spec1 = importlib.util.spec_from_file_location("idea_gen_direct", idea_gen_path)
    idea_gen = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(idea_gen)
    
    # Load Paper from semantic_scholar
    parent_dir = os.path.dirname(current_dir)
    ss_path = os.path.join(parent_dir, "utils", "semantic_scholar.py")
    spec2 = importlib.util.spec_from_file_location("ss_direct", ss_path)
    ss_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ss_module)
    
    return idea_gen.FullProposal, idea_gen.SeedIdea, ss_module.Paper

FullProposal, SeedIdea, Paper = _load_classes()


# ============================================================================
# PROMPTS
# ============================================================================

NOVELTY_CHECK_PROMPT = """You are a research reviewer checking if a proposed idea is novel compared to existing work.

Proposed Idea:
Title: {idea_title}
Method: {idea_method}

Existing Paper:
Title: {paper_title}
Abstract: {paper_abstract}

Is the proposed idea essentially the same as this existing paper? 
Consider it the same if:
- The core method is identical (not just similar)
- The main contribution would be redundant

Answer with only "SAME" or "DIFFERENT" followed by a one-sentence explanation."""


FEASIBILITY_CHECK_PROMPT = """You are evaluating the feasibility of a research proposal for a typical academic lab.

Proposal:
{proposal_text}

Check for these feasibility issues:
1. Does it require extensive manual labor (e.g., large-scale human annotation)?
2. Does it require hardware resources beyond a typical academic lab (e.g., training 100B+ parameter models)?
3. Are there inconsistencies in the experimental setup?
   - Example: Claims to use only black-box API access but requires internal model weights
   - Example: Proposes to fine-tune a model but also claims no GPU compute needed

Answer with "FEASIBLE" or "NOT_FEASIBLE" followed by a brief explanation of any issues found."""


BATCH_NOVELTY_CHECK_PROMPT = """You are a research reviewer checking if a proposed idea is novel compared to a set of existing papers.

Proposed Idea:
Title: {idea_title}
Method: {idea_method}

Existing Papers:
{papers_list}

Is the proposed idea essentially the same as ANY of these existing papers?
Consider it the same if the core method is identical (not just similar) and the main contribution would be redundant.

Answer with:
- "NOVEL" if the idea is different from all papers
- "NOT_NOVEL: [paper_number]" if it's essentially the same as one of the papers

Then provide a one-sentence explanation."""


# ============================================================================
# EMBEDDING-BASED SIMILARITY (for finding similar papers)
# ============================================================================

# Lazy loading of embedding model
_EMBEDDING_MODEL = None

def _get_embedding_model():
    """Load embedding model lazily."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("[Filtering] Loading embedding model...")
            _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            print("[Filtering] Warning: sentence-transformers not installed, using random similarity")
            return None
    return _EMBEDDING_MODEL


def find_similar_papers(
    proposal: FullProposal,
    papers: List,
    top_k: int = 10
) -> List:
    """
    Find the top-k most similar papers to the proposal using embeddings.
    
    Args:
        proposal: The proposal to compare
        papers: List of Paper objects
        top_k: Number of similar papers to return
    
    Returns:
        List of (paper, similarity_score) tuples, sorted by similarity
    """
    if not papers:
        return []
    
    model = _get_embedding_model()
    
    if model is None:
        # Fallback: return random sample if no embedding model
        import random
        sample = random.sample(papers, min(top_k, len(papers)))
        return [(p, 0.5) for p in sample]
    
    # Create proposal text for embedding
    proposal_text = f"{proposal.title}\n{proposal.proposed_method}"
    
    # Create paper texts
    paper_texts = []
    for paper in papers:
        text = f"{paper.title}\n{paper.abstract if paper.abstract else ''}"
        paper_texts.append(text)
    
    # Compute embeddings
    proposal_embedding = model.encode([proposal_text], convert_to_numpy=True)[0]
    paper_embeddings = model.encode(paper_texts, convert_to_numpy=True)
    
    # Compute cosine similarities
    similarities = []
    for i, paper_emb in enumerate(paper_embeddings):
        sim = np.dot(proposal_embedding, paper_emb) / (
            np.linalg.norm(proposal_embedding) * np.linalg.norm(paper_emb) + 1e-10
        )
        similarities.append((papers[i], float(sim)))
    
    # Sort by similarity and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# ============================================================================
# NOVELTY CHECK
# ============================================================================

def check_novelty_against_paper(
    proposal: FullProposal,
    paper,
    client,
    model_name: str
) -> Tuple[bool, str]:
    """
    Check if a proposal is novel compared to a single paper.
    
    Returns:
        (is_different, explanation)
    """
    prompt = NOVELTY_CHECK_PROMPT.format(
        idea_title=proposal.title,
        idea_method=proposal.proposed_method[:1000],  # Limit length
        paper_title=paper.title,
        paper_abstract=paper.abstract[:1000] if paper.abstract else "No abstract available"
    )
    
    response = _call_llm(client, model_name, prompt, max_tokens=100)
    response = response.strip().upper()
    
    is_different = response.startswith("DIFFERENT")
    explanation = response.split("\n")[0] if "\n" in response else response
    
    return is_different, explanation


def check_novelty(
    proposal: FullProposal,
    papers: List,
    client,
    model_name: str,
    top_k: int = 10,
    use_batch: bool = True
) -> Tuple[bool, str]:
    """
    Check if proposal is novel compared to existing papers.
    
    1. Use embedding similarity to find top_k most similar papers
    2. Ask LLM to compare proposal against similar papers
    3. Return (is_novel, explanation)
    
    Filter out (return False) if ANY paper is judged as equivalent.
    
    Args:
        proposal: The proposal to check
        papers: List of Paper objects from RAG
        client: OpenAI client
        model_name: Model to use
        top_k: Number of similar papers to check against
        use_batch: If True, check all papers in one prompt (faster)
    
    Returns:
        (is_novel, explanation)
    """
    if not papers:
        return True, "No papers to compare against"
    
    # Find most similar papers using embeddings
    similar_papers = find_similar_papers(proposal, papers, top_k)
    
    if not similar_papers:
        return True, "No similar papers found"
    
    if use_batch:
        # Batch check: compare against all similar papers in one prompt
        return _check_novelty_batch(proposal, similar_papers, client, model_name)
    else:
        # Individual check: compare against each paper separately
        return _check_novelty_individual(proposal, similar_papers, client, model_name)


def _check_novelty_batch(
    proposal: FullProposal,
    similar_papers: List[Tuple],  # List of (paper, similarity) tuples
    client,
    model_name: str
) -> Tuple[bool, str]:
    """Check novelty against multiple papers in one LLM call."""
    
    # Format papers list
    papers_list = ""
    for i, (paper, sim) in enumerate(similar_papers, 1):
        abstract = paper.abstract[:300] if paper.abstract else "No abstract"
        papers_list += f"\n[Paper {i}] {paper.title}\nAbstract: {abstract}...\n"
    
    prompt = BATCH_NOVELTY_CHECK_PROMPT.format(
        idea_title=proposal.title,
        idea_method=proposal.proposed_method[:800],
        papers_list=papers_list
    )
    
    response = _call_llm(client, model_name, prompt, max_tokens=150)
    response_upper = response.strip().upper()
    
    is_novel = response_upper.startswith("NOVEL") and "NOT_NOVEL" not in response_upper
    
    return is_novel, response.strip()


def _check_novelty_individual(
    proposal: FullProposal,
    similar_papers: List[Tuple],
    client,
    model_name: str
) -> Tuple[bool, str]:
    """Check novelty against each paper individually."""
    
    for paper, similarity in similar_papers:
        is_different, explanation = check_novelty_against_paper(
            proposal, paper, client, model_name
        )
        
        if not is_different:
            return False, f"Similar to '{paper.title}': {explanation}"
    
    return True, "Novel compared to all checked papers"


# ============================================================================
# FEASIBILITY CHECK
# ============================================================================

def check_feasibility(
    proposal: FullProposal,
    client,
    model_name: str
) -> Tuple[bool, str]:
    """
    Check if proposal is feasible for a typical academic lab.
    
    Checks for:
    - Extensive manual labor requirements
    - Hardware resources beyond typical academic labs
    - Inconsistencies in experimental setup
    
    Args:
        proposal: The proposal to check
        client: OpenAI client
        model_name: Model to use
    
    Returns:
        (is_feasible, explanation)
    """
    # Create proposal text (include key sections)
    proposal_text = f"""Title: {proposal.title}

Proposed Method: {proposal.proposed_method}

Experiment Plan: {proposal.experiment_plan}"""
    
    # Limit length
    if len(proposal_text) > 2000:
        proposal_text = proposal_text[:2000] + "..."
    
    prompt = FEASIBILITY_CHECK_PROMPT.format(proposal_text=proposal_text)
    
    response = _call_llm(client, model_name, prompt, max_tokens=200)
    response_upper = response.strip().upper()
    
    is_feasible = response_upper.startswith("FEASIBLE") and "NOT_FEASIBLE" not in response_upper
    
    return is_feasible, response.strip()


# ============================================================================
# MAIN FILTERING FUNCTION
# ============================================================================

def filter_proposals(
    proposals: List[FullProposal],
    papers: List,
    client,
    model_name: str,
    check_novelty_flag: bool = True,
    check_feasibility_flag: bool = True,
    novelty_top_k: int = 10,
    show_progress: bool = True
) -> Tuple[List[FullProposal], Dict]:
    """
    Filter proposals based on novelty and feasibility.
    
    Paper reports filtering about 1% of proposals.
    
    Args:
        proposals: List of FullProposal objects to filter
        papers: List of Paper objects for novelty comparison
        client: OpenAI client
        model_name: Model to use
        check_novelty_flag: Whether to check novelty
        check_feasibility_flag: Whether to check feasibility
        novelty_top_k: Number of similar papers to check for novelty
        show_progress: Whether to show progress bar
    
    Returns:
        Tuple of (filtered_proposals, stats_dict)
    """
    print(f"\n[Filtering] Starting to filter {len(proposals)} proposals")
    print(f"  Novelty check: {'ON' if check_novelty_flag else 'OFF'}")
    print(f"  Feasibility check: {'ON' if check_feasibility_flag else 'OFF'}")
    if check_novelty_flag:
        print(f"  Papers for comparison: {len(papers)}")
        print(f"  Top-k similar papers: {novelty_top_k}")
    
    filtered = []
    stats = {
        "total": len(proposals),
        "passed": 0,
        "failed_novelty": 0,
        "failed_feasibility": 0,
        "novelty_reasons": [],
        "feasibility_reasons": []
    }
    
    iterator = proposals
    if show_progress:
        iterator = tqdm(proposals, desc="Filtering proposals")
    
    for proposal in iterator:
        # Check novelty
        if check_novelty_flag and papers:
            is_novel, novelty_reason = check_novelty(
                proposal, papers, client, model_name, 
                top_k=novelty_top_k
            )
            
            if not is_novel:
                stats["failed_novelty"] += 1
                stats["novelty_reasons"].append({
                    "title": proposal.title,
                    "reason": novelty_reason
                })
                continue
        
        # Check feasibility
        if check_feasibility_flag:
            is_feasible, feasibility_reason = check_feasibility(
                proposal, client, model_name
            )
            
            if not is_feasible:
                stats["failed_feasibility"] += 1
                stats["feasibility_reasons"].append({
                    "title": proposal.title,
                    "reason": feasibility_reason
                })
                continue
        
        # Passed all checks
        filtered.append(proposal)
        stats["passed"] += 1
    
    # Print summary
    print(f"\n[Filtering] Complete!")
    print(f"  Total: {stats['total']}")
    print(f"  Passed: {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
    if check_novelty_flag:
        print(f"  Failed novelty: {stats['failed_novelty']} ({stats['failed_novelty']/stats['total']*100:.1f}%)")
    if check_feasibility_flag:
        print(f"  Failed feasibility: {stats['failed_feasibility']} ({stats['failed_feasibility']/stats['total']*100:.1f}%)")
    
    return filtered, stats


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _call_llm(client, model_name: str, prompt: str, max_tokens: int = 256) -> str:
    """Call OpenAI LLM and return response text."""
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def save_filtering_report(
    stats: Dict,
    filepath: str
):
    """Save a detailed filtering report to file."""
    with open(filepath, 'w') as f:
        f.write("IDEA FILTERING REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total proposals: {stats['total']}\n")
        f.write(f"Passed: {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)\n")
        f.write(f"Failed novelty: {stats['failed_novelty']}\n")
        f.write(f"Failed feasibility: {stats['failed_feasibility']}\n\n")
        
        if stats['novelty_reasons']:
            f.write("NOVELTY FAILURES:\n")
            f.write("-" * 40 + "\n")
            for item in stats['novelty_reasons']:
                f.write(f"\nTitle: {item['title']}\n")
                f.write(f"Reason: {item['reason']}\n")
        
        if stats['feasibility_reasons']:
            f.write("\nFEASIBILITY FAILURES:\n")
            f.write("-" * 40 + "\n")
            for item in stats['feasibility_reasons']:
                f.write(f"\nTitle: {item['title']}\n")
                f.write(f"Reason: {item['reason']}\n")
    
    print(f"[Filtering] Report saved to {filepath}")


# ============================================================================
# STANDALONE CHECKS (for individual testing)
# ============================================================================

def quick_novelty_check(
    idea_title: str,
    idea_method: str,
    paper_title: str,
    paper_abstract: str,
    client,
    model_name: str
) -> Tuple[bool, str]:
    """
    Quick novelty check for testing purposes.
    
    Returns:
        (is_different, explanation)
    """
    prompt = NOVELTY_CHECK_PROMPT.format(
        idea_title=idea_title,
        idea_method=idea_method,
        paper_title=paper_title,
        paper_abstract=paper_abstract
    )
    
    response = _call_llm(client, model_name, prompt, max_tokens=100)
    is_different = response.strip().upper().startswith("DIFFERENT")
    
    return is_different, response.strip()


def quick_feasibility_check(
    proposal_text: str,
    client,
    model_name: str
) -> Tuple[bool, str]:
    """
    Quick feasibility check for testing purposes.
    
    Returns:
        (is_feasible, explanation)
    """
    prompt = FEASIBILITY_CHECK_PROMPT.format(proposal_text=proposal_text)
    
    response = _call_llm(client, model_name, prompt, max_tokens=200)
    is_feasible = "NOT_FEASIBLE" not in response.strip().upper()
    
    return is_feasible, response.strip()
