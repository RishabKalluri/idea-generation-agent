"""
Idea Ranking Module using Swiss System Tournament.

Implements a Swiss system tournament for pairwise comparison and ranking of proposals.
The Swiss system efficiently ranks items by pairing competitors with similar scores,
requiring only O(n log n) comparisons instead of O(n^2) for full pairwise comparison.
"""

import os
import sys
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

# Load FullProposal class directly to avoid anthropic dependency
import importlib.util

def _load_full_proposal():
    """Load FullProposal class without triggering anthropic import."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    idea_gen_path = os.path.join(current_dir, "idea_generation.py")
    
    spec = importlib.util.spec_from_file_location("idea_gen_direct", idea_gen_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FullProposal, module.SeedIdea

FullProposal, SeedIdea = _load_full_proposal()


# ============================================================================
# PROMPTS
# ============================================================================

PAIRWISE_COMPARISON_PROMPT = """You are an expert research reviewer comparing two research proposals.

Proposal A:
{proposal_a}

Proposal B:
{proposal_b}

Which proposal is better overall? Consider:
- Novelty and creativity of the idea
- Potential impact on the field
- Technical soundness of the approach
- Feasibility of execution
- Clarity of the experiment plan

Answer with only "A" or "B" to indicate which proposal is better."""


DETAILED_COMPARISON_PROMPT = """You are an expert research reviewer comparing two research proposals.

Proposal A:
Title: {title_a}
Problem: {problem_a}
Method: {method_a}

Proposal B:
Title: {title_b}
Problem: {problem_b}
Method: {method_b}

Compare these proposals on the following criteria (score each 1-5):

1. Novelty: How original and creative is the idea?
2. Impact: How significant would the contribution be to the field?
3. Soundness: How technically sound is the approach?
4. Feasibility: How practical is it to execute?
5. Clarity: How clear is the experiment plan?

After scoring, indicate which proposal is better overall.

Format your response as:
Novelty: A=[score] B=[score]
Impact: A=[score] B=[score]
Soundness: A=[score] B=[score]
Feasibility: A=[score] B=[score]
Clarity: A=[score] B=[score]
Winner: [A or B]"""


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MatchResult:
    """Result of a pairwise comparison."""
    proposal_a_id: str
    proposal_b_id: str
    winner_id: str
    round_num: int
    raw_response: str = ""


@dataclass
class TournamentState:
    """Tracks the state of a Swiss tournament."""
    proposals: List
    scores: Dict[str, int]
    match_history: List[MatchResult] = field(default_factory=list)
    current_round: int = 0
    
    def get_proposal_by_id(self, proposal_id: str):
        """Get proposal by its ID (title)."""
        for p in self.proposals:
            if p.title == proposal_id:
                return p
        return None


# ============================================================================
# PAIRWISE COMPARISON
# ============================================================================

def format_proposal_for_comparison(proposal: FullProposal, max_length: int = 800) -> str:
    """Format a proposal for pairwise comparison, limiting length."""
    text = f"""Title: {proposal.title}

Problem: {proposal.problem_statement[:200] if proposal.problem_statement else 'N/A'}

Proposed Method: {proposal.proposed_method[:400] if proposal.proposed_method else 'N/A'}

Experiment Plan: {proposal.experiment_plan[:200] if proposal.experiment_plan else 'N/A'}"""
    
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text


def pairwise_compare(
    proposal_a: FullProposal,
    proposal_b: FullProposal,
    client,
    model_name: str,
    detailed: bool = False
) -> Tuple[str, str]:
    """
    Compare two proposals, return winner ('A' or 'B') and raw response.
    
    Args:
        proposal_a: First proposal
        proposal_b: Second proposal
        client: OpenAI client
        model_name: Model to use
        detailed: If True, use detailed comparison prompt with scores
    
    Returns:
        Tuple of (winner, raw_response) where winner is 'A' or 'B'
    """
    if detailed:
        prompt = DETAILED_COMPARISON_PROMPT.format(
            title_a=proposal_a.title,
            problem_a=proposal_a.problem_statement[:300] if proposal_a.problem_statement else 'N/A',
            method_a=proposal_a.proposed_method[:500] if proposal_a.proposed_method else 'N/A',
            title_b=proposal_b.title,
            problem_b=proposal_b.problem_statement[:300] if proposal_b.problem_statement else 'N/A',
            method_b=proposal_b.proposed_method[:500] if proposal_b.proposed_method else 'N/A'
        )
        max_tokens = 300
    else:
        prompt = PAIRWISE_COMPARISON_PROMPT.format(
            proposal_a=format_proposal_for_comparison(proposal_a),
            proposal_b=format_proposal_for_comparison(proposal_b)
        )
        max_tokens = 50
    
    response = _call_llm(client, model_name, prompt, max_tokens)
    response_text = response.strip()
    
    # Extract winner from response
    # Look for 'A' or 'B' in the response
    response_upper = response_text.upper()
    
    if "WINNER: A" in response_upper or response_upper.startswith("A"):
        winner = 'A'
    elif "WINNER: B" in response_upper or response_upper.startswith("B"):
        winner = 'B'
    elif 'A' in response_upper and 'B' not in response_upper:
        winner = 'A'
    elif 'B' in response_upper and 'A' not in response_upper:
        winner = 'B'
    else:
        # Default to random if unclear (shouldn't happen often)
        winner = random.choice(['A', 'B'])
    
    return winner, response_text


def run_pairwise_comparison(
    proposal_a: FullProposal,
    proposal_b: FullProposal,
    client,
    model_name: str
) -> str:
    """
    Convenience function that returns just the winner.
    
    Returns:
        'A' or 'B' indicating the winner
    """
    winner, _ = pairwise_compare(proposal_a, proposal_b, client, model_name)
    return winner


# ============================================================================
# SWISS TOURNAMENT
# ============================================================================

def create_pairings(
    proposals: List,
    scores: Dict[str, int],
    match_history: List[MatchResult] = None
) -> List[Tuple]:
    """
    Create pairings for a Swiss tournament round.
    
    Pairs proposals with similar scores, avoiding repeat matchups when possible.
    
    Args:
        proposals: List of proposals
        scores: Current scores dict (proposal.title -> score)
        match_history: Previous match results to avoid repeats
    
    Returns:
        List of (proposal_a, proposal_b) tuples
    """
    # Sort by score (descending)
    sorted_proposals = sorted(
        proposals, 
        key=lambda p: scores.get(p.title, 0), 
        reverse=True
    )
    
    # Track previous opponents
    previous_opponents = {}
    if match_history:
        for match in match_history:
            if match.proposal_a_id not in previous_opponents:
                previous_opponents[match.proposal_a_id] = set()
            if match.proposal_b_id not in previous_opponents:
                previous_opponents[match.proposal_b_id] = set()
            previous_opponents[match.proposal_a_id].add(match.proposal_b_id)
            previous_opponents[match.proposal_b_id].add(match.proposal_a_id)
    
    # Create pairings
    pairs = []
    paired = set()
    
    for i, prop_a in enumerate(sorted_proposals):
        if prop_a.title in paired:
            continue
        
        # Find best opponent (similar score, not previously matched if possible)
        best_opponent = None
        best_opponent_idx = None
        
        for j, prop_b in enumerate(sorted_proposals[i+1:], start=i+1):
            if prop_b.title in paired:
                continue
            
            # Check if previously matched
            prev_opps = previous_opponents.get(prop_a.title, set())
            was_matched = prop_b.title in prev_opps
            
            if best_opponent is None or not was_matched:
                best_opponent = prop_b
                best_opponent_idx = j
                
                # If we found someone we haven't played, use them
                if not was_matched:
                    break
        
        if best_opponent:
            pairs.append((prop_a, best_opponent))
            paired.add(prop_a.title)
            paired.add(best_opponent.title)
    
    return pairs


def swiss_tournament_round(
    proposals: List,
    scores: Dict[str, int],
    client,
    model_name: str,
    match_history: List[MatchResult] = None,
    round_num: int = 0,
    show_progress: bool = True
) -> Tuple[Dict[str, int], List[MatchResult]]:
    """
    Run one round of Swiss system tournament.
    
    1. Sort proposals by current score
    2. Pair adjacent proposals (similar scores compete)
    3. Run pairwise comparisons
    4. Update scores (winner gets +1 point)
    
    Args:
        proposals: List of proposals
        scores: Current scores dict
        client: OpenAI client
        model_name: Model to use
        match_history: Previous match results
        round_num: Current round number
        show_progress: Show progress bar
    
    Returns:
        Tuple of (updated_scores, new_match_results)
    """
    if match_history is None:
        match_history = []
    
    # Create pairings
    pairs = create_pairings(proposals, scores, match_history)
    
    new_matches = []
    
    # Run comparisons
    iterator = pairs
    if show_progress:
        iterator = tqdm(pairs, desc=f"Round {round_num + 1} matches", leave=False)
    
    for prop_a, prop_b in iterator:
        winner, raw_response = pairwise_compare(prop_a, prop_b, client, model_name)
        
        # Update scores
        if winner == 'A':
            scores[prop_a.title] = scores.get(prop_a.title, 0) + 1
            winner_id = prop_a.title
        else:
            scores[prop_b.title] = scores.get(prop_b.title, 0) + 1
            winner_id = prop_b.title
        
        # Record match
        match = MatchResult(
            proposal_a_id=prop_a.title,
            proposal_b_id=prop_b.title,
            winner_id=winner_id,
            round_num=round_num,
            raw_response=raw_response
        )
        new_matches.append(match)
    
    # Handle bye (odd number of proposals)
    paired_titles = set()
    for prop_a, prop_b in pairs:
        paired_titles.add(prop_a.title)
        paired_titles.add(prop_b.title)
    
    for prop in proposals:
        if prop.title not in paired_titles:
            # This proposal got a bye, give them a point
            scores[prop.title] = scores.get(prop.title, 0) + 1
    
    return scores, new_matches


def rank_proposals(
    proposals: List,
    client,
    model_name: str,
    num_rounds: int = 5,
    show_progress: bool = True
) -> List[Tuple]:
    """
    Rank proposals using Swiss system tournament.
    
    The Swiss system is efficient because:
    - Each round, proposals with similar scores compete
    - After k rounds, proposals are roughly sorted
    - Requires only O(n * k) comparisons vs O(n^2) for full pairwise
    
    Args:
        proposals: List of proposals to rank
        client: OpenAI client
        model_name: Model for comparisons
        num_rounds: Number of tournament rounds (paper uses 5)
        show_progress: Show progress bars
    
    Returns:
        List of (proposal, score) tuples sorted by score descending
    """
    if len(proposals) == 0:
        return []
    
    if len(proposals) == 1:
        return [(proposals[0], 0)]
    
    print(f"\n[Ranking] Starting Swiss tournament")
    print(f"  Proposals: {len(proposals)}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Total comparisons: ~{len(proposals) // 2 * num_rounds}")
    
    # Initialize scores
    scores = {p.title: 0 for p in proposals}
    all_matches = []
    
    # Run tournament rounds
    for round_num in range(num_rounds):
        if show_progress:
            print(f"\n  Round {round_num + 1}/{num_rounds}")
        
        scores, new_matches = swiss_tournament_round(
            proposals=proposals,
            scores=scores,
            client=client,
            model_name=model_name,
            match_history=all_matches,
            round_num=round_num,
            show_progress=show_progress
        )
        
        all_matches.extend(new_matches)
        
        if show_progress:
            # Show current standings
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_3 = sorted_scores[:3]
            print(f"    Top 3: ", end="")
            for title, score in top_3:
                short_title = title[:30] + "..." if len(title) > 30 else title
                print(f"{short_title}({score}) ", end="")
            print()
    
    # Sort by final score
    ranked = []
    for p in proposals:
        ranked.append((p, scores[p.title]))
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n[Ranking] Tournament complete!")
    print(f"  Total matches: {len(all_matches)}")
    
    return ranked


def run_swiss_tournament(
    proposals: List,
    client,
    model_name: str,
    num_rounds: int = 5
) -> List[Tuple]:
    """
    Alias for rank_proposals for backward compatibility.
    """
    return rank_proposals(proposals, client, model_name, num_rounds)


# ============================================================================
# ANALYSIS AND REPORTING
# ============================================================================

def get_tournament_statistics(
    ranked_proposals: List[Tuple],
    match_history: List[MatchResult] = None
) -> Dict:
    """
    Get statistics from tournament results.
    
    Returns:
        Dict with various statistics
    """
    if not ranked_proposals:
        return {}
    
    scores = [score for _, score in ranked_proposals]
    
    stats = {
        "num_proposals": len(ranked_proposals),
        "max_score": max(scores),
        "min_score": min(scores),
        "avg_score": sum(scores) / len(scores),
        "score_distribution": {},
    }
    
    # Score distribution
    for score in scores:
        stats["score_distribution"][score] = stats["score_distribution"].get(score, 0) + 1
    
    # Top proposals
    stats["top_5"] = [
        {"title": p.title, "score": s} 
        for p, s in ranked_proposals[:5]
    ]
    
    return stats


def save_ranking_report(
    ranked_proposals: List[Tuple],
    filepath: str,
    match_history: List[MatchResult] = None
):
    """Save ranking results to a file."""
    with open(filepath, 'w') as f:
        f.write("IDEA RANKING REPORT - SWISS TOURNAMENT\n")
        f.write("=" * 60 + "\n\n")
        
        stats = get_tournament_statistics(ranked_proposals, match_history)
        
        f.write(f"Total proposals: {stats.get('num_proposals', 0)}\n")
        f.write(f"Max score: {stats.get('max_score', 0)}\n")
        f.write(f"Min score: {stats.get('min_score', 0)}\n")
        f.write(f"Avg score: {stats.get('avg_score', 0):.2f}\n\n")
        
        f.write("FINAL RANKINGS:\n")
        f.write("-" * 60 + "\n\n")
        
        for rank, (proposal, score) in enumerate(ranked_proposals, 1):
            f.write(f"Rank {rank}: Score {score}\n")
            f.write(f"  Title: {proposal.title}\n")
            if proposal.problem_statement:
                f.write(f"  Problem: {proposal.problem_statement[:100]}...\n")
            f.write("\n")
    
    print(f"[Ranking] Report saved to {filepath}")


def get_top_proposals(
    ranked_proposals: List[Tuple],
    top_k: int = 10
) -> List:
    """
    Get the top-k ranked proposals.
    
    Returns:
        List of FullProposal objects
    """
    return [proposal for proposal, score in ranked_proposals[:top_k]]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _call_llm(client, model_name: str, prompt: str, max_tokens: int = 100) -> str:
    """Call OpenAI LLM and return response text."""
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def quick_compare(
    title_a: str,
    method_a: str,
    title_b: str,
    method_b: str,
    client,
    model_name: str
) -> Tuple[str, str]:
    """
    Quick comparison of two ideas by title and method.
    Useful for testing.
    
    Returns:
        (winner, explanation)
    """
    prompt = f"""Compare these two research ideas:

Idea A: {title_a}
Method: {method_a}

Idea B: {title_b}
Method: {method_b}

Which is more promising? Answer with "A" or "B" and a brief explanation."""

    response = _call_llm(client, model_name, prompt, max_tokens=100)
    
    winner = 'A' if response.strip().upper().startswith('A') else 'B'
    
    return winner, response.strip()
