"""
Pairwise Paper Grading Module.

Generates all pairs from a set of proposals, sends each pair to GPT-5.2
with web search enabled, and asks it to choose the better paper across
7 criteria. Aggregates results into win-rate rankings.
"""

import os
import json
import random
import itertools
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple
from datetime import datetime

from openai import OpenAI

from .converter import Proposal


# ============================================================================
# CRITERIA
# ============================================================================

CRITERIA = [
    "Originality",
    "Importance of Research Question",
    "Whether Claims Are Well-Supported",
    "Soundness of Experiments",
    "Clarity of Writing",
    "Value to the Research Community",
    "Contextualization Relative to Prior Work",
]

CRITERIA_BULLETS = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(CRITERIA))


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PairResult:
    """Result of comparing two proposals."""
    paper_a_id: str
    paper_a_title: str
    paper_b_id: str
    paper_b_title: str
    winner_id: str          # arxiv ID of the winner ("tie" if tie)
    winner_label: str       # "A", "B", or "tie"
    reasoning: str          # LLM explanation
    criteria_scores: dict   # per-criterion winner labels
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "PairResult":
        return PairResult(**d)


@dataclass
class Ranking:
    """Aggregated ranking for a single proposal."""
    arxiv_id: str
    title: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    total_matchups: int = 0
    win_rate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# GRADING PROMPT
# ============================================================================

GRADING_PROMPT = """You are an expert AI research reviewer. You will compare two research papers and determine which one is better overall.

**Evaluation Criteria:**
{criteria}

For each criterion, decide which paper is better (Paper A or Paper B) or if they are equal.

Then give your **overall verdict**: which paper is better overall, considering all criteria.

You are encouraged to search the web to look up the actual papers, verify claims, check citations, and assess the impact and reception of each paper. Use the arXiv IDs provided to find them.

---

**Paper A:**
- Title: {title_a}
- arXiv ID: {arxiv_id_a}
- Abstract: {abstract_a}

**Paper B:**
- Title: {title_b}
- arXiv ID: {arxiv_id_b}
- Abstract: {abstract_b}

---

Respond in EXACTLY this format:

CRITERION SCORES:
1. Originality: [A/B/tie]
2. Importance of Research Question: [A/B/tie]
3. Whether Claims Are Well-Supported: [A/B/tie]
4. Soundness of Experiments: [A/B/tie]
5. Clarity of Writing: [A/B/tie]
6. Value to the Research Community: [A/B/tie]
7. Contextualization Relative to Prior Work: [A/B/tie]

OVERALL WINNER: [A/B/tie]

REASONING:
[Your detailed explanation of why you chose the winner, referencing specific criteria and evidence from your web search.]"""


# ============================================================================
# PAIR GENERATION
# ============================================================================

def generate_pairs(
    proposals: List[Proposal],
    max_pairs: int = None,
    seed: int = 42,
) -> List[Tuple[Proposal, Proposal]]:
    """
    Generate proposal pairs for comparison.

    If the total number of pairs exceeds max_pairs, randomly sample.
    Otherwise, return all pairs.

    Args:
        proposals: List of Proposal objects
        max_pairs: Maximum number of pairs (None = all)
        seed: Random seed for reproducibility

    Returns:
        List of (Proposal, Proposal) tuples
    """
    all_pairs = list(itertools.combinations(proposals, 2))

    if max_pairs and len(all_pairs) > max_pairs:
        rng = random.Random(seed)
        all_pairs = rng.sample(all_pairs, max_pairs)

    # Randomize order within each pair to avoid position bias
    rng = random.Random(seed + 1)
    shuffled = []
    for a, b in all_pairs:
        if rng.random() < 0.5:
            shuffled.append((b, a))
        else:
            shuffled.append((a, b))

    return shuffled


# ============================================================================
# JUDGING
# ============================================================================

def judge_pair(
    paper_a: Proposal,
    paper_b: Proposal,
    client: OpenAI,
    model_name: str = "gpt-5.2",
) -> PairResult:
    """
    Use the Responses API with web search to judge a pair of papers.

    Args:
        paper_a: First proposal
        paper_b: Second proposal
        client: OpenAI client
        model_name: Model to use for grading

    Returns:
        PairResult with winner and per-criterion scores
    """
    # Use problem_statement as a proxy for abstract (since proposals
    # were converted from papers, the problem_statement captures the gist)
    abstract_a = paper_a.problem_statement
    abstract_b = paper_b.problem_statement

    prompt = GRADING_PROMPT.format(
        criteria=CRITERIA_BULLETS,
        title_a=paper_a.source_title or paper_a.title,
        arxiv_id_a=paper_a.source_arxiv_id,
        abstract_a=abstract_a,
        title_b=paper_b.source_title or paper_b.title,
        arxiv_id_b=paper_b.source_arxiv_id,
        abstract_b=abstract_b,
    )

    # Call Responses API with web search (gpt-5.2 with medium reasoning)
    response = client.responses.create(
        model=model_name,
        tools=[{"type": "web_search"}],
        input=prompt,
        reasoning={"effort": "medium"},
    )

    text = response.output_text

    # Parse response
    criteria_scores = _parse_criteria_scores(text)
    winner_label = _parse_overall_winner(text)
    reasoning = _parse_reasoning(text)

    # Map winner label to arxiv ID
    if winner_label == "A":
        winner_id = paper_a.source_arxiv_id
    elif winner_label == "B":
        winner_id = paper_b.source_arxiv_id
    else:
        winner_id = "tie"

    return PairResult(
        paper_a_id=paper_a.source_arxiv_id,
        paper_a_title=paper_a.source_title or paper_a.title,
        paper_b_id=paper_b.source_arxiv_id,
        paper_b_title=paper_b.source_title or paper_b.title,
        winner_id=winner_id,
        winner_label=winner_label,
        reasoning=reasoning,
        criteria_scores=criteria_scores,
        timestamp=datetime.now().isoformat(),
    )


# ============================================================================
# PARSING HELPERS
# ============================================================================

def _parse_criteria_scores(text: str) -> dict:
    """Parse per-criterion scores from the response."""
    import re

    scores = {}
    for criterion in CRITERIA:
        # Look for pattern like "1. Originality: A" or "Originality: B"
        pattern = rf"(?:\d+\.\s*)?{re.escape(criterion)}:\s*(A|B|tie)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).upper()
            scores[criterion] = val if val in ("A", "B") else "tie"
        else:
            scores[criterion] = "unknown"

    return scores


def _parse_overall_winner(text: str) -> str:
    """Parse the overall winner from the response."""
    import re

    match = re.search(r"OVERALL\s+WINNER:\s*(A|B|tie)", text, re.IGNORECASE)
    if match:
        val = match.group(1).upper()
        return val if val in ("A", "B") else "tie"

    # Fallback: count per-criterion wins
    return "tie"


def _parse_reasoning(text: str) -> str:
    """Extract reasoning section from the response."""
    import re

    match = re.search(r"REASONING:\s*\n?(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return text  # Return full text as fallback


# ============================================================================
# RANKING
# ============================================================================

def compute_rankings(
    results: List[PairResult],
    proposals: List[Proposal],
) -> List[Ranking]:
    """
    Compute win-rate rankings from pairwise results.

    Args:
        results: List of PairResult objects
        proposals: Original proposals for metadata

    Returns:
        List of Ranking objects sorted by win_rate descending
    """
    # Build lookup
    id_to_title = {}
    for p in proposals:
        id_to_title[p.source_arxiv_id] = p.source_title or p.title

    # Aggregate
    stats = {}
    for r in results:
        for pid in [r.paper_a_id, r.paper_b_id]:
            if pid not in stats:
                stats[pid] = {"wins": 0, "losses": 0, "ties": 0}

        if r.winner_label == "A":
            stats[r.paper_a_id]["wins"] += 1
            stats[r.paper_b_id]["losses"] += 1
        elif r.winner_label == "B":
            stats[r.paper_b_id]["wins"] += 1
            stats[r.paper_a_id]["losses"] += 1
        else:
            stats[r.paper_a_id]["ties"] += 1
            stats[r.paper_b_id]["ties"] += 1

    # Build ranking objects
    rankings = []
    for pid, s in stats.items():
        total = s["wins"] + s["losses"] + s["ties"]
        win_rate = (s["wins"] + 0.5 * s["ties"]) / total if total > 0 else 0.0

        rankings.append(Ranking(
            arxiv_id=pid,
            title=id_to_title.get(pid, pid),
            wins=s["wins"],
            losses=s["losses"],
            ties=s["ties"],
            total_matchups=total,
            win_rate=round(win_rate, 4),
        ))

    rankings.sort(key=lambda r: r.win_rate, reverse=True)
    return rankings


# ============================================================================
# FULL GRADING PIPELINE
# ============================================================================

def grade_dataset(
    proposals: List[Proposal],
    client: OpenAI,
    model_name: str = "gpt-5.2",
    max_pairs: int = None,
    seed: int = 42,
    output_dir: str = None,
) -> Tuple[List[PairResult], List[Ranking]]:
    """
    Run the full pairwise grading pipeline.

    Args:
        proposals: Proposals to compare
        client: OpenAI client
        model_name: Grading model
        max_pairs: Limit pairs (None = all)
        seed: Random seed
        output_dir: Directory to save results

    Returns:
        (pair_results, rankings)
    """
    n = len(proposals)
    total_possible = n * (n - 1) // 2
    pairs = generate_pairs(proposals, max_pairs=max_pairs, seed=seed)

    print(f"\n[Grader] Pairwise comparison")
    print(f"  Proposals    : {n}")
    print(f"  Total pairs  : {total_possible}")
    print(f"  Pairs to run : {len(pairs)}")
    print(f"  Model        : {model_name}")
    print(f"  Web search   : enabled")

    results = []
    for i, (pa, pb) in enumerate(pairs):
        print(f"\n  [{i+1}/{len(pairs)}] {pa.source_title[:40]}... vs {pb.source_title[:40]}...")

        try:
            result = judge_pair(pa, pb, client, model_name)
            results.append(result)
            winner_display = result.paper_a_title[:30] if result.winner_label == "A" else (
                result.paper_b_title[:30] if result.winner_label == "B" else "TIE"
            )
            print(f"    → Winner: {winner_display}")
        except Exception as e:
            print(f"    ✗ Error: {e}")

        # Be polite to the API
        if i < len(pairs) - 1:
            time.sleep(1)

    # Compute rankings
    rankings = compute_rankings(results, proposals)

    # Save results
    if output_dir:
        _save_results(results, rankings, output_dir)

    # Print summary
    _print_rankings(rankings)

    return results, rankings


# ============================================================================
# I/O HELPERS
# ============================================================================

def _save_results(
    results: List[PairResult],
    rankings: List[Ranking],
    output_dir: str,
):
    """Save grading results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save pair results
    results_path = os.path.join(output_dir, "pair_results.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    print(f"\n  ✓ Pair results saved to {results_path}")

    # Save rankings
    rankings_path = os.path.join(output_dir, "rankings.json")
    with open(rankings_path, "w") as f:
        json.dump([r.to_dict() for r in rankings], f, indent=2)
    print(f"  ✓ Rankings saved to {rankings_path}")

    # Save human-readable summary
    summary_path = os.path.join(output_dir, "rankings_summary.txt")
    with open(summary_path, "w") as f:
        f.write("ORACL Paper Rankings\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total matchups: {len(results)}\n")
        f.write("=" * 80 + "\n\n")

        for i, r in enumerate(rankings):
            f.write(f"#{i+1}  Win Rate: {r.win_rate:.1%}  "
                    f"({r.wins}W/{r.losses}L/{r.ties}T)  "
                    f"{r.title}\n"
                    f"     arXiv: {r.arxiv_id}\n\n")

    print(f"  ✓ Summary saved to {summary_path}")


def _print_rankings(rankings: List[Ranking]):
    """Print rankings to console."""
    print(f"\n{'='*70}")
    print(f"  RANKINGS")
    print(f"{'='*70}")
    print(f"  {'#':<4} {'Win Rate':<10} {'W/L/T':<10} Title")
    print(f"  {'-'*3:<4} {'-'*8:<10} {'-'*7:<10} {'-'*40}")

    for i, r in enumerate(rankings):
        wlt = f"{r.wins}/{r.losses}/{r.ties}"
        title = r.title[:50] + ("..." if len(r.title) > 50 else "")
        print(f"  {i+1:<4} {r.win_rate:<10.1%} {wlt:<10} {title}")

    print(f"{'='*70}\n")


def load_pair_results(path: str) -> List[PairResult]:
    """Load pair results from a JSONL file."""
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(PairResult.from_dict(json.loads(line)))
    return results
