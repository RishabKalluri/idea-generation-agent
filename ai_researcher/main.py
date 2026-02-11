#!/usr/bin/env python3
"""
AI Research Idea Generation Pipeline

Full pipeline for generating, filtering, and ranking AI research ideas.
Based on the methodology from "Can LLMs Generate Novel Research Ideas?"

Usage:
    python main.py --topic factuality --output_dir outputs
    python main.py --topic math --num_ideas 100 --output_dir outputs
"""

import os
import sys
import json
import argparse
import importlib.util
from datetime import datetime
from typing import List, Tuple, Dict, Any

# Load modules directly to avoid anthropic dependency
def _load_module(name: str, path: str):
    """Load a module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load all modules
settings = _load_module("settings", os.path.join(SCRIPT_DIR, "config/settings.py"))
paper_retrieval = _load_module("paper_retrieval", os.path.join(SCRIPT_DIR, "modules/paper_retrieval.py"))
idea_generation = _load_module("idea_generation", os.path.join(SCRIPT_DIR, "modules/idea_generation.py"))
deduplication = _load_module("deduplication", os.path.join(SCRIPT_DIR, "modules/deduplication.py"))
idea_filtering = _load_module("idea_filtering", os.path.join(SCRIPT_DIR, "modules/idea_filtering.py"))
idea_ranking = _load_module("idea_ranking", os.path.join(SCRIPT_DIR, "modules/idea_ranking.py"))
style_normalization = _load_module("style_normalization", os.path.join(SCRIPT_DIR, "modules/style_normalization.py"))
semantic_scholar = _load_module("semantic_scholar", os.path.join(SCRIPT_DIR, "utils/semantic_scholar.py"))
feedback_utils = _load_module("feedback", os.path.join(SCRIPT_DIR, "utils/feedback.py"))

# Import tqdm for progress bars
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

# Research topics from the paper
RESEARCH_TOPICS = {
    "bias": "novel prompting methods to reduce social biases and stereotypes of large language models",
    "coding": "novel prompting methods for large language models to improve code generation",
    "safety": "novel prompting methods to improve large language models' robustness against adversarial attacks or improve their security or privacy",
    "multilingual": "novel prompting methods to improve large language models' performance on multilingual tasks or low-resource languages and vernacular languages",
    "factuality": "novel prompting methods that can improve factuality and reduce hallucination of large language models",
    "math": "novel prompting methods for large language models to improve mathematical problem solving",
    "uncertainty": "novel prompting methods that can better quantify uncertainty or calibrate the confidence of large language models"
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_openai_client():
    """Initialize and return OpenAI client."""
    # Use the key from settings (which loads from .env)
    api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Add it to config/.env or set as environment variable")
    
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def ensure_output_dir(output_dir: str, topic_key: str) -> str:
    """Create output directory structure."""
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, topic_key, timestamp)
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "intermediate"), exist_ok=True)
    
    return run_dir


def save_papers(papers: List, output_path: str):
    """Save retrieved papers to JSON."""
    paper_data = []
    for p in papers:
        paper_data.append({
            "paper_id": p.paper_id,
            "title": p.title,
            "abstract": p.abstract[:500] if p.abstract else "",
            "year": p.year,
            "citation_count": p.citation_count,
            "authors": p.authors[:5]  # First 5 authors
        })
    
    with open(output_path, 'w') as f:
        json.dump(paper_data, f, indent=2)


def save_seed_ideas(ideas: List, output_path: str):
    """Save seed ideas to file."""
    with open(output_path, 'w') as f:
        for i, idea in enumerate(ideas, 1):
            f.write(f"{'='*60}\n")
            f.write(f"SEED IDEA {i}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Title: {idea.title}\n\n")
            f.write(f"Problem: {idea.problem}\n\n")
            f.write(f"Existing Methods: {idea.existing_methods}\n\n")
            f.write(f"Motivation: {idea.motivation}\n\n")
            f.write(f"Proposed Method: {idea.proposed_method}\n\n")
            f.write(f"Experiment Plan: {idea.experiment_plan}\n\n")
            f.write(f"RAG Used: {idea.rag_used}\n")
            f.write("\n\n")


def save_proposals(proposals: List, output_path: str):
    """Save full proposals to file."""
    with open(output_path, 'w') as f:
        for i, proposal in enumerate(proposals, 1):
            f.write(f"{'='*60}\n")
            f.write(f"PROPOSAL {i}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Title: {proposal.title}\n\n")
            f.write(f"1. Problem Statement:\n{proposal.problem_statement}\n\n")
            f.write(f"2. Motivation:\n{proposal.motivation}\n\n")
            f.write(f"3. Proposed Method:\n{proposal.proposed_method}\n\n")
            f.write(f"4. Experiment Plan:\n{proposal.experiment_plan}\n\n")
            f.write(f"5. Test Case Examples:\n{proposal.test_case_examples}\n\n")
            f.write(f"6. Fallback Plan:\n{proposal.fallback_plan}\n")
            f.write("\n\n")


def save_ranked_proposals(ranked: List[Tuple], output_path: str):
    """Save ranked proposals with scores."""
    with open(output_path, 'w') as f:
        f.write("RANKED PROPOSALS\n")
        f.write("="*60 + "\n\n")
        
        for rank, (proposal, score) in enumerate(ranked, 1):
            f.write(f"Rank #{rank} (Score: {score})\n")
            f.write(f"-"*40 + "\n")
            f.write(f"Title: {proposal.title}\n\n")
            f.write(f"Problem: {proposal.problem_statement[:300]}...\n\n")
            f.write(f"Method: {proposal.proposed_method[:300]}...\n")
            f.write("\n" + "="*60 + "\n\n")


def save_results(
    normalized_proposals: List,
    ranked_proposals: List[Tuple],
    output_dir: str,
    topic_key: str,
    run_dir: str = None
):
    """Save all final results."""
    if run_dir is None:
        run_dir = ensure_output_dir(output_dir, topic_key)
    
    # Save normalized top proposals
    save_proposals(normalized_proposals, os.path.join(run_dir, "top_proposals_normalized.txt"))
    
    # Save full rankings
    save_ranked_proposals(ranked_proposals, os.path.join(run_dir, "rankings.txt"))
    
    # Save summary
    summary = {
        "topic": topic_key,
        "topic_description": RESEARCH_TOPICS[topic_key],
        "timestamp": datetime.now().isoformat(),
        "total_ranked": len(ranked_proposals),
        "top_normalized": len(normalized_proposals),
        "top_10_titles": [p.title for p, _ in ranked_proposals[:10]]
    }
    
    with open(os.path.join(run_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {run_dir}")


def print_banner(text: str):
    """Print a formatted banner."""
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(
    topic_key: str,
    output_dir: str,
    num_ideas: int = None,
    num_papers: int = None,
    skip_retrieval: bool = False,
    papers_file: str = None,
    model_name: str = None,
    retrieval_method: str = None
):
    """
    Run the full idea generation pipeline for a topic.
    
    Steps:
    1. Retrieve relevant papers via RAG
    2. Generate seed ideas (default 4000)
    3. Deduplicate to unique ideas
    4. Expand to full proposals
    5. Filter for novelty and feasibility
    6. Rank using Swiss tournament
    7. Normalize styles
    8. Save results
    
    Args:
        topic_key: Research topic key
        output_dir: Base output directory
        num_ideas: Number of seed ideas to generate (default from settings)
        num_papers: Number of papers to retrieve (default from settings)
        skip_retrieval: Skip paper retrieval and use provided papers_file
        papers_file: Path to pre-retrieved papers JSON
        model_name: Override model name
        retrieval_method: Paper retrieval method ("llm_guided", "keyword", or custom)
    """
    topic = RESEARCH_TOPICS[topic_key]
    
    # Use settings defaults if not provided
    num_ideas = num_ideas or settings.NUM_SEED_IDEAS
    num_papers = num_papers or settings.NUM_RETRIEVED_PAPERS
    model = model_name or settings.OPENAI_MODEL_NAME
    
    # Initialize client
    client = get_openai_client()
    
    # Create output directory
    run_dir = ensure_output_dir(output_dir, topic_key)
    
    print_banner(f"AI Research Idea Generation Pipeline")
    print(f"Topic: {topic_key}")
    print(f"Description: {topic}")
    print(f"Model: {model}")
    print(f"Target ideas: {num_ideas}")
    print(f"Output: {run_dir}")
    
    # -------------------------------------------------------------------------
    # Step 1: Paper Retrieval
    # -------------------------------------------------------------------------
    print_banner("Step 1: Retrieving relevant papers")
    
    if skip_retrieval and papers_file:
        print(f"Loading papers from: {papers_file}")
        with open(papers_file, 'r') as f:
            papers_data = json.load(f)
        
        # Convert to Paper objects
        Paper = semantic_scholar.Paper
        papers = [
            Paper(
                paper_id=p["paper_id"],
                title=p["title"],
                abstract=p.get("abstract", ""),
                year=p.get("year", 0),
                citation_count=p.get("citation_count", 0),
                authors=p.get("authors", [])
            )
            for p in papers_data
        ]
    else:
        papers = paper_retrieval.retrieve_papers(
            topic=topic,
            client=client,
            model_name=model,
            target_papers=num_papers,
            method=retrieval_method
        )
    
    print(f"✓ Retrieved {len(papers)} relevant papers")
    
    # Save intermediate
    save_papers(papers, os.path.join(run_dir, "intermediate", "papers.json"))
    
    # -------------------------------------------------------------------------
    # Step 2: Generate Seed Ideas
    # -------------------------------------------------------------------------
    print_banner("Step 2: Generating seed ideas")
    
    seed_ideas = idea_generation.generate_seed_ideas(
        topic=topic,
        papers=papers,
        client=client,
        model_name=model,
        num_ideas=num_ideas,
        rag_rate=settings.RAG_APPLICATION_RATE,
        num_demo_examples=settings.NUM_DEMO_EXAMPLES
    )
    
    print(f"✓ Generated {len(seed_ideas)} seed ideas")
    
    # Save intermediate
    save_seed_ideas(seed_ideas, os.path.join(run_dir, "intermediate", "seed_ideas.txt"))
    
    # -------------------------------------------------------------------------
    # Step 3: Deduplicate
    # -------------------------------------------------------------------------
    print_banner("Step 3: Deduplicating ideas")
    
    # Use target-based deduplication if target retention is set
    target_retention = getattr(settings, 'TARGET_RETENTION_PERCENT', None)
    if target_retention:
        unique_ideas = deduplication.deduplicate_to_target(
            seed_ideas,
            target_percent=target_retention
        )
    else:
        unique_ideas = deduplication.deduplicate_ideas(
            seed_ideas, 
            threshold=settings.SIMILARITY_THRESHOLD
        )
    
    # Hard cap at configured percentage (minimum 1 idea)
    hard_cap_percent = getattr(settings, 'HARD_CAP_PERCENT', None)
    if hard_cap_percent and len(seed_ideas) > 0:
        max_ideas = max(1, int(len(seed_ideas) * hard_cap_percent))
        if len(unique_ideas) > max_ideas:
            print(f"  Hard cap ({hard_cap_percent*100:.0f}%) applied: {len(unique_ideas)} → {max_ideas}")
            unique_ideas = unique_ideas[:max_ideas]
    
    retention_rate = 100 * len(unique_ideas) / len(seed_ideas) if seed_ideas else 0
    print(f"✓ Reduced to {len(unique_ideas)} unique ideas ({retention_rate:.1f}% retained)")
    
    # Analyze diversity
    diversity_stats = deduplication.analyze_diversity(unique_ideas)
    print(f"  Diversity stats: avg similarity = {diversity_stats.get('average_similarity', 0):.3f}")
    
    # Save intermediate
    save_seed_ideas(unique_ideas, os.path.join(run_dir, "intermediate", "unique_ideas.txt"))
    
    # -------------------------------------------------------------------------
    # Step 4: Expand to Full Proposals
    # -------------------------------------------------------------------------
    print_banner("Step 4: Expanding to full proposals")
    
    full_proposals = []
    for idea in tqdm(unique_ideas, desc="Expanding ideas"):
        try:
            proposal = idea_generation.expand_to_full_proposal(idea, client, model)
            if proposal is not None:
                full_proposals.append(proposal)
        except Exception as e:
            print(f"  Warning: Failed to expand idea '{idea.title[:30]}...': {e}")
    
    print(f"✓ Expanded {len(full_proposals)} proposals")
    
    # Save intermediate
    save_proposals(full_proposals, os.path.join(run_dir, "intermediate", "full_proposals.txt"))
    
    # -------------------------------------------------------------------------
    # Step 5: Filter
    # -------------------------------------------------------------------------
    print_banner("Step 5: Filtering proposals")
    
    filtered_proposals, filter_stats = idea_filtering.filter_proposals(
        full_proposals, 
        papers, 
        client, 
        model
    )
    
    if isinstance(filtered_proposals, tuple):
        filtered_proposals = filtered_proposals[0]
    
    retention_rate = 100 * len(filtered_proposals) / len(full_proposals) if full_proposals else 0
    print(f"✓ Filtered to {len(filtered_proposals)} proposals ({retention_rate:.1f}% kept)")
    
    # Save intermediate
    save_proposals(filtered_proposals, os.path.join(run_dir, "intermediate", "filtered_proposals.txt"))
    
    # -------------------------------------------------------------------------
    # Step 6: Rank
    # -------------------------------------------------------------------------
    print_banner("Step 6: Ranking proposals")
    
    ranked_proposals = idea_ranking.rank_proposals(
        filtered_proposals,
        client,
        model,
        num_rounds=settings.NUM_RANKING_ROUNDS
    )
    
    print(f"✓ Ranking complete ({len(ranked_proposals)} proposals ranked)")
    
    # Show top 5
    print("\nTop 5 proposals:")
    for i, (prop, score) in enumerate(ranked_proposals[:5], 1):
        print(f"  {i}. [{score} pts] {prop.title[:50]}...")
    
    # -------------------------------------------------------------------------
    # Step 7: Normalize Styles
    # -------------------------------------------------------------------------
    print_banner("Step 7: Normalizing styles")
    
    # Only normalize top proposals for efficiency
    top_n = min(50, len(ranked_proposals))
    top_proposals = [p for p, score in ranked_proposals[:top_n]]
    
    template_path = os.path.join(SCRIPT_DIR, "data/demo_examples/full_proposal_example.txt")
    
    if os.path.exists(template_path):
        normalized_proposals = style_normalization.normalize_all_proposals(
            top_proposals,
            template_path,
            client,
            model
        )
    else:
        print(f"  Warning: Template not found at {template_path}")
        print("  Using quick normalization instead...")
        normalized_proposals = style_normalization.normalize_all_quick(
            top_proposals,
            client,
            model
        )
    
    print(f"✓ Normalized {len(normalized_proposals)} proposals")
    
    # -------------------------------------------------------------------------
    # Step 8: Save Results
    # -------------------------------------------------------------------------
    print_banner("Step 8: Saving results")
    
    save_results(normalized_proposals, ranked_proposals, output_dir, topic_key, run_dir)
    
    # Final summary
    print_banner(f"Pipeline Complete for '{topic_key}'")
    print(f"Papers retrieved: {len(papers)}")
    print(f"Seed ideas generated: {len(seed_ideas)}")
    print(f"After deduplication: {len(unique_ideas)}")
    print(f"After filtering: {len(filtered_proposals)}")
    print(f"Final ranked: {len(ranked_proposals)}")
    print(f"Top normalized: {len(normalized_proposals)}")
    print(f"\nResults saved to: {run_dir}")
    
    # -------------------------------------------------------------------------
    # Human-in-the-Loop Feedback
    # -------------------------------------------------------------------------
    human_in_loop = getattr(settings, 'HUMAN_IN_THE_LOOP', False)
    if human_in_loop:
        # Show existing feedback if any
        existing = feedback_utils.load_feedback()
        if existing:
            print(f"\n[Feedback] Existing feedback on file ({len(existing)} chars)")
        
        feedback_utils.collect_feedback_interactive(topic=topic_key)
    
    return normalized_proposals, ranked_proposals


def run_pipeline_lite(
    topic_key: str,
    output_dir: str,
    num_ideas: int = 10,
    model_name: str = None
):
    """
    Run a lightweight version of the pipeline for testing.
    Generates fewer ideas and skips some expensive steps.
    """
    topic = RESEARCH_TOPICS[topic_key]
    model = model_name or settings.OPENAI_MODEL_NAME
    
    client = get_openai_client()
    run_dir = ensure_output_dir(output_dir, topic_key + "_lite")
    
    print_banner("Lite Pipeline (for testing)")
    print(f"Topic: {topic_key}")
    print(f"Ideas: {num_ideas}")
    
    # Step 1: Retrieve papers (fewer)
    print("\n[1/4] Retrieving papers...")
    papers = paper_retrieval.retrieve_papers(
        topic=topic,
        client=client,
        model_name=model,
        target_papers=20,
        max_retrieval_steps=5
    )
    print(f"  ✓ {len(papers)} papers")
    
    # Step 2: Generate ideas
    print("\n[2/4] Generating ideas...")
    seed_ideas = idea_generation.generate_seed_ideas(
        topic=topic,
        papers=papers,
        client=client,
        model_name=model,
        num_ideas=num_ideas,
        rag_rate=0.5,
        num_demo_examples=2
    )
    print(f"  ✓ {len(seed_ideas)} ideas")
    
    # Step 3: Deduplicate
    print("\n[3/4] Deduplicating...")
    unique_ideas = deduplication.deduplicate_ideas(seed_ideas, threshold=0.85)
    print(f"  ✓ {len(unique_ideas)} unique ideas")
    
    # Step 4: Expand top ideas only
    print("\n[4/4] Expanding top ideas...")
    top_ideas = unique_ideas[:5]
    proposals = []
    for idea in top_ideas:
        try:
            proposal = idea_generation.expand_to_full_proposal(idea, client, model)
            proposals.append(proposal)
        except Exception as e:
            print(f"  Warning: {e}")
    
    # Save results
    save_seed_ideas(seed_ideas, os.path.join(run_dir, "seed_ideas.txt"))
    save_proposals(proposals, os.path.join(run_dir, "proposals.txt"))
    
    print(f"\n✓ Lite pipeline complete! Results in: {run_dir}")
    return proposals


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Research Idea Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --topic factuality
  python main.py --topic math --num_ideas 100 --output_dir ./my_outputs
  python main.py --topic bias --lite
  
Available topics: """ + ", ".join(RESEARCH_TOPICS.keys())
    )
    
    parser.add_argument(
        "--topic", 
        type=str, 
        required=True, 
        choices=list(RESEARCH_TOPICS.keys()),
        help="Research topic to generate ideas for"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    parser.add_argument(
        "--num_ideas", 
        type=int, 
        default=None,
        help=f"Number of seed ideas to generate (default: {settings.NUM_SEED_IDEAS})"
    )
    parser.add_argument(
        "--num_papers", 
        type=int, 
        default=None,
        help=f"Number of papers to retrieve (default: {settings.NUM_RETRIEVED_PAPERS})"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Model name to use (default: from settings)"
    )
    parser.add_argument(
        "--lite", 
        action="store_true",
        help="Run lightweight version for testing (10 ideas, skip filtering/ranking)"
    )
    parser.add_argument(
        "--skip_retrieval", 
        action="store_true",
        help="Skip paper retrieval, load from --papers_file"
    )
    parser.add_argument(
        "--papers_file", 
        type=str, 
        default=None,
        help="Path to pre-retrieved papers JSON (use with --skip_retrieval)"
    )
    parser.add_argument(
        "--retrieval_method",
        type=str,
        default=None,
        choices=["llm_guided", "keyword", "tavily"],
        help="Paper retrieval method (default: from settings.RETRIEVAL_METHOD)"
    )
    parser.add_argument(
        "--human_feedback",
        action="store_true",
        help="Enable human-in-the-loop: pause after results to collect feedback"
    )
    parser.add_argument(
        "--clear_feedback",
        action="store_true",
        help="Clear all accumulated human feedback before running"
    )
    
    args = parser.parse_args()
    
    # Handle feedback flags
    if args.human_feedback:
        settings.HUMAN_IN_THE_LOOP = True
    if args.clear_feedback:
        feedback_utils.clear_feedback()
        print("✓ All previous feedback cleared.")
    
    # Show feedback status
    existing_feedback = feedback_utils.load_feedback()
    if existing_feedback:
        print(f"[Feedback] Loaded {len(existing_feedback)} chars of accumulated feedback")
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Run pipeline
    if args.lite:
        run_pipeline_lite(
            topic_key=args.topic,
            output_dir=args.output_dir,
            num_ideas=args.num_ideas or 10,
            model_name=args.model
        )
    else:
        run_pipeline(
            topic_key=args.topic,
            output_dir=args.output_dir,
            num_ideas=args.num_ideas,
            num_papers=args.num_papers,
            skip_retrieval=args.skip_retrieval,
            papers_file=args.papers_file,
            model_name=args.model,
            retrieval_method=args.retrieval_method
        )
