"""Modules for AI Research Idea Generation Pipeline."""

from .paper_retrieval import (
    retrieve_papers, 
    build_rag_context, 
    retrieve_diverse_papers,
    deduplicate_papers
)
from .idea_generation import generate_seed_ideas, generate_full_proposal
from .deduplication import compute_embeddings, deduplicate_ideas
from .idea_filtering import check_novelty, check_feasibility, filter_ideas
from .idea_ranking import run_pairwise_comparison, run_swiss_tournament
from .style_normalization import normalize_style, batch_normalize
