"""
Configuration settings for the ORACL dataset generation pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from ai_researcher config (shared API keys)
env_path = Path(__file__).parent.parent.parent / "ai_researcher" / "config" / ".env"
load_dotenv(env_path)

# Also load local .env if it exists
local_env = Path(__file__).parent / ".env"
if local_env.exists():
    load_dotenv(local_env, override=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model
OPENAI_MODEL_NAME = "gpt-5-mini"

# ============================================================================
# arXiv Scraping Settings
# ============================================================================

# Conference categories on arXiv
# Map conference names to arXiv category codes
CONFERENCE_CATEGORIES = {
    "ACL": "cs.CL",
    "EMNLP": "cs.CL",
    "NAACL": "cs.CL",
    "NeurIPS": "cs.LG",
    "ICML": "cs.LG",
    "ICLR": "cs.LG",
    "AAAI": "cs.AI",
    "CVPR": "cs.CV",
    "ICCV": "cs.CV",
    "ECCV": "cs.CV",
}

# Default scraping parameters
DEFAULT_CONFERENCE = "ACL"
DEFAULT_YEAR = 2024
DEFAULT_MONTH_CUTOFF = 1  # Papers from January onward
MAX_PAPERS_PER_QUERY = 5  # arXiv API batch size
MAX_TOTAL_PAPERS = 5  # Total papers to fetch per run

# ============================================================================
# Conversion Settings
# ============================================================================

# Max characters of paper content to send to LLM
MAX_PAPER_CONTENT_LENGTH = 20000

# Concurrency for LLM calls
BATCH_SIZE = 5

# ============================================================================
# Storage Settings
# ============================================================================

# Where to store the dataset
DATASET_DIR = os.path.join(Path(__file__).parent.parent, "dataset")

# ============================================================================
# Grading Settings
# ============================================================================

# Model for pairwise grading (uses Responses API with web search)
GRADING_MODEL_NAME = "gpt-5.2"

# Maximum number of pairs to evaluate (None = all pairs)
# For N proposals, total pairs = N*(N-1)/2.  Set a cap for large datasets.
MAX_GRADING_PAIRS = None

# Where to save grading outputs
GRADING_OUTPUT_DIR = os.path.join(Path(__file__).parent.parent, "grading_results")

# ============================================================================
# OpenReview Settings
# ============================================================================

# Target ratio of accepted papers when sampling from OpenReview.
# 0.5 = balanced (50% accepted, 50% rejected).
OPENREVIEW_ACCEPTED_RATIO = 0.5

# Conferences that are available on OpenReview (not on CMT).
# CVPR/ICCV/ECCV use CMT and are NOT supported for acceptance lookup.
OPENREVIEW_CONFERENCES = {
    "ICLR", "NeurIPS", "ICML", "AAAI",
    "ACL", "EMNLP", "NAACL",
}
