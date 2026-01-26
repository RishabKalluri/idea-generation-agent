"""
Configuration settings for the AI Research Idea Generation Pipeline.
"""

import os

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# Model Settings
MODEL_NAME = "claude-sonnet-4-20250514"  # For Anthropic
OPENAI_MODEL_NAME = "gpt-4"  # For OpenAI (or "gpt-3.5-turbo" for cheaper/faster)

# Idea Generation Parameters
NUM_SEED_IDEAS = 100

# Deduplication Settings
SIMILARITY_THRESHOLD = 0.8

# Paper Retrieval Settings
NUM_RETRIEVED_PAPERS = 120
TOP_K_PAPERS_PER_QUERY = 20

# Prompt Engineering Settings
NUM_DEMO_EXAMPLES = 6
RAG_APPLICATION_RATE = 0.5

# Ranking Settings
NUM_RANKING_ROUNDS = 5
