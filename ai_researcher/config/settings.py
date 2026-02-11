"""
Configuration settings for the AI Research Idea Generation Pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from config directory
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Model Settings
MODEL_NAME = "claude-sonnet-4-20250514"  # For Anthropic
OPENAI_MODEL_NAME = "gpt-5-mini"  # Fast & cheap - use "gpt-5" for higher quality  

# Research Topic
# Set the topic for idea generation. Can be a key from RESEARCH_TOPICS or a custom description.
# Built-in topic keys: "bias", "coding", "safety", "multilingual", "factuality", "math", "uncertainty"
# Or provide a custom topic description string directly.
RESEARCH_TOPIC = "factuality"

# Custom topic descriptions (used when RESEARCH_TOPIC is a key)
RESEARCH_TOPICS = {
    "bias": "novel prompting methods to reduce social biases and stereotypes of large language models",
    "coding": "novel prompting methods for large language models to improve code generation",
    "safety": "novel prompting methods to improve large language models' robustness against adversarial attacks or improve their security or privacy",
    "multilingual": "novel prompting methods to improve large language models' performance on multilingual tasks or low-resource languages and vernacular languages",
    "factuality": "novel prompting methods that can improve factuality and reduce hallucination of large language models",
    "math": "novel prompting methods for large language models to improve mathematical problem solving",
    "uncertainty": "novel prompting methods that can better quantify uncertainty or calibrate the confidence of large language models"
}

# Idea Generation Parameters
NUM_SEED_IDEAS = 4 #4000

# Deduplication Settings
# Lower threshold = more aggressive deduplication (0.7 for ~5% retention, 0.8 for ~20%)
SIMILARITY_THRESHOLD = 0.7
TARGET_RETENTION_PERCENT = 0.05  # Target 5% retention (4000 -> ~200)
HARD_CAP_PERCENT = 0.75  # Hard cap at 5% - will never keep more than this percentage

# Paper Retrieval Settings
NUM_RETRIEVED_PAPERS = 12 #120
TOP_K_PAPERS_PER_QUERY = 20 #20

# Retrieval Method: Choose which paper retrieval strategy to use
# Options:
#   "llm_guided" - LLM agent iteratively searches Semantic Scholar (default)
#   "keyword" - Simple keyword-based search with predefined queries
#   "tavily" - Tavily search + arXiv metadata + PDF summaries
#   "custom" - Use a custom retrieval function (must register via paper_retrieval.register_retrieval_method)
RETRIEVAL_METHOD = "llm_guided"

# Minimum relevance score for papers (1-10, used by llm_guided and tavily methods)
MIN_PAPER_RELEVANCE_SCORE = 7

# Tavily + arXiv specific settings
TAVILY_MAX_PAPERS = 20
TAVILY_SUMMARIZE_THRESHOLD = 8  # Papers with score >= this get full PDF summaries

# Prompt Engineering Settings
NUM_DEMO_EXAMPLES = 6
RAG_APPLICATION_RATE = 0.5

# Ranking Settings
NUM_RANKING_ROUNDS = 5

# Human-in-the-Loop Settings
# When enabled, the pipeline will pause after generating proposals to collect
# feedback from the user. Feedback is saved locally and injected into all
# future LLM calls to guide output quality.
HUMAN_IN_THE_LOOP = True
HUMAN_FEEDBACK_FILE = os.path.join(Path(__file__).parent, "human_feedback.txt")
