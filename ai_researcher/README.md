# AI Research Idea Generation Pipeline

An automated pipeline for generating, filtering, and ranking novel AI research ideas using LLMs and retrieval-augmented generation (RAG).

## Project Structure

```
ai_researcher/
├── config/
│   ├── __init__.py
│   └── settings.py          # API keys, model settings, hyperparameters
├── modules/
│   ├── __init__.py
│   ├── paper_retrieval.py   # Semantic Scholar RAG
│   ├── idea_generation.py   # Seed idea and full proposal generation
│   ├── deduplication.py     # Embedding-based deduplication
│   ├── idea_filtering.py    # Novelty and feasibility checks
│   ├── idea_ranking.py      # Swiss system tournament ranking
│   └── style_normalization.py # Style standardization
├── prompts/
│   ├── __init__.py
│   └── templates.py         # All LLM prompt templates
├── data/
│   └── demo_examples/       # Demonstration examples for few-shot prompting
├── utils/
│   ├── __init__.py
│   ├── api_client.py        # Wrapper for Claude/OpenAI API calls
│   └── semantic_scholar.py  # Semantic Scholar API wrapper
├── main.py                  # Main orchestration script
├── requirements.txt
└── README.md
```

## Configuration

The pipeline is configured through `config/settings.py` with the following parameters:

### API Keys
- `ANTHROPIC_API_KEY`: Required for Claude API access
- `SEMANTIC_SCHOLAR_API_KEY`: Optional, but recommended for higher rate limits

### Model Settings
- `MODEL_NAME`: "claude-sonnet-4-20250514"

### Pipeline Parameters
- `NUM_SEED_IDEAS`: 4000 - Number of initial seed ideas to generate
- `SIMILARITY_THRESHOLD`: 0.8 - Threshold for deduplication
- `NUM_RETRIEVED_PAPERS`: 120 - Total papers to retrieve
- `TOP_K_PAPERS_PER_QUERY`: 20 - Papers per retrieval query
- `NUM_DEMO_EXAMPLES`: 6 - Examples for few-shot prompting
- `RAG_APPLICATION_RATE`: 0.5 - Rate of applying RAG to proposals
- `NUM_RANKING_ROUNDS`: 5 - Swiss tournament rounds

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- **anthropic**: Claude API client
- **sentence-transformers**: Embeddings (all-MiniLM-L6-v2 model)
- **requests**: Semantic Scholar API calls
- **numpy**: Similarity calculations
- **tqdm**: Progress bars

## Usage

Set your API keys as environment variables:

```bash
export ANTHROPIC_API_KEY="your-key-here"
export SEMANTIC_SCHOLAR_API_KEY="your-key-here"  # Optional
```

Run the pipeline:

```bash
python main.py
```

## Pipeline Stages

1. **Seed Idea Generation**: Generate initial research ideas
2. **Paper Retrieval**: Fetch relevant papers from Semantic Scholar
3. **Full Proposal Generation**: Expand seeds into detailed proposals with RAG
4. **Deduplication**: Remove similar ideas using embeddings
5. **Filtering**: Check novelty and feasibility
6. **Ranking**: Swiss tournament for pairwise comparison
7. **Style Normalization**: Standardize writing style

## Module Descriptions

- **paper_retrieval.py**: Interfaces with Semantic Scholar API to retrieve relevant academic papers
- **idea_generation.py**: Generates both seed ideas and full research proposals using LLMs
- **deduplication.py**: Uses sentence transformers to compute embeddings and remove duplicate ideas
- **idea_filtering.py**: Evaluates ideas for novelty (vs. existing work) and feasibility
- **idea_ranking.py**: Implements Swiss system tournament for ranking ideas through pairwise comparisons
- **style_normalization.py**: Ensures consistent writing style across all generated ideas

## Notes

This is a skeleton structure. Implementation details for each module will be added separately.
