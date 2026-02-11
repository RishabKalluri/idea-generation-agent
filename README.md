# AI Research Idea Generation Pipeline

Automated pipeline that uses LLMs to generate, filter, and rank novel AI research ideas. It retrieves relevant papers, generates seed ideas informed by that literature, deduplicates them, expands the best into full proposals, and ranks them using a tournament system.

## Setup

```bash
cd ai_researcher
conda env create -f environment.yml
conda activate ai-researcher
```

Create a `config/.env` file with your API keys:

```
OPENAI_API_KEY=your-openai-key
SEMANTIC_SCHOLAR_API_KEY=your-semantic-scholar-key   # optional, improves paper retrieval rate limits
TAVILY_API_KEY=your-tavily-key                       # only needed if using tavily retrieval method
```

## Running

Set your topic in the settings!!!!!!!!!!!!

and slightly play around with the other settings. Only worth really messing with seed ideas or some of the other numbers with #.

```bash
cd ai_researcher

# Quick test run (fewer ideas, skips filtering/ranking)
python main.py --topic factuality --lite

# Full pipeline
python main.py --topic factuality

```
More in depth readme in the folder.

Results are saved to `outputs/<topic>/<timestamp>/`.
