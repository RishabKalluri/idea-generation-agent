# AI Research Idea Generation Pipeline

An automated pipeline for generating, filtering, and ranking AI research ideas using LLMs. Based on the methodology from "Can LLMs Generate Novel Research Ideas?"

## Quick Start

```bash
# 1. Setup conda environment
export PATH="/opt/conda/bin:$PATH"
conda env create -f environment.yml
conda activate ai-researcher

# 2. Set your API key
export OPENAI_API_KEY='your-key-here'

# 3. Run a test
python main.py --topic factuality --lite

# 4. Run full pipeline
python main.py --topic factuality
```

## Project Structure

```
ai_researcher/
├── main.py                 # Main entry point
├── environment.yml         # Conda environment (GPU-enabled)
├── requirements.txt        # Pip dependencies (alternative)
├── setup_env.sh           # Setup helper script
│
├── config/
│   └── settings.py        # Configuration & hyperparameters
│
├── modules/
│   ├── paper_retrieval.py     # Semantic Scholar RAG
│   ├── idea_generation.py     # Seed ideas & proposal expansion
│   ├── deduplication.py       # Embedding-based deduplication
│   ├── idea_filtering.py      # Novelty & feasibility checks
│   ├── idea_ranking.py        # Swiss tournament ranking
│   └── style_normalization.py # Consistent formatting
│
├── utils/
│   ├── semantic_scholar.py    # Semantic Scholar API client
│   └── api_client.py          # LLM API utilities
│
├── data/demo_examples/        # Few-shot examples for prompting
│
├── prompts/
│   └── templates.py           # Prompt templates
│
└── tests/
    └── test_pipeline.py       # Comprehensive tests
```

## Usage

### Command Line

```bash
# Lite mode (quick test, ~$1-2)
python main.py --topic factuality --lite

# Full pipeline (~$50-80 with gpt-4o)
python main.py --topic factuality

# Custom parameters
python main.py --topic math --num_ideas 100 --output_dir ./results

# Skip paper retrieval (use cached papers)
python main.py --topic coding --skip_retrieval --papers_file papers.json

# Use a different retrieval method
python main.py --topic factuality --retrieval_method tavily

# Enable human-in-the-loop feedback
python main.py --topic factuality --human_feedback

# Clear accumulated feedback before running
python main.py --topic factuality --clear_feedback
```

### CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--topic` | string | *required* | Research topic (`bias`, `coding`, `safety`, `multilingual`, `factuality`, `math`, `uncertainty`) |
| `--output_dir` | string | `outputs` | Directory to save results |
| `--num_ideas` | int | from settings | Number of seed ideas to generate |
| `--num_papers` | int | from settings | Number of papers to retrieve |
| `--model` | string | from settings | Override the LLM model name |
| `--lite` | flag | off | Quick test run (10 ideas, skips filtering/ranking) |
| `--skip_retrieval` | flag | off | Skip paper retrieval, load from `--papers_file` instead |
| `--papers_file` | string | none | Path to pre-retrieved papers JSON (use with `--skip_retrieval`) |
| `--retrieval_method` | string | from settings | Paper retrieval strategy: `llm_guided`, `keyword`, or `tavily` |
| `--human_feedback` | flag | off | Pause after results to collect human feedback |
| `--clear_feedback` | flag | off | Clear all accumulated human feedback before running |

### Available Topics

| Topic | Description |
|-------|-------------|
| `bias` | Reduce social biases and stereotypes |
| `coding` | Improve code generation |
| `safety` | Improve robustness, security, privacy |
| `multilingual` | Multilingual/low-resource languages |
| `factuality` | Reduce hallucination |
| `math` | Mathematical problem solving |
| `uncertainty` | Quantify uncertainty/calibration |

### Pipeline Stages

1. **Paper Retrieval** - RAG from Semantic Scholar (~120 papers)
2. **Seed Idea Generation** - Generate 4000 ideas with few-shot prompting
3. **Deduplication** - Embedding-based duplicate removal (~200 unique)
4. **Proposal Expansion** - Expand to detailed proposals
5. **Filtering** - Novelty and feasibility checks
6. **Ranking** - Swiss tournament pairwise comparison
7. **Style Normalization** - Consistent academic formatting
8. **Save Results** - Timestamped output directory

## Configuration

Edit `config/settings.py`:

```python
# API Keys (use environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")  # Optional

# Model
OPENAI_MODEL_NAME = "gpt-4o"  # or "gpt-4o-mini" for cheaper runs

# Hyperparameters
NUM_SEED_IDEAS = 4000          # Ideas to generate
SIMILARITY_THRESHOLD = 0.8     # Deduplication threshold
NUM_RETRIEVED_PAPERS = 120     # Papers for RAG
RAG_APPLICATION_RATE = 0.5     # % of ideas using RAG
NUM_RANKING_ROUNDS = 5         # Swiss tournament rounds
```

## Output

Results are saved to `outputs/<topic>/<timestamp>/`:

```
outputs/factuality/20260127_143052/
├── summary.json                    # Run metadata
├── rankings.txt                    # Full ranked list
├── top_proposals_normalized.txt    # Top 50 formatted proposals
└── intermediate/
    ├── papers.json                 # Retrieved papers
    ├── seed_ideas.txt              # All generated ideas
    ├── unique_ideas.txt            # After deduplication
    ├── full_proposals.txt          # Expanded proposals
    └── filtered_proposals.txt      # After filtering
```

## Testing

```bash
# Run all tests
python tests/test_pipeline.py

# With pytest (more verbose)
pytest tests/test_pipeline.py -v
```

## Cost Estimates

| Mode | Ideas | Tokens | GPT-4o Cost |
|------|-------|--------|-------------|
| Lite | 10 | ~50K | ~$1-2 |
| Medium | 100 | ~500K | ~$5-10 |
| Full | 4000 | ~10M | ~$50-80 |

## GPU Cluster (SLURM)

```bash
# Create job script
cat > run_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=ai-researcher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

source /opt/conda/etc/profile.d/conda.sh
conda activate ai-researcher
export OPENAI_API_KEY='your-key'

python main.py --topic factuality
EOF

# Submit
sbatch run_job.sh
```

## API Keys

### OpenAI (Required)
Get from: https://platform.openai.com/api-keys

```bash
export OPENAI_API_KEY='sk-...'
```

### Semantic Scholar (Optional but Recommended)
Get from: https://www.semanticscholar.org/product/api

```bash
export SEMANTIC_SCHOLAR_API_KEY='...'
```

Without an API key, you're limited to ~100 requests/5 minutes.

## License

MIT
