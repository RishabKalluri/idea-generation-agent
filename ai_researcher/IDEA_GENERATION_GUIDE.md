# Idea Generation Module Guide

## Overview

The idea generation module creates novel research ideas using LLMs with few-shot prompting and optional RAG (Retrieval-Augmented Generation).

## SeedIdea Format

Each generated idea follows this structure:

```
Title: [Concise title for the idea]

Problem: [1-2 sentences describing the problem]

Existing Methods: [1-2 sentences on current approaches and their limitations]

Motivation: [2-3 sentences on why the proposed approach should work better]

Proposed Method: [3-5 sentences describing the key steps of the method]

Experiment Plan: [2-3 sentences on how to evaluate the method]
```

## SeedIdea Dataclass

```python
@dataclass
class SeedIdea:
    title: str
    problem: str
    existing_methods: str
    motivation: str
    proposed_method: str
    experiment_plan: str
    raw_text: str = ""
    rag_used: bool = False
```

## Quick Start

```python
import os
import importlib.util
from openai import OpenAI

# Load module directly (avoids anthropic dependency)
spec = importlib.util.spec_from_file_location('idea_gen', 'modules/idea_generation.py')
idea_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(idea_gen)

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Generate ideas
ideas = idea_gen.generate_seed_ideas(
    topic="prompting methods to reduce LLM hallucinations",
    papers=[],  # Optional: pass papers for RAG
    client=client,
    model_name="gpt-4",
    num_ideas=100,
    rag_rate=0.5
)

print(f"Generated {len(ideas)} ideas")
for idea in ideas[:5]:
    print(f"- {idea.title}")
```

## Main Function

### `generate_seed_ideas()`

```python
generate_seed_ideas(
    topic: str,              # Research topic
    papers: List[Paper],     # Papers for RAG context
    client,                  # OpenAI client
    model_name: str,         # Model name (e.g., "gpt-4")
    num_ideas: int = 4000,   # Number of ideas to generate
    rag_rate: float = 0.5,   # Fraction using RAG (0.0-1.0)
    num_demo_examples: int = 6,  # Demo examples in prompt
    papers_per_rag: int = 10,    # Papers per RAG context
    show_progress: bool = True   # Show progress bar
) -> List[SeedIdea]
```

**Returns:** List of `SeedIdea` objects

## How It Works

### 1. Demo Examples (Few-Shot Learning)

The module loads 6 demonstration examples showing the expected format:
- Located in `data/demo_examples/example_01.txt` through `example_06.txt`
- Based on real papers (Chain-of-Verification, Self-Refine, etc.)
- If no files found, uses built-in examples

### 2. RAG Integration (50% of ideas)

For `rag_rate` fraction of ideas:
- Randomly selects papers from the provided list
- Formats their titles/abstracts as context
- Encourages ideas inspired by (but not copying) the papers

### 3. Diversity Tracking

- Tracks previously generated titles
- Includes last 20 titles in prompt to avoid duplication
- LLM instructed to generate different ideas

### 4. Parsing & Validation

- Parses LLM responses into structured `SeedIdea` objects
- Handles various formatting variations
- Skips unparseable responses

## Demo Examples

Located in `data/demo_examples/`:

| File | Based On | Key Concept |
|------|----------|-------------|
| `example_01.txt` | Chain-of-Verification | Self-verification for factuality |
| `example_02.txt` | Self-Refine | Iterative self-refinement |
| `example_03.txt` | Constitutional AI | Principle-guided generation |
| `example_04.txt` | Tree-of-Thoughts | Parallel reasoning exploration |
| `example_05.txt` | ReAct | Interleaved reasoning & action |
| `example_06.txt` | Adaptive RAG | Query-based retrieval |

## Configuration

Default settings in `config/settings.py`:

```python
NUM_SEED_IDEAS = 4000
RAG_APPLICATION_RATE = 0.5
NUM_DEMO_EXAMPLES = 6
```

## Usage Examples

### Generate Ideas Without RAG

```python
ideas = generate_seed_ideas(
    topic="efficient fine-tuning methods for LLMs",
    papers=[],  # No papers
    client=client,
    model_name="gpt-4",
    num_ideas=100,
    rag_rate=0.0  # No RAG
)
```

### Generate Ideas With RAG

```python
# First retrieve papers
from modules.paper_retrieval_openai import retrieve_papers

papers = retrieve_papers(
    topic="efficient fine-tuning methods",
    client=client,
    model_name="gpt-4"
)

# Then generate ideas with paper context
ideas = generate_seed_ideas(
    topic="efficient fine-tuning methods for LLMs",
    papers=papers,
    client=client,
    model_name="gpt-4",
    num_ideas=100,
    rag_rate=0.5  # 50% use RAG
)
```

### Save Ideas to File

```python
from modules.idea_generation import save_ideas_to_file

save_ideas_to_file(ideas, "generated_ideas.txt")
```

### Load Ideas from File

```python
from modules.idea_generation import load_ideas_from_file

ideas = load_ideas_from_file("generated_ideas.txt")
```

## Cost Estimation

For 4000 ideas:

| Model | Cost | Time |
|-------|------|------|
| GPT-4 | ~$40-60 | ~4-6 hours |
| GPT-4 Turbo | ~$20-30 | ~2-3 hours |
| GPT-3.5-Turbo | ~$4-6 | ~1-2 hours |

## Testing

```bash
export OPENAI_API_KEY="your-key"
cd ai_researcher
python3.9 test_idea_generation.py
```

This generates 5 test ideas and saves them to `test_ideas_output.txt`.

## Output Example

```
[Seed Idea Generation] Starting generation
  Topic: novel prompting methods...
  Target: 100 ideas
  RAG rate: 50%
  Model: gpt-4
  Demo examples: 6
  Papers for RAG: 45

Generating ideas: 100%|████████████| 100/100 [12:34<00:00]

[Seed Idea Generation] Complete!
  Generated: 98 ideas
  Failed: 2 attempts
  With RAG: 49 (50.0%)
```

## Integration with Pipeline

```python
# Full pipeline:
# 1. Retrieve papers
papers = retrieve_papers(topic, client, model)

# 2. Generate seed ideas
ideas = generate_seed_ideas(topic, papers, client, model)

# 3. Deduplicate ideas (future module)
unique_ideas = deduplicate_ideas(ideas)

# 4. Filter ideas (future module)
filtered_ideas = filter_ideas(unique_ideas)

# 5. Rank ideas (future module)
ranked_ideas = rank_ideas(filtered_ideas)
```

## Customization

### Add Custom Demo Examples

Create files in `data/demo_examples/`:
- Name: `example_NN.txt` (e.g., `example_07.txt`)
- Format: Follow the SeedIdea format exactly

### Modify Prompts

Edit the templates in `modules/idea_generation.py`:
- `SEED_IDEA_FORMAT` - Output format
- `SEED_IDEA_PROMPT` - Main generation prompt
- `RAG_SECTION_TEMPLATE` - RAG context format

## Troubleshooting

### "ModuleNotFoundError: No module named 'anthropic'"

Use direct imports:
```python
import importlib.util
spec = importlib.util.spec_from_file_location('idea_gen', 'modules/idea_generation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
```

### Low quality ideas

- Use GPT-4 instead of GPT-3.5
- Add more demo examples
- Use RAG with relevant papers
- Make topic more specific

### Many parsing failures

- Check LLM is following the format
- Review demo examples for clarity
- Increase `max_tokens` if responses are truncated

### Duplicate ideas

- The module tracks previous titles
- Try increasing topic specificity
- RAG helps with diversity

## Summary

The idea generation module:
- ✅ Generates structured research ideas
- ✅ Uses few-shot prompting with 6 demo examples
- ✅ Supports RAG with retrieved papers
- ✅ Tracks previous ideas to encourage diversity
- ✅ Parses responses into structured format
- ✅ Works with OpenAI (no anthropic needed)
