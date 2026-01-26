# Paper Retrieval Module Guide

## Overview

The paper retrieval module uses an **LLM-guided search** to intelligently find relevant papers from Semantic Scholar. Instead of simple keyword searches, it uses Claude to iteratively explore the research space, following promising leads and trying different search strategies.

## How It Works

### 1. LLM as Research Assistant

The LLM acts as a research assistant that can:
- Generate diverse keyword searches
- Follow references of relevant papers
- Explore different sub-aspects of a topic
- Adapt its search strategy based on what it finds

### 2. Tool Access

The LLM has access to three Semantic Scholar API functions:
```python
KeywordQuery(keywords: str)      # Search by keywords (returns up to 20 papers)
PaperQuery(paper_id: str)        # Get details about a specific paper
GetReferences(paper_id: str)     # Get papers cited by a paper (returns up to 20)
```

### 3. Iterative Process

```
1. LLM suggests next search action
2. Execute API call, get results
3. Add new papers to collection
4. Show LLM the results and progress
5. Repeat until ~120 papers found
```

### 4. Relevance Scoring

After retrieval, each paper is scored 1-10 by the LLM based on:
- Direct relevance to the topic
- Empirical/computational focus (not surveys/position papers)
- Potential to inspire new research

Only papers scoring ≥7 are kept.

## Usage

### Basic Usage

```python
from anthropic import Anthropic
from modules.paper_retrieval import retrieve_papers
from config import ANTHROPIC_API_KEY, MODEL_NAME

# Initialize client
client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Define your research topic
topic = ("novel prompting methods that can improve factuality and "
         "reduce hallucination of large language models")

# Retrieve papers
papers = retrieve_papers(
    topic=topic,
    client=client,
    model_name=MODEL_NAME,
    target_papers=120,    # Target number to retrieve
    min_score=7           # Minimum relevance score
)

# Use the papers
print(f"Found {len(papers)} relevant papers")
for paper in papers[:5]:
    print(f"- {paper.title} ({paper.year})")
```

### Parameters

**`retrieve_papers(topic, client, model_name, target_papers=120, min_score=7)`**

- `topic` (str): Research topic description. Be specific!
- `client`: Anthropic client instance
- `model_name` (str): Claude model to use (e.g., "claude-sonnet-4-20250514")
- `target_papers` (int): Target number of papers to retrieve before scoring
- `min_score` (int): Minimum relevance score (1-10) to keep papers

**Returns:** List of `Paper` objects, sorted by relevance score (highest first)

## Example Output

```
[Paper Retrieval] Starting LLM-guided retrieval for topic:
  novel prompting methods that can improve factuality and reduce hallucination of large language models
  Target: 120 papers

  [1] KeywordQuery("chain of thought prompting factuality")... -> 20 results, 20 new (total: 20)
  [2] KeywordQuery("hallucination reduction LLM prompting")... -> 20 results, 18 new (total: 38)
  [3] GetReferences("204e3073870fae3d05bcbc2f6a8e263d9b72e776")... -> 20 results, 15 new (total: 53)
  [4] KeywordQuery("self-consistency prompting accuracy")... -> 20 results, 12 new (total: 65)
  ...
  [12] KeywordQuery("fact verification language models")... -> 20 results, 8 new (total: 122)

✓ Target reached: 122 papers found

[Paper Scoring] Scoring papers for relevance...
  Scored 10/122 papers...
  Scored 20/122 papers...
  ...
  Scored 122/122 papers ✓

[Paper Scoring] Kept 45 papers with score >= 7
                Average score: 8.1
```

## The Retrieval Process

### Phase 1: Exploration (Iterations 1-5)

The LLM typically starts with broad keyword searches:
```
KeywordQuery("prompting methods factuality")
KeywordQuery("hallucination reduction LLM")
KeywordQuery("chain of thought reasoning")
```

### Phase 2: Following Leads (Iterations 6-15)

As it finds relevant papers, it explores their references:
```
GetReferences("highly_relevant_paper_id")
KeywordQuery("specific technique mentioned in papers")
```

### Phase 3: Filling Gaps (Iterations 15+)

Targeted searches to reach the target count:
```
KeywordQuery("related subtopic")
KeywordQuery("alternative approaches")
```

## Paper Scoring Criteria

The LLM scores each paper using this rubric:

**Score 9-10:** Directly addresses the topic, empirical work, highly inspiring
**Score 7-8:** Relevant and useful, empirical, good inspiration potential
**Score 5-6:** Somewhat relevant, may be survey/position paper
**Score 3-4:** Tangentially related, limited applicability
**Score 1-2:** Not relevant or not empirical

## Building RAG Context

Once you have papers, build context for LLM prompting:

```python
from modules.paper_retrieval import build_rag_context

# Build context from top papers
context = build_rag_context(papers, max_papers=10)

# Use in your prompts
prompt = f"""
{context}

Based on these papers, generate a novel research idea...
"""
```

## Tips for Good Results

### 1. Write Specific Topics

✅ Good:
```python
topic = "few-shot prompting techniques that improve reasoning in math word problems"
```

❌ Too vague:
```python
topic = "AI and machine learning"
```

### 2. Include Key Aspects

✅ Good:
```python
topic = ("methods to reduce catastrophic forgetting in continual learning "
         "for vision transformers")
```

This gives the LLM clear search directions:
- "catastrophic forgetting"
- "continual learning"
- "vision transformers"

### 3. Balance Specificity

Too specific → Not enough papers found
Too broad → Too many irrelevant papers

Aim for a topic that would yield 200-500 papers on Google Scholar.

## Cost Estimation

For 120 papers with scoring:

**API Calls:**
- Retrieval: ~10-15 iterations × 1 LLM call = 15 LLM calls
- Scoring: 120 papers × 1 LLM call = 120 LLM calls
- **Total: ~135 LLM calls**

**Cost (Claude Sonnet 4):**
- Input: ~100-200 tokens per call
- Output: ~20-50 tokens per call
- **Estimated cost: $0.50-$1.50 per retrieval**

**Semantic Scholar API:**
- ~10-15 API calls (within free tier)

## Troubleshooting

### "Rate limit exceeded" from Semantic Scholar

**Solution:**
```bash
export SEMANTIC_SCHOLAR_API_KEY="your-key"
```
Get a key at: https://www.semanticscholar.org/product/api

### Too few papers found

**Possible causes:**
- Topic too specific
- LLM being too conservative

**Solutions:**
- Broaden your topic
- Increase `target_papers` parameter
- Lower `min_score` threshold

### Papers not relevant enough

**Possible causes:**
- Topic description unclear
- Scoring threshold too low

**Solutions:**
- Refine topic description to be more specific
- Increase `min_score` parameter
- Review top papers and adjust topic wording

### LLM keeps searching the same terms

This is rare but can happen. The retrieval agent tracks history to avoid this, but if it occurs:
- The topic might be too narrow
- Try rephrasing the topic

## Advanced Usage

### Custom Target and Scoring

```python
# Get more papers, but be more selective
papers = retrieve_papers(
    topic=topic,
    client=client,
    model_name=MODEL_NAME,
    target_papers=200,    # Retrieve more
    min_score=8           # Higher threshold
)
```

### Multiple Topics

```python
topics = [
    "prompt engineering for code generation",
    "in-context learning for program synthesis",
    "few-shot learning for software development"
]

all_papers = []
for topic in topics:
    papers = retrieve_papers(topic, client, model_name)
    all_papers.extend(papers)

# Deduplicate
from modules.paper_retrieval import deduplicate_papers
unique_papers = deduplicate_papers(all_papers)
```

## Testing

Run the test script to see it in action:

```bash
cd ai_researcher

# Set up API keys
export ANTHROPIC_API_KEY="your-anthropic-key"
export SEMANTIC_SCHOLAR_API_KEY="your-ss-key"  # Optional but recommended

# Run test
python3 test_paper_retrieval.py
```

This will retrieve papers on a sample topic and show detailed statistics.

## Integration with Pipeline

In the full idea generation pipeline:

```python
# 1. Generate seed idea
seed_idea = generate_seed_idea(...)

# 2. Retrieve relevant papers
papers = retrieve_papers(
    topic=seed_idea,
    client=client,
    model_name=model_name
)

# 3. Build RAG context
context = build_rag_context(papers, max_papers=10)

# 4. Generate full proposal with context
proposal = generate_full_proposal(
    seed_idea=seed_idea,
    rag_context=context,
    client=client,
    model_name=model_name
)
```

## Why LLM-Guided Retrieval?

### vs. Simple Keyword Search
- **Adaptive:** Adjusts strategy based on results
- **Diverse:** Explores multiple angles
- **Intelligent:** Follows promising leads

### vs. Static Query List
- **Dynamic:** Responds to what it finds
- **Efficient:** Stops when target reached
- **Contextual:** Uses previous results to inform next queries

### vs. Embedding-Based Search
- **Actionable:** Uses actual API tools available
- **Transparent:** Can see what searches were tried
- **Controllable:** Easy to adjust via prompt

## Summary

The LLM-guided paper retrieval system:
1. ✅ Uses Claude to intelligently search Semantic Scholar
2. ✅ Retrieves ~120 papers through iterative exploration
3. ✅ Scores papers for relevance using LLM evaluation
4. ✅ Returns only high-quality, relevant papers
5. ✅ Provides transparency into the search process

This creates a high-quality paper corpus for RAG-based idea generation!
