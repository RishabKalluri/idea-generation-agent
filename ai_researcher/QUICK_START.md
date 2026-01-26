# Quick Start Guide - Paper Retrieval

## 🚀 Minimal Working Example

```python
from anthropic import Anthropic
from modules.paper_retrieval import retrieve_papers

# Setup
client = Anthropic(api_key="your-anthropic-key")
model = "claude-sonnet-4-20250514"

# Retrieve papers
topic = "chain of thought prompting for reasoning"
papers = retrieve_papers(topic, client, model)

# Use results
for paper in papers[:5]:
    print(f"{paper.title} ({paper.year})")
```

## 📋 Prerequisites

```bash
# Set API keys
export ANTHROPIC_API_KEY="your-anthropic-key"
export SEMANTIC_SCHOLAR_API_KEY="your-ss-key"  # Optional but recommended

# Install dependencies (if not already done)
pip3 install anthropic requests
```

## 🎯 Function Reference

### Main Function

```python
retrieve_papers(
    topic: str,              # Research topic description
    client,                  # Anthropic client instance
    model_name: str,         # Model to use
    target_papers: int = 120,  # Papers to retrieve before scoring
    min_score: int = 7       # Minimum relevance score (1-10)
) -> List[Paper]
```

**Returns:** List of `Paper` objects sorted by relevance (highest first)

### Paper Object

```python
@dataclass
class Paper:
    paper_id: str          # Semantic Scholar ID
    title: str             # Paper title
    abstract: str          # Abstract (may be empty)
    year: int              # Publication year
    citation_count: int    # Number of citations
    authors: List[str]     # Author names
```

### Helper Functions

```python
# Build RAG context from papers
build_rag_context(
    papers: List[Paper],
    max_papers: int = 10
) -> str

# Remove duplicate papers
deduplicate_papers(
    papers: List[Paper]
) -> List[Paper]
```

## 💡 Common Use Cases

### Use Case 1: Basic Retrieval

```python
papers = retrieve_papers(
    "multimodal learning with vision and language",
    client,
    model_name
)
```

### Use Case 2: Higher Quality Threshold

```python
# Get only highly relevant papers (score >= 8)
papers = retrieve_papers(
    topic,
    client,
    model_name,
    min_score=8
)
```

### Use Case 3: More Papers

```python
# Retrieve 200 papers before filtering
papers = retrieve_papers(
    topic,
    client,
    model_name,
    target_papers=200
)
```

### Use Case 4: Build RAG Context

```python
papers = retrieve_papers(topic, client, model_name)
context = build_rag_context(papers, max_papers=10)

# Use in prompt
prompt = f"{context}\n\nGenerate a research idea..."
```

## 🧪 Testing

### Test Semantic Scholar API
```bash
python3 check_api_status.py
```

### Test Full Retrieval
```bash
python3 test_paper_retrieval.py
```

### Run Examples
```bash
python3 USAGE_EXAMPLE.py
```

## 📊 What to Expect

**Retrieval Phase:**
```
[Paper Retrieval] Starting LLM-guided retrieval
  [1] KeywordQuery("...") -> 20 results, 20 new (total: 20)
  [2] KeywordQuery("...") -> 20 results, 18 new (total: 38)
  ...
  [12] KeywordQuery("...") -> 20 results, 5 new (total: 121)
✓ Target reached: 121 papers found
```

**Scoring Phase:**
```
[Paper Scoring] Scoring papers for relevance...
  Scored 10/121 papers...
  Scored 20/121 papers...
  ...
  Scored 121/121 papers ✓
[Paper Scoring] Kept 45 papers with score >= 7
                Average score: 8.1
```

## ⚡ Performance

- **Time:** 2-5 minutes for 120 papers (depends on LLM speed)
- **Cost:** ~$0.50-$1.50 per retrieval (Claude Sonnet 4)
- **API Calls:** 
  - Semantic Scholar: 10-15 calls
  - Anthropic: ~135 calls (15 retrieval + 120 scoring)

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Rate limit (429) | Get Semantic Scholar API key |
| Too few papers | Broaden topic or lower `min_score` |
| Not relevant | Make topic more specific |
| Import errors | Check Python version (3.7+) |

## 📚 Documentation

- **PAPER_RETRIEVAL_GUIDE.md** - Comprehensive guide
- **SEMANTIC_SCHOLAR_API.md** - API wrapper documentation
- **USAGE_EXAMPLE.py** - Code examples
- **test_paper_retrieval.py** - Working test

## 🎓 Pro Tips

1. **Be Specific:** "few-shot prompting for math reasoning" beats "prompting methods"
2. **Use Context:** Build RAG context for better idea generation
3. **Adjust Thresholds:** `min_score=8` for quality, `min_score=6` for quantity
4. **Cache Results:** The system caches Semantic Scholar calls automatically
5. **Check Papers:** Review top 10 papers to verify retrieval quality

## 🔗 Integration

This module integrates with the full pipeline:

```python
# 1. Generate seed idea (future module)
seed = generate_seed_idea(...)

# 2. Retrieve papers
papers = retrieve_papers(seed, client, model)

# 3. Build context
context = build_rag_context(papers)

# 4. Generate full proposal (future module)
proposal = generate_full_proposal(seed, context, client, model)
```

## ✅ Checklist

Before running:
- [ ] Anthropic API key set
- [ ] Semantic Scholar API key set (recommended)
- [ ] Dependencies installed
- [ ] Test scripts run successfully

Ready to retrieve papers? Run:
```bash
python3 test_paper_retrieval.py
```
