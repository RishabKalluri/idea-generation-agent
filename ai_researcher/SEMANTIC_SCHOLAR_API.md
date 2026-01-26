# Semantic Scholar API Wrapper Documentation

## Overview

The Semantic Scholar API wrapper provides a simple, robust interface for retrieving academic papers from the Semantic Scholar database. It includes built-in rate limiting, caching, and error handling.

## Features

- ✅ **Three main query methods**: KeywordQuery, PaperQuery, GetReferences
- ✅ **Rate limiting**: Automatic rate limit management (100 req/5min without API key, 1000 req/5min with key)
- ✅ **Caching**: Avoid duplicate API calls
- ✅ **Error handling**: Graceful handling of missing fields and network errors
- ✅ **Type hints**: Full type annotations for better IDE support
- ✅ **Paper dataclass**: Clean, structured paper data

## Installation

The required dependencies are already in `requirements.txt`:

```bash
pip install requests
```

## Configuration

Set your Semantic Scholar API key (optional but recommended for higher rate limits):

```bash
export SEMANTIC_SCHOLAR_API_KEY="your-api-key-here"
```

Get your API key at: https://www.semanticscholar.org/product/api

## Quick Start

```python
from utils.semantic_scholar import SemanticScholarClient

# Initialize client
client = SemanticScholarClient()

# Search for papers
papers = client.KeywordQuery("transformers attention mechanism", limit=20)

# Get detailed paper info
paper = client.PaperQuery("paper_id_here")

# Get paper references
references = client.GetReferences("paper_id_here", limit=20)
```

## Paper Dataclass

All methods return `Paper` objects with the following structure:

```python
@dataclass
class Paper:
    paper_id: str          # Semantic Scholar paper ID
    title: str             # Paper title
    abstract: str          # Abstract (may be empty)
    year: int              # Publication year
    citation_count: int    # Number of citations
    authors: List[str]     # List of author names
```

## API Methods

### 1. KeywordQuery(keywords: str, limit: int = 20) -> List[Paper]

Search for papers using keywords.

**Parameters:**
- `keywords` (str): Search query string
- `limit` (int): Number of results to return (default: 20, max: 100)

**Returns:**
- `List[Paper]`: List of Paper objects matching the query

**Example:**

```python
client = SemanticScholarClient()

# Basic search
papers = client.KeywordQuery("machine learning")

# Limit results
papers = client.KeywordQuery("neural networks", limit=10)

# Print results
for paper in papers:
    print(f"{paper.title} ({paper.year})")
    print(f"Citations: {paper.citation_count}")
    print(f"Authors: {', '.join(paper.authors)}")
    print()
```

### 2. PaperQuery(paper_id: str) -> Optional[Paper]

Get detailed information for a specific paper.

**Parameters:**
- `paper_id` (str): Semantic Scholar paper ID

**Returns:**
- `Paper`: Paper object or None if not found

**Example:**

```python
client = SemanticScholarClient()

# Get paper by ID
paper = client.PaperQuery("649def34f8be52c8b66281af98ae884c09aef38b")

if paper:
    print(f"Title: {paper.title}")
    print(f"Abstract: {paper.abstract}")
    print(f"Citations: {paper.citation_count}")
```

### 3. GetReferences(paper_id: str, limit: int = 20) -> List[Paper]

Get the references cited by a paper.

**Parameters:**
- `paper_id` (str): Semantic Scholar paper ID
- `limit` (int): Number of references to return (default: 20)

**Returns:**
- `List[Paper]`: List of Paper objects representing cited papers

**Example:**

```python
client = SemanticScholarClient()

# Get references
references = client.GetReferences("649def34f8be52c8b66281af98ae884c09aef38b", limit=10)

print(f"Found {len(references)} references:")
for ref in references:
    print(f"- {ref.title} ({ref.year})")
```

## Rate Limiting

The wrapper automatically handles rate limiting:

- **Without API key**: 100 requests per 5 minutes
- **With API key**: 1000 requests per 5 minutes (configurable)

When the rate limit is reached, the wrapper will automatically wait before making the next request.

```python
client = SemanticScholarClient()

# The wrapper handles rate limiting automatically
for query in ["AI", "ML", "NLP", ...]:  # Many queries
    papers = client.KeywordQuery(query)
    # Automatic waiting if rate limit reached
```

## Caching

The wrapper caches all API responses to avoid duplicate requests:

```python
client = SemanticScholarClient()

# First call hits the API
papers1 = client.KeywordQuery("deep learning")

# Second call uses cached result (instant)
papers2 = client.KeywordQuery("deep learning")

# Check cache size
print(f"Cached requests: {client.get_cache_size()}")

# Clear cache if needed
client.clear_cache()
```

## Error Handling

The wrapper handles common errors gracefully:

### Missing Fields

Some papers may not have all fields (especially abstracts):

```python
paper = client.PaperQuery("some_id")
if paper:
    if paper.abstract:
        print(f"Abstract: {paper.abstract}")
    else:
        print("Abstract not available")
```

### Network Errors

Network errors are caught and logged:

```python
papers = client.KeywordQuery("query")
# Returns empty list on error, prints error message
if not papers:
    print("No papers found or error occurred")
```

## Integration with Paper Retrieval Module

The wrapper is integrated with the `modules/paper_retrieval.py` module:

```python
from modules import retrieve_papers, build_rag_context

# Retrieve papers
papers = retrieve_papers("transformers", num_papers=20)

# Build RAG context for LLM
context = build_rag_context(papers, max_papers=10)
print(context)
```

## Advanced Usage

### Multiple Queries for Diverse Coverage

```python
from modules import retrieve_diverse_papers, deduplicate_papers

queries = [
    "transformers attention mechanism",
    "BERT language models",
    "GPT autoregressive models"
]

# Retrieve papers from multiple queries
all_papers = retrieve_diverse_papers(queries, papers_per_query=20)

# Remove duplicates
unique_papers = deduplicate_papers(all_papers)

print(f"Retrieved {len(unique_papers)} unique papers")
```

### Filtering by Citation Count

```python
papers = client.KeywordQuery("machine learning", limit=50)

# Filter highly cited papers
highly_cited = [p for p in papers if p.citation_count > 100]

# Sort by citation count
sorted_papers = sorted(papers, key=lambda p: p.citation_count, reverse=True)

print(f"Top paper: {sorted_papers[0].title}")
print(f"Citations: {sorted_papers[0].citation_count}")
```

### Recent Papers Only

```python
import datetime

papers = client.KeywordQuery("large language models", limit=50)

# Filter papers from last 3 years
current_year = datetime.datetime.now().year
recent_papers = [p for p in papers if p.year >= current_year - 3]

print(f"Found {len(recent_papers)} recent papers")
```

## Testing

Run the example test script:

```bash
cd ai_researcher
python test_semantic_scholar.py
```

This will demonstrate all three API methods with real queries.

## API Endpoints Reference

The wrapper uses the following Semantic Scholar API endpoints:

1. **Paper Search**: `GET /graph/v1/paper/search`
   - Parameters: `query`, `limit`, `fields`

2. **Paper Details**: `GET /graph/v1/paper/{paper_id}`
   - Parameters: `fields`

3. **Paper References**: `GET /graph/v1/paper/{paper_id}/references`
   - Parameters: `limit`, `fields`

## Troubleshooting

### Rate Limit Errors

If you're hitting rate limits frequently:
1. Get an API key from Semantic Scholar (increases limit to 1000 req/5min)
2. Reduce the number of queries
3. Use caching effectively

### Empty Results

If queries return empty results:
1. Check your query syntax
2. Try broader search terms
3. Verify internet connectivity
4. Check if the API is accessible

### Missing Abstracts

Some papers don't have abstracts in Semantic Scholar:
- This is normal and handled gracefully
- Check `if paper.abstract:` before using

## Best Practices

1. **Use API key**: Always set `SEMANTIC_SCHOLAR_API_KEY` for production use
2. **Cache results**: Reuse client instances to benefit from caching
3. **Handle None**: Always check if `PaperQuery` returns None
4. **Check fields**: Verify abstract and other fields exist before using
5. **Limit queries**: Use appropriate `limit` values to avoid unnecessary data transfer

## Support

- Semantic Scholar API Documentation: https://api.semanticscholar.org/
- API Status: https://status.semanticscholar.org/

## License

This wrapper is part of the AI Research Idea Generation Pipeline.
