# Which API Should I Use?

## Quick Answer

| You Have | Use This |
|----------|----------|
| OpenAI API key | ✅ `paper_retrieval_openai.py` + `test_paper_retrieval_openai.py` |
| Anthropic API key | ✅ `paper_retrieval.py` + `test_paper_retrieval.py` |
| Both | Either! They work the same way |
| Neither | Get OpenAI key (easier to obtain) |

## Setup Commands

### If You Have OpenAI Key
```bash
export OPENAI_API_KEY="your-key"
export SEMANTIC_SCHOLAR_API_KEY="your-key"  # Optional
pip3 install --user openai requests
python3 test_paper_retrieval_openai.py
```

### If You Have Anthropic Key
```bash
export ANTHROPIC_API_KEY="your-key"
export SEMANTIC_SCHOLAR_API_KEY="your-key"  # Optional
pip3 install --user anthropic requests
python3 test_paper_retrieval.py
```

## Code Examples

### OpenAI Version
```python
from openai import OpenAI
from modules.paper_retrieval_openai import retrieve_papers

client = OpenAI(api_key="your-key")
papers = retrieve_papers(topic, client, "gpt-4")
```

### Anthropic Version
```python
from anthropic import Anthropic
from modules.paper_retrieval import retrieve_papers

client = Anthropic(api_key="your-key")
papers = retrieve_papers(topic, client, "claude-sonnet-4-20250514")
```

## Comparison

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| **Quality** | GPT-4: Excellent | Claude Sonnet 4: Excellent |
| **Cost** | $1-2 per retrieval | $0.50-1.50 per retrieval |
| **Speed** | Fast | Fast |
| **API Access** | Easy to get | Requires application |
| **Cheaper Option** | GPT-3.5: $0.10-0.20 | No cheaper tier |

## Recommendation

### For Quick Testing (You Have OpenAI Key)
✅ Use **OpenAI** - It's ready to go!

### For Production (You Have Time to Get Keys)
✅ Use **Anthropic** - Slightly cheaper, excellent quality

### For Budget-Conscious (Need Many Retrievals)
✅ Use **GPT-3.5-Turbo** - 10x cheaper (quality trade-off)

## Both Work Great!

The functionality is **identical**. Choose based on what API key you have available. Both produce high-quality results for the paper retrieval task.

## Documentation

- **OPENAI_QUICKSTART.md** - OpenAI-specific guide
- **QUICK_START.md** - General guide (Anthropic-focused)
- **PAPER_RETRIEVAL_GUIDE.md** - Detailed guide (applies to both)
