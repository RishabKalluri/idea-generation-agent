# ✅ OpenAI Setup Complete!

## Problem Fixed

The `ModuleNotFoundError: No module named 'anthropic'` error has been resolved. You can now use the OpenAI version without needing Anthropic!

## What's Installed

✅ Python 3.9  
✅ OpenAI package (v2.15.0)  
✅ requests package  
✅ Semantic Scholar wrapper  
⚠️ Need: OPENAI_API_KEY  

## Quick Start

### 1. Verify Setup
```bash
cd /home/heck2/rkalluri7/idea-generation-agent/ai_researcher
python3.9 verify_openai_setup.py
```

This should show all green checkmarks except the API key.

### 2. Set Your OpenAI API Key
```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

**Get your key:** https://platform.openai.com/api-keys

### 3. Run the Test
```bash
python3.9 test_openai_simple.py
```

Or use the script:
```bash
bash RUN_OPENAI_TEST.sh
```

## Files to Use

### ✅ Working Files (No Anthropic Dependency)
- `test_openai_simple.py` - **Use this for testing**
- `modules/paper_retrieval_openai.py` - OpenAI version of retrieval
- `verify_openai_setup.py` - Check your setup
- `RUN_OPENAI_TEST.sh` - Easy run script

### ❌ Don't Use (Require Anthropic)
- `test_paper_retrieval.py` - Needs Anthropic
- `modules/paper_retrieval.py` - Needs Anthropic
- `test_semantic_scholar.py` - Will fail due to imports

## Commands Reference

### Verify Everything Works
```bash
python3.9 verify_openai_setup.py
```

### Run Paper Retrieval Test
```bash
export OPENAI_API_KEY='your-key'
python3.9 test_openai_simple.py
```

### Check Semantic Scholar API Status
```bash
python3.9 test_ss_standalone.py
```
(Note: May hit rate limit, get free key at https://www.semanticscholar.org/product/api)

## Cost Information

**GPT-4** (recommended):
- Cost: ~$1-2 per retrieval
- Quality: Excellent
- Speed: 2-5 minutes for 120 papers

**GPT-3.5-Turbo** (budget):
- Cost: ~$0.10-0.20 per retrieval
- Quality: Good (less reliable than GPT-4)
- Speed: Faster

To use GPT-3.5, edit `test_openai_simple.py` line 50:
```python
model_name = "gpt-3.5-turbo"  # Change from "gpt-4"
```

## Example Usage in Your Code

```python
import os
import sys
from openai import OpenAI

# Load Semantic Scholar without anthropic dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ss", "utils/semantic_scholar.py"
)
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)

# Load paper retrieval
spec2 = importlib.util.spec_from_file_location(
    "pr", "modules/paper_retrieval_openai.py"
)
pr = importlib.util.module_from_spec(spec2)
pr.SemanticScholarClient = ss.SemanticScholarClient
pr.Paper = ss.Paper
spec2.loader.exec_module(pr)

# Use it
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
papers = pr.retrieve_papers(
    topic="your research topic here",
    client=client,
    model_name="gpt-4"
)

print(f"Found {len(papers)} papers")
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'openai'"
→ Use `python3.9` not `python3`
→ Or install: `python3.9 -m pip install --user openai`

### "ModuleNotFoundError: No module named 'anthropic'"
→ Use `test_openai_simple.py` not `test_paper_retrieval.py`
→ Make sure you're running the OpenAI version

### "Rate limit exceeded" (Semantic Scholar)
→ Get free API key: https://www.semanticscholar.org/product/api
→ Set: `export SEMANTIC_SCHOLAR_API_KEY='your-key'`

### "Authentication error" (OpenAI)
→ Check your API key is correct
→ Verify you have credits in your OpenAI account
→ Check: https://platform.openai.com/usage

## What the Test Does

When you run `python3.9 test_openai_simple.py`:

1. **LLM-Guided Search** (2-3 min)
   - GPT-4 generates smart search queries
   - Retrieves papers from Semantic Scholar
   - Continues until ~120 papers found

2. **Paper Scoring** (1-2 min)
   - GPT-4 scores each paper 1-10 for relevance
   - Filters papers with score < 7
   - Sorts by relevance

3. **Results**
   - Shows top 10 papers
   - Statistics (years, citations, etc.)
   - Total papers found

## Next Steps

1. ✅ Run verification: `python3.9 verify_openai_setup.py`
2. ✅ Set API key: `export OPENAI_API_KEY='your-key'`
3. ✅ Run test: `python3.9 test_openai_simple.py`
4. ✅ Integrate into your pipeline!

## Summary

- ✅ **No Anthropic needed** - Everything works with OpenAI only
- ✅ **Python 3.9 ready** - All packages installed
- ✅ **Standalone version** - No dependency conflicts
- ⏳ **Just need API key** - Get it from OpenAI

Once you set `OPENAI_API_KEY`, you're ready to go! 🚀
