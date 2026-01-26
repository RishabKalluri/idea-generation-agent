# Setup Complete! ✅

## Your System

- ✅ **Python 3.9** installed and ready
- ✅ **OpenAI package** installed (v2.15.0)
- ✅ **requests** package installed
- ⚠️ Need to set **OPENAI_API_KEY**

## Next Steps

### 1. Set Your OpenAI API Key

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

**Where to get it:**
- Go to: https://platform.openai.com/api-keys
- Create a new API key
- Copy it and paste into the command above

### 2. (Optional) Set Semantic Scholar API Key

```bash
export SEMANTIC_SCHOLAR_API_KEY='your-ss-key'
```

**Why?** Higher rate limits (1000 vs 100 requests per 5 min)

**Where to get it:**
- Go to: https://www.semanticscholar.org/product/api
- Sign up and get your key

### 3. Run the Test

**Easy way (using the script):**
```bash
cd /home/heck2/rkalluri7/idea-generation-agent/ai_researcher
bash RUN_OPENAI_TEST.sh
```

**Manual way:**
```bash
cd /home/heck2/rkalluri7/idea-generation-agent/ai_researcher
python3.9 test_paper_retrieval_openai.py
```

## Important: Use Python 3.9

⚠️ **Always use `python3.9`** (not `python3`)
- Python 3.6 is too old for OpenAI library
- Python 3.9 has everything installed

## Quick Test

To verify everything is working:

```bash
# Test 1: Check OpenAI is installed
python3.9 -c "from openai import OpenAI; print('✓ OpenAI ready')"

# Test 2: Check Semantic Scholar API
python3.9 check_api_status.py

# Test 3: Full retrieval test (needs API key)
export OPENAI_API_KEY='your-key'
python3.9 test_paper_retrieval_openai.py
```

## Cost Information

**GPT-4:**
- Best quality
- ~$1-2 per retrieval (120 papers)

**GPT-3.5-Turbo:**
- Good enough for testing
- ~$0.10-0.20 per retrieval
- 10x cheaper

To use GPT-3.5-Turbo instead, edit the test file:
```python
# Change this line:
model_name = "gpt-4"

# To this:
model_name = "gpt-3.5-turbo"
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'openai'"
→ Use `python3.9` instead of `python3`

### "Rate limit exceeded" from Semantic Scholar
→ Get a Semantic Scholar API key (free)

### "OpenAI API error"
→ Check your API key is set correctly
→ Verify you have credits in your OpenAI account

## Your Files

All ready to use:
- `test_paper_retrieval_openai.py` - Test script
- `modules/paper_retrieval_openai.py` - Main module
- `RUN_OPENAI_TEST.sh` - Easy run script
- `OPENAI_QUICKSTART.md` - Complete guide

## Example Usage in Your Code

```python
from openai import OpenAI
from modules.paper_retrieval_openai import retrieve_papers

# Setup
client = OpenAI(api_key="your-key")

# Retrieve papers
topic = "prompt engineering for code generation"
papers = retrieve_papers(topic, client, "gpt-4")

# Use results
for paper in papers[:10]:
    print(f"{paper.title} ({paper.year})")
```

## Ready to Go! 🚀

Once you set your OPENAI_API_KEY:

```bash
export OPENAI_API_KEY='sk-...'
bash RUN_OPENAI_TEST.sh
```

This will retrieve papers on a sample topic and show you how it works!
