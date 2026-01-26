# Quick Start with OpenAI

## ✅ Yes, OpenAI Works!

You can use your **OpenAI API key** instead of Anthropic. I've created OpenAI-compatible versions of all the modules.

## 🚀 Quick Test

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-key-here"

# Optional but recommended
export SEMANTIC_SCHOLAR_API_KEY="your-ss-key"

# Install OpenAI package
pip3 install --user openai

# Run the test
cd ai_researcher
python3 test_paper_retrieval_openai.py
```

## 📝 Code Example

```python
from openai import OpenAI
from modules.paper_retrieval_openai import retrieve_papers

# Initialize OpenAI client
client = OpenAI(api_key="your-key")

# Choose model
model = "gpt-4"  # or "gpt-3.5-turbo" for faster/cheaper

# Retrieve papers
topic = "chain of thought prompting for reasoning"
papers = retrieve_papers(topic, client, model)

# Use the papers
print(f"Found {len(papers)} papers")
for paper in papers[:5]:
    print(f"- {paper.title} ({paper.year})")
```

## 🤖 Model Options

### GPT-4 (Recommended)
- **Best quality** - More reliable at following instructions
- **Cost**: ~$1-2 per retrieval (120 papers)
- **Speed**: Moderate
- **Use**: `model="gpt-4"`

### GPT-4-Turbo
- **Good quality** - Faster than GPT-4
- **Cost**: ~$0.50-1 per retrieval
- **Speed**: Fast
- **Use**: `model="gpt-4-turbo-preview"`

### GPT-3.5-Turbo
- **Lower cost** - 10x cheaper
- **Cost**: ~$0.10-0.20 per retrieval
- **Speed**: Very fast
- **Note**: Less reliable, may need more iterations
- **Use**: `model="gpt-3.5-turbo"`

## 📊 Cost Comparison

| Provider | Model | Cost per Retrieval | Quality |
|----------|-------|-------------------|---------|
| Anthropic | Claude Sonnet 4 | $0.50-1.50 | ⭐⭐⭐⭐⭐ |
| OpenAI | GPT-4 | $1.00-2.00 | ⭐⭐⭐⭐⭐ |
| OpenAI | GPT-4 Turbo | $0.50-1.00 | ⭐⭐⭐⭐ |
| OpenAI | GPT-3.5-Turbo | $0.10-0.20 | ⭐⭐⭐ |

*Estimates for retrieving 120 papers with scoring*

## 📁 Files to Use

### For OpenAI
- `modules/paper_retrieval_openai.py` - Main module
- `test_paper_retrieval_openai.py` - Test script
- Use `from openai import OpenAI`

### For Anthropic (original)
- `modules/paper_retrieval.py` - Main module
- `test_paper_retrieval.py` - Test script
- Use `from anthropic import Anthropic`

## 🔄 Switching Between APIs

You can use both! Just import the appropriate version:

```python
# Using OpenAI
from openai import OpenAI
from modules.paper_retrieval_openai import retrieve_papers

client = OpenAI(api_key=openai_key)
papers = retrieve_papers(topic, client, "gpt-4")

# Using Anthropic
from anthropic import Anthropic
from modules.paper_retrieval import retrieve_papers

client = Anthropic(api_key=anthropic_key)
papers = retrieve_papers(topic, client, "claude-sonnet-4-20250514")
```

## ⚠️ Known Differences

### OpenAI-Specific Considerations

1. **GPT-3.5-Turbo** may occasionally:
   - Generate invalid function calls
   - Be less consistent in paper scoring
   - Require more iterations

2. **GPT-4** is recommended for:
   - Production use
   - When you need high-quality results
   - Complex research topics

3. **Rate Limits**:
   - OpenAI free tier: Limited requests per minute
   - Paid tier: Higher limits
   - Consider adding delays if you hit limits

## 🧪 Testing Your Setup

### Step 1: Check OpenAI Connection
```bash
python3 -c "from openai import OpenAI; client = OpenAI(); print('✓ OpenAI installed')"
```

### Step 2: Test Semantic Scholar
```bash
python3 check_api_status.py
```

### Step 3: Full Test
```bash
python3 test_paper_retrieval_openai.py
```

## 💡 Tips for Best Results

1. **Use GPT-4 for important work**
   - More reliable instruction following
   - Better at paper scoring
   - Worth the extra cost

2. **Start with small tests**
   ```python
   # Test with fewer papers first
   papers = retrieve_papers(topic, client, model, target_papers=30)
   ```

3. **Monitor costs**
   - Check your OpenAI dashboard
   - Each retrieval = ~150 API calls
   - Set usage limits in OpenAI settings

4. **Handle rate limits**
   ```python
   import time
   # Add small delays between retrievals
   time.sleep(1)
   ```

## 🐛 Troubleshooting

### "Module openai not found"
```bash
pip3 install --user openai
```

### "Rate limit exceeded"
```bash
# Wait a minute, or upgrade to paid tier
# Or use gpt-3.5-turbo which has higher limits
```

### "Invalid function call format"
- GPT-3.5 may struggle with function formatting
- Switch to GPT-4 for better reliability
- Or increase max_iterations in code

### Papers not relevant enough
- GPT-3.5 may score less accurately
- Use GPT-4 for scoring
- Or increase min_score threshold

## 📚 Full Documentation

The OpenAI version works identically to the Anthropic version. See:
- **PAPER_RETRIEVAL_GUIDE.md** - Comprehensive guide (applies to both)
- **QUICK_START.md** - General quick start
- **USAGE_EXAMPLE.py** - Code examples (adapt for OpenAI)

## 🎯 Summary

✅ **Yes, your OpenAI key works!**

**Quick commands:**
```bash
export OPENAI_API_KEY="your-key"
pip3 install --user openai
python3 test_paper_retrieval_openai.py
```

**In code:**
```python
from openai import OpenAI
from modules.paper_retrieval_openai import retrieve_papers

client = OpenAI(api_key="your-key")
papers = retrieve_papers(topic, client, "gpt-4")
```

**Recommended:** Use GPT-4 for best results, or GPT-3.5-Turbo if you want faster/cheaper.
