#!/bin/bash
# Quick script to run the OpenAI paper retrieval test

echo "================================================================"
echo "OpenAI Paper Retrieval Test"
echo "================================================================"
echo ""

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY not set"
    echo ""
    echo "To set your API key:"
    echo "  export OPENAI_API_KEY='your-openai-api-key-here'"
    echo ""
    echo "Then run this script again:"
    echo "  bash RUN_OPENAI_TEST.sh"
    echo ""
    exit 1
fi

echo "✓ OPENAI_API_KEY is set"

# Check for Semantic Scholar key (optional)
if [ -z "$SEMANTIC_SCHOLAR_API_KEY" ]; then
    echo "⚠ SEMANTIC_SCHOLAR_API_KEY not set (optional but recommended)"
    echo "  Get one at: https://www.semanticscholar.org/product/api"
else
    echo "✓ SEMANTIC_SCHOLAR_API_KEY is set"
fi

echo ""
echo "Starting paper retrieval test with Python 3.9..."
echo "This may take a few minutes..."
echo ""

# Run with Python 3.9 (which has OpenAI installed)
# Using standalone version to avoid anthropic dependency
python3.9 test_openai_simple.py
