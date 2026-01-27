#!/bin/bash
# Setup script for AI Research Idea Generation Pipeline
# 
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh

set -e  # Exit on error

echo "=============================================="
echo "AI Research Idea Generation - Environment Setup"
echo "=============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Environment name
ENV_NAME="ai-researcher"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to update it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Updating environment..."
        conda env update -f environment.yml --prune
    else
        echo "Skipping environment update."
    fi
else
    echo ""
    echo "Creating conda environment '${ENV_NAME}'..."
    conda env create -f environment.yml
fi

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To set your API key:"
echo "  export OPENAI_API_KEY='your-key-here'"
echo ""
echo "To run a quick test:"
echo "  python tests/test_pipeline.py"
echo ""
echo "To run the pipeline:"
echo "  python main.py --topic factuality --lite"
echo ""
