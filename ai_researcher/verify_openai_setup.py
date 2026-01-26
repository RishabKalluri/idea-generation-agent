"""
Quick verification that OpenAI setup works without anthropic.
"""

import sys
import os

print("=" * 60)
print("OpenAI Setup Verification")
print("=" * 60)

# Test 1: Check Python version
print("\n[1/5] Checking Python version...")
version = sys.version_info
print(f"  Python {version.major}.{version.minor}.{version.micro}")
if version.major == 3 and version.minor >= 7:
    print("  ✓ Python version OK (3.7+)")
else:
    print(f"  ✗ Python {version.major}.{version.minor} too old, need 3.7+")
    print("  Run with: python3.9 verify_openai_setup.py")
    sys.exit(1)

# Test 2: Check OpenAI package
print("\n[2/5] Checking OpenAI package...")
try:
    from openai import OpenAI
    print("  ✓ OpenAI package installed")
except ImportError as e:
    print(f"  ✗ OpenAI not installed: {e}")
    print("  Install with: python3.9 -m pip install --user openai")
    sys.exit(1)

# Test 3: Check requests package
print("\n[3/5] Checking requests package...")
try:
    import requests
    print("  ✓ requests package installed")
except ImportError:
    print("  ✗ requests not installed")
    print("  Install with: pip3 install --user requests")
    sys.exit(1)

# Test 4: Check Semantic Scholar module (without anthropic)
print("\n[4/5] Checking Semantic Scholar module...")
try:
    # Import directly from semantic_scholar.py to avoid __init__.py with anthropic
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "semantic_scholar",
        "utils/semantic_scholar.py"
    )
    ss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss_module)
    SemanticScholarClient = ss_module.SemanticScholarClient
    Paper = ss_module.Paper
    print("  ✓ Semantic Scholar module loads correctly")
except Exception as e:
    print(f"  ✗ Error loading Semantic Scholar: {e}")
    sys.exit(1)

# Test 5: Check API keys
print("\n[5/5] Checking API keys...")
openai_key = os.getenv("OPENAI_API_KEY")
ss_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

if openai_key:
    print(f"  ✓ OPENAI_API_KEY is set ({openai_key[:10]}...)")
else:
    print("  ✗ OPENAI_API_KEY not set")
    print("  Set with: export OPENAI_API_KEY='your-key'")

if ss_key:
    print(f"  ✓ SEMANTIC_SCHOLAR_API_KEY is set ({ss_key[:10]}...)")
else:
    print("  ⚠ SEMANTIC_SCHOLAR_API_KEY not set (optional)")

print("\n" + "=" * 60)

if openai_key:
    print("✓ All checks passed! Ready to run paper retrieval.")
    print("\nRun the test with:")
    print("  python3.9 test_openai_simple.py")
    print("\nOr use the script:")
    print("  bash RUN_OPENAI_TEST.sh")
else:
    print("⚠ Setup incomplete: Set OPENAI_API_KEY to continue")
    print("\nSet your key:")
    print("  export OPENAI_API_KEY='your-key-here'")
    print("\nThen run the test:")
    print("  python3.9 test_openai_simple.py")

print("=" * 60)
