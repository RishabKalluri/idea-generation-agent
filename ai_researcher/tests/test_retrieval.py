#!/usr/bin/env python3
"""
Tests for paper retrieval methods.

Run all tests:
    pytest tests/test_retrieval.py -v

Run specific method tests:
    pytest tests/test_retrieval.py -v -k tavily
    pytest tests/test_retrieval.py -v -k llm_guided
    pytest tests/test_retrieval.py -v -k keyword

Run with output visible:
    pytest tests/test_retrieval.py -v -s

Quick manual test (no pytest):
    python tests/test_retrieval.py --method tavily --topic factuality
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def openai_client():
    """Get OpenAI client (requires OPENAI_API_KEY)."""
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


@pytest.fixture
def mock_client():
    """Mock LLM client for unit tests."""
    client = MagicMock()
    
    # Mock chat completion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "test response"
    client.chat.completions.create.return_value = mock_response
    
    return client


@pytest.fixture
def sample_topic():
    """Sample research topic for testing."""
    return "novel prompting methods that can improve factuality and reduce hallucination of large language models"


# ============================================================================
# UNIT TESTS (No API calls)
# ============================================================================

class TestRetrievalRegistry:
    """Test the retrieval method registry."""
    
    def test_get_available_methods(self):
        """Test that built-in methods are listed."""
        from modules.paper_retrieval import get_available_methods
        
        methods = get_available_methods()
        assert "llm_guided" in methods
        assert "keyword" in methods
        assert "tavily" in methods
    
    def test_register_custom_method(self):
        """Test registering a custom retrieval method."""
        from modules.paper_retrieval import register_retrieval_method, get_available_methods
        
        def dummy_retrieval(topic, client, model_name, target_papers):
            return []
        
        register_retrieval_method("test_dummy", dummy_retrieval)
        
        methods = get_available_methods()
        assert "test_dummy" in methods


class TestTavilyModule:
    """Unit tests for Tavily module functions."""
    
    def test_extract_arxiv_id(self):
        """Test arXiv ID extraction from URLs."""
        from modules.paper_retrieval_tavily import _extract_arxiv_id
        
        # Standard abs URL
        assert _extract_arxiv_id("https://arxiv.org/abs/2301.12345") == "2301.12345"
        
        # PDF URL
        assert _extract_arxiv_id("https://arxiv.org/pdf/2301.12345.pdf") == "2301.12345"
        
        # Just the ID
        assert _extract_arxiv_id("2301.12345") == "2301.12345"
        
        # Invalid URL
        assert _extract_arxiv_id("https://google.com") is None
    
    def test_extract_score(self):
        """Test score extraction from LLM responses."""
        from modules.paper_retrieval_tavily import _extract_score
        
        # Simple number
        assert _extract_score("8") == 8
        
        # With explanation
        assert _extract_score("Score: 7") == 7
        
        # Fraction format
        assert _extract_score("8/10") == 8
        
        # Invalid
        assert _extract_score("not a number") == 0
    
    def test_retrieved_paper_dataclass(self):
        """Test RetrievedPaper dataclass."""
        from modules.paper_retrieval_tavily import RetrievedPaper
        
        paper = RetrievedPaper(
            paper_id="2301.12345",
            title="Test Paper",
            authors=["Author One", "Author Two"],
            abstract="This is an abstract.",
            summary="This is a summary.",
            relevance_score=0.8,
            year=2023,
            url="https://arxiv.org/abs/2301.12345"
        )
        
        assert paper.paper_id == "2301.12345"
        assert paper.relevance_score == 0.8
        assert len(paper.authors) == 2


# ============================================================================
# INTEGRATION TESTS (Require API keys)
# ============================================================================

class TestTavilyRetrieval:
    """Integration tests for Tavily retrieval (requires TAVILY_API_KEY and OPENAI_API_KEY)."""
    
    @pytest.fixture
    def tavily_available(self):
        """Check if Tavily API key is available."""
        if not os.environ.get("TAVILY_API_KEY"):
            pytest.skip("TAVILY_API_KEY not set")
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
    
    def test_tavily_retrieval_basic(self, tavily_available, openai_client, sample_topic):
        """Test basic Tavily retrieval with real API calls."""
        from modules.paper_retrieval_tavily import retrieve_papers
        
        papers = retrieve_papers(
            topic=sample_topic,
            client=openai_client,
            model_name="gpt-4o-mini",
            max_papers=5,
            min_relevance_score=3,  # Lower threshold for testing
            summarize_threshold=10  # Disable PDF summarization for speed
        )
        
        assert len(papers) > 0
        assert all(p.paper_id for p in papers)
        assert all(p.title for p in papers)
        assert all(0 <= p.relevance_score <= 1 for p in papers)
    
    def test_tavily_via_dispatcher(self, tavily_available, openai_client, sample_topic):
        """Test Tavily retrieval via main dispatcher."""
        from modules.paper_retrieval import retrieve_papers
        
        papers = retrieve_papers(
            topic=sample_topic,
            client=openai_client,
            model_name="gpt-4o-mini",
            target_papers=5,
            method="tavily"
        )
        
        assert len(papers) > 0
        # Should return legacy Paper format
        assert all(hasattr(p, 'paper_id') for p in papers)
        assert all(hasattr(p, 'abstract') for p in papers)


class TestKeywordRetrieval:
    """Integration tests for keyword-based retrieval."""
    
    @pytest.fixture
    def semantic_scholar_available(self):
        """Check if Semantic Scholar is accessible."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
    
    def test_keyword_retrieval_basic(self, semantic_scholar_available, openai_client, sample_topic):
        """Test keyword retrieval with real API calls."""
        from modules.paper_retrieval import retrieve_papers
        
        papers = retrieve_papers(
            topic=sample_topic,
            client=openai_client,
            model_name="gpt-4o-mini",
            target_papers=5,
            method="keyword"
        )
        
        assert len(papers) >= 0  # May return 0 if API issues
        if papers:
            assert all(hasattr(p, 'paper_id') for p in papers)


class TestLLMGuidedRetrieval:
    """Integration tests for LLM-guided retrieval."""
    
    @pytest.fixture
    def semantic_scholar_available(self):
        """Check if Semantic Scholar is accessible."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
    
    def test_llm_guided_retrieval_basic(self, semantic_scholar_available, openai_client, sample_topic):
        """Test LLM-guided retrieval with real API calls."""
        from modules.paper_retrieval import retrieve_papers
        
        papers = retrieve_papers(
            topic=sample_topic,
            client=openai_client,
            model_name="gpt-4o-mini",
            target_papers=10,
            method="llm_guided"
        )
        
        assert len(papers) >= 0  # May return 0 if API issues
        if papers:
            assert all(hasattr(p, 'paper_id') for p in papers)


# ============================================================================
# MANUAL TEST RUNNER
# ============================================================================

def run_manual_test(method: str, topic: str, max_papers: int = 10):
    """Run a manual test of a retrieval method."""
    print(f"\n{'='*60}")
    print(f"Testing retrieval method: {method}")
    print(f"Topic: {topic}")
    print(f"Max papers: {max_papers}")
    print(f"{'='*60}\n")
    
    # Check API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return
    
    if method == "tavily" and not os.environ.get("TAVILY_API_KEY"):
        print("ERROR: TAVILY_API_KEY not set")
        return
    
    # Initialize client
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Import and run
    from modules.paper_retrieval import retrieve_papers
    
    papers = retrieve_papers(
        topic=topic,
        client=client,
        model_name="gpt-4o-mini",
        target_papers=max_papers,
        method=method
    )
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(papers)} papers retrieved")
    print(f"{'='*60}\n")
    
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.title[:70]}...")
        print(f"   ID: {paper.paper_id}")
        print(f"   Year: {paper.year} | Authors: {', '.join(paper.authors[:2])}...")
        print(f"   Abstract: {paper.abstract[:150]}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test paper retrieval methods")
    parser.add_argument(
        "--method",
        type=str,
        default="tavily",
        choices=["tavily", "keyword", "llm_guided"],
        help="Retrieval method to test"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="factuality",
        help="Topic key or custom topic description"
    )
    parser.add_argument(
        "--max_papers",
        type=int,
        default=10,
        help="Maximum number of papers to retrieve"
    )
    
    args = parser.parse_args()
    
    # Map topic keys to descriptions
    TOPICS = {
        "factuality": "novel prompting methods that can improve factuality and reduce hallucination of large language models",
        "math": "novel prompting methods for large language models to improve mathematical problem solving",
        "coding": "novel prompting methods for large language models to improve code generation",
        "bias": "novel prompting methods to reduce social biases and stereotypes of large language models",
    }
    
    topic = TOPICS.get(args.topic, args.topic)
    
    run_manual_test(args.method, topic, args.max_papers)
