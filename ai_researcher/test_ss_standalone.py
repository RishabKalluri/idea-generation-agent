"""
Standalone test for Semantic Scholar API (no external dependencies except requests).
"""

import os
import time
import requests
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from collections import deque


@dataclass
class Paper:
    """Dataclass representing a research paper."""
    paper_id: str
    title: str
    abstract: str
    year: int
    citation_count: int
    authors: List[str]


class SemanticScholarClient:
    """Wrapper for Semantic Scholar API with rate limiting and caching."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self):
        """Initialize the Semantic Scholar client."""
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
        
        # Rate limiting
        self.rate_limit_window = 300
        self.max_requests = 100 if not self.api_key else 1000
        self.request_times = deque()
        
        # Simple cache
        self.cache = {}
    
    def _wait_if_rate_limited(self):
        """Implement rate limiting by tracking request times."""
        current_time = time.time()
        
        while self.request_times and current_time - self.request_times[0] > self.rate_limit_window:
            self.request_times.popleft()
        
        if len(self.request_times) >= self.max_requests:
            sleep_time = self.rate_limit_window - (current_time - self.request_times[0])
            if sleep_time > 0:
                print(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                current_time = time.time()
                while self.request_times and current_time - self.request_times[0] > self.rate_limit_window:
                    self.request_times.popleft()
        
        self.request_times.append(current_time)
    
    def _make_request(self, url, params=None):
        """Make an API request with rate limiting and error handling."""
        cache_key = f"{url}?{str(params)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        self._wait_if_rate_limited()
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            self.cache[cache_key] = result
            return result
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            raise
    
    def _parse_paper(self, paper_data):
        """Parse paper data from API response into Paper dataclass."""
        try:
            authors = []
            if paper_data.get('authors'):
                authors = [author.get('name', 'Unknown') for author in paper_data['authors']]
            
            return Paper(
                paper_id=paper_data.get('paperId', ''),
                title=paper_data.get('title', 'Untitled'),
                abstract=paper_data.get('abstract', ''),
                year=paper_data.get('year', 0) or 0,
                citation_count=paper_data.get('citationCount', 0) or 0,
                authors=authors
            )
        except Exception as e:
            print(f"Error parsing paper data: {e}")
            return None
    
    def KeywordQuery(self, keywords, limit=20):
        """Search papers by keywords."""
        url = f"{self.BASE_URL}/paper/search"
        params = {
            'query': keywords,
            'limit': limit,
            'fields': 'paperId,title,abstract,year,citationCount,authors'
        }
        
        try:
            response = self._make_request(url, params)
            papers = []
            
            if 'data' in response:
                for paper_data in response['data']:
                    paper = self._parse_paper(paper_data)
                    if paper:
                        papers.append(paper)
            
            return papers
        except Exception as e:
            print(f"Error in KeywordQuery for '{keywords}': {e}")
            return []
    
    def PaperQuery(self, paper_id):
        """Get detailed information for a specific paper."""
        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {
            'fields': 'paperId,title,abstract,year,citationCount,authors,references'
        }
        
        try:
            response = self._make_request(url, params)
            return self._parse_paper(response)
        except Exception as e:
            print(f"Error in PaperQuery for paper_id '{paper_id}': {e}")
            return None
    
    def GetReferences(self, paper_id, limit=20):
        """Get references of a paper."""
        url = f"{self.BASE_URL}/paper/{paper_id}/references"
        params = {
            'limit': limit,
            'fields': 'paperId,title,abstract,year,citationCount,authors'
        }
        
        try:
            response = self._make_request(url, params)
            papers = []
            
            if 'data' in response:
                for reference_data in response['data']:
                    if 'citedPaper' in reference_data:
                        paper = self._parse_paper(reference_data['citedPaper'])
                        if paper:
                            papers.append(paper)
            
            return papers
        except Exception as e:
            print(f"Error in GetReferences for paper_id '{paper_id}': {e}")
            return []
    
    def get_cache_size(self):
        """Get the number of cached requests."""
        return len(self.cache)


def main():
    """Test Semantic Scholar API wrapper functionality."""
    
    print("=" * 80)
    print("Semantic Scholar API Test")
    print("=" * 80)
    
    # Check for API key
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        print(f"\n✓ API Key found (higher rate limits enabled)")
    else:
        print(f"\n⚠ No API Key found (using default rate limits: 100 req/5min)")
        print(f"  To get higher limits, set: export SEMANTIC_SCHOLAR_API_KEY='your-key'")
    
    # Initialize the client
    print("\n[Initializing Client]")
    try:
        client = SemanticScholarClient()
        print("✓ Client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        return
    
    # Test 1: Keyword Search
    print("\n[Test 1] KeywordQuery: Searching for 'attention mechanism'")
    print("-" * 80)
    try:
        papers = client.KeywordQuery("attention mechanism", limit=5)
        
        if papers:
            print(f"✓ Found {len(papers)} papers:")
            for i, paper in enumerate(papers, 1):
                print(f"\n{i}. {paper.title}")
                print(f"   Year: {paper.year}, Citations: {paper.citation_count}")
                print(f"   Authors: {', '.join(paper.authors[:3])}")
                if len(paper.authors) > 3:
                    print(f"            ... and {len(paper.authors) - 3} more")
                print(f"   Paper ID: {paper.paper_id}")
                if paper.abstract:
                    abstract_preview = paper.abstract[:100] + "..." if len(paper.abstract) > 100 else paper.abstract
                    print(f"   Abstract: {abstract_preview}")
                else:
                    print(f"   Abstract: [Not available]")
            
            first_paper_id = papers[0].paper_id
            
            # Test 2: Paper Query
            print("\n\n[Test 2] PaperQuery: Getting details for first paper")
            print("-" * 80)
            try:
                paper = client.PaperQuery(first_paper_id)
                if paper:
                    print(f"✓ Successfully retrieved paper details")
                    print(f"   Title: {paper.title}")
                    print(f"   Year: {paper.year}")
                    print(f"   Citations: {paper.citation_count}")
                    print(f"   Authors: {', '.join(paper.authors[:5])}")
                else:
                    print(f"✗ Paper not found")
            except Exception as e:
                print(f"✗ Error: {e}")
            
            # Test 3: Get References
            print("\n\n[Test 3] GetReferences: Getting references for first paper")
            print("-" * 80)
            try:
                references = client.GetReferences(first_paper_id, limit=5)
                if references:
                    print(f"✓ Found {len(references)} references:")
                    for i, ref in enumerate(references, 1):
                        print(f"\n{i}. {ref.title}")
                        print(f"   Year: {ref.year}, Citations: {ref.citation_count}")
                else:
                    print("✓ No references found (paper may not have references)")
            except Exception as e:
                print(f"✗ Error: {e}")
        else:
            print("✗ No papers found")
    except Exception as e:
        print(f"✗ Error during search: {e}")
        import traceback
        traceback.print_exc()
    
    # Show statistics
    print("\n\n[Statistics]")
    print("-" * 80)
    print(f"Cached requests: {client.get_cache_size()}")
    print(f"API calls made: {len(client.request_times)}")
    
    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
