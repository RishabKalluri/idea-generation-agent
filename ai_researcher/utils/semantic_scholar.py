"""
Semantic Scholar API Wrapper.

Handles all interactions with the Semantic Scholar API for paper retrieval.
"""

import time
import requests
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from collections import deque
from config import SEMANTIC_SCHOLAR_API_KEY


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
        self.api_key = SEMANTIC_SCHOLAR_API_KEY
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
        
        # Rate limiting: 100 requests per 5 minutes (300 seconds) without API key
        # With API key, limits are higher but we'll still track
        self.rate_limit_window = 300  # 5 minutes in seconds
        self.max_requests = 100 if not self.api_key else 1000
        self.request_times = deque()
        
        # Simple cache to avoid duplicate API calls
        self.cache: Dict[str, Any] = {}
    
    def _wait_if_rate_limited(self):
        """Implement rate limiting by tracking request times."""
        current_time = time.time()
        
        # Remove requests outside the current window
        while self.request_times and current_time - self.request_times[0] > self.rate_limit_window:
            self.request_times.popleft()
        
        # If we've hit the rate limit, wait
        if len(self.request_times) >= self.max_requests:
            sleep_time = self.rate_limit_window - (current_time - self.request_times[0])
            if sleep_time > 0:
                print(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                # Clear old requests after waiting
                current_time = time.time()
                while self.request_times and current_time - self.request_times[0] > self.rate_limit_window:
                    self.request_times.popleft()
        
        # Record this request
        self.request_times.append(current_time)
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """
        Make an API request with rate limiting and error handling.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        # Check cache first
        cache_key = f"{url}?{str(params)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Apply rate limiting
        self._wait_if_rate_limited()
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Cache the result
            self.cache[cache_key] = result
            
            return result
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            raise
    
    def _parse_paper(self, paper_data: Dict) -> Optional[Paper]:
        """
        Parse paper data from API response into Paper dataclass.
        
        Args:
            paper_data: Raw paper data from API
            
        Returns:
            Paper object or None if required fields are missing
        """
        try:
            # Extract author names
            authors = []
            if paper_data.get('authors'):
                authors = [author.get('name', 'Unknown') for author in paper_data['authors']]
            
            return Paper(
                paper_id=paper_data.get('paperId', ''),
                title=paper_data.get('title', 'Untitled'),
                abstract=paper_data.get('abstract', ''),  # May be empty
                year=paper_data.get('year', 0) or 0,  # Handle None
                citation_count=paper_data.get('citationCount', 0) or 0,  # Handle None
                authors=authors
            )
        except Exception as e:
            print(f"Error parsing paper data: {e}")
            return None
    
    def KeywordQuery(self, keywords: str, limit: int = 20) -> List[Paper]:
        """
        Search papers by keywords.
        
        Args:
            keywords: Search query string
            limit: Number of results to return (default: 20)
            
        Returns:
            List of Paper objects
        """
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
    
    def PaperQuery(self, paper_id: str) -> Optional[Paper]:
        """
        Get detailed information for a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Paper object or None if not found
        """
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
    
    def GetReferences(self, paper_id: str, limit: int = 20) -> List[Paper]:
        """
        Get references of a paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            limit: Number of references to return (default: 20)
            
        Returns:
            List of Paper objects representing references
        """
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
                    # References are nested under 'citedPaper'
                    if 'citedPaper' in reference_data:
                        paper = self._parse_paper(reference_data['citedPaper'])
                        if paper:
                            papers.append(paper)
            
            return papers
        except Exception as e:
            print(f"Error in GetReferences for paper_id '{paper_id}': {e}")
            return []
    
    def clear_cache(self):
        """Clear the request cache."""
        self.cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the number of cached requests."""
        return len(self.cache)
