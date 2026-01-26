"""
Check Semantic Scholar API status and rate limits.
"""

import requests
import os


def check_api_status():
    """Check if the Semantic Scholar API is accessible."""
    
    print("=" * 80)
    print("Semantic Scholar API Status Check")
    print("=" * 80)
    
    # Check for API key
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
        print("\n✓ Using API Key")
    else:
        print("\n⚠ No API Key (limited rate: 100 requests per 5 minutes)")
        print("  Get one at: https://www.semanticscholar.org/product/api")
    
    # Try a simple paper lookup (less likely to be rate limited than search)
    print("\n[Testing API Access]")
    print("-" * 80)
    
    # Use a well-known paper ID (Attention is All You Need)
    test_paper_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{test_paper_id}"
    params = {'fields': 'title,year'}
    
    try:
        print(f"Requesting: {url}")
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ API is working!")
            print(f"  Test paper: {data.get('title', 'Unknown')}")
            print(f"  Year: {data.get('year', 'Unknown')}")
            print(f"\n✓ Your Semantic Scholar API wrapper should work fine!")
            print(f"\n  Run: python3 test_ss_standalone.py")
            return True
        elif response.status_code == 429:
            print(f"\n⚠ Rate Limit Exceeded (429)")
            print(f"  This means too many requests were made recently from this IP.")
            print(f"\n  Solutions:")
            print(f"  1. Wait 5 minutes and try again")
            print(f"  2. Get an API key: https://www.semanticscholar.org/product/api")
            print(f"     Then: export SEMANTIC_SCHOLAR_API_KEY='your-key'")
            return False
        elif response.status_code == 403:
            print(f"\n✗ Access Forbidden (403)")
            print(f"  You may need an API key to access this service.")
            print(f"  Get one at: https://www.semanticscholar.org/product/api")
            return False
        else:
            print(f"\n⚠ Unexpected status code: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Connection Error: {e}")
        print(f"  Check your internet connection.")
        return False
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    check_api_status()
