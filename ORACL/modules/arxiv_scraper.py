"""
arXiv Paper Scraper Module.

Fetches papers from arXiv filtered by category (conference proxy) and date,
downloads PDFs, and extracts text content.
"""

import os
import re
import time
import tempfile
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime


@dataclass
class ArxivPaper:
    """A paper fetched from arXiv."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str  # ISO date string
    pdf_url: str
    content_md: str = ""  # Extracted markdown content from PDF

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ArxivPaper":
        return ArxivPaper(**d)


def fetch_papers(
    category: str,
    year: int,
    month_start: int = 1,
    month_end: int = 12,
    max_papers: int = 500,
    search_query: str = None,
) -> List[ArxivPaper]:
    """
    Fetch papers from arXiv matching a category and date range.

    Args:
        category: arXiv category code (e.g. "cs.CL", "cs.LG")
        year: Publication year to filter
        month_start: Start month (inclusive)
        month_end: End month (inclusive)
        max_papers: Maximum papers to return
        search_query: Optional additional keyword query to narrow results

    Returns:
        List of ArxivPaper objects (without PDF content yet)
    """
    import arxiv

    # Build query with arXiv date-range filter (submittedDate)
    # Format: YYYYMMDDHHMM
    date_from = f"{year}{month_start:02d}01"
    date_to = f"{year}{month_end:02d}31"

    query_parts = [f"cat:{category}"]
    query_parts.append(f"submittedDate:[{date_from}0000 TO {date_to}2359]")
    if search_query:
        query_parts.append(f"all:{search_query}")

    query = " AND ".join(query_parts)

    print(f"\n[arXiv Scraper] Fetching papers")
    print(f"  Category: {category}")
    print(f"  Date range: {year}-{month_start:02d} to {year}-{month_end:02d}")
    print(f"  Query: {query}")
    print(f"  Max papers: {max_papers}")

    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,
        num_retries=3,
    )

    search = arxiv.Search(
        query=query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    skipped = 0

    for result in client.results(search):
        pub_date = result.published

        # Secondary date filter (safety check)
        if pub_date.year != year:
            skipped += 1
            continue
        if pub_date.month < month_start or pub_date.month > month_end:
            skipped += 1
            continue

        arxiv_id = result.entry_id.split("/")[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        paper = ArxivPaper(
            arxiv_id=arxiv_id,
            title=result.title.replace("\n", " ").strip(),
            authors=[a.name for a in result.authors],
            abstract=result.summary.replace("\n", " ").strip(),
            categories=[c for c in result.categories],
            published=pub_date.isoformat(),
            pdf_url=result.pdf_url,
        )
        papers.append(paper)

        if len(papers) % 50 == 0:
            print(f"  Collected {len(papers)} papers so far...")

        if len(papers) >= max_papers:
            break

    print(f"  ✓ Fetched {len(papers)} papers (skipped {skipped} outside date range)")
    return papers


def download_and_extract(
    papers: List[ArxivPaper],
    max_content_chars: int = 20000,
) -> List[ArxivPaper]:
    """
    Download PDFs and extract markdown content for each paper.

    Args:
        papers: List of ArxivPaper objects
        max_content_chars: Truncate extracted content to this length

    Returns:
        Same papers list with content_md populated
    """
    try:
        import pymupdf4llm
        import requests
    except ImportError as e:
        raise ImportError(
            f"Required packages not installed: {e}\n"
            "Install with: pip install pymupdf4llm requests"
        )

    print(f"\n[PDF Extraction] Processing {len(papers)} papers...")

    success = 0
    failed = 0

    for i, paper in enumerate(papers):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(papers)} ({success} success, {failed} failed)")

        try:
            # Download PDF
            response = requests.get(paper.pdf_url, timeout=30)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            try:
                md_content = pymupdf4llm.to_markdown(tmp_path)

                # Truncate
                if len(md_content) > max_content_chars:
                    md_content = md_content[:max_content_chars] + "\n\n[Content truncated...]"

                paper.content_md = md_content
                success += 1
            finally:
                os.unlink(tmp_path)

            # Be nice to arXiv servers
            time.sleep(1.0)

        except Exception as e:
            print(f"    Warning: Failed for '{paper.title[:40]}...': {e}")
            paper.content_md = ""
            failed += 1

    print(f"  ✓ Extracted {success} papers, {failed} failed")
    return papers
