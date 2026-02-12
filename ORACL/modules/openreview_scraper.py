"""
OpenReview Paper Scraper Module.

Fetches papers (accepted AND rejected) from OpenReview for conferences
that use the platform (ICLR, NeurIPS, ICML, ACL/ARR, etc.).
Downloads PDFs and extracts text content.

Conferences NOT on OpenReview (CVPR, ICCV, ECCV) are not supported
by this module — use arxiv_scraper for those.
"""

import os
import re
import time
import random
import tempfile
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict
from datetime import datetime

try:
    import openreview
except ImportError:
    raise ImportError(
        "openreview-py is required. Install with: pip install openreview-py"
    )


# ============================================================================
# VENUE ID MAPPING
# ============================================================================
# Maps (conference, year) → OpenReview venue ID and submission invitation.
# These follow patterns like "ICLR.cc/2024/Conference/-/Submission".
# Some older years or smaller venues may have different patterns.

VENUE_MAP: Dict[Tuple[str, int], dict] = {
    # --- ICLR ---
    ("ICLR", 2024): {
        "venue_id": "ICLR.cc/2024/Conference",
        "submission_inv": "ICLR.cc/2024/Conference/-/Submission",
    },
    ("ICLR", 2023): {
        "venue_id": "ICLR.cc/2023/Conference",
        "submission_inv": "ICLR.cc/2023/Conference/-/Blind_Submission",
    },
    ("ICLR", 2022): {
        "venue_id": "ICLR.cc/2022/Conference",
        "submission_inv": "ICLR.cc/2022/Conference/-/Blind_Submission",
    },
    ("ICLR", 2021): {
        "venue_id": "ICLR.cc/2021/Conference",
        "submission_inv": "ICLR.cc/2021/Conference/-/Blind_Submission",
    },
    # --- NeurIPS ---
    ("NeurIPS", 2024): {
        "venue_id": "NeurIPS.cc/2024/Conference",
        "submission_inv": "NeurIPS.cc/2024/Conference/-/Submission",
    },
    ("NeurIPS", 2023): {
        "venue_id": "NeurIPS.cc/2023/Conference",
        "submission_inv": "NeurIPS.cc/2023/Conference/-/Submission",
    },
    ("NeurIPS", 2022): {
        "venue_id": "NeurIPS.cc/2022/Conference",
        "submission_inv": "NeurIPS.cc/2022/Conference/-/Submission",
    },
    ("NeurIPS", 2021): {
        "venue_id": "NeurIPS.cc/2021/Conference",
        "submission_inv": "NeurIPS.cc/2021/Conference/-/Blind_Submission",
    },
    # --- ICML ---
    ("ICML", 2024): {
        "venue_id": "ICML.cc/2024/Conference",
        "submission_inv": "ICML.cc/2024/Conference/-/Submission",
    },
    ("ICML", 2023): {
        "venue_id": "ICML.cc/2023/Conference",
        "submission_inv": "ICML.cc/2023/Conference/-/Submission",
    },
    # --- AAAI --- (limited OpenReview presence)
    ("AAAI", 2025): {
        "venue_id": "AAAI.org/2025/Conference",
        "submission_inv": "AAAI.org/2025/Conference/-/Submission",
    },
    # --- ACL (via ARR + commitment) ---
    ("ACL", 2024): {
        "venue_id": "aclweb.org/ACL/2024/Conference",
        "submission_inv": "aclweb.org/ACL/2024/Conference/-/Submission",
    },
    ("EMNLP", 2024): {
        "venue_id": "EMNLP/2024/Conference",
        "submission_inv": "EMNLP/2024/Conference/-/Submission",
    },
    ("NAACL", 2024): {
        "venue_id": "aclweb.org/NAACL/2024/Conference",
        "submission_inv": "aclweb.org/NAACL/2024/Conference/-/Submission",
    },
}


# Accept-like strings in OpenReview venue fields
_ACCEPT_KEYWORDS = [
    "accept", "poster", "spotlight", "oral", "notable",
    "award", "best paper", "top", "long paper", "short paper",
]
_REJECT_KEYWORDS = [
    "reject", "withdrawn", "desk reject",
]


@dataclass
class OpenReviewPaper:
    """A paper fetched from OpenReview with decision metadata."""
    openreview_id: str        # OpenReview note ID
    title: str
    authors: List[str]
    abstract: str
    venue: str                # Raw venue string from OpenReview
    accepted: Optional[bool]  # True=accepted, False=rejected, None=unknown
    decision_raw: str         # The raw decision/venue string
    pdf_url: str
    published: str            # ISO date string (creation date)
    content_md: str = ""      # Extracted markdown content from PDF

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "OpenReviewPaper":
        return OpenReviewPaper(**d)


def _classify_decision(venue_str: str, decision_str: str = "") -> Optional[bool]:
    """
    Determine accept/reject from OpenReview venue or decision strings.

    Returns True (accepted), False (rejected), or None (ambiguous).
    """
    combined = f"{venue_str} {decision_str}".lower()

    for kw in _ACCEPT_KEYWORDS:
        if kw in combined:
            return True
    for kw in _REJECT_KEYWORDS:
        if kw in combined:
            return False

    # If venue string contains the conference name (e.g. "ICLR 2024 poster"),
    # that typically means accepted
    if venue_str and "submitted to" not in venue_str.lower():
        # Has a concrete venue → likely accepted
        return True

    return None


def get_supported_conferences() -> List[str]:
    """Return list of (conference, year) combos available on OpenReview."""
    return sorted(set(VENUE_MAP.keys()))


def is_supported(conference: str, year: int) -> bool:
    """Check if a conference/year combo is available on OpenReview."""
    return (conference.upper(), year) in VENUE_MAP


def fetch_papers_openreview(
    conference: str,
    year: int,
    max_papers: int = 200,
    min_month: int = 1,
    accepted_ratio: float = 0.5,
) -> List[OpenReviewPaper]:
    """
    Fetch papers from OpenReview for a conference/year.

    Retrieves BOTH accepted and rejected submissions and samples them
    to achieve approximately the target accepted_ratio for data diversity.

    Args:
        conference: Conference name (e.g. "ICLR", "NeurIPS")
        year: Conference year
        max_papers: Maximum total papers to return
        min_month: Only include papers created on or after this month
        accepted_ratio: Target fraction of accepted papers (0.5 = balanced)

    Returns:
        List of OpenReviewPaper objects (without PDF content yet)
    """
    conf_upper = conference.upper()
    key = (conf_upper, year)

    if key not in VENUE_MAP:
        supported = [f"{c} {y}" for (c, y) in sorted(VENUE_MAP.keys())]
        raise ValueError(
            f"Conference {conf_upper} {year} not in OpenReview venue map.\n"
            f"Supported: {', '.join(supported)}\n"
            f"(CVPR/ICCV/ECCV are not on OpenReview — use the arXiv pipeline instead.)"
        )

    venue_info = VENUE_MAP[key]
    venue_id = venue_info["venue_id"]
    submission_inv = venue_info["submission_inv"]

    print(f"\n[OpenReview Scraper] Fetching papers")
    print(f"  Conference    : {conf_upper} {year}")
    print(f"  Venue ID      : {venue_id}")
    print(f"  Invitation    : {submission_inv}")
    print(f"  Max papers    : {max_papers}")
    print(f"  Accepted ratio: {accepted_ratio:.0%}")
    print(f"  Min month     : {min_month}")

    # Connect to OpenReview API v2 (guest access, no login required)
    client = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net"
    )

    print(f"  Fetching submissions (this may take a minute)...")

    try:
        notes = client.get_all_notes(
            invitation=submission_inv,
            details="directReplies",
        )
    except Exception as e:
        # Fallback: try without details
        print(f"  Warning: Failed with details, retrying without: {e}")
        notes = client.get_all_notes(invitation=submission_inv)

    print(f"  Retrieved {len(notes)} total submissions from OpenReview")

    # Parse all notes into structured papers
    accepted_papers = []
    rejected_papers = []
    unknown_papers = []

    for note in notes:
        content = note.content or {}

        # Extract title
        title_field = content.get("title", {})
        title = title_field.get("value", "") if isinstance(title_field, dict) else str(title_field)
        if not title:
            continue

        # Extract abstract
        abstract_field = content.get("abstract", {})
        abstract = abstract_field.get("value", "") if isinstance(abstract_field, dict) else str(abstract_field)

        # Extract authors
        authors_field = content.get("authors", {})
        authors = authors_field.get("value", []) if isinstance(authors_field, dict) else []
        if isinstance(authors, str):
            authors = [authors]

        # Extract venue / decision
        venue_field = content.get("venue", {})
        venue_str = venue_field.get("value", "") if isinstance(venue_field, dict) else str(venue_field)

        venueid_field = content.get("venueid", {})
        venueid_str = venueid_field.get("value", "") if isinstance(venueid_field, dict) else str(venueid_field)

        # Also check for explicit decision in direct replies
        decision_str = ""
        if hasattr(note, "details") and note.details:
            replies = note.details.get("directReplies", [])
            for reply in replies:
                reply_inv = reply.get("invitation", "") or ""
                if "Decision" in reply_inv or "decision" in reply_inv:
                    rc = reply.get("content", {})
                    dec = rc.get("decision", {})
                    decision_str = dec.get("value", "") if isinstance(dec, dict) else str(dec)
                    break

        # Classify
        accepted = _classify_decision(venue_str or venueid_str, decision_str)

        # Get creation date
        cdate = note.cdate or note.tcdate
        if cdate:
            pub_dt = datetime.fromtimestamp(cdate / 1000)
            published = pub_dt.isoformat()
            # Filter by min_month
            if pub_dt.month < min_month:
                continue
        else:
            published = ""

        # PDF URL
        pdf_field = content.get("pdf", {})
        pdf_val = pdf_field.get("value", "") if isinstance(pdf_field, dict) else str(pdf_field)
        if pdf_val and not pdf_val.startswith("http"):
            pdf_url = f"https://openreview.net{pdf_val}"
        elif pdf_val:
            pdf_url = pdf_val
        else:
            pdf_url = f"https://openreview.net/pdf?id={note.id}"

        paper = OpenReviewPaper(
            openreview_id=note.id,
            title=title.replace("\n", " ").strip(),
            authors=authors[:20],  # Cap author list
            abstract=abstract.replace("\n", " ").strip(),
            venue=venue_str,
            accepted=accepted,
            decision_raw=decision_str or venue_str,
            pdf_url=pdf_url,
            published=published,
        )

        if accepted is True:
            accepted_papers.append(paper)
        elif accepted is False:
            rejected_papers.append(paper)
        else:
            unknown_papers.append(paper)

    print(f"  Classified: {len(accepted_papers)} accepted, "
          f"{len(rejected_papers)} rejected, {len(unknown_papers)} unknown")

    # ----------------------------------------------------------------
    # Balanced sampling: target accepted_ratio
    # ----------------------------------------------------------------
    n_target_accepted = int(max_papers * accepted_ratio)
    n_target_rejected = max_papers - n_target_accepted

    # Shuffle for random sampling
    random.shuffle(accepted_papers)
    random.shuffle(rejected_papers)

    sampled_accepted = accepted_papers[:n_target_accepted]
    sampled_rejected = rejected_papers[:n_target_rejected]

    # If one bucket is short, fill from the other
    shortfall_accepted = n_target_accepted - len(sampled_accepted)
    shortfall_rejected = n_target_rejected - len(sampled_rejected)

    if shortfall_accepted > 0 and len(rejected_papers) > n_target_rejected:
        extra = rejected_papers[n_target_rejected:n_target_rejected + shortfall_accepted]
        sampled_rejected.extend(extra)

    if shortfall_rejected > 0 and len(accepted_papers) > n_target_accepted:
        extra = accepted_papers[n_target_accepted:n_target_accepted + shortfall_rejected]
        sampled_accepted.extend(extra)

    result = sampled_accepted + sampled_rejected
    random.shuffle(result)

    actual_acc = sum(1 for p in result if p.accepted is True)
    actual_rej = sum(1 for p in result if p.accepted is False)
    print(f"  Sampled {len(result)} papers: {actual_acc} accepted, {actual_rej} rejected")

    return result


def download_and_extract_openreview(
    papers: List[OpenReviewPaper],
    max_content_chars: int = 20000,
) -> List[OpenReviewPaper]:
    """
    Download PDFs from OpenReview and extract markdown content.

    Args:
        papers: List of OpenReviewPaper objects
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
            response = requests.get(paper.pdf_url, timeout=30)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            try:
                md_content = pymupdf4llm.to_markdown(tmp_path)

                if len(md_content) > max_content_chars:
                    md_content = md_content[:max_content_chars] + "\n\n[Content truncated...]"

                paper.content_md = md_content
                success += 1
            finally:
                os.unlink(tmp_path)

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"    Warning: Failed for '{paper.title[:50]}...': {e}")
            paper.content_md = ""
            failed += 1

    print(f"  ✓ Extracted {success} papers, {failed} failed")
    return papers
