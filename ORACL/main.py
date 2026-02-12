"""
ORACL Dataset Generation Pipeline.

Scrapes arXiv papers for a given conference/category and date range,
converts them into research proposal format, and stores them in a
JSONL dataset.

Usage:
    python -m ORACL.main --conference ACL --year 2024
    python -m ORACL.main --conference NeurIPS --year 2024 --month_start 1 --month_end 6
    python -m ORACL.main --conference CVPR --year 2023 --max_papers 50 --skip_pdf
    python -m ORACL.main --list   # Show all stored datasets
"""

import argparse
import os
import sys
import time
from datetime import datetime

from openai import OpenAI

from .config.settings import (
    OPENAI_API_KEY,
    OPENAI_MODEL_NAME,
    CONFERENCE_CATEGORIES,
    DEFAULT_CONFERENCE,
    DEFAULT_YEAR,
    MAX_TOTAL_PAPERS,
    MAX_PAPER_CONTENT_LENGTH,
    DATASET_DIR,
    GRADING_MODEL_NAME,
    MAX_GRADING_PAIRS,
    GRADING_OUTPUT_DIR,
    OPENREVIEW_ACCEPTED_RATIO,
    OPENREVIEW_CONFERENCES,
)
from .modules.arxiv_scraper import fetch_papers, download_and_extract
from .modules.converter import convert_batch, convert_openreview_batch
from .modules.dataset_store import save_proposals, load_proposals, list_datasets, get_stats, get_existing_ids
from .modules.grader import grade_dataset
from .modules.openreview_scraper import (
    fetch_papers_openreview,
    download_and_extract_openreview,
    is_supported as openreview_is_supported,
    VENUE_MAP,
)


def run_pipeline(
    conference: str,
    year: int,
    month_start: int = 1,
    month_end: int = 12,
    max_papers: int = None,
    skip_pdf: bool = False,
    search_query: str = None,
):
    """
    Run the full dataset generation pipeline.

    Steps:
        1. Resolve conference → arXiv category
        2. Fetch papers from arXiv
        3. Download PDFs and extract content (optional)
        4. Convert papers to proposal format via LLM
        5. Save proposals to JSONL dataset
    """
    max_papers = max_papers or MAX_TOTAL_PAPERS
    start_time = time.time()

    # ------------------------------------------------------------------
    # Step 1: Resolve conference to arXiv category
    # ------------------------------------------------------------------
    conference_upper = conference.upper()
    if conference_upper not in CONFERENCE_CATEGORIES:
        print(f"\nError: Unknown conference '{conference}'")
        print(f"Available: {', '.join(CONFERENCE_CATEGORIES.keys())}")
        sys.exit(1)

    category = CONFERENCE_CATEGORIES[conference_upper]
    print(f"\n{'='*60}")
    print(f"ORACL Dataset Generation Pipeline")
    print(f"{'='*60}")
    print(f"  Conference : {conference_upper}")
    print(f"  Category   : {category}")
    print(f"  Year       : {year}")
    print(f"  Months     : {month_start}-{month_end}")
    print(f"  Max papers : {max_papers}")
    print(f"  Skip PDF   : {skip_pdf}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 2: Fetch papers from arXiv
    # ------------------------------------------------------------------
    print(f"\n[Step 1/4] Fetching papers from arXiv...")
    papers = fetch_papers(
        category=category,
        year=year,
        month_start=month_start,
        month_end=month_end,
        max_papers=max_papers,
        search_query=search_query,
    )

    if not papers:
        print("  No papers found. Try a different date range or conference.")
        return

    # ------------------------------------------------------------------
    # Deduplicate: skip papers already in the dataset
    # ------------------------------------------------------------------
    existing_ids = get_existing_ids(conference_upper, year, DATASET_DIR)
    if existing_ids:
        before = len(papers)
        papers = [p for p in papers if p.arxiv_id not in existing_ids]
        skipped = before - len(papers)
        print(f"  Skipped {skipped} papers already in dataset ({len(papers)} new)")
        if not papers:
            print("  All fetched papers are already in the dataset. Nothing to do.")
            return

    # ------------------------------------------------------------------
    # Step 3: Download PDFs and extract content
    # ------------------------------------------------------------------
    if not skip_pdf:
        print(f"\n[Step 2/4] Downloading PDFs and extracting content...")
        papers = download_and_extract(
            papers,
            max_content_chars=MAX_PAPER_CONTENT_LENGTH,
        )
        papers_with_content = [p for p in papers if p.content_md]
        print(f"  ✓ {len(papers_with_content)}/{len(papers)} papers have extracted content")
    else:
        print(f"\n[Step 2/4] Skipping PDF download (--skip_pdf)")
        print(f"  Papers will be converted using abstracts only")

    # ------------------------------------------------------------------
    # Step 4: Convert papers to proposal format
    # ------------------------------------------------------------------
    print(f"\n[Step 3/4] Converting papers to proposal format...")

    if not OPENAI_API_KEY:
        print("  Error: OPENAI_API_KEY not set. Add it to ai_researcher/config/.env")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    proposals = convert_batch(
        papers=papers,
        client=client,
        model_name=OPENAI_MODEL_NAME,
    )

    print(f"  ✓ Converted {len(proposals)}/{len(papers)} papers to proposals")

    if not proposals:
        print("  No proposals generated. Check LLM connectivity and paper content.")
        return

    # ------------------------------------------------------------------
    # Step 5: Save to dataset
    # ------------------------------------------------------------------
    print(f"\n[Step 4/4] Saving proposals to dataset...")
    save_proposals(
        proposals=proposals,
        conference=conference_upper,
        year=year,
        dataset_dir=DATASET_DIR,
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"{'='*60}\n")


def run_openreview_pipeline(
    conference: str,
    min_year: int,
    min_month: int = 1,
    max_papers: int = None,
    skip_pdf: bool = False,
    accepted_ratio: float = None,
):
    """
    Run the OpenReview-based dataset generation pipeline.

    Fetches papers directly from OpenReview (both accepted AND rejected),
    downloads PDFs, converts to proposals, and tags each with acceptance
    status.

    Steps:
        1. Verify conference is on OpenReview
        2. Fetch papers (balanced accept/reject sampling)
        3. Deduplicate against existing dataset
        4. Download PDFs and extract content
        5. Convert to proposals via LLM
        6. Save to JSONL with acceptance labels
    """
    max_papers = max_papers or MAX_TOTAL_PAPERS
    accepted_ratio = accepted_ratio if accepted_ratio is not None else OPENREVIEW_ACCEPTED_RATIO
    start_time = time.time()

    conference_upper = conference.upper()

    # ------------------------------------------------------------------
    # Step 1: Verify conference is on OpenReview
    # ------------------------------------------------------------------
    if not openreview_is_supported(conference_upper, min_year):
        print(f"\nError: {conference_upper} {min_year} is not available on OpenReview.")
        if conference_upper in {"CVPR", "ICCV", "ECCV"}:
            print(f"  {conference_upper} uses CMT, not OpenReview.")
            print(f"  Use the arXiv pipeline instead: python -m ORACL.main --conference {conference_upper} --year {min_year}")
        else:
            supported = sorted(set(f"{c} {y}" for (c, y) in VENUE_MAP.keys()))
            print(f"  Supported: {', '.join(supported)}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"ORACL OpenReview Pipeline")
    print(f"{'='*60}")
    print(f"  Conference      : {conference_upper}")
    print(f"  Year            : {min_year}")
    print(f"  Min month       : {min_month}")
    print(f"  Max papers      : {max_papers}")
    print(f"  Accepted ratio  : {accepted_ratio:.0%}")
    print(f"  Skip PDF        : {skip_pdf}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 2: Fetch papers from OpenReview
    # ------------------------------------------------------------------
    print(f"\n[Step 1/4] Fetching papers from OpenReview...")
    papers = fetch_papers_openreview(
        conference=conference_upper,
        year=min_year,
        max_papers=max_papers,
        min_month=min_month,
        accepted_ratio=accepted_ratio,
    )

    if not papers:
        print("  No papers found. Check the conference/year.")
        return

    # ------------------------------------------------------------------
    # Deduplicate: skip papers already in the dataset
    # ------------------------------------------------------------------
    existing_ids = get_existing_ids(conference_upper, min_year, DATASET_DIR)
    if existing_ids:
        before = len(papers)
        papers = [p for p in papers if p.openreview_id not in existing_ids]
        skipped = before - len(papers)
        print(f"  Skipped {skipped} papers already in dataset ({len(papers)} new)")
        if not papers:
            print("  All fetched papers are already in the dataset. Nothing to do.")
            return

    # ------------------------------------------------------------------
    # Step 3: Download PDFs and extract content
    # ------------------------------------------------------------------
    if not skip_pdf:
        print(f"\n[Step 2/4] Downloading PDFs and extracting content...")
        papers = download_and_extract_openreview(
            papers,
            max_content_chars=MAX_PAPER_CONTENT_LENGTH,
        )
        papers_with_content = [p for p in papers if p.content_md]
        print(f"  ✓ {len(papers_with_content)}/{len(papers)} papers have extracted content")
    else:
        print(f"\n[Step 2/4] Skipping PDF download (--skip_pdf)")
        print(f"  Papers will be converted using abstracts only")

    # ------------------------------------------------------------------
    # Step 4: Convert papers to proposal format
    # ------------------------------------------------------------------
    print(f"\n[Step 3/4] Converting papers to proposal format...")

    if not OPENAI_API_KEY:
        print("  Error: OPENAI_API_KEY not set. Add it to ai_researcher/config/.env")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    proposals = convert_openreview_batch(
        papers=papers,
        client=client,
        model_name=OPENAI_MODEL_NAME,
    )

    print(f"  ✓ Converted {len(proposals)}/{len(papers)} papers to proposals")

    n_acc = sum(1 for p in proposals if p.accepted is True)
    n_rej = sum(1 for p in proposals if p.accepted is False)
    print(f"  Breakdown: {n_acc} accepted, {n_rej} rejected")

    if not proposals:
        print("  No proposals generated. Check LLM connectivity and paper content.")
        return

    # ------------------------------------------------------------------
    # Step 5: Save to dataset
    # ------------------------------------------------------------------
    print(f"\n[Step 4/4] Saving proposals to dataset...")
    save_proposals(
        proposals=proposals,
        conference=conference_upper,
        year=min_year,
        dataset_dir=DATASET_DIR,
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"{'='*60}\n")


def grade_papers(
    conference: str,
    year: int,
    max_pairs: int = None,
    grading_model: str = None,
):
    """
    Grade all proposals in a dataset via pairwise comparison.

    Loads proposals from the dataset store, generates pairs, judges
    them with the grading model (web search enabled), and saves rankings.
    """
    conference_upper = conference.upper()
    grading_model = grading_model or GRADING_MODEL_NAME
    max_pairs = max_pairs or MAX_GRADING_PAIRS

    # Load proposals
    proposals = load_proposals(conference_upper, year, DATASET_DIR)
    if not proposals:
        print(f"\nNo proposals found for {conference_upper} {year}.")
        print("Run the scrape+convert pipeline first.")
        return

    print(f"\n{'='*60}")
    print(f"ORACL Paper Grading")
    print(f"{'='*60}")
    print(f"  Conference : {conference_upper}")
    print(f"  Year       : {year}")
    print(f"  Proposals  : {len(proposals)}")
    print(f"  Model      : {grading_model}")
    print(f"{'='*60}")

    if not OPENAI_API_KEY:
        print("  Error: OPENAI_API_KEY not set.")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Output dir: grading_results/CONFERENCE/YEAR_TIMESTAMP/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        GRADING_OUTPUT_DIR, conference_upper, f"{year}_{timestamp}"
    )

    results, rankings = grade_dataset(
        proposals=proposals,
        client=client,
        model_name=grading_model,
        max_pairs=max_pairs,
        output_dir=output_dir,
    )

    print(f"\nGrading complete. Results in {output_dir}")


def show_datasets():
    """Print a summary of all stored datasets."""
    stats = get_stats(DATASET_DIR)

    if stats["num_datasets"] == 0:
        print("\nNo datasets found. Run the pipeline to generate one.")
        return

    print(f"\n{'='*60}")
    print(f"ORACL Dataset Store")
    print(f"{'='*60}")
    print(f"  Total proposals : {stats['total_proposals']}")
    print(f"  Datasets        : {stats['num_datasets']}")
    print(f"{'='*60}")
    print(f"\n  {'Conference':<12} {'Year':<8} {'Count':<8} Path")
    print(f"  {'-'*10:<12} {'-'*6:<8} {'-'*5:<8} {'-'*30}")

    for ds in stats["datasets"]:
        print(f"  {ds['conference']:<12} {ds['year']:<8} {ds['count']:<8} {ds['path']}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="ORACL: Generate research proposal datasets from arXiv papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # arXiv pipeline (all conferences)
  python -m ORACL.main --conference ACL --year 2024
  python -m ORACL.main --conference NeurIPS --year 2024 --month_start 1 --month_end 6
  python -m ORACL.main --conference CVPR --year 2023 --max_papers 50 --skip_pdf

  # OpenReview pipeline (ICLR, NeurIPS, ICML, ACL, etc.)
  python -m ORACL.main --openreview --conference ICLR --year 2024
  python -m ORACL.main --openreview --conference NeurIPS --year 2023 --max_papers 100
  python -m ORACL.main --openreview --conference ICLR --year 2024 --min_month 1 --accepted_ratio 0.5

  # Grading
  python -m ORACL.main --grade --conference CVPR --year 2023
  python -m ORACL.main --grade --conference CVPR --year 2023 --max_pairs 10

  # List datasets
  python -m ORACL.main --list

Available conferences (arXiv):   """ + ", ".join(CONFERENCE_CATEGORIES.keys()) + """
Available conferences (OpenReview): """ + ", ".join(sorted(OPENREVIEW_CONFERENCES))
    )

    parser.add_argument(
        "--conference",
        type=str,
        default=DEFAULT_CONFERENCE,
        help=f"Conference name (default: {DEFAULT_CONFERENCE}). "
             f"Options: {', '.join(CONFERENCE_CATEGORIES.keys())}",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Publication year to filter (default: {DEFAULT_YEAR})",
    )
    parser.add_argument(
        "--month_start",
        type=int,
        default=1,
        help="Start month, inclusive (default: 1)",
    )
    parser.add_argument(
        "--month_end",
        type=int,
        default=12,
        help="End month, inclusive (default: 12)",
    )
    parser.add_argument(
        "--max_papers",
        type=int,
        default=None,
        help=f"Maximum papers to fetch (default: {MAX_TOTAL_PAPERS})",
    )
    parser.add_argument(
        "--skip_pdf",
        action="store_true",
        help="Skip PDF download; convert using abstracts only (faster but lower quality)",
    )
    parser.add_argument(
        "--search_query",
        type=str,
        default=None,
        help="Optional additional search terms to narrow arXiv results",
    )
    parser.add_argument(
        "--openreview",
        action="store_true",
        help="Use OpenReview to fetch papers with accept/reject labels "
             "(supports ICLR, NeurIPS, ICML, ACL, EMNLP, NAACL)",
    )
    parser.add_argument(
        "--min_month",
        type=int,
        default=1,
        help="For --openreview: only include papers from this month onward (default: 1)",
    )
    parser.add_argument(
        "--accepted_ratio",
        type=float,
        default=None,
        help=f"For --openreview: target fraction of accepted papers (default: {OPENREVIEW_ACCEPTED_RATIO}). "
             f"Use 0.5 for balanced accept/reject.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_datasets",
        help="List all stored datasets and exit",
    )
    parser.add_argument(
        "--grade",
        action="store_true",
        help="Grade proposals in the dataset via pairwise comparison (requires existing dataset)",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Maximum number of pairs to evaluate during grading (default: all)",
    )
    parser.add_argument(
        "--grading_model",
        type=str,
        default=None,
        help=f"Model for grading (default: {GRADING_MODEL_NAME})",
    )

    args = parser.parse_args()

    if args.list_datasets:
        show_datasets()
        return

    if args.grade:
        grade_papers(
            conference=args.conference,
            year=args.year,
            max_pairs=args.max_pairs,
            grading_model=args.grading_model,
        )
        return

    if args.openreview:
        run_openreview_pipeline(
            conference=args.conference,
            min_year=args.year,
            min_month=args.min_month,
            max_papers=args.max_papers,
            skip_pdf=args.skip_pdf,
            accepted_ratio=args.accepted_ratio,
        )
        return

    run_pipeline(
        conference=args.conference,
        year=args.year,
        month_start=args.month_start,
        month_end=args.month_end,
        max_papers=args.max_papers,
        skip_pdf=args.skip_pdf,
        search_query=args.search_query,
    )


if __name__ == "__main__":
    main()
