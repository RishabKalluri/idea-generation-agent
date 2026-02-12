"""
Dataset Storage Module.

Stores converted proposals in JSONL files organized by conference and year.
Supports appending, loading, and querying the dataset.
"""

import os
import json
from typing import List, Optional
from datetime import datetime

from .converter import Proposal


def _get_dataset_dir() -> str:
    """Get dataset directory from settings."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")


def get_store_path(conference: str, year: int, dataset_dir: str = None) -> str:
    """
    Get the JSONL file path for a conference/year combo.

    Example: dataset/ACL/2024.jsonl
    """
    dataset_dir = dataset_dir or _get_dataset_dir()
    conf_dir = os.path.join(dataset_dir, conference.upper())
    os.makedirs(conf_dir, exist_ok=True)
    return os.path.join(conf_dir, f"{year}.jsonl")


def get_existing_ids(
    conference: str,
    year: int,
    dataset_dir: str = None,
) -> set:
    """
    Get the set of arXiv IDs and OpenReview IDs already stored for a conference/year.

    Use this to skip papers that have already been processed.
    """
    path = get_store_path(conference, year, dataset_dir)
    ids = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    aid = entry.get("source_arxiv_id", "")
                    if aid:
                        ids.add(aid)
                    orid = entry.get("source_openreview_id", "")
                    if orid:
                        ids.add(orid)
                except json.JSONDecodeError:
                    continue
    return ids


def save_proposals(
    proposals: List[Proposal],
    conference: str,
    year: int,
    dataset_dir: str = None,
) -> str:
    """
    Append proposals to the JSONL store for a conference/year.

    Args:
        proposals: List of Proposal objects to save
        conference: Conference name (e.g. "ACL")
        year: Publication year
        dataset_dir: Override dataset directory

    Returns:
        Path to the JSONL file
    """
    path = get_store_path(conference, year, dataset_dir)

    # Load existing IDs to avoid duplicates
    existing_ids = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    aid = entry.get("source_arxiv_id", "")
                    if aid:
                        existing_ids.add(aid)
                    orid = entry.get("source_openreview_id", "")
                    if orid:
                        existing_ids.add(orid)
                except json.JSONDecodeError:
                    continue

    new_count = 0
    with open(path, "a") as f:
        for proposal in proposals:
            # Check both ID types for dedup
            pid = proposal.source_arxiv_id or proposal.source_openreview_id
            if pid in existing_ids:
                continue
            f.write(json.dumps(proposal.to_dict()) + "\n")
            existing_ids.add(pid)
            new_count += 1

    print(f"  ✓ Saved {new_count} new proposals to {path} ({len(existing_ids)} total)")
    return path


def load_proposals(
    conference: str,
    year: int,
    dataset_dir: str = None,
) -> List[Proposal]:
    """
    Load all proposals for a conference/year.

    Args:
        conference: Conference name
        year: Publication year
        dataset_dir: Override dataset directory

    Returns:
        List of Proposal objects
    """
    path = get_store_path(conference, year, dataset_dir)

    if not os.path.exists(path):
        return []

    proposals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                proposals.append(Proposal.from_dict(d))
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  Warning: Skipping malformed entry: {e}")

    return proposals


def list_datasets(dataset_dir: str = None) -> List[dict]:
    """
    List all available datasets.

    Returns:
        List of dicts with keys: conference, year, count, path
    """
    dataset_dir = dataset_dir or _get_dataset_dir()

    if not os.path.exists(dataset_dir):
        return []

    datasets = []
    for conf in sorted(os.listdir(dataset_dir)):
        conf_dir = os.path.join(dataset_dir, conf)
        if not os.path.isdir(conf_dir):
            continue

        for fname in sorted(os.listdir(conf_dir)):
            if not fname.endswith(".jsonl"):
                continue

            year = fname.replace(".jsonl", "")
            path = os.path.join(conf_dir, fname)

            # Count lines
            count = 0
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        count += 1

            datasets.append({
                "conference": conf,
                "year": year,
                "count": count,
                "path": path,
            })

    return datasets


def get_stats(dataset_dir: str = None) -> dict:
    """Get summary statistics for the full dataset store."""
    datasets = list_datasets(dataset_dir)
    total = sum(d["count"] for d in datasets)

    return {
        "total_proposals": total,
        "num_datasets": len(datasets),
        "datasets": datasets,
    }
