"""
Data Preparation for Pairwise Preference Fine-Tuning.

Loads proposals from ORACL JSONL datasets, forms (accepted, rejected) cross-pairs
with swap augmentation (each pair appears in both orders), strips identifying
metadata, and exports train/val splits in SFT format.

Key anti-bias measures:
  1. Swap augmentation: every pair produces TWO examples (A=acc,B=rej AND A=rej,B=acc)
  2. Metadata stripping: no arXiv IDs, OpenReview IDs, author names, venue info
  3. Balanced position labels: exactly 50% of labels are "A", 50% are "B"
  4. Cross-pairing: all accepted × rejected combinations (not just natural pairs)

Usage:
    python -m ORACL.finetuning.prepare_data --out_dir ORACL/finetuning/data
    python -m ORACL.finetuning.prepare_data --conferences ICLR NeurIPS --years 2023 2024
    python -m ORACL.finetuning.prepare_data --max_pairs_per_conf 500 --val_ratio 0.15
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ORACL.modules.converter import Proposal
from ORACL.modules.dataset_store import load_proposals, list_datasets
from ORACL.config.settings import DATASET_DIR


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

SYSTEM_PROMPT = """You are an expert AI research reviewer. You will be shown two research proposals. One was accepted at a top venue and the other was rejected. Your task is to determine which proposal was accepted based purely on the quality of the research ideas, methodology, and experimental design.

Analyze both proposals carefully, then respond with ONLY "A" or "B" to indicate which proposal was accepted."""

PAIR_TEMPLATE = """**Proposal A:**

1. Title: {title_a}

2. Problem Statement: {problem_a}

3. Motivation: {motivation_a}

4. Proposed Method: {method_a}

5. Step-by-Step Experiment Plan: {experiment_a}

6. Test Case Examples: {tests_a}

7. Fallback Plan: {fallback_a}

---

**Proposal B:**

1. Title: {title_b}

2. Problem Statement: {problem_b}

3. Motivation: {motivation_b}

4. Proposed Method: {method_b}

5. Step-by-Step Experiment Plan: {experiment_b}

6. Test Case Examples: {tests_b}

7. Fallback Plan: {fallback_b}

---

Which proposal was accepted at the venue? Answer with ONLY "A" or "B"."""


# ============================================================================
# PAIR GENERATION
# ============================================================================

def load_all_proposals(
    conferences: List[str] = None,
    years: List[int] = None,
    dataset_dir: str = None,
) -> Tuple[List[Proposal], List[Proposal]]:
    """
    Load proposals and split into accepted/rejected lists.

    Returns:
        (accepted_proposals, rejected_proposals)
    """
    dataset_dir = dataset_dir or DATASET_DIR

    # Discover available datasets
    all_datasets = list_datasets(dataset_dir)

    if not all_datasets:
        print("No datasets found. Run the OpenReview pipeline first.")
        sys.exit(1)

    accepted = []
    rejected = []

    for ds in all_datasets:
        conf = ds["conference"]
        year = int(ds["year"])

        # Filter by requested conferences/years
        if conferences and conf not in [c.upper() for c in conferences]:
            continue
        if years and year not in years:
            continue

        proposals = load_proposals(conf, year, dataset_dir)
        for p in proposals:
            if p.accepted is True:
                accepted.append(p)
            elif p.accepted is False:
                rejected.append(p)
            # Skip p.accepted is None (arXiv-only, no label)

    return accepted, rejected


def format_proposal_fields(p: Proposal) -> dict:
    """Extract only the 7 content fields (no metadata) for prompt formatting."""
    return {
        "title": p.title,
        "problem": p.problem_statement,
        "motivation": p.motivation,
        "method": p.proposed_method,
        "experiment": p.experiment_plan,
        "tests": p.test_case_examples,
        "fallback": p.fallback_plan,
    }


def make_pair_example(
    accepted: Proposal,
    rejected: Proposal,
    accepted_is_a: bool,
) -> dict:
    """
    Create a single training example from an (accepted, rejected) pair.

    Args:
        accepted: The accepted proposal
        rejected: The rejected proposal
        accepted_is_a: If True, accepted goes in position A; otherwise position B

    Returns:
        Dict with keys: system, instruction, output, metadata
    """
    acc = format_proposal_fields(accepted)
    rej = format_proposal_fields(rejected)

    if accepted_is_a:
        a_fields, b_fields = acc, rej
        label = "A"
    else:
        a_fields, b_fields = rej, acc
        label = "B"

    instruction = PAIR_TEMPLATE.format(
        title_a=a_fields["title"],
        problem_a=a_fields["problem"],
        motivation_a=a_fields["motivation"],
        method_a=a_fields["method"],
        experiment_a=a_fields["experiment"],
        tests_a=a_fields["tests"],
        fallback_a=a_fields["fallback"],
        title_b=b_fields["title"],
        problem_b=b_fields["problem"],
        motivation_b=b_fields["motivation"],
        method_b=b_fields["method"],
        experiment_b=b_fields["experiment"],
        tests_b=b_fields["tests"],
        fallback_b=b_fields["fallback"],
    )

    return {
        "system": SYSTEM_PROMPT,
        "instruction": instruction,
        "output": label,
        # Metadata (not used in training, for debugging)
        "_accepted_title": accepted.title,
        "_rejected_title": rejected.title,
        "_accepted_is_a": accepted_is_a,
        "_accepted_id": accepted.source_openreview_id or accepted.source_arxiv_id,
        "_rejected_id": rejected.source_openreview_id or rejected.source_arxiv_id,
    }


def generate_pairs(
    accepted: List[Proposal],
    rejected: List[Proposal],
    max_pairs_per_conf: int = None,
    seed: int = 42,
) -> List[dict]:
    """
    Generate swap-augmented training pairs.

    For each (accepted, rejected) combination, creates TWO examples:
      - One with accepted as Proposal A (label = "A")
      - One with accepted as Proposal B (label = "B")

    This ensures the model can't learn a position shortcut.

    Args:
        accepted: List of accepted proposals
        rejected: List of rejected proposals
        max_pairs_per_conf: Cap on unique pairs before swap aug (None = all)
        seed: Random seed for reproducibility

    Returns:
        List of training examples (each pair produces 2 examples)
    """
    rng = random.Random(seed)

    # All cross-combinations
    all_pairs = [(a, r) for a in accepted for r in rejected]
    rng.shuffle(all_pairs)

    # Cap if requested
    if max_pairs_per_conf and len(all_pairs) > max_pairs_per_conf:
        all_pairs = all_pairs[:max_pairs_per_conf]

    # Swap augmentation: each pair → 2 examples
    examples = []
    for acc, rej in all_pairs:
        examples.append(make_pair_example(acc, rej, accepted_is_a=True))
        examples.append(make_pair_example(acc, rej, accepted_is_a=False))

    return examples


def train_val_split(
    examples: List[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """
    Split examples into train/val sets.

    Swap-augmented pairs are kept together (both orderings go to the same
    split) to prevent data leakage.
    """
    rng = random.Random(seed)

    # Group by pair identity (swap pairs share the same accepted+rejected IDs)
    pair_groups = {}
    for ex in examples:
        key = (ex["_accepted_id"], ex["_rejected_id"])
        pair_groups.setdefault(key, []).append(ex)

    keys = list(pair_groups.keys())
    rng.shuffle(keys)

    n_val = max(1, int(len(keys) * val_ratio))
    val_keys = set(keys[:n_val])

    train = []
    val = []
    for key in keys:
        if key in val_keys:
            val.extend(pair_groups[key])
        else:
            train.extend(pair_groups[key])

    # Shuffle within each split
    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


# ============================================================================
# EXPORT
# ============================================================================

def save_jsonl(examples: List[dict], path: str, strip_metadata: bool = True):
    """Save examples to JSONL, optionally stripping debug metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        for ex in examples:
            row = {k: v for k, v in ex.items() if not k.startswith("_")} if strip_metadata else ex
            f.write(json.dumps(row) + "\n")

    print(f"  ✓ Saved {len(examples)} examples to {path}")


def export_sharegpt_format(examples: List[dict], path: str):
    """
    Export in ShareGPT/conversation format (compatible with LLaMA-Factory,
    axolotl, and most fine-tuning frameworks).

    Format: {"conversations": [{"from": "system", ...}, {"from": "human", ...}, {"from": "gpt", ...}]}
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        for ex in examples:
            row = {
                "conversations": [
                    {"from": "system", "value": ex["system"]},
                    {"from": "human", "value": ex["instruction"]},
                    {"from": "gpt", "value": ex["output"]},
                ]
            }
            f.write(json.dumps(row) + "\n")

    print(f"  ✓ Saved {len(examples)} examples (ShareGPT) to {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare pairwise preference data for fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ORACL.finetuning.prepare_data
  python -m ORACL.finetuning.prepare_data --conferences ICLR NeurIPS --years 2023 2024
  python -m ORACL.finetuning.prepare_data --max_pairs 500 --val_ratio 0.15
  python -m ORACL.finetuning.prepare_data --out_dir ORACL/finetuning/data --debug
        """,
    )

    parser.add_argument("--conferences", nargs="+", default=None,
                        help="Filter by conference(s). Default: all")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Filter by year(s). Default: all")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Max unique pairs before swap augmentation (default: all)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of pairs for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data"),
                        help="Output directory (default: ORACL/finetuning/data/)")
    parser.add_argument("--debug", action="store_true",
                        help="Keep metadata fields (prefixed with _) in output")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"ORACL Fine-Tuning Data Preparation")
    print(f"{'='*60}")

    # Load proposals
    accepted, rejected = load_all_proposals(
        conferences=args.conferences,
        years=args.years,
    )

    print(f"  Loaded {len(accepted)} accepted + {len(rejected)} rejected proposals")

    if not accepted or not rejected:
        print("\n  Error: Need at least 1 accepted and 1 rejected proposal.")
        print("  Run the OpenReview pipeline first:")
        print("    python -m ORACL.main --openreview --conference ICLR --year 2024 --max_papers 50")
        sys.exit(1)

    # Generate pairs with swap augmentation
    examples = generate_pairs(
        accepted=accepted,
        rejected=rejected,
        max_pairs_per_conf=args.max_pairs,
        seed=args.seed,
    )

    n_label_a = sum(1 for ex in examples if ex["output"] == "A")
    n_label_b = sum(1 for ex in examples if ex["output"] == "B")
    print(f"  Generated {len(examples)} examples ({n_label_a} label-A, {n_label_b} label-B)")

    # Split
    train, val = train_val_split(examples, val_ratio=args.val_ratio, seed=args.seed)
    print(f"  Split: {len(train)} train, {len(val)} val")

    # Save both formats
    strip = not args.debug
    save_jsonl(train, os.path.join(args.out_dir, "train.jsonl"), strip_metadata=strip)
    save_jsonl(val, os.path.join(args.out_dir, "val.jsonl"), strip_metadata=strip)

    # ShareGPT format (for LLaMA-Factory / axolotl)
    export_sharegpt_format(train, os.path.join(args.out_dir, "train_sharegpt.jsonl"))
    export_sharegpt_format(val, os.path.join(args.out_dir, "val_sharegpt.jsonl"))

    # Save config for reproducibility
    config = {
        "conferences": args.conferences,
        "years": args.years,
        "max_pairs": args.max_pairs,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "n_train": len(train),
        "n_val": len(val),
        "n_total_examples": len(examples),
    }
    config_path = os.path.join(args.out_dir, "data_config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved data config to {config_path}")

    print(f"\n{'='*60}")
    print(f"Done. Next steps:")
    print(f"  1. Collect more data:  python -m ORACL.main --openreview --conference ICLR --year 2024 --max_papers 200")
    print(f"  2. Train:              python -m ORACL.finetuning.train --data_dir {args.out_dir}")
    print(f"  3. Evaluate:           python -m ORACL.finetuning.eval --data_dir {args.out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
