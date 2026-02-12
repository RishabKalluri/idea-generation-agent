"""
Evaluation Script for Pairwise Preference Model.

Measures:
  1. Accuracy — fraction of pairs where the model correctly identifies the accepted proposal
  2. Position consistency — fraction of swap-pairs where the model gives the same answer
     regardless of which proposal is A vs B (detects position bias)
  3. Position bias — whether the model systematically favors A or B

Usage:
    python -m ORACL.finetuning.eval --model_dir ORACL/finetuning/checkpoints/final
    python -m ORACL.finetuning.eval --model_dir ORACL/finetuning/checkpoints/final --data_dir ORACL/finetuning/data
    python -m ORACL.finetuning.eval --base_model Qwen/Qwen2.5-7B-Instruct  # eval base model (no fine-tuning)
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Pick best GPU BEFORE importing torch
def _pick_best_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            capture_output=True, text=True
        )
        free_mem = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        best_gpu = max(range(len(free_mem)), key=lambda i: free_mem[i])
        print(f"  Selected GPU {best_gpu} ({free_mem[best_gpu]} MiB free)")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
    except Exception:
        pass

_pick_best_gpu()

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============================================================================
# INFERENCE
# ============================================================================

def load_model_for_eval(
    model_dir: str = None,
    base_model: str = None,
    use_4bit: bool = True,
):
    """
    Load model for evaluation.

    Args:
        model_dir: Path to fine-tuned LoRA checkpoint (adapter_model/)
        base_model: HuggingFace model ID for base model eval (no adapter)
        use_4bit: Whether to use 4-bit quantization
    """
    if model_dir:
        # Load from fine-tuned checkpoint
        # Detect if this is a LoRA adapter or full model
        adapter_config = os.path.join(model_dir, "adapter_config.json")
        if os.path.exists(adapter_config):
            with open(adapter_config) as f:
                cfg = json.load(f)
            base_name = cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
            print(f"  Loading base model: {base_name}")
            print(f"  Loading LoRA adapter: {model_dir}")
        else:
            base_name = model_dir
            adapter_config = None
            print(f"  Loading full model: {model_dir}")
    elif base_model:
        base_name = base_model
        adapter_config = None
        print(f"  Loading base model (no fine-tuning): {base_name}")
    else:
        print("Error: Provide either --model_dir or --base_model")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(
        base_name,
        trust_remote_code=True,
        padding_side="left",  # left-pad for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": "cuda:0",
    }

    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)

    if model_dir and adapter_config:
        model = PeftModel.from_pretrained(model, model_dir)

    model.eval()
    return model, tokenizer


def predict_single(model, tokenizer, example: dict) -> str:
    """
    Run inference on a single pair and return the predicted label ("A" or "B").
    """
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["instruction"]},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Extract A or B from response
    response_upper = response.upper()
    if "A" in response_upper and "B" not in response_upper:
        return "A"
    elif "B" in response_upper and "A" not in response_upper:
        return "B"
    elif response_upper.startswith("A"):
        return "A"
    elif response_upper.startswith("B"):
        return "B"
    else:
        return response[:10]  # Return raw for debugging


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(results: list) -> dict:
    """
    Compute accuracy, position consistency, and position bias.

    Args:
        results: List of dicts with keys: predicted, expected, accepted_is_a, pair_id

    Returns:
        Dict of metric values
    """
    total = len(results)
    correct = sum(1 for r in results if r["predicted"] == r["expected"])
    accuracy = correct / total if total else 0

    # Position bias: how often does the model predict "A" regardless of truth?
    n_pred_a = sum(1 for r in results if r["predicted"] == "A")
    n_pred_b = sum(1 for r in results if r["predicted"] == "B")
    n_other = total - n_pred_a - n_pred_b
    a_rate = n_pred_a / total if total else 0

    # Position consistency: group swap pairs and check if they agree
    # Swap pairs share the same pair_id
    pair_groups = defaultdict(list)
    for r in results:
        pair_groups[r["pair_id"]].append(r)

    n_consistent = 0
    n_swap_pairs = 0
    for pid, group in pair_groups.items():
        if len(group) == 2:
            n_swap_pairs += 1
            # Both should predict the accepted proposal, but from different positions
            # If both are correct, OR both are wrong in a consistent way, they're consistent
            both_correct = all(r["predicted"] == r["expected"] for r in group)
            # More precisely: do both point to the same underlying proposal?
            # group[0] has accepted_is_a=True, group[1] has accepted_is_a=False (or vice versa)
            # If group[0] predicts "A" and group[1] predicts "B", they both pick the accepted → consistent
            # If group[0] predicts "B" and group[1] predicts "A", they both pick the rejected → consistent
            # Otherwise → inconsistent (position-biased)
            preds = {r["accepted_is_a"]: r["predicted"] for r in group}
            if True in preds and False in preds:
                # accepted_is_a=True and predicted "A" means picks accepted
                # accepted_is_a=False and predicted "B" means picks accepted
                picks_accepted_when_a = (preds.get(True) == "A")
                picks_accepted_when_b = (preds.get(False) == "B")
                if picks_accepted_when_a == picks_accepted_when_b:
                    n_consistent += 1

    consistency = n_consistent / n_swap_pairs if n_swap_pairs else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "position_consistency": consistency,
        "consistent_pairs": n_consistent,
        "total_swap_pairs": n_swap_pairs,
        "pred_A_rate": a_rate,
        "pred_A": n_pred_a,
        "pred_B": n_pred_b,
        "pred_other": n_other,
    }


# ============================================================================
# MAIN
# ============================================================================

def run_eval(
    model,
    tokenizer,
    data_path: str,
    output_path: str = None,
):
    """Run evaluation on a JSONL file."""
    # Load data
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"  Evaluating on {len(examples)} examples...")

    results = []
    for i, ex in enumerate(examples):
        predicted = predict_single(model, tokenizer, ex)
        expected = ex["output"]

        # Build pair_id from instruction hash for grouping swap pairs
        # Since we stripped metadata, use a hash of sorted titles from instruction
        import hashlib
        pair_id = hashlib.md5(
            "".join(sorted(ex["instruction"].split("Title: ")[1:3])).encode()
        ).hexdigest()[:12]

        result = {
            "predicted": predicted,
            "expected": expected,
            "correct": predicted == expected,
            "accepted_is_a": expected == "A",
            "pair_id": pair_id,
        }
        results.append(result)

        # Progress
        if (i + 1) % 10 == 0 or i == len(examples) - 1:
            running_acc = sum(r["correct"] for r in results) / len(results)
            print(f"    [{i+1}/{len(examples)}] Running accuracy: {running_acc:.1%}")

    # Compute metrics
    metrics = compute_metrics(results)

    # Print report
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy           : {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Position consistency: {metrics['position_consistency']:.1%} ({metrics['consistent_pairs']}/{metrics['total_swap_pairs']} swap pairs)")
    print(f"  Pred-A rate        : {metrics['pred_A_rate']:.1%} ({metrics['pred_A']}A / {metrics['pred_B']}B / {metrics['pred_other']}other)")
    print(f"{'='*60}")

    # Interpretation
    if metrics["pred_A_rate"] > 0.65:
        print(f"  ⚠ Position bias detected: model favors Proposal A ({metrics['pred_A_rate']:.0%})")
    elif metrics["pred_A_rate"] < 0.35:
        print(f"  ⚠ Position bias detected: model favors Proposal B ({1-metrics['pred_A_rate']:.0%})")
    else:
        print(f"  ✓ No significant position bias (A rate: {metrics['pred_A_rate']:.0%})")

    if metrics["position_consistency"] < 0.6:
        print(f"  ⚠ Low position consistency ({metrics['position_consistency']:.0%}) — model may be guessing by position")
    else:
        print(f"  ✓ Good position consistency ({metrics['position_consistency']:.0%})")

    random_baseline = 0.5
    if metrics["accuracy"] <= random_baseline + 0.05:
        print(f"  ⚠ Accuracy near random ({metrics['accuracy']:.0%}) — model may not have learned quality signals")
    else:
        print(f"  ✓ Above random baseline ({metrics['accuracy']:.0%} vs {random_baseline:.0%})")

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "metrics": metrics,
                "predictions": results,
            }, f, indent=2)
        print(f"\n  ✓ Results saved to {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pairwise preference model",
    )

    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to fine-tuned checkpoint (LoRA adapter)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="HuggingFace model ID for base model eval (no fine-tuning)")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data"),
                        help="Directory with val.jsonl")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"],
                        help="Which split to evaluate (default: val)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization")

    args = parser.parse_args()

    if not args.model_dir and not args.base_model:
        print("Error: Provide either --model_dir (fine-tuned) or --base_model (baseline)")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"ORACL Preference Model Evaluation")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model_for_eval(
        model_dir=args.model_dir,
        base_model=args.base_model,
        use_4bit=not args.no_4bit,
    )

    # Eval
    data_path = os.path.join(args.data_dir, f"{args.split}.jsonl")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        sys.exit(1)

    output_path = args.output or os.path.join(
        args.model_dir or ".",
        f"eval_{args.split}.json",
    )

    run_eval(model, tokenizer, data_path, output_path)


if __name__ == "__main__":
    main()
