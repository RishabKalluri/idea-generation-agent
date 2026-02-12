"""
Fine-Tuning Script for Pairwise Preference Model.

Trains a Qwen2.5 model (with LoRA) to predict which of two research
proposals was accepted at a venue.  Uses SFT on swap-augmented pairs
so the model learns quality signals rather than positional shortcuts.

Usage:
    python -m ORACL.finetuning.train
    python -m ORACL.finetuning.train --model Qwen/Qwen2.5-7B-Instruct --epochs 3
    python -m ORACL.finetuning.train --config ORACL/finetuning/config.yaml
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Pick best GPU BEFORE importing torch (must happen before CUDA init)
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
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig


# ============================================================================
# DEFAULTS
# ============================================================================

DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "data_dir": os.path.join(os.path.dirname(__file__), "data"),
    "output_dir": os.path.join(os.path.dirname(__file__), "checkpoints"),
    "epochs": 3,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "max_seq_length": 8192,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "bf16": True,
    "gradient_checkpointing": True,
    # LoRA
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
    # Quantization
    "use_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "use_double_quant": True,
    # Logging
    "logging_steps": 5,
    "save_steps": 50,
    "eval_steps": 50,
    "save_total_limit": 3,
    "seed": 42,
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(data_dir: str):
    """Load train/val JSONL files and return HF Datasets."""
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run prepare_data.py first.")
        sys.exit(1)

    def read_jsonl(path):
        rows = []
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    train_data = read_jsonl(train_path)
    val_data = read_jsonl(val_path) if os.path.exists(val_path) else []

    print(f"  Loaded {len(train_data)} train, {len(val_data)} val examples")
    return Dataset.from_list(train_data), Dataset.from_list(val_data) if val_data else None


def format_chat(example, tokenizer):
    """Format a single example into chat template string."""
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_quantization(cfg: dict) -> BitsAndBytesConfig:
    """Configure 4-bit quantization."""
    compute_dtype = getattr(torch, cfg["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["use_double_quant"],
    )


def setup_lora(cfg: dict) -> LoraConfig:
    """Configure LoRA adapter."""
    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(cfg: dict):
    """Load model with quantization and apply LoRA."""
    print(f"  Loading model: {cfg['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if cfg["bf16"] else torch.float16,
        "device_map": "cuda:0",
    }

    if cfg["use_4bit"]:
        model_kwargs["quantization_config"] = setup_quantization(cfg)

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **model_kwargs)

    if cfg["use_4bit"]:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg["gradient_checkpointing"],
        )

    # Apply LoRA
    lora_config = setup_lora(cfg)
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# ============================================================================
# TRAINING
# ============================================================================

def train(cfg: dict):
    """Run the full training pipeline."""
    print(f"\n{'='*60}")
    print(f"ORACL Pairwise Preference Fine-Tuning")
    print(f"{'='*60}")
    print(f"  Model          : {cfg['model_name']}")
    print(f"  Data           : {cfg['data_dir']}")
    print(f"  Output         : {cfg['output_dir']}")
    print(f"  Epochs         : {cfg['epochs']}")
    print(f"  Batch size     : {cfg['per_device_batch_size']} × {cfg['gradient_accumulation_steps']} accum")
    print(f"  Learning rate  : {cfg['learning_rate']}")
    print(f"  LoRA rank      : {cfg['lora_r']}")
    print(f"  4-bit quant    : {cfg['use_4bit']}")
    print(f"  Max seq length : {cfg['max_seq_length']}")
    print(f"{'='*60}")

    # Load data
    print(f"\n[1/3] Loading data...")
    train_dataset, val_dataset = load_data(cfg["data_dir"])

    # Load model
    print(f"\n[2/3] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Format data using chat template
    def formatting_func(example):
        messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Training args (SFTConfig extends TrainingArguments with SFT-specific fields)
    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        per_device_eval_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        bf16=cfg["bf16"],
        fp16=not cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=cfg["eval_steps"] if val_dataset else None,
        save_total_limit=cfg["save_total_limit"],
        seed=cfg["seed"],
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit" if cfg["use_4bit"] else "adamw_torch",
        lr_scheduler_type="cosine",
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg["gradient_checkpointing"] else None,
        max_length=cfg["max_seq_length"],
        packing=False,
    )

    # Trainer
    print(f"\n[3/3] Starting training...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        args=training_args,
    )

    trainer.train()

    # Save final model
    final_dir = os.path.join(cfg["output_dir"], "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n  ✓ Model saved to {final_dir}")

    # Save config
    config_path = os.path.join(cfg["output_dir"], "train_config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  ✓ Config saved to {config_path}")

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Next: python -m ORACL.finetuning.eval --model_dir {final_dir}")
    print(f"{'='*60}\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen for pairwise proposal acceptance prediction",
    )

    parser.add_argument("--model", type=str, default=DEFAULTS["model_name"],
                        help=f"HuggingFace model ID (default: {DEFAULTS['model_name']})")
    parser.add_argument("--data_dir", type=str, default=DEFAULTS["data_dir"],
                        help="Directory with train.jsonl/val.jsonl")
    parser.add_argument("--output_dir", type=str, default=DEFAULTS["output_dir"],
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["per_device_batch_size"])
    parser.add_argument("--grad_accum", type=int, default=DEFAULTS["gradient_accumulation_steps"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--max_seq_length", type=int, default=DEFAULTS["max_seq_length"])
    parser.add_argument("--lora_r", type=int, default=DEFAULTS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=DEFAULTS["lora_alpha"])
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (overrides CLI args)")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    args = parser.parse_args()

    # Start with defaults
    cfg = dict(DEFAULTS)

    # Override from YAML config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        cfg.update(yaml_cfg)
        print(f"  Loaded config from {args.config}")

    # Override from CLI
    cfg["model_name"] = args.model
    cfg["data_dir"] = args.data_dir
    cfg["output_dir"] = args.output_dir
    cfg["epochs"] = args.epochs
    cfg["per_device_batch_size"] = args.batch_size
    cfg["gradient_accumulation_steps"] = args.grad_accum
    cfg["learning_rate"] = args.lr
    cfg["max_seq_length"] = args.max_seq_length
    cfg["lora_r"] = args.lora_r
    cfg["lora_alpha"] = args.lora_alpha
    cfg["use_4bit"] = not args.no_4bit
    cfg["seed"] = args.seed

    train(cfg)


if __name__ == "__main__":
    main()
