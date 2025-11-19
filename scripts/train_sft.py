#!/usr/bin/env python3
"""
QLoRA SFT training script for medical assistant using TRL/PEFT.

Default inputs:
- Dataset: data/sft/train_med.jsonl
- Output : checkpoints/qwen2-1_5b-sft-med

Usage (PowerShell):
  .venv/Scripts/python.exe scripts/train_sft.py `
    --model Qwen/Qwen2.5-7B-Instruct `
    --train data/sft/train_med.jsonl `
    --eval data/sft/dev_med.jsonl `
    --out checkpoints/qwen2_5-7b-sft-med

Notes:
- Uses 4-bit QLoRA (bitsandbytes). If it fails on Windows, consider WSL2.
- Expects JSONL entries with keys: instruction, input, output, system.
- Memory-saving defaults: gradient checkpointing, SDPA attention, smaller seq_len via --seq_len.
"""

import argparse
import json
from typing import Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


def build_messages(example: Dict) -> Dict:
    system = example.get("system") or "You are a clinical assistant."
    instruction = example.get("instruction") or "Answer strictly in JSON."
    _input = example.get("input") or ""
    output = example.get("output") or "{}"

    # Expect the output field to already be a JSON string (target)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{instruction}\n\n{_input}".strip()},
        {"role": "assistant", "content": output}
    ]
    return {"messages": messages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train", default="data/sft/train_med.jsonl")
    ap.add_argument("--eval", default="data/sft/dev_med.jsonl")
    ap.add_argument("--out", default="checkpoints/qwen2_5-7b-sft-med")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=0, help="If > 0, overrides epochs and trains for this many steps")
    ap.add_argument("--lora_r", type=int, default=8, help="LoRA rank (smaller is lighter, e.g., 4 or 8)")
    args = ap.parse_args()

    print("Loading datasets...")
    train_ds = load_dataset("json", data_files=args.train, split="train")
    eval_ds = load_dataset("json", data_files=args.eval, split="train") if args.eval else None

    # Map to chat messages
    train_ds = train_ds.map(build_messages, remove_columns=train_ds.column_names)
    if eval_ds:
        eval_ds = eval_ds.map(build_messages, remove_columns=eval_ds.column_names)

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit QLoRA config
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="sdpa",
    )

    # Enable memory savings
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    # Transformers convention to allow gradient checkpointing
    if hasattr(model, "config"):
        model.config.use_cache = False

    # LoRA target modules cover Q/K/V/Proj blocks for Qwen2
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    print("Starting SFT training...")
    training_args = SFTConfig(
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 0,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        save_steps=500,
        bf16=True,
        output_dir=args.out,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=500 if eval_ds else None,
        save_total_limit=2,
        gradient_checkpointing=True,
        group_by_length=True,
        optim="paged_adamw_8bit",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )

    trainer.train()
    print("Training completed. Saving adapter and tokenizer...")
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print("Saved to:", args.out)


if __name__ == "__main__":
    main()
