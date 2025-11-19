#!/usr/bin/env python3
"""
Merge LoRA adapter into base model to produce a standalone HF model.

Usage (PowerShell):
  .venv/Scripts/python.exe scripts/merge_lora.py `
    --base Qwen/Qwen2-1.5B-Instruct `
    --adapter checkpoints/qwen2-1_5b-sft-med `
    --out artifacts/qwen2-1_5b-sft-med-merged
"""

import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2-1.5B-Instruct")
    ap.add_argument("--adapter", default="checkpoints/qwen2-1_5b-sft-med")
    ap.add_argument("--out", default="artifacts/qwen2-1_5b-sft-med-merged")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading base model on CPU (no offload)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)

    print("Loading adapter and merging...")
    merged = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False)
    merged = merged.merge_and_unload()  # bake LoRA into base weights

    print("Saving merged model...")
    merged.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("Merged model saved to:", args.out)


if __name__ == "__main__":
    main()

