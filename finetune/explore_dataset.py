"""Minimal QLoRA fine-tuning script (simplified, no defensive checks).
Assumes:
- config.yaml provides model_name and TTS_dataset
- Dataset has a 'text' column
- bitsandbytes, transformers, peft, datasets installed
"""
import yaml
import torch
from datasets import load_dataset


config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

ds = load_dataset(dsn, split="train")
# Add per-example length and a stable index, then sort by length descending
ds = ds.map(lambda ex: {"_len": len(ex["input_ids"])})
ds = ds.sort("_len", reverse=True)
N_skip = 0
ds = ds.select(range(N_skip, ds.num_rows))


for i, ex in enumerate(ds):
    print(f"{i}: {ex['_len']}")