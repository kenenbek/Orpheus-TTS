"""Load a dataset and print all example lengths in tokens (no defensive checks).
Assumes:
- finetune/config.yaml provides TTS_dataset
- Dataset has an 'input_ids' column
"""
import os
import yaml
import datasets

# Resolve config.yaml relative to this script's directory
config_file = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

# Load, compute token lengths, and sort (no defensive checks)
ds = datasets.load_dataset(dsn, split="train")
ds = ds.map(lambda ex: {"_len": len(ex["input_ids"])})
ds = ds.sort("_len", reverse=True)

# Print all lengths, one per line
for L in ds["_len"]:
    print(int(L))
