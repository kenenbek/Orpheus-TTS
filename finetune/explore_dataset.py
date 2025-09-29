"""Minimal QLoRA fine-tuning script (simplified, no defensive checks).
Assumes:
- config.yaml provides model_name and TTS_dataset
- Dataset has a 'text' column
- bitsandbytes, transformers, peft, datasets installed
"""
import yaml
import torch
from datasets import load_dataset, concatenate_datasets


config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

# Support multiple datasets: config can define TTS_datasets: [list, ...] or legacy TTS_dataset: "name"
dataset_names = []
if "TTS_datasets" in config and config["TTS_datasets"]:
    if isinstance(config["TTS_datasets"], list):
        dataset_names = config["TTS_datasets"]
    else:
        raise ValueError("TTS_datasets must be a list of dataset identifiers.")
elif "TTS_dataset" in config:
    dataset_names = [config["TTS_dataset"]]
else:
    raise ValueError("You must specify either TTS_dataset or TTS_datasets in config.yaml")
print(f"[INFO] Loading {len(dataset_names)} dataset(s): {dataset_names}")

# Build training dataset (concatenate if multiple)
loaded_parts = []
for name in dataset_names:
    print(f"[INFO] Loading split 'train' from {name}")
    part = load_dataset(name, split="train")
    # Short info
    cols = list(part.features.keys())
    info_msg = f"[DATASET] {name}: rows={part.num_rows}, columns={cols}"
    if "input_ids" in part.features:
        try:
            sample_n = min(2048, part.num_rows)
            if sample_n > 0:
                subset = part.select(range(sample_n))
                lengths = [len(ids) for ids in subset["input_ids"]]
                if lengths:
                    avg_len = sum(lengths) / len(lengths)
                    info_msg += f", sample_avg_input_ids_len~{avg_len:.1f} (first {sample_n})"
        except Exception:
            pass
    print(info_msg)
    loaded_parts.append(part)

if len(loaded_parts) == 1:
    ds = loaded_parts[0]
else:
    # Simple concatenation (optionally could use interleave_datasets for sampling)
    ds = concatenate_datasets(loaded_parts)
    print(f"[INFO] Combined dataset size: {ds.num_rows}")


ds = ds.map(lambda ex: {"_len": len(ex["input_ids"])})
ds = ds.sort("_len", reverse=False)
N_skip = 0
ds = ds.select(range(N_skip, ds.num_rows))
print(f"[INFO] Final training dataset size after skip: {ds.num_rows}")

def print_row_element_lengths(ds, index=0, keys=None):
    """
    Print length of each element (column) in dataset row `index`.
    If an element has no length (e.g. int), print its type and value.
    """
    row = ds[index]
    keys = keys or list(row.keys())
    for k in keys:
        v = row.get(k)
        try:
            l = len(v)
            print(f"[ROW {index}] {k}: len={l}")
        except Exception:
            print(f"[ROW {index}] {k}: type={type(v).__name__}, value={v}")

# Example: print lengths for the first 5 rows
for i in range(ds.num_rows):
    print_row_element_lengths(ds, index=i)
    print("-" * 40)