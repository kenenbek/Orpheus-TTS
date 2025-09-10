"""Minimal QLoRA fine-tuning script (simplified, no defensive checks).
Assumes:
- config.yaml provides model_name and either TTS_dataset (str) or TTS_datasets (list of str)
- Dataset(s) have required columns (e.g., already tokenized with input_ids)
- bitsandbytes, transformers, peft, datasets installed
"""
import wandb
import yaml
import torch
import os
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel

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

model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]
gradient_accumulation_steps = config["gradient_accumulation_steps"]
save_total_limit = config["save_total_limit"]

# Optional: path to an existing LoRA adapter to continue training / reuse
lora_checkpoint = config.get("lora_checkpoint")  # e.g. "ft-checkpoints/merged" or a LoRA adapter dir
logging_steps = config["logging_steps"]

lora_rank = 32
lora_alpha = 64
lora_dropout = 0.05

wandb.init(
        project=project_name,
        name=run_name,
        config={
            "model_name": model_name,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
        reinit=True,
    )

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
)
model.gradient_checkpointing_enable()
model.config.use_cache = False

# LoRA init: either load existing adapter or create a new one
if lora_checkpoint:
    if os.path.isdir(lora_checkpoint) and os.path.exists(os.path.join(lora_checkpoint, "adapter_config.json")):
        print(f"[INFO] Loading existing LoRA adapter from: {lora_checkpoint}")
        model = PeftModel.from_pretrained(model, lora_checkpoint, is_trainable=True)
    else:
        print(f"[WARN] Provided lora_checkpoint '{lora_checkpoint}' does not look like a LoRA adapter directory (missing adapter_config.json). Initializing a fresh LoRA adapter instead.")
        lora_checkpoint = None  # fall through to new adapter creation

if not lora_checkpoint:
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["lm_head", "embed_tokens"],
        use_rslora=True
    )
    model = get_peft_model(model, lora_cfg)

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

# Add per-example length for sorting (expects 'input_ids' already present)
if "input_ids" not in ds.features:
    raise ValueError("Expected 'input_ids' in dataset features. Add a tokenization step before training.")

ds = ds.map(lambda ex: {"_len": len(ex["input_ids"])})
ds = ds.sort("_len", reverse=True)
N_skip = 10
ds = ds.select(range(N_skip, ds.num_rows))
print(f"[INFO] Final training dataset size after skip: {ds.num_rows}")

args = TrainingArguments(
    output_dir=f"./{base_repo_id}",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    save_steps=save_steps,
    bf16=torch.cuda.is_available(),
    report_to=["wandb"],
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    save_total_limit=save_total_limit,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
)

trainer.train()

merged_model = model.merge_and_unload()

merged_model.save_pretrained(f"./{base_repo_id}/merged")
tokenizer.save_pretrained(f"./{base_repo_id}/merged")
