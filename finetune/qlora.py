"""Minimal QLoRA fine-tuning script (simplified, no defensive checks).
Assumes:
- config.yaml provides model_name and TTS_dataset
- Dataset has a 'text' column
- bitsandbytes, transformers, peft, datasets installed
"""
import os
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

MODEL_NAME = cfg["model_name"]
DATASET_NAME = cfg["TTS_dataset"]
OUTPUT_DIR = f"./{cfg.get('save_folder','checkpoints')}-qlora"
BASE_REPO_ID = cfg.get('save_folder','checkpoints')  # for merged save path parity with lora.py
EPOCHS = cfg.get("epochs", 1)
BATCH = cfg.get("batch_size", 1)
LR = cfg.get("learning_rate", 5e-5)
SAVE_STEPS = cfg.get("save_steps", 1000)
RUN_NAME = cfg.get("run_name", "qlora")
PROJECT_NAME = cfg.get("project_name", "qlora")
PAD_TOKEN_ID = cfg.get("pad_token")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    PAD_TOKEN_ID = tokenizer.pad_token_id if PAD_TOKEN_ID is None else PAD_TOKEN_ID
else:
    if PAD_TOKEN_ID is None:
        PAD_TOKEN_ID = tokenizer.pad_token_id

# 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quant_config,
    attn_implementation="sdpa",
)
model.gradient_checkpointing_enable()
model.config.use_cache = False

# LoRA config (fixed simple values)
lora_cfg = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
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
)
model = get_peft_model(model, lora_cfg)

# Dataset (assumes 'text' column)
ds = load_dataset(DATASET_NAME, split="train")
# take a small shuffled subset for quick/fitting training
ds = ds.shuffle(seed=42)
ds = ds.select(range(500))


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=8,
    learning_rate=LR,
    logging_steps=10,
    save_steps=SAVE_STEPS,
    bf16=torch.cuda.is_available(),
    report_to=[],
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
)

trainer.train()

# Save adapter
adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
trainer.model.save_pretrained(adapter_dir)
try:
    tokenizer.save_pretrained(adapter_dir)
except Exception:
    pass

# Merge and save full model (same pattern as lora.py)
merged_model = model.merge_and_unload()
merged_path = f"./{BASE_REPO_ID}/merged"
merged_model.save_pretrained(merged_path)
try:
    tokenizer.save_pretrained(merged_path)
except Exception:
    pass
