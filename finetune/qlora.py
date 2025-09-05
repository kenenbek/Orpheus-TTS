"""Minimal QLoRA fine-tuning script (simplified, no defensive checks).
Assumes:
- config.yaml provides model_name and TTS_dataset
- Dataset has a 'text' column
- bitsandbytes, transformers, peft, datasets installed
"""
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

config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

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

lora_rank = 32
lora_alpha = 64
lora_dropout = 0.05

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
    attn_implementation="sdpa",
)
model.gradient_checkpointing_enable()
model.config.use_cache = False

# LoRA config (fixed simple values)
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

ds = load_dataset(dsn, split="train")
ds = ds.shuffle(seed=42)
#ds = ds.select(range(6))


args = TrainingArguments(
    output_dir=f"./{base_repo_id}",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=16,
    learning_rate=learning_rate,
    logging_steps=50,
    save_steps=save_steps,
    bf16=torch.cuda.is_available(),
    report_to=["wandb"],
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,

    save_total_limit=5
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
