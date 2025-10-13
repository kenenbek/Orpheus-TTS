import logging
import yaml
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import wandb
import torch
from utils import get_last_checkpoint


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune.log")]
)
logger = logging.getLogger(__name__)

config_file = "config.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

MODEL_NAME = config["MODEL_NAME"]
DATASET = config["DATASET"]
OUTPUT_DIR = config["OUTPUT_DIR"]
KEEP_LAST_N_CHECKPOINTS = config["KEEP_LAST_N_CHECKPOINTS"]
LOGGING_STEPS = config["LOGGING_STEPS"]
SAVE_STEPS = config["SAVE_STEPS"]
NUM_EPOCHS = config["NUM_EPOCHS"]
BATCH_SIZE = config["BATCH_SIZE"]
GRADIENT_ACCUMULATION_STEPS = config["GRADIENT_ACCUMULATION_STEPS"]
GRADIENT_CHECKPOINTING = config["GRADIENT_CHECKPOINTING"]
LEARNING_RATE = float(config["LEARNING_RATE"])
SEED = config["SEED"]
PAD_TOKEN = config["PAD_TOKEN"]

LORA_RANK = config["LORA_RANK"]
LORA_ALPHA = config["LORA_ALPHA"]
LORA_DROPOUT = config["LORA_DROPOUT"]

# CUDA_VISIBLE_DEVICES=3 python training.py

logger.info("Loading dataset...")
ds = load_dataset(DATASET, split="train")
logger.info(f"Dataset loaded: {len(ds)} samples")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                            torch_dtype=torch.bfloat16)

# model.gradient_checkpointing_enable()
# model.config.use_cache = False

logger.info("Applying LoRA configuration...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj",  "o_proj", "gate_proj", "down_proj", "up_proj"],
    bias="none",
    modules_to_save=["lm_head", "embed_tokens"], # Optional to train the embeddings and lm head
    task_type="CAUSAL_LM",
    use_rslora=True,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

logger.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=KEEP_LAST_N_CHECKPOINTS,
    seed=SEED,
    bf16=True,
    report_to="wandb",
    logging_dir=f"{OUTPUT_DIR}/logs",
)

logger.info("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
)

# Check for existing checkpoints and resume if available
last_checkpoint_path = get_last_checkpoint(OUTPUT_DIR)

logger.info("Starting training...")
trainer.train(resume_from_checkpoint=last_checkpoint_path)

logger.info("Training complete. Merging and saving final model...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"./{OUTPUT_DIR}/merged")

logger.info(f"Final merged model saved to ./{OUTPUT_DIR}/merged")