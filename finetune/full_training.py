import os
import logging
import yaml
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoProcessor
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune.log")]
)
logger = logging.getLogger(__name__)

config_file = "config_2.yaml"
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

LORA_RANK = config["LORA_RANK"]
LORA_ALPHA = config["LORA_ALPHA"]
LORA_DROPOUT = config["LORA_DROPOUT"]

# CUDA_VISIBLE_DEVICES=3 python full_training.py


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                            attn_implementation="sdpa")

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj",  "o_proj", "gate_proj", "down_proj", "up_proj"],
    bias="none",
    modules_to_save=["lm_head", "embed_tokens"], # Optional to train the embeddings and lm head
    task_type="CAUSAL_LM",
    use_rslora=True,
)

model = get_peft_model(model, lora_config)
ds = load_dataset(DATASET)

wandb.init(project="OrpheusTTS")

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=LOGGING_STEPS,
    output_dir=f"./{OUTPUT_DIR}",
    report_to="wandb",
    save_steps=SAVE_STEPS,
    remove_unused_columns=True,
    learning_rate=LEARNING_RATE,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
)

trainer.train()

merged_model = model.merge_and_unload()

merged_model.save_pretrained(f"./{OUTPUT_DIR}/merged")