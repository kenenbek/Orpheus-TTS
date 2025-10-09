import logging
import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from peft import PeftModel
from utils import get_last_checkpoint


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def merge_lora_checkpoint(config_file="config.yaml", output_subdir="merged_checkpoint"):
    """
    Load the last checkpoint with LoRA adapters and merge them with the base model.
    Saves the merged model ready for inference.
    
    Args:
        config_file: Path to the config.yaml file
        output_subdir: Subdirectory name for saving the merged model
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_file}...")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    
    MODEL_NAME = config["MODEL_NAME"]
    OUTPUT_DIR = config["OUTPUT_DIR"]
    
    # Find the last checkpoint
    logger.info(f"Searching for checkpoints in {OUTPUT_DIR}...")
    last_checkpoint_path = get_last_checkpoint(OUTPUT_DIR)
    
    if last_checkpoint_path is None:
        logger.error("No checkpoint found! Please train the model first.")
        return None
    
    logger.info(f"Loading base model: {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    logger.info(f"Loading LoRA adapters from checkpoint: {last_checkpoint_path}...")
    model = PeftModel.from_pretrained(base_model, last_checkpoint_path)
    
    logger.info("Merging LoRA adapters with base model...")
    merged_model = model.merge_and_unload()
    
    # Save the merged model
    output_path = Path(OUTPUT_DIR) / output_subdir
    logger.info(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)
    
    # Also save the tokenizer
    logger.info("Saving tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"✓ Successfully merged and saved model to: {output_path}")
    logger.info(f"You can now load this model for inference using:")
    logger.info(f"  model = OrpheusModel(model_name='{output_path}', ...)")
    
    return str(output_path)


if __name__ == "__main__":
    merge_lora_checkpoint()

