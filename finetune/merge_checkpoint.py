"""
Merge LoRA checkpoint with base model for Orpheus TTS inference
This script converts HuggingFace Trainer checkpoints into a merged model
that can be loaded directly with OrpheusModel.
"""

import os
import sys
import argparse
import logging
import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_file="config_2.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_file} not found. Using defaults.")
        return None


def merge_checkpoint(checkpoint_path, output_path, base_model_path=None, 
                     dtype=torch.bfloat16, save_tokenizer=True):
    """
    Merge LoRA checkpoint with base model
    
    Args:
        checkpoint_path: Path to the LoRA checkpoint directory
        output_path: Path to save the merged model
        base_model_path: Path to base model (if None, reads from checkpoint config)
        dtype: Data type for model weights
        save_tokenizer: Whether to save tokenizer with merged model
    """
    
    logger.info(f"Starting merge process...")
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Output path: {output_path}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Check if this is a LoRA checkpoint
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise ValueError(f"No adapter_config.json found in {checkpoint_path}. "
                        "This doesn't appear to be a LoRA checkpoint.")
    
    try:
        # Load PEFT config to get base model path
        logger.info("Loading PEFT configuration...")
        peft_config = PeftConfig.from_pretrained(checkpoint_path)
        
        if base_model_path is None:
            base_model_path = peft_config.base_model_name_or_path
            logger.info(f"Using base model from checkpoint config: {base_model_path}")
        
        # Load base model
        logger.info(f"Loading base model from {base_model_path}...")
        logger.info("This may take a few minutes...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        logger.info(f"Loading LoRA weights from {checkpoint_path}...")
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            torch_dtype=dtype
        )
        
        # Merge LoRA weights with base model
        logger.info("Merging LoRA weights with base model...")
        merged_model = model.merge_and_unload()
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save merged model
        logger.info(f"Saving merged model to {output_path}...")
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        
        # Save tokenizer if requested
        if save_tokenizer:
            logger.info("Saving tokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model_path)
                tokenizer.save_pretrained(output_path)
            except Exception as e:
                logger.warning(f"Could not save tokenizer: {e}")
        
        logger.info("="*60)
        logger.info("✓ MERGE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Merged model saved to: {output_path}")
        logger.info(f"\nYou can now load this model with:")
        logger.info(f'model = OrpheusModel(model_name="{output_path}", max_model_len=2048)')
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error during merge: {e}")
        raise


def merge_all_checkpoints(checkpoints_dir, output_base_dir, base_model_path=None, 
                          checkpoint_pattern="checkpoint-*"):
    """
    Merge all checkpoints in a directory
    
    Args:
        checkpoints_dir: Directory containing checkpoint folders
        output_base_dir: Base directory for merged models
        base_model_path: Path to base model (optional)
        checkpoint_pattern: Pattern to match checkpoint folders
    """
    
    logger.info(f"Searching for checkpoints in {checkpoints_dir}")
    
    # Find all checkpoint directories
    checkpoint_dirs = sorted([
        d for d in Path(checkpoints_dir).glob(checkpoint_pattern)
        if d.is_dir() and (d / "adapter_config.json").exists()
    ])
    
    if not checkpoint_dirs:
        logger.warning(f"No checkpoints found matching pattern '{checkpoint_pattern}' in {checkpoints_dir}")
        return []
    
    logger.info(f"Found {len(checkpoint_dirs)} checkpoints to merge")
    
    merged_paths = []
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_name = checkpoint_dir.name
        output_path = os.path.join(output_base_dir, f"{checkpoint_name}_merged")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {checkpoint_name}...")
        logger.info(f"{'='*60}")
        
        try:
            merged_path = merge_checkpoint(
                str(checkpoint_dir),
                output_path,
                base_model_path=base_model_path
            )
            merged_paths.append(merged_path)
        except Exception as e:
            logger.error(f"Failed to merge {checkpoint_name}: {e}")
            continue
    
    return merged_paths


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA checkpoints with base model for Orpheus TTS"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to specific checkpoint to merge"
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="./finetuned_model",
        help="Directory containing multiple checkpoints to merge (default: ./finetuned_model)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for merged model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Path to base model (optional, will read from checkpoint if not provided)"
    )
    parser.add_argument(
        "--merge-all",
        action="store_true",
        help="Merge all checkpoints in checkpoints-dir"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_2.yaml",
        help="Path to config file (default: config_2.yaml)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights (default: bfloat16)"
    )
    
    args = parser.parse_args()
    
    # Load config file if exists
    config = load_config(args.config)
    
    # Set dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Get base model path
    base_model_path = args.base_model
    if base_model_path is None and config:
        base_model_path = config.get("MODEL_NAME")
    
    # Merge all checkpoints or a single one
    if args.merge_all:
        output_base = args.output if args.output else "./merged_models"
        merged_paths = merge_all_checkpoints(
            args.checkpoints_dir,
            output_base,
            base_model_path=base_model_path
        )
        
        if merged_paths:
            logger.info("\n" + "="*60)
            logger.info("ALL CHECKPOINTS MERGED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"Merged {len(merged_paths)} checkpoints:")
            for path in merged_paths:
                logger.info(f"  - {path}")
        else:
            logger.warning("No checkpoints were successfully merged")
            
    elif args.checkpoint:
        # Merge single checkpoint
        if not args.output:
            checkpoint_name = Path(args.checkpoint).name
            args.output = f"./{checkpoint_name}_merged"
        
        merge_checkpoint(
            args.checkpoint,
            args.output,
            base_model_path=base_model_path,
            dtype=dtype
        )
    
    else:
        # Interactive mode - find latest checkpoint
        checkpoints_dir = args.checkpoints_dir
        
        if not os.path.exists(checkpoints_dir):
            logger.error(f"Checkpoints directory not found: {checkpoints_dir}")
            logger.info("\nUsage examples:")
            logger.info("  # Merge specific checkpoint:")
            logger.info("  python merge_checkpoint.py --checkpoint ./finetuned_model/checkpoint-36000 --output ./merged_model")
            logger.info("\n  # Merge all checkpoints:")
            logger.info("  python merge_checkpoint.py --merge-all --checkpoints-dir ./finetuned_model")
            logger.info("\n  # Merge latest checkpoint (auto-detect):")
            logger.info("  python merge_checkpoint.py")
            return
        
        # Find latest checkpoint
        checkpoint_dirs = sorted([
            d for d in Path(checkpoints_dir).glob("checkpoint-*")
            if d.is_dir() and (d / "adapter_config.json").exists()
        ])
        
        if not checkpoint_dirs:
            logger.error(f"No checkpoints found in {checkpoints_dir}")
            return
        
        latest_checkpoint = checkpoint_dirs[-1]
        logger.info(f"Found {len(checkpoint_dirs)} checkpoints")
        logger.info(f"Using latest checkpoint: {latest_checkpoint}")
        
        output_path = args.output if args.output else f"./{latest_checkpoint.name}_merged"
        
        merge_checkpoint(
            str(latest_checkpoint),
            output_path,
            base_model_path=base_model_path,
            dtype=dtype
        )


if __name__ == "__main__":
    main()

