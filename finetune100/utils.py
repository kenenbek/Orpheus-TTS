import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_last_checkpoint(output_dir):
    """
    Find the most recent checkpoint in the output directory.
    Returns the checkpoint path or None if no checkpoints exist.
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        logger.info(f"Output directory '{output_dir}' does not exist. Starting training from scratch.")
        return None
    
    # Find all checkpoint directories (format: checkpoint-XXXX)
    checkpoint_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    
    if not checkpoint_dirs:
        logger.info(f"No checkpoints found in '{output_dir}'. Starting training from scratch.")
        return None
    
    # Sort by step number (extract number from checkpoint-XXXX)
    checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[-1]))
    last_checkpoint = checkpoint_dirs[-1]
    
    logger.info(f"Found {len(checkpoint_dirs)} checkpoint(s). Resuming from: {last_checkpoint}")
    return str(last_checkpoint)

