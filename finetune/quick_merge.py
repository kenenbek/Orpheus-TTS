"""
Quick and simple checkpoint merger - just run this script!
It will automatically find your latest checkpoint and merge it.
"""

import os
import sys
from merge_checkpoint import merge_checkpoint, load_config
import torch
from pathlib import Path

def quick_merge():
    """Quick merge of the latest checkpoint"""
    
    print("\n" + "="*60)
    print("Quick Checkpoint Merger for Orpheus TTS")
    print("="*60)
    
    # Load config
    config = load_config("config_2.yaml")
    
    if config is None:
        print("ERROR: config_2.yaml not found!")
        return
    
    base_model = config.get("MODEL_NAME")
    checkpoints_dir = config.get("OUTPUT_DIR", "./finetuned_model")
    
    # Find all checkpoints
    checkpoint_dirs = sorted([
        d for d in Path(checkpoints_dir).glob("checkpoint-*")
        if d.is_dir() and (d / "adapter_config.json").exists()
    ])
    
    if not checkpoint_dirs:
        print(f"\nNo checkpoints found in {checkpoints_dir}")
        print("Please run full_training.py first to create checkpoints.")
        return
    
    print(f"\nFound {len(checkpoint_dirs)} checkpoint(s):")
    for i, ckpt in enumerate(checkpoint_dirs):
        print(f"  {i+1}. {ckpt.name}")
    
    # Get user choice
    if len(checkpoint_dirs) == 1:
        selected = checkpoint_dirs[0]
        print(f"\nUsing checkpoint: {selected.name}")
    else:
        print(f"\nLatest checkpoint: {checkpoint_dirs[-1].name}")
        choice = input("\nMerge latest checkpoint? (y/n, or enter number): ").strip().lower()
        
        if choice == 'y' or choice == '':
            selected = checkpoint_dirs[-1]
        elif choice == 'n':
            print("Aborted.")
            return
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(checkpoint_dirs):
                    selected = checkpoint_dirs[idx]
                else:
                    print("Invalid selection!")
                    return
            except ValueError:
                print("Invalid input!")
                return
    
    # Set output path
    output_path = f"./{selected.name}_merged"
    
    print(f"\nMerging {selected.name}...")
    print(f"Output: {output_path}")
    print(f"Base model: {base_model}")
    
    # Perform merge
    try:
        merge_checkpoint(
            str(selected),
            output_path,
            base_model_path=base_model,
            dtype=torch.bfloat16
        )
        
        print("\n" + "="*60)
        print("SUCCESS! Your merged model is ready to use.")
        print("="*60)
        print(f"\nUpdate your generate script with:")
        print(f'CHECKPOINT_PATH = "./{selected.name}_merged"')
        
    except Exception as e:
        print(f"\nERROR during merge: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_merge()

