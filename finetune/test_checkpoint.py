"""
Test and diagnose checkpoint loading issues
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def test_checkpoint(checkpoint_path):
    """Test if checkpoint can be loaded properly"""
    
    print("="*60)
    print("Orpheus Checkpoint Diagnostic Tool")
    print("="*60)
    
    if not os.path.exists(checkpoint_path):
        print(f"\n✗ ERROR: Checkpoint not found at {checkpoint_path}")
        return False
    
    print(f"\n✓ Checkpoint directory exists: {checkpoint_path}")
    
    # Check required files
    required_files = ["config.json", "model.safetensors"]
    optional_files = ["tokenizer.json", "tokenizer_config.json"]
    
    print("\nChecking required files:")
    for file in required_files:
        file_path = os.path.join(checkpoint_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024**3)  # GB
            print(f"  ✓ {file} ({size:.2f} GB)")
        else:
            print(f"  ✗ {file} - MISSING!")
            return False
    
    print("\nChecking optional files:")
    for file in optional_files:
        file_path = os.path.join(checkpoint_path, file)
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ⚠ {file} - not found (may use default)")
    
    # Try loading tokenizer
    print("\n" + "="*60)
    print("Testing tokenizer loading...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Pad token: {tokenizer.pad_token_id}")
    except Exception as e:
        print(f"✗ Tokenizer loading failed: {e}")
        return False
    
    # Try loading model config
    print("\n" + "="*60)
    print("Testing model config...")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(checkpoint_path)
        print(f"✓ Config loaded successfully")
        print(f"  Model type: {config.model_type}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")
        print(f"  Vocab size: {config.vocab_size}")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False
    
    # Try loading model (without putting on GPU to save memory)
    print("\n" + "="*60)
    print("Testing model loading (CPU, low memory mode)...")
    print("This may take a minute...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        print(f"✓ Model loaded successfully on CPU")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test GPU loading if available
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("Testing GPU loading...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print(f"✓ Model loaded successfully on GPU")
        except Exception as e:
            print(f"✗ GPU loading failed: {e}")
            print(f"Note: CPU loading works, so you can still use the model")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour checkpoint is valid and ready to use.")
    print("\nRecommendation: Use generate_examples_direct.py instead of")
    print("generate_examples_for_listening.py to avoid vLLM issues.")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "./checkpoint-36000_merged/"
    
    test_checkpoint(checkpoint_path)

