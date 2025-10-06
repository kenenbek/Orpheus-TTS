# Troubleshooting vLLM EngineDeadError

## The Problem

You're getting a `vllm.v1.engine.exceptions.EngineDeadError` when trying to use `OrpheusModel` with your fine-tuned checkpoint. This happens because:

1. **vLLM engine crashes** when loading certain model configurations
2. **Fine-tuned models** sometimes have compatibility issues with vLLM
3. **Version mismatches** between vLLM and your model can cause this

## The Solution: Use Direct Inference

I've created **`generate_examples_direct.py`** which bypasses vLLM entirely and uses transformers directly. This is more stable for fine-tuned models.

## Quick Start

### Step 1: Test your checkpoint
```bash
cd /home/kenenbek/PycharmProjects/Orpheus-TTS/finetune
python test_checkpoint.py ./checkpoint-36000_merged/
```

This will verify your checkpoint is valid.

### Step 2: Generate audio with direct inference
```bash
python generate_examples_direct.py
```

This uses the more stable transformers-based approach.

## Files Created

1. **`generate_examples_direct.py`** - Main inference script (no vLLM)
2. **`test_checkpoint.py`** - Diagnostic tool to test checkpoint validity
3. **`merge_checkpoint.py`** - Merge LoRA checkpoints with base model
4. **`quick_merge.py`** - Quick checkpoint merger

## Differences Between Scripts

### `generate_examples_for_listening.py` (Original - vLLM)
- ✗ Uses vLLM async engine
- ✗ Can crash with `EngineDeadError`
- ✓ Faster for production if it works
- ✓ Streaming inference

### `generate_examples_direct.py` (New - Transformers)
- ✓ Uses transformers directly
- ✓ More stable with fine-tuned models
- ✓ Easier to debug
- ✓ Works with your checkpoint
- ✗ Slightly slower (but still fast)

## Troubleshooting

### If you still get errors with direct inference:

1. **Check CUDA memory:**
```bash
nvidia-smi
```

2. **Reduce model precision:**
Edit `generate_examples_direct.py`:
```python
torch_dtype=torch.float16  # instead of bfloat16
```

3. **Use CPU inference:**
Edit the initialization:
```python
model = OrpheusDirectInference(CHECKPOINT_PATH, device="cpu")
```

### If you want to fix vLLM:

The vLLM issue is usually caused by:

1. **Version incompatibility** - Try:
```bash
pip install vllm==0.7.3  # Revert to stable version
```

2. **Model config issues** - vLLM v1 has stricter requirements

3. **GPU memory** - vLLM needs more VRAM than transformers

## Customization

### Change voices in `generate_examples_direct.py`:
```python
VOICES = ["timur", "aiganysh", "your_voice_name"]
```

### Add more emotions:
```python
EMOTIONS = ["happy", "sad", "excited", "your_emotion"]
```

### Generate single audio:
```python
from generate_examples_direct import OrpheusDirectInference, generate_audio

model = OrpheusDirectInference("./checkpoint-36000_merged/")
generate_audio(
    model, 
    "саламатсызбы!", 
    voice="timur", 
    emotion="happy",
    output_filename="test.wav"
)
```

## Performance Comparison

| Method | Speed | Stability | Memory | Streaming |
|--------|-------|-----------|--------|-----------|
| vLLM | Fast | Low | High | Yes |
| Direct | Medium | High | Medium | No |

For fine-tuned models, **direct inference is recommended** until vLLM compatibility is confirmed.

## Next Steps

1. Run `python test_checkpoint.py` to verify your checkpoint
2. Run `python generate_examples_direct.py` to generate audio
3. Listen to the generated files in `./generated_audio_samples/`
4. Customize the scripts for your specific voices and texts

## Need More Help?

- Check if your checkpoint was merged properly: `ls -lh checkpoint-36000_merged/`
- Verify you have the required dependencies: `pip list | grep -E "transformers|torch|snac"`
- Try with a smaller max_tokens value (e.g., 800 instead of 1200)

