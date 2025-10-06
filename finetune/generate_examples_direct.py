"""
Generate audio examples using direct transformers inference (no vLLM)
This is more stable for fine-tuned checkpoints and avoids vLLM engine issues
"""

import os
import sys
import wave
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Import SNAC decoder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'orpheus_tts_pypi'))

# Configuration
CHECKPOINT_PATH = "./checkpoint-36000_merged/"  # Path to your merged checkpoint
OUTPUT_DIR = "./generated_audio_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Available voices (speakers)
VOICES = ["timur", "aiganysh"]

# Available emotions/tones
EMOTIONS = [
    "happy", "normal", "sad", "frustrated", "excited",
    "whisper", "panicky", "curious", "surprise", "fast",
    "crying", "deep", "sleepy", "angry", "shout", "slow"
]

# Kyrgyz language example texts
KYRGYZ_TEXTS = [
    "саламатсызбы, менин атым айгүл. мен кыргыз тилинде сүйлөп жатам.",
    "бишкек кыргызстандын борбору жана эң чоң шаары.",
    "ыссык-көл дүйнөдөгү эң чоң тоо көлдөрүнүн бири.",
    "кыргыз тили өтө сулуу жана бай тил.",
    "бүгүн аба ырайы абдан жакшы."
]


class OrpheusDirectInference:
    """Direct inference class without vLLM - more stable for fine-tuned models"""
    
    def __init__(self, model_path, device="cuda"):
        print(f"Loading model from {model_path}...")
        self.device = device
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model
        print("Loading model (this may take a few minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # Load SNAC decoder for audio generation
        print("Loading SNAC audio decoder...")
        from snac import SNAC
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        snac_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.snac_model = self.snac_model.to(snac_device)
        self.snac_device = snac_device
        
        print("✓ Model loaded successfully!")
    
    def format_prompt(self, text, voice=None):
        """Format prompt in Orpheus style"""
        if voice:
            adapted_prompt = f"{voice}: {text}"
        else:
            adapted_prompt = text
            
        prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        
        return all_input_ids
    
    def generate_tokens(self, text, voice=None, temperature=0.6, top_p=0.8, 
                       max_new_tokens=1200, repetition_penalty=1.3):
        """Generate speech tokens from text"""
        input_ids = self.format_prompt(text, voice).to(self.device)
        
        print(f"Generating tokens...")
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or 128263,
                eos_token_id=49158,  # Stop token
                return_dict_in_generate=True,
                output_scores=False
            )
        
        # Get only the generated tokens (exclude input)
        generated_tokens = outputs.sequences[0][input_ids.shape[1]:]
        return generated_tokens
    
    def tokens_to_audio(self, tokens):
        """Convert tokens to audio using SNAC decoder"""
        # Convert tokens to audio codes
        tokens_list = tokens.cpu().tolist()
        
        # Process tokens in groups of 7 (SNAC format)
        audio_chunks = []
        
        # Filter and prepare tokens
        processed_tokens = []
        for token in tokens_list:
            # Convert token ID to SNAC code
            # Tokens are offset by 128264 in Orpheus
            if token >= 128264:
                code = token - 128264
                processed_tokens.append(code)
        
        # Group into frames of 7
        num_frames = len(processed_tokens) // 7
        
        if num_frames == 0:
            print("Warning: Not enough tokens generated for audio")
            return None
        
        for frame_idx in range(num_frames):
            frame_start = frame_idx * 7
            frame = processed_tokens[frame_start:frame_start + 7]
            
            if len(frame) < 7:
                break
            
            # Decode frame to audio
            try:
                audio_chunk = self._decode_frame(frame)
                if audio_chunk is not None:
                    audio_chunks.append(audio_chunk)
            except Exception as e:
                print(f"Warning: Could not decode frame {frame_idx}: {e}")
                continue
        
        if not audio_chunks:
            return None
        
        # Concatenate all audio chunks
        audio = np.concatenate(audio_chunks)
        return audio
    
    def _decode_frame(self, frame):
        """Decode a single frame using SNAC"""
        # SNAC uses hierarchical codes
        codes_0 = torch.tensor([frame[0]], device=self.snac_device, dtype=torch.int32).unsqueeze(0)
        codes_1 = torch.tensor([frame[1], frame[4]], device=self.snac_device, dtype=torch.int32).unsqueeze(0)
        codes_2 = torch.tensor([frame[2], frame[3], frame[5], frame[6]], device=self.snac_device, dtype=torch.int32).unsqueeze(0)
        
        # Check bounds
        if (torch.any(codes_0 < 0) or torch.any(codes_0 > 4096) or
            torch.any(codes_1 < 0) or torch.any(codes_1 > 4096) or
            torch.any(codes_2 < 0) or torch.any(codes_2 > 4096)):
            return None
        
        codes = [codes_0, codes_1, codes_2]
        
        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)
        
        # Extract the audio slice
        audio_slice = audio_hat[0, 0, :].detach().cpu().numpy()
        return audio_slice


def generate_audio(model, text, voice, emotion=None, output_filename=None,
                   temperature=0.6, top_p=0.8, max_tokens=1200):
    """Generate audio from text"""
    
    # Format prompt with emotion if specified
    if emotion:
        prompt = f"[{emotion}] {text}"
    else:
        prompt = text
    
    print(f"\n{'='*60}")
    print(f"Generating audio:")
    print(f"  Voice: {voice}")
    print(f"  Emotion: {emotion if emotion else 'None'}")
    print(f"  Text: {text[:50]}..." if len(text) > 50 else f"  Text: {text}")
    print(f"{'='*60}")
    
    start_time = time.monotonic()
    
    # Generate tokens
    tokens = model.generate_tokens(
        prompt,
        voice=voice,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_tokens
    )
    
    # Convert tokens to audio
    print("Converting tokens to audio...")
    audio = model.tokens_to_audio(tokens)
    
    if audio is None:
        print("✗ Failed to generate audio")
        return 0, 0
    
    # Save audio to file
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_int16.tobytes())
    
    duration = len(audio) / 24000
    end_time = time.monotonic()
    generation_time = end_time - start_time
    
    print(f"✓ Generated {duration:.2f}s of audio in {generation_time:.2f}s")
    print(f"✓ Saved to: {output_filename}")
    
    return duration, generation_time


def main():
    """Main function to generate audio examples"""
    
    print("\n" + "="*60)
    print("Orpheus TTS - Direct Inference (No vLLM)")
    print("="*60)
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n⚠ ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please ensure you have merged your checkpoint first.")
        print("Run: python merge_checkpoint.py")
        return
    
    # Initialize model
    model = OrpheusDirectInference(CHECKPOINT_PATH)
    
    # Example 1: Different voices
    print("\n" + "="*60)
    print("EXAMPLE 1: Different Speakers/Voices")
    print("="*60)
    
    sample_text = KYRGYZ_TEXTS[0]
    for voice in VOICES:
        output_file = os.path.join(OUTPUT_DIR, f"kyrgyz_voice_{voice}_direct.wav")
        generate_audio(model, sample_text, voice, output_filename=output_file)
    
    # Example 2: Different emotions
    print("\n" + "="*60)
    print("EXAMPLE 2: Different Emotions/Tones")
    print("="*60)
    
    sample_text = KYRGYZ_TEXTS[1]
    sample_voice = VOICES[0]
    
    for emotion in ["normal", "happy", "sad", "excited"]:
        output_file = os.path.join(OUTPUT_DIR, f"kyrgyz_emotion_{emotion}_direct.wav")
        generate_audio(model, sample_text, sample_voice, emotion=emotion, output_filename=output_file)
    
    # Example 3: Multiple texts
    print("\n" + "="*60)
    print("EXAMPLE 3: Multiple Kyrgyz Texts")
    print("="*60)
    
    for idx, text in enumerate(KYRGYZ_TEXTS):
        voice = VOICES[idx % len(VOICES)]
        output_file = os.path.join(OUTPUT_DIR, f"kyrgyz_text_{idx+1}_direct.wav")
        generate_audio(model, text, voice, output_filename=output_file)
    
    print("\n" + "="*60)
    print("✓ ALL EXAMPLES GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nAll audio files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

