"""
Generate audio examples from fine-tuned Orpheus TTS checkpoints
Supports different speakers, emotions/tones, and Kyrgyz language
"""

import os
import sys
import wave
import time
import torch
from pathlib import Path

# Add parent directory to path to import orpheus_tts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'orpheus_tts_pypi'))

from orpheus_tts import OrpheusModel

# Configuration
CHECKPOINT_PATH = "./finetuned_model/merged"  # Path to your merged checkpoint from full_training.py
# Alternative: Use specific checkpoint like "./finetuned_model/checkpoint-1000"
OUTPUT_DIR = "./generated_audio_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Available voices (speakers) - these are the default voices from Orpheus
# You can experiment with other names if your dataset has different speakers
VOICES = ["timur", "aiganysh"]

# Available emotions/tones that can be added to prompts
EMOTIONS = [
    "happy", "normal", "sad", "frustrated", "excited",
    "whisper", "panicky", "curious", "surprise", "fast",
    "crying", "deep", "sleepy", "angry", "shout", "slow"
]

# Kyrgyz language example texts
KYRGYZ_TEXTS = [
    "Саламатсызбы, менин атым Айгүл. Мен кыргыз тилинде сүйлөп жатам.".lower(),
    "Бишкек Кыргызстандын борбору жана эң чоң шаары.".lower(),
    "Ыссык-Көл дүйнөдөгү эң чоң тоо көлдөрүнүн бири.".lower(),
    "Кыргыз тили өтө сулуу жана бай тил.".lower(),
    "Бүгүн аба ырайы абдан жакшы.".lower()
]

# English examples for comparison (if needed)
ENGLISH_TEXTS = [
    "Hello, my name is assistant. I can speak in different emotions.",
    "This is a test of the Orpheus text to speech system.",
    "The weather is beautiful today."
]


def generate_audio(model, text, voice, emotion=None, output_filename=None,
                   temperature=0.6, top_p=0.8, max_tokens=1200):
    """
    Generate audio from text using the fine-tuned model

    Args:
        model: OrpheusModel instance
        text: Text to convert to speech
        voice: Speaker voice name
        emotion: Optional emotion/tone (e.g., "happy", "sad")
        output_filename: Path to save the audio file
        temperature: Sampling temperature (default 0.6)
        top_p: Top-p sampling parameter (default 0.8)
        max_tokens: Maximum number of tokens to generate
    """
    # Format the prompt with emotion if specified
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

    # Generate speech tokens
    syn_tokens = model.generate_speech(
        prompt=prompt,
        voice=voice,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    # Save audio to file
    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        chunk_counter = 0
        for audio_chunk in syn_tokens:
            chunk_counter += 1
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)

        duration = total_frames / wf.getframerate()

    end_time = time.monotonic()
    generation_time = end_time - start_time

    print(f"✓ Generated {duration:.2f}s of audio in {generation_time:.2f}s")
    print(f"✓ Saved to: {output_filename}")

    return duration, generation_time


def main():
    """Main function to generate multiple audio examples"""

    print("\n" + "="*60)
    print("Orpheus TTS - Audio Generation from Fine-tuned Checkpoint")
    print("="*60)

    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n⚠ ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please ensure you have run full_training.py and the checkpoint exists.")
        print("\nAlternative: Update CHECKPOINT_PATH in this script to point to your checkpoint.")
        return

    print(f"\n✓ Loading model from: {CHECKPOINT_PATH}")
    print("This may take a few minutes...")

    # Initialize the model with your fine-tuned checkpoint
    model = OrpheusModel(
        model_name=CHECKPOINT_PATH,
        max_model_len=2048,
        dtype=torch.bfloat16
    )

    print("✓ Model loaded successfully!")

    # Example 1: Generate samples with different speakers (voices)
    print("\n" + "="*60)
    print("EXAMPLE 1: Different Speakers/Voices")
    print("="*60)

    sample_text = KYRGYZ_TEXTS[0]
    for voice in VOICES[:3]:  # Use first 3 voices
        output_file = os.path.join(OUTPUT_DIR, f"kyrgyz_voice_{voice}.wav")
        generate_audio(model, sample_text, voice, output_filename=output_file)

    # Example 2: Generate samples with different emotions/tones
    print("\n" + "="*60)
    print("EXAMPLE 2: Different Emotions/Tones")
    print("="*60)

    sample_text = KYRGYZ_TEXTS[1]
    sample_voice = VOICES[0]

    for emotion in ["normal", "happy", "sad", "excited"]:
        output_file = os.path.join(OUTPUT_DIR, f"kyrgyz_emotion_{emotion}.wav")
        generate_audio(model, sample_text, sample_voice, emotion=emotion, output_filename=output_file)

    # Example 3: Generate multiple Kyrgyz texts
    print("\n" + "="*60)
    print("EXAMPLE 3: Multiple Kyrgyz Texts")
    print("="*60)

    for idx, text in enumerate(KYRGYZ_TEXTS):
        voice = VOICES[idx % len(VOICES)]
        output_file = os.path.join(OUTPUT_DIR, f"kyrgyz_text_{idx+1}.wav")
        generate_audio(model, text, voice, output_filename=output_file)

    # Example 4: Advanced - Combining voice, emotion, and custom parameters
    print("\n" + "="*60)
    print("EXAMPLE 4: Advanced Examples")
    print("="*60)

    advanced_examples = [
        {
            "text": KYRGYZ_TEXTS[2],
            "voice": "julia",
            "emotion": "excited",
            "temperature": 0.7,
            "output": "kyrgyz_advanced_excited_julia.wav"
        },
        {
            "text": KYRGYZ_TEXTS[3],
            "voice": "leo",
            "emotion": "whisper",
            "temperature": 0.5,
            "output": "kyrgyz_advanced_whisper_leo.wav"
        },
        {
            "text": KYRGYZ_TEXTS[4],
            "voice": "tara",
            "emotion": None,
            "temperature": 0.6,
            "output": "kyrgyz_advanced_neutral_tara.wav"
        }
    ]

    for example in advanced_examples:
        output_file = os.path.join(OUTPUT_DIR, example["output"])
        generate_audio(
            model,
            example["text"],
            example["voice"],
            emotion=example["emotion"],
            output_filename=output_file,
            temperature=example["temperature"]
        )

    print("\n" + "="*60)
    print("✓ ALL EXAMPLES GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nAll audio files saved to: {OUTPUT_DIR}")
    print(f"Total files generated: {len(os.listdir(OUTPUT_DIR))}")
    print("\nYou can now listen to the generated audio files!")


def generate_custom_audio(text, voice="tara", emotion=None, output_file=None):
    """
    Convenience function to generate a single audio file
    Can be imported and used in other scripts

    Usage:
        from generate_examples_for_listening import generate_custom_audio
        generate_custom_audio("Саламатсызбы!", voice="julia", emotion="happy", output_file="test.wav")
    """
    model = OrpheusModel(
        model_name=CHECKPOINT_PATH,
        max_model_len=2048,
        dtype=torch.bfloat16
    )

    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "custom_output.wav")

    return generate_audio(model, text, voice, emotion, output_file)


if __name__ == "__main__":
    main()

