#!/usr/bin/env python3
"""
Example usage of OrpheusOfflineModel for text-to-speech generation.
"""

import torch
from orpheus_vllm import OrpheusOfflineModel

def main():
    # Initialize the model (replace with your actual model path)
    model_path = "/mnt/d/OrpheusTTS-checkpoints/merged_checkpoint"  # Update this path
    model = OrpheusOfflineModel(
        model_path=model_path,
        dtype=torch.bfloat16,
        tokenizer='canopylabs/orpheus-3b-0.1-ft',
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Text to convert to speech
    text = "Бишкек Кыргызстандын борбору жана эң чоң шаары болуп саналат."

    # Generate speech
    print(f"Generating speech for: '{text}'")
    model.generate(text)

    print("Speech generation complete. Audio files saved as audio_0.wav, audio_1.wav, etc.")

if __name__ == "__main__":
    main()
