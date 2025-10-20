#!/usr/bin/env python3
"""
Example usage of OrpheusOfflineModel for text-to-speech generation.
"""

import torch
from orpheus_vllm import OrpheusOfflineModel

def main():
    # Initialize the model (replace with your actual model path)
    model_path = "/path/to/your/orpheus/model"  # Update this path
    model = OrpheusOfflineModel(
        model_path=model_path,
        dtype=torch.bfloat16,
        tokenizer='canopylabs/orpheus-3b-0.1-ft',
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Text to convert to speech
    text = "Hello, this is a test of the Orpheus text-to-speech system."

    # Generate speech
    print(f"Generating speech for: '{text}'")
    generated_ids = model.generate(text)
    model.parse_output_as_speech(generated_ids)

    print("Speech generation complete. Audio files saved as audio_0.wav, audio_1.wav, etc.")

if __name__ == "__main__":
    main()
