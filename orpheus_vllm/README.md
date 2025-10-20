# Orpheus vLLM

Orpheus TTS with vLLM backend for fast inference.

## Installation

Install in editable mode for development:

```bash
pip install -e .
```

Or install from the directory:

```bash
pip install .
```

## Usage

```python
from orpheus_vllm import OrpheusOfflineModel

# Initialize the model
model = OrpheusOfflineModel(
    model_path="/path/to/your/model",
    dtype=torch.bfloat16,
    tokenizer='canopylabs/orpheus-3b-0.1-ft',
    device="cuda"
)

# Generate speech
text = "Hello, world!"
generated_ids = model.generate(text)
model.parse_output_as_speech(generated_ids)
```

This will generate audio files named `audio_0.wav`, `audio_1.wav`, etc.

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- vLLM >= 0.2.0
- Transformers >= 4.30.0
- SNAC >= 0.1.0
- SciPy >= 1.10.0

## Features

- Fast inference using vLLM
- Multi-speaker support (Timur, Aiganysh)
- Emotion control (neutral, strict)
- Direct audio file output
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="orpheus-vllm",
    version="0.1.0",
    author="Orpheus TTS Team",
    description="Orpheus TTS with vLLM backend for fast inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/orpheus-tts",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "vllm>=0.2.0",
        "transformers>=4.30.0",
        "snac>=0.1.0",
        "scipy>=1.10.0",
    ],
)

