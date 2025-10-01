from flask import Flask, Response, request, send_from_directory
import os
import struct
import time
import sys

# Ensure local package is importable without installation
CURRENT_DIR = os.path.dirname(__file__)
LOCAL_PKG_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "orpheus_tts_pypi"))
if LOCAL_PKG_DIR not in sys.path:
    sys.path.insert(0, LOCAL_PKG_DIR)

# --- Simple .env loader (no extra deps) ---
ENV_PATHS = [
    os.path.join(CURRENT_DIR, ".env"),
    os.path.join(CURRENT_DIR, "config.env"),
]

def _load_env_files(paths):
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    # Support lines like KEY=VALUE or KEY="VALUE"
                    if "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    # Don't override existing env
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            # Ignore malformed files silently
            pass

_load_env_files(ENV_PATHS)

from orpheus_tts.engine_class import OrpheusModel

app = Flask(__name__)

# Environment-configurable settings
MODEL_NAME = os.getenv("ORPHEUS_MODEL", "canopylabs/orpheus-tts-0.1-finetune-prod")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))

# Initialize the TTS engine
engine = OrpheusModel(
    model_name=MODEL_NAME,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
)


def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    # For streaming we put 0 as data size; many players accept progressive WAV.
    data_size = 0

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        24000,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header


@app.route('/')
def index():
    # Serve the client page from this directory
    return send_from_directory(os.path.dirname(__file__), 'client.html')


@app.route('/healthz')
def healthz():
    return {"status": "ok", "model": MODEL_NAME, "max_model_len": MAX_MODEL_LEN}


@app.route('/tts', methods=['GET'])
def tts():
    # Read query params with defaults matching the example UI
    prompt = request.args.get('prompt', 'Hey there, looks like you forgot to provide a prompt!')
    voice = request.args.get('voice', 'tara')

    def to_float(name, default):
        try:
            return float(request.args.get(name, default))
        except Exception:
            return default

    def to_int(name, default):
        try:
            return int(request.args.get(name, default))
        except Exception:
            return default

    temperature = to_float('temperature', 0.4)
    top_p = to_float('top_p', 0.9)
    max_tokens = to_int('max_tokens', 2000)

    # Optional params
    repetition_penalty = to_float('repetition_penalty', 1.1)

    def generate_audio_stream():
        start_time = time.monotonic()
        yield create_wav_header()

        syn_tokens = engine.generate_speech(
            prompt=prompt,
            voice=voice,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[128258],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        total_frames = 0
        sample_width = 2  # 16-bit PCM
        channels = 1
        frame_size = sample_width * channels

        for chunk in syn_tokens:
            # Count frames for logging/debugging
            if isinstance(chunk, (bytes, bytearray)):
                total_frames += len(chunk) // frame_size
            yield chunk

        duration = total_frames / 24000.0
        elapsed = time.monotonic() - start_time
        app.logger.info("Streamed %.2fs of audio in %.2fs", duration, elapsed)

    return Response(generate_audio_stream(), mimetype='audio/wav')


if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8080'))
    app.run(host=host, port=port, threaded=True)
