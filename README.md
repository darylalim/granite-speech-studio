# Granite Speech Pipeline

Transcribe and translate audio and video files using the IBM Granite 4.0 1B Speech model on Apple Silicon with MLX.

## Features

- **Pipeline processing** — run multiple transcription and translation tasks on the same audio
- **Transcription** — English, French, German, Spanish, Portuguese, Japanese
- **Translation** — English ↔ French, German, Spanish, Portuguese, Italian, Japanese, Mandarin Chinese (Italian and Mandarin: English source only)
- **VAD segmentation** — automatic speech detection with timestamped per-segment output (togglable; disable to process whole audio in one pass)
- **Safety** — automatic toxicity detection on English output (transcription or translation to English) via Granite Guardian HAP 38m
- **Source language** — pick once; valid tasks update accordingly
- **Audio input** — upload audio (WAV, FLAC, M4A, MP3, OGG, AAC) or video (MP4, MOV, WebM, MKV — audio track is extracted) or record from microphone
- **Side-by-side results** — compare outputs in a column grid (up to 3 columns)
- **Deferred loading** — models load on first pipeline run for instant page startup
- **Export** — download per-task transcriptions and translations as text

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.12+

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Usage

1. Upload an audio or video file, or record from your microphone
2. Pick the source language of your audio
3. Pick tasks (transcribe, translate to a language)
4. Optionally toggle **VAD segmentation** (on by default)
5. Click **Transcribe** to process all selected tasks
6. View side-by-side results and download as text

## Development

```bash
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run ty check              # Typecheck
uv run pytest                # Test
```

## Models

- [Granite 4.0 1b Speech 8bit](https://huggingface.co/mlx-community/granite-4.0-1b-speech-8bit) — transcription and translation (MLX, 8-bit quantized)
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection for speech segmentation
- [Granite Guardian HAP 38m](https://huggingface.co/ibm-granite/granite-guardian-hap-38m) — English toxicity detection

## Resources

- [Granite 4.0 1b Speech 8bit (MLX)](https://huggingface.co/mlx-community/granite-4.0-1b-speech-8bit)
- [Granite Speech Models](https://huggingface.co/collections/ibm-granite/granite-speech)
- [Technical Report](https://arxiv.org/abs/2505.08699)
