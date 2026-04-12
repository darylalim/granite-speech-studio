# Granite Speech Pipeline

Streamlit web app for speech-to-text and translation using IBM's Granite Speech models via [MLX](https://github.com/Blaizzy/mlx-audio). Requires Apple Silicon. Process audio through multiple tasks simultaneously, with automatic VAD-based segmentation and English toxicity detection.

## Features

- **Pipeline processing** — run multiple transcription and translation tasks on the same audio
- **Translation** — French, German, Spanish, Portuguese, Italian, Japanese, Mandarin Chinese, English
- **VAD segmentation** — automatic speech detection with timestamped per-segment output
- **Safety** — automatic toxicity detection on English transcriptions via Granite Guardian HAP 38m
- **Task selection** — pick tasks via pills (Transcribe preselected by default)
- **Audio input** — record from microphone or upload files (WAV, FLAC, M4A)
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

1. Pick tasks (transcribe, translate to a language)
2. Record from your microphone or upload an audio file
3. Click **Transcribe** to process all selected tasks
4. View side-by-side results and download as text

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
