# Granite Speech Pipeline

Streamlit web app for speech-to-text and translation using IBM's Granite Speech models. Process audio through multiple tasks simultaneously with preset task groups. Includes automatic punctuation/capitalization and English toxicity detection.

## Features

- **Pipeline processing** — run multiple transcription and translation tasks on the same audio
- **Translation** — French, German, Spanish, Portuguese, Italian, Japanese, Mandarin Chinese, English
- **VAD segmentation** — automatic speech detection with timestamped per-segment output
- **Punctuation** — automatic capitalization and punctuation on English transcriptions
- **Safety** — automatic toxicity detection on English transcriptions via Granite Guardian HAP 38m
- **Presets** — All Tasks, European Languages, Asian Languages, Transcribe Only
- **Custom selection** — pick individual tasks via multiselect
- **Audio input** — upload files (WAV, MP3, M4A, OGG, FLAC, WebM, AAC) or record from microphone
- **Side-by-side results** — compare outputs in a column grid (up to 3 columns)
- **Deferred loading** — models load on first pipeline run for instant page startup
- **Metrics** — audio duration, word count, processing time per task
- **Export** — download per-task text/JSON or combined JSON for all results (safety scores included for transcriptions)

## Hardware Support

Automatically uses the best available device:

- Apple Silicon (MPS)
- NVIDIA GPU (CUDA)
- CPU (fallback)

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Usage

1. Select a preset or pick individual tasks (transcribe, translate to a language)
2. Upload an audio file or record from your microphone
3. Click "Run Pipeline" to process all selected tasks
4. View side-by-side results and download as text or JSON

## Development

```bash
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run ty check              # Typecheck
uv run pytest                # Test
```

## Models

- [Granite 4.0 1b Speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) — transcription and translation
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection for speech segmentation
- pcs_en (via [punctuators](https://pypi.org/project/punctuators/)) — English punctuation and capitalization
- [Granite Guardian HAP 38m](https://huggingface.co/ibm-granite/granite-guardian-hap-38m) — English toxicity detection

## Resources

- [Granite Speech Models](https://huggingface.co/collections/ibm-granite/granite-speech)
- [Technical Report](https://arxiv.org/abs/2505.08699)
