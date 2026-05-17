# Granite Speech Pipeline

Transcribe and translate audio and video files using the IBM Granite 4.0 1B Speech model on Apple Silicon with MLX.

## Features

- **Pipeline processing** — run multiple transcription and translation tasks on the same audio (Transcribe + one translation runs as a single inference per segment via chain-of-thought prompting)
- **Transcription** — English, French, German, Spanish, Portuguese, Japanese
- **Translation** — English ↔ French, German, Spanish, Portuguese, Italian, Japanese, Mandarin Chinese (Italian and Mandarin: English source only)
- **Keywords** — bias recognition toward up to 15 user-provided terms (proper nouns, acronyms, jargon)
- **VAD segmentation** — automatic speech detection with timestamped per-segment output (togglable; disable to process whole audio in one pass)
- **Toxicity check** — togglable (on by default); surfaces toxicity score on English output (transcription or translation to English) via Granite Guardian HAP 125m
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
5. Optionally add **Keywords** (proper nouns, acronyms, jargon)
6. Optionally toggle **Toxicity check** (on by default)
7. Click **Transcribe** to process all selected tasks
8. View side-by-side results and download as text
