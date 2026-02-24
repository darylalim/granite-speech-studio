# Granite Speech Pipeline

Streamlit web app for automatic speech recognition and translation using IBM's Granite Speech models.

## Features

- **Transcription** — speech to text
- **Translation** — speech to French, German, Spanish, or Portuguese
- **Audio Input** — upload files (WAV, MP3, M4A, OGG, FLAC, WebM, AAC) or record from microphone
- **Metrics** — audio duration, word count, processing time
- **Export** — download transcript as text or JSON

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

1. Select a task (transcribe or translate to a language)
2. Upload an audio file or record from your microphone
3. Click "Transcribe" or "Translate" to process
4. View results and download as text or JSON

## Development

```bash
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run ty check              # Typecheck
uv run pytest                # Test
```

## Resources

- [Granite Speech Models](https://huggingface.co/collections/ibm-granite/granite-speech)
- [Technical Report](https://arxiv.org/abs/2505.08699)
