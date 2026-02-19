# Granite Speech Pipeline

Streamlit web app for automatic speech recognition and translation using IBM's Granite Speech models.

## Features

- **Model Selection** — Granite Speech 3.3 2B (default) or 8B
- **Transcription** — speech to text
- **Translation** — speech to French, German, Spanish, or Portuguese
- **Audio Formats** — WAV, MP3, M4A, OGG, FLAC, WebM, AAC
- **Metrics** — model, audio duration, word count, eval duration
- **JSON Export** — download results

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

1. Select a model (8B for higher quality, 2B for faster inference)
2. Upload an audio file
3. Choose a prompt (transcribe or translate)
4. Click "Transcribe" to process
5. View results and download as JSON

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
