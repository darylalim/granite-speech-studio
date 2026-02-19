# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app for automatic speech recognition and translation using IBM's [Granite Speech](https://huggingface.co/collections/ibm-granite/granite-speech) models.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- `RuntimeError` for known transcription failures (no custom exception class)
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

- `transformers` — Hugging Face model loading
- `accelerate` — device mapping for model loading
- `peft` — LoRA adapter loading for Granite Speech models
- `torch` — tensor operations
- `torchaudio` — audio loading and resampling
- `torchcodec` — audio decoding backend for torchaudio
- `streamlit` — web user interface
- `ruff` — linting/formatting (dev)
- `ty` — type checking (dev)
- `pytest` — testing (dev)

## Configuration

`pyproject.toml` — ruff isort (`combine-as-imports`) and ty (`python-version = "3.12"`).

## Architecture

`streamlit_app.py` — single-file app.

### Models

- [Granite Speech 3.3 2b](https://huggingface.co/ibm-granite/granite-speech-3.3-2b)
- [Granite Speech 3.3 8b](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)

### Languages

- English (transcription)
- French, German, Spanish, Portuguese (translation)

### Audio Formats

wav, mp3, m4a, ogg, flac, webm, aac

### Performance

- Best available device: MPS > CUDA > CPU
- `@st.cache_resource` to cache models
- `@torch.inference_mode()` on inference functions
- `io.BytesIO` for in-memory audio loading (no temp files)
- bfloat16 on MPS/CUDA, float32 on CPU
- `time.perf_counter()` for timing

### Error Handling

- `RuntimeError` caught explicitly for transcription failures
- Unexpected exceptions shown with `st.exception()`

### JSON Download

Fields in the downloadable JSON via `st.download_button`:

- `model` (string) — model name
- `audio_duration` (float) — duration in seconds
- `transcript` (string) — generated text
- `num_words` (int) — word count
- `eval_duration` (float) — transcription time in seconds (rounded to 2 decimal places)

Model shown via `st.caption`. `st.metric` displays audio duration, words, and eval duration.

### Tests

`tests/test_streamlit_app.py` — unit tests for device detection, model options, prompt choices, supported formats, audio loading, and error handling.

## Resources

- [Granite Speech Models](https://huggingface.co/collections/ibm-granite/granite-speech)
- [Technical Report](https://arxiv.org/abs/2505.08699)
- [Finetune on custom data](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/fine_tuning_granite_speech.ipynb)
- [Two-Pass Spoken Question Answering](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/two_pass_spoken_qa.ipynb)
