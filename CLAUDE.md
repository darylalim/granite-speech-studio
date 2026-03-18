# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app for speech-to-text and translation using IBM's [Granite Speech](https://huggingface.co/collections/ibm-granite/granite-speech) models. Supports multi-task pipeline processing with preset task groups. Includes VAD-based audio segmentation, automatic punctuation/capitalization, and English toxicity detection.

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
- `torch` — tensor operations
- `torchaudio` — audio loading and resampling
- `torchcodec` — audio decoding backend for torchaudio
- `punctuators` — English punctuation and capitalization (ONNX)
- `silero-vad` — Voice Activity Detection for audio segmentation
- `streamlit` — web user interface
- `ruff` — linting/formatting (dev)
- `ty` — type checking (dev)
- `pytest` — testing (dev)

## Configuration

`pyproject.toml` — ruff isort (`combine-as-imports`) and ty (`python-version = "3.12"`).

## Architecture

`streamlit_app.py` — single-file app.

### Functions

- `format_timestamp` — formats seconds to `M:SS` or `H:MM:SS`
- `silero_vad` — runs Silero VAD on waveform, returns `(start, end)` tuples in seconds
- `get_speech_segments` — post-processes VAD output with buffering and merging
- `load_vad_model` — cached Silero VAD model loader
- Pipeline supports optional VAD segmentation with per-segment transcription and timestamped output

### Models

- [Granite 4.0 1b Speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) — transcription and translation
- [Granite Guardian HAP 38m](https://huggingface.co/ibm-granite/granite-guardian-hap-38m) — English toxicity detection (runs on CPU)
- pcs_en (via `punctuators`) — English punctuation and capitalization (runs on CPU, ONNX)
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice Activity Detection for speech segmentation (runs on CPU)

### Languages

- English (transcription)
- French, German, Spanish, Portuguese, Italian, Japanese, Mandarin Chinese (translation)

### UI Layout

- **Task selection** — `st.pills` for presets (`TASK_PRESETS` dict) + `st.multiselect` for custom task selection, resolved via `get_selected_tasks`
- **Audio input** — `st.tabs` with Upload (`st.file_uploader`) and Record (`st.audio_input`)
- **Segmentation** — `st.checkbox` for VAD segmentation toggle (default on)
- **Results** — pipeline results persisted in `st.session_state`, displayed in a side-by-side column grid (up to 3 columns) via `_render_result_card` helper
- **Safety** — transcription results show `st.success` (safe) or `st.warning` (toxic) banner with toxicity score (English only)
- **Footer** — model name, punctuation model name, safety model name, device, links to model cards

### Audio Formats

wav, mp3, m4a, ogg, flac, webm, aac

### Performance

- Best available device: MPS > CUDA > CPU
- Deferred model loading — speech model loads on first pipeline run, not on page load
- `@st.cache_resource` to cache models
- `@torch.inference_mode()` on inference functions
- `io.BytesIO` for in-memory audio loading (no temp files)
- bfloat16 on MPS/CUDA, float32 on CPU
- `time.perf_counter()` for timing
- Guardian model runs on CPU with default dtype (38M params, fast inference)
- Punctuation model runs on CPU via ONNX Runtime (no `@torch.inference_mode()`)
- Silero VAD model runs on CPU (~3MB)

### Error Handling

- `RuntimeError` caught explicitly for transcription failures
- Unexpected exceptions shown with `st.exception()`

### Downloads

- **Per-task Text** — plain transcript as `.txt`
- **Per-task JSON** — `model`, `task`, `audio_duration`, `transcript`, `num_words`, `eval_duration`, plus `is_toxic` and `toxicity_score` for transcription only
- **Combined JSON** — "Download All" with `model`, `audio_duration`, and all `results` keyed by task name

`st.metric` displays audio duration, words, and processing time per task.

### Tests

`tests/test_streamlit_app.py` — unit tests for device detection, prompt choices, supported formats, task presets, task selection, audio loading, model loading, guardian model loading, punctuation model loading, punctuation application, safety checking, transcription, pipeline execution, result card rendering, and error handling. `TestFormatTimestamp`, `TestSileroVad`, `TestGetSpeechSegments`, `TestLoadVadModel`, plus new segmentation cases in `TestRunPipeline`.

## Resources

- [Granite Guardian HAP 38m](https://huggingface.co/ibm-granite/granite-guardian-hap-38m)
- [Granite Speech Models](https://huggingface.co/collections/ibm-granite/granite-speech)
- [Technical Report](https://arxiv.org/abs/2505.08699)
- [Finetune on custom data](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/fine_tuning_granite_speech.ipynb)
- [Two-Pass Spoken Question Answering](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/two_pass_spoken_qa.ipynb)
