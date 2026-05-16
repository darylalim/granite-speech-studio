# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Transcribe and translate audio and video files using the IBM Granite 4.0 1B Speech model on Apple Silicon with MLX. Streamlit web app with multi-task pipeline processing, VAD-based audio segmentation, and English toxicity detection.

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

- `mlx-audio` — MLX-based speech model loading and inference (Apple Silicon)
- `transformers` — guardian model loading (toxicity detection)
- `torch` — tensor operations (VAD, guardian model)
- `torchaudio` — audio loading and resampling (for VAD preprocessing)
- `torchcodec` — audio decoding backend for torchaudio
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

- `build_tasks` — returns ordered `{task_name: prompt}` dict for a given source language
- `produces_english` — predicate: does `(source, task)` yield English output (drives safety check)
- `result_title` — display title for result cards (transcription shows source; translation shows target)
- `result_slug` — filename slug for downloads (transcription includes source; translation uses target)
- `is_video` — predicate by extension, drives `st.video` vs `st.audio` preview
- `format_timestamp` — formats seconds to `M:SS` or `H:MM:SS`
- `silero_vad` — runs Silero VAD on waveform, returns `(start, end)` tuples in seconds
- `get_speech_segments` — post-processes VAD output with buffering and merging
- `load_vad_model` — cached Silero VAD model loader
- `run_pipeline` — takes `tasks: dict[str, str]` (task→prompt), `safety_tasks: set[str]`, and `use_segmentation: bool`; when segmentation is on, runs VAD then transcribes each segment; when off, treats the full audio as a single segment. Emits timestamped output and runs safety check only for tasks in `safety_tasks`.

### Models

- [Granite 4.0 1b Speech 8bit](https://huggingface.co/mlx-community/granite-4.0-1b-speech-8bit) — transcription and translation (MLX, 8-bit quantized)
- [Granite Guardian HAP 38m](https://huggingface.co/ibm-granite/granite-guardian-hap-38m) — English toxicity detection (runs on CPU)
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice Activity Detection for speech segmentation (runs on CPU)

### Languages

- Source languages: English, French, German, Spanish, Portuguese, Japanese (model-supported ASR set)
- Transcription: available for any source language
- Translation: English source → French, German, Spanish, Portuguese, Italian, Japanese, Mandarin Chinese; non-English source → English only (matches model's En↔X capability)

### UI Layout (top to bottom)

- **Title + description** — `st.title` plus `st.markdown` linking to the IBM Granite 4.0 1B Speech model card
- **Audio input** — `st.tabs` with Upload (`st.file_uploader`) first, then Record (`st.audio_input`); labels hidden via `label_visibility="collapsed"`
- **Audio/video preview** — `st.video` for video containers, `st.audio` otherwise; selected via `is_video(filename)`. `st.caption` shows filename or "Recorded audio".
- **Source language** — `st.segmented_control` (single-select), `English` default; drives the task option list via `build_tasks(source)`
- **Task selection** — `st.pills` with `selection_mode="multi"`, label hidden, `Transcribe` preselected; widget keyed by source so options reset when source changes
- **VAD segmentation** — `st.columns([15, 1], vertical_alignment="center")`: `st.markdown("VAD segmentation", help=...)` on the left, `st.toggle` defaulting to `True` on the right. When off, VAD model load is skipped and `run_pipeline` treats the full audio as a single segment. Part of `_last_input_key` so toggling invalidates cached results.
- **Run button** — `st.button("Transcribe", type="primary", width="stretch")` placed in a right-aligned column via `st.columns([4, 1])`; disabled until audio is loaded and at least one task is selected
- **Results** — pipeline results, stem, and source captured at run time in `st.session_state`; displayed in a side-by-side column grid (up to 3 columns) via `_render_result_card` helper
- **Safety** — results show `st.success` (safe) or `st.warning` (toxic) banner with toxicity score whenever output is English (English source transcription, or X→English translation)

### Audio Formats

Audio: wav, flac, m4a, mp3, ogg, aac. Video containers (audio track extracted via torchcodec): mp4, mov, webm, mkv. `VIDEO_FORMATS` set drives conditional preview (`st.video` vs `st.audio`). Upload size limit raised to 500 MB in `.streamlit/config.toml`.

### Performance

- Speech model runs via MLX on Apple Silicon GPU (8-bit quantized, ~2.9GB)
- Deferred model loading — speech and VAD models load on first pipeline run, not on page load
- `@st.cache_resource` to cache models
- `@torch.inference_mode()` on safety check and pipeline (for guardian model)
- `io.BytesIO` for in-memory audio loading (no temp files)
- `time.perf_counter()` for timing
- Guardian model runs on CPU with default dtype (38M params, fast inference)
- Silero VAD model runs on CPU (~3MB)
- `max_tokens=512` per segment (prevents truncation on long speech)

### Error Handling

- `RuntimeError` caught explicitly for transcription failures
- Unexpected exceptions shown with `st.exception()`

### Downloads

- **Per-task Text** — plain transcript as `.txt`, icon-only download button (`:material/download:`) with context-aware tooltip ("Download transcription" or "Download translation")

### Tests

`tests/test_streamlit_app.py` — unit tests for constants, helpers (`build_tasks`, `produces_english`, `result_title`, `result_slug`, `is_video`, `format_timestamp`, `silero_vad`, `get_speech_segments`), model loaders, audio loading (wav + mp4 video), safety check, transcription, pipeline execution (multi-task, multi-segment, VAD on/off), and result card rendering. `TestRunPipeline` patches `get_speech_segments` via `setup_method` with a default single-segment fixture, and overrides it per-test for multi-segment cases. Test fixtures in `tests/data/audio/` (`sample_10s.wav`, `sample_10s_video.mp4`).

## Resources

- [Granite 4.0 1b Speech 8bit (MLX)](https://huggingface.co/mlx-community/granite-4.0-1b-speech-8bit)
- [Granite Guardian HAP 38m](https://huggingface.co/ibm-granite/granite-guardian-hap-38m)
- [Granite Speech Models](https://huggingface.co/collections/ibm-granite/granite-speech)
- [Technical Report](https://arxiv.org/abs/2505.08699)
- [Finetune on custom data](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/fine_tuning_granite_speech.ipynb)
- [Two-Pass Spoken Question Answering](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/two_pass_spoken_qa.ipynb)
