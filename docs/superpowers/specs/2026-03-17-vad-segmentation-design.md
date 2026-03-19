# VAD Segmentation Design

## Overview

Add Voice Activity Detection (VAD) segmentation to the speech pipeline using Silero VAD. When enabled, audio is split into speech segments before transcription, producing timestamped output lines. This improves output quality for longer audio files by giving the model focused speech regions rather than the full recording.

## Decisions

- **Output format:** Timestamped lines (`[0:00 - 0:15] text`) concatenated with newlines into a single transcript string
- **User control:** Checkbox toggle ("VAD segmentation"), default on
- **Punctuation:** Applied per-segment (punctuation models work best on short sentence-like inputs)
- **Safety check:** Runs once on the full joined text without timestamps (one toxic phrase anywhere flags the whole transcript)
- **No-speech fallback:** Falls back silently to processing the full audio as one segment
- **VAD engine:** Silero VAD (neural network, CPU, ~3MB, depends only on torch)

## VAD Model Loading

New cached loader following the existing `@st.cache_resource` pattern:

```python
@st.cache_resource(show_spinner=False)
def load_vad_model() -> torch.nn.Module:
    return load_silero_vad()
```

Runs on CPU. Loaded in `main()` before pipeline execution when the segmentation checkbox is enabled.

## Segmentation Functions

### `silero_vad(wav, model, sample_rate) -> list[tuple[float, float]]`

Runs Silero VAD on a waveform tensor. Takes the loaded VAD model as a parameter (consistent with how `transcribe_audio` and `check_safety` receive their models). Returns a list of `(start, end)` tuples in seconds.

### `get_speech_segments(wav, model, sample_rate) -> list[dict]`

Post-processes VAD output. Takes the loaded VAD model as a parameter (passes it through to `silero_vad`):
- Adds 0.3s start buffer and 0.3s end buffer to catch cut-off word beginnings and trailing syllables
- Clamps end buffer to audio duration
- Merges segments with gaps < 0.5s to avoid over-fragmentation
- Falls back to a single segment covering the full audio if no speech detected
- Returns `list[dict]` with `{"start": float, "end": float}`

### `format_timestamp(seconds) -> str`

Formats seconds to `M:SS` or `H:MM:SS` (e.g., `0:15`, `1:02:30`).

## Pipeline Integration

`run_pipeline()` gets two new parameters:
- `vad_model: torch.nn.Module | None = None`
- `use_segmentation: bool = False`

### When segmentation is enabled (and vad_model provided):

1. Run `get_speech_segments()` once on the full waveform
2. For each task, loop through segments:
   - Slice the waveform: `wav[:, start_sample:end_sample]`
   - Transcribe the segment via existing `transcribe_audio()`
   - If English task, punctuate the segment text via existing `apply_punctuation()`
   - Prepend timestamp: `[0:00 - 0:15] Punctuated text.`
3. Join all segment lines with `\n` to form the full transcript
4. If English task, run safety check once on the joined text without timestamps (guardian tokenizer uses `truncation=True`, so very long transcripts will be truncated — acceptable since toxicity in early content is still caught, consistent with existing non-segmented behavior)
5. `num_words` is summed from per-segment word counts (counted before timestamps are prepended, avoiding timestamp tokens in the count)
6. `eval_duration` is wall-clock time for all segments (including punctuation overhead), rounded to 2 decimal places

Segment boundaries are computed once and reused across all tasks.

### When segmentation is disabled (or no vad_model):

Existing behavior, unchanged.

## UI Changes

### Checkbox

Added in `main()` below the audio input tabs:

```python
use_segmentation = st.checkbox("VAD segmentation", value=True)
```

### Model loading

When `use_segmentation` is True, load the VAD model inside a spinner alongside the other conditional model loads.

### Input key

Add `use_segmentation` to the `input_key` tuple so changing the toggle clears stale results.

### No changes to:

- `_render_result_card()` — `st.code()` handles multi-line timestamped text naturally
- Downloads — text and JSON exports include the timestamped transcript as-is
- Footer — VAD is a processing option, not a persistent model worth listing

## Dependencies

Add `silero-vad` to `pyproject.toml`. This is the PyPI package that bundles the Silero VAD model (avoids runtime downloads via `torch.hub`). Only depends on torch (already a dependency). The `load_silero_vad()` function is imported from `silero_vad`.

## Inference Mode

`silero_vad()` and `get_speech_segments()` do not need `@torch.inference_mode()` — Silero VAD manages its own inference context internally. When called from within `run_pipeline()` (which has the decorator), VAD runs inside the existing inference mode context. When called from tests via `__wrapped__` (bypassing the decorator), VAD still works correctly since it handles its own context.

## CLAUDE.md Updates

After implementation, update CLAUDE.md:
- Add `silero-vad` to Dependencies section
- Add Silero VAD to Models section
- Add `silero_vad`, `get_speech_segments`, `format_timestamp` to Architecture notes
- Update Performance section to note VAD runs on CPU
- Update Tests section to list new test classes

## Testing

New test classes:
- `TestLoadVadModel` — cached loader calls `load_silero_vad()`
- `TestSileroVad` — mock Silero model, verify `(start, end)` tuples in seconds
- `TestGetSpeechSegments` — start buffering, end buffering, end buffer clamping, gap merging, no-speech fallback
- `TestFormatTimestamp` — seconds to `M:SS` / `H:MM:SS` formatting

New cases in `TestRunPipeline`:
- Segmented transcription produces timestamped lines joined by `\n`
- Punctuation applied per-segment
- Safety check runs on joined text without timestamps
- Pipeline unchanged when `use_segmentation=False`
- Pipeline unchanged when `vad_model=None`

All tests use mocks (no real Silero model downloads). `TestLoadVadModel` mocks the `silero_vad` module's `load_silero_vad` function (different from the HuggingFace `from_pretrained` pattern used by other loaders).

## What Stays the Same

- `load_and_preprocess_audio()` — returns single waveform; slicing happens in `run_pipeline()`
- `transcribe_audio()` — called per-segment with sliced waveform
- `check_safety()` — called with joined text
- `apply_punctuation()` — called per-segment
- `_render_result_card()` — transcript is just multi-line now
- Result dict structure — same keys
- JSON export formats — same structure, `transcript` value is multi-line when segmented
