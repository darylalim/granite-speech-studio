# Punctuation & Capitalization for English Transcriptions

## Summary

Add automatic punctuation and capitalization to English transcription output using `PunctCapSegModelONNX` from the `punctuators` library (`from punctuators.models import PunctCapSegModelONNX`).

## Scope

- **English transcriptions only** — gated by `ENGLISH_TASKS`, same as Guardian safety checks
- **Always applied** — no user toggle; raw lowercase output is rarely desired
- **Applied before Guardian** — safety check receives properly formatted text

## Approach

Use `PunctCapSegModelONNX.from_pretrained("pcs_en")` from the `punctuators` library. ONNX-based, small, fast, CPU-only.

## Design

### Model Loading

New `load_punctuation_model(model_id: str) -> PunctCapSegModelONNX` function:

- Cached with `@st.cache_resource`
- Loads on demand — only when pipeline includes an English task, in the same `if ENGLISH_TASKS.intersection(tasks):` block in `main()` as Guardian, with a spinner (e.g., "Loading punctuation model...")
- No device parameter needed (ONNX runs on CPU)
- No `@torch.inference_mode()` — uses ONNX Runtime, not PyTorch
- Model ID constant: `PUNCTUATION_MODEL_ID = "pcs_en"`

### Post-processing

New `apply_punctuation(text: str, model: PunctCapSegModelONNX) -> str` function:

- Calls `model.infer([text])` which returns `list[list[str]]` — a list of sentence-segmented outputs per input
- Takes `result[0]` (the sentences for our single input) and joins with `" "`
- Replaces `<unk>` and `<Unk>` tokens with a single space, then collapses double spaces and strips
- No `@torch.inference_mode()` — ONNX, not PyTorch

### Pipeline Integration

In `run_pipeline`, for English tasks, the flow becomes:

1. Transcribe audio
2. Apply punctuation/capitalization
3. Run Guardian safety check

The punctuation model is passed into `run_pipeline` as an optional parameter: `punct_model: PunctCapSegModelONNX | None = None`. The type is imported at module level alongside the other model imports.

### Result Data

No new fields. `"transcript"` stores the punctuated text directly. No raw transcript preserved — keeps JSON output clean.

### UI Changes

None beyond the footer. Punctuated text flows through existing `_render_result_card` for display, downloads, and word count. Footer adds `Punctuation: pcs_en` label (no link — it's a model name, not a HF repo path).

### Dependencies

Add `punctuators` to `pyproject.toml` dependencies with a minimum version pin.

### Documentation

Update `CLAUDE.md`:
- Add `punctuators` to Dependencies section
- Add `pcs_en` to Models section
- Update pipeline flow description in Architecture

## Testing

- `TestLoadPunctuationModel` — verifies `from_pretrained` called with `"pcs_en"`, mocks `st.cache_resource`
- `TestApplyPunctuation` — verifies `model.infer` called with `[text]`, result unpacked from `result[0]`, `<unk>` cleanup works, empty string handling
- `TestRunPipeline` additions:
  - Punctuation applied before safety check for English tasks
  - Skipped for translation tasks
  - Pipeline works when punctuation model is `None`
