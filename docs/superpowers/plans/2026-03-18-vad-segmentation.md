# VAD Segmentation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Silero VAD segmentation to split audio into speech segments before transcription, producing timestamped output lines.

**Architecture:** Silero VAD runs on CPU to detect speech regions. Segments are buffered, merged, and reused across all tasks. Each segment is transcribed independently, punctuated per-segment (English), then joined with timestamps. Safety check runs once on the full joined text.

**Tech Stack:** silero-vad (PyPI package), torch, existing Granite speech pipeline

**Spec:** `docs/superpowers/specs/2026-03-17-vad-segmentation-design.md`

---

## Chunk 1: Dependencies and Utility Functions

### Task 1: Add silero-vad dependency

**Files:**
- Modify: `pyproject.toml:6-14`

- [ ] **Step 1: Add silero-vad to dependencies**

In `pyproject.toml`, add `silero-vad` to the dependencies list (alphabetical order):

```toml
dependencies = [
    "accelerate>=1.12.0",
    "punctuators>=0.0.5",
    "silero-vad",
    "streamlit",
    "torch",
    "torchaudio",
    "torchcodec>=0.10.0",
    "transformers",
]
```

- [ ] **Step 2: Run uv sync**

Run: `uv sync`
Expected: Resolves and installs silero-vad successfully

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "from silero_vad import load_silero_vad; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add silero-vad dependency"
```

### Task 2: Add format_timestamp function

**Files:**
- Modify: `streamlit_app.py:54` (insert after `ENGLISH_TASKS`)
- Modify: `tests/test_streamlit_app.py`

- [ ] **Step 1: Write failing tests**

Add to the test file imports:

```python
from streamlit_app import (
    ...
    format_timestamp,
    ...
)
```

Add new test class after `TestGetSelectedTasks`:

```python
class TestFormatTimestamp:
    def test_zero_seconds(self) -> None:
        assert format_timestamp(0.0) == "0:00"

    def test_seconds_only(self) -> None:
        assert format_timestamp(15.0) == "0:15"

    def test_minutes_and_seconds(self) -> None:
        assert format_timestamp(62.0) == "1:02"

    def test_hours(self) -> None:
        assert format_timestamp(3661.0) == "1:01:01"

    def test_fractional_seconds_truncated(self) -> None:
        assert format_timestamp(15.7) == "0:15"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestFormatTimestamp -v`
Expected: FAIL — `ImportError: cannot import name 'format_timestamp'`

- [ ] **Step 3: Write implementation**

Add to `streamlit_app.py` after the `ENGLISH_TASKS` constant (line 52), before `get_selected_tasks`:

```python
def format_timestamp(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestFormatTimestamp -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add format_timestamp utility function"
```

### Task 3: Add silero_vad function

**Files:**
- Modify: `streamlit_app.py` (add import + function after `format_timestamp`)
- Modify: `tests/test_streamlit_app.py`

- [ ] **Step 1: Write failing tests**

Add `silero_vad` to the test file imports from `streamlit_app`.

Add new test class after `TestFormatTimestamp`:

```python
class TestSileroVad:
    def test_returns_tuples_in_seconds(self) -> None:
        model = MagicMock()
        mock_timestamps = [
            {"start": 16000, "end": 48000},
            {"start": 64000, "end": 96000},
        ]
        with patch("streamlit_app.get_speech_timestamps", return_value=mock_timestamps):
            result = silero_vad(torch.zeros(1, 160000), model)
        assert result == [(1.0, 3.0), (4.0, 6.0)]

    def test_empty_audio_returns_empty_list(self) -> None:
        model = MagicMock()
        with patch("streamlit_app.get_speech_timestamps", return_value=[]):
            result = silero_vad(torch.zeros(1, 16000), model)
        assert result == []

    def test_passes_model_and_sample_rate(self) -> None:
        model = MagicMock()
        wav = torch.zeros(1, 16000)
        with patch("streamlit_app.get_speech_timestamps", return_value=[]) as mock_fn:
            silero_vad(wav, model)
        mock_fn.assert_called_once()
        call_args = mock_fn.call_args
        assert torch.equal(call_args[0][0], wav.squeeze())
        assert call_args[0][1] is model
        assert call_args[1]["sampling_rate"] == 16000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestSileroVad -v`
Expected: FAIL — `ImportError: cannot import name 'silero_vad'`

- [ ] **Step 3: Write implementation**

Add import at the top of `streamlit_app.py` (after `from punctuators.models import PunctCapSegModelONNX`):

```python
from silero_vad import get_speech_timestamps
```

Add function after `format_timestamp`:

```python
def silero_vad(
    wav: torch.Tensor, model: torch.nn.Module, sample_rate: int = 16000
) -> list[tuple[float, float]]:
    speech_timestamps = get_speech_timestamps(
        wav.squeeze(), model, sampling_rate=sample_rate
    )
    return [
        (ts["start"] / sample_rate, ts["end"] / sample_rate)
        for ts in speech_timestamps
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestSileroVad -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add silero_vad speech detection function"
```

### Task 4: Add get_speech_segments function

**Files:**
- Modify: `streamlit_app.py` (add function after `silero_vad`)
- Modify: `tests/test_streamlit_app.py`

- [ ] **Step 1: Write failing tests**

Add `get_speech_segments` to the test file imports from `streamlit_app`.

Add new test class after `TestSileroVad`:

```python
class TestGetSpeechSegments:
    def test_adds_start_buffer(self) -> None:
        model = MagicMock()
        with patch(
            "streamlit_app.silero_vad", return_value=[(1.0, 2.0)]
        ):
            result = get_speech_segments(torch.zeros(1, 160000), model)
        assert result[0]["start"] == pytest.approx(0.7)

    def test_adds_end_buffer(self) -> None:
        model = MagicMock()
        with patch(
            "streamlit_app.silero_vad", return_value=[(1.0, 2.0)]
        ):
            result = get_speech_segments(torch.zeros(1, 160000), model)
        assert result[0]["end"] == pytest.approx(2.3)

    def test_clamps_start_buffer_to_zero(self) -> None:
        model = MagicMock()
        with patch(
            "streamlit_app.silero_vad", return_value=[(0.1, 1.0)]
        ):
            result = get_speech_segments(torch.zeros(1, 160000), model)
        assert result[0]["start"] == 0.0

    def test_clamps_end_buffer_to_duration(self) -> None:
        model = MagicMock()
        wav = torch.zeros(1, 32000)  # 2 seconds at 16kHz
        with patch(
            "streamlit_app.silero_vad", return_value=[(0.5, 1.9)]
        ):
            result = get_speech_segments(wav, model)
        assert result[0]["end"] == 2.0

    def test_merges_close_segments(self) -> None:
        model = MagicMock()
        with patch(
            "streamlit_app.silero_vad",
            return_value=[(1.0, 2.0), (2.3, 3.0)],
        ):
            result = get_speech_segments(torch.zeros(1, 160000), model)
        assert len(result) == 1
        assert result[0]["start"] == pytest.approx(0.7)
        assert result[0]["end"] == pytest.approx(3.3)

    def test_keeps_distant_segments_separate(self) -> None:
        model = MagicMock()
        with patch(
            "streamlit_app.silero_vad",
            return_value=[(1.0, 2.0), (5.0, 6.0)],
        ):
            result = get_speech_segments(torch.zeros(1, 160000), model)
        assert len(result) == 2

    def test_no_speech_falls_back_to_full_audio(self) -> None:
        model = MagicMock()
        wav = torch.zeros(1, 160000)  # 10 seconds at 16kHz
        with patch("streamlit_app.silero_vad", return_value=[]):
            result = get_speech_segments(wav, model)
        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 10.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestGetSpeechSegments -v`
Expected: FAIL — `ImportError: cannot import name 'get_speech_segments'`

- [ ] **Step 3: Write implementation**

Add function after `silero_vad` in `streamlit_app.py`:

```python
def get_speech_segments(
    wav: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: int = 16000,
) -> list[dict[str, float]]:
    duration = wav.shape[-1] / sample_rate
    vad_segments = silero_vad(wav, model, sample_rate)
    if not vad_segments:
        return [{"start": 0.0, "end": duration}]
    start_buffer = 0.3
    end_buffer = 0.3
    min_gap = 0.5
    segments: list[dict[str, float]] = []
    for start, end in vad_segments:
        buffered_start = max(0.0, start - start_buffer)
        buffered_end = min(duration, end + end_buffer)
        if segments and buffered_start - segments[-1]["end"] < min_gap:
            segments[-1]["end"] = max(segments[-1]["end"], buffered_end)
        else:
            segments.append({"start": buffered_start, "end": buffered_end})
    return segments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestGetSpeechSegments -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Run all tests to verify no regressions**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add get_speech_segments with buffering and merging"
```

## Chunk 2: Model Loading and Pipeline Integration

### Task 5: Add load_vad_model function

**Files:**
- Modify: `streamlit_app.py` (add import + function after `load_punctuation_model`)
- Modify: `tests/test_streamlit_app.py`

- [ ] **Step 1: Write failing test**

Add `load_vad_model` to the test file imports from `streamlit_app`.

Add new test class after `TestLoadPunctuationModel`:

```python
class TestLoadVadModel:
    @patch("streamlit_app.load_silero_vad")
    @patch("streamlit_app.st")
    def test_loads_model(
        self,
        _mock_st: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        result = load_vad_model.__wrapped__()  # type: ignore[attr-defined]
        mock_load.assert_called_once()
        assert result == mock_load.return_value
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestLoadVadModel -v`
Expected: FAIL — `ImportError: cannot import name 'load_vad_model'`

- [ ] **Step 3: Write implementation**

Add import at the top of `streamlit_app.py` (update the existing `silero_vad` import line):

```python
from silero_vad import get_speech_timestamps, load_silero_vad
```

Add function after `load_punctuation_model`:

```python
@st.cache_resource(show_spinner=False)
def load_vad_model() -> torch.nn.Module:
    return load_silero_vad()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streamlit_app.py::TestLoadVadModel -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add cached VAD model loader"
```

### Task 6: Integrate segmentation into run_pipeline

This is the largest task. `run_pipeline` gains two new parameters and a segmented code path.

**Files:**
- Modify: `streamlit_app.py:155-193` (`run_pipeline` function)
- Modify: `tests/test_streamlit_app.py` (`TestRunPipeline` class)

- [ ] **Step 1: Write failing tests for segmented pipeline**

Add these new test methods to the existing `TestRunPipeline` class. The `_make_mocks` helper already returns `(model, processor, guardian_model, guardian_tokenizer, punct_model)`. We need a VAD model mock too.

```python
    def test_segmented_transcript_has_timestamps(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        vad_model = MagicMock()
        wav = torch.zeros(1, 48000)  # 3 seconds
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                ["Transcribe"],
                model,
                processor,
                "cpu",
                guardian_model,
                guardian_tokenizer,
                None,
                None,
                vad_model,
                True,
            )

        transcript = results["Transcribe"]["transcript"]
        assert "[0:00 - 0:01]" in transcript
        assert "[0:01 - 0:03]" in transcript
        assert "\n" in transcript

    def test_segmented_eval_duration_is_sum(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        vad_model = MagicMock()
        wav = torch.zeros(1, 48000)
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                ["Transcribe"],
                model,
                processor,
                "cpu",
                guardian_model,
                guardian_tokenizer,
                None,
                None,
                vad_model,
                True,
            )

        assert results["Transcribe"]["eval_duration"] > 0

    def test_segmented_num_words_excludes_timestamps(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        vad_model = MagicMock()
        wav = torch.zeros(1, 48000)
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                ["Transcribe"],
                model,
                processor,
                "cpu",
                guardian_model,
                guardian_tokenizer,
                None,
                None,
                vad_model,
                True,
            )

        # "decoded text" = 2 words per segment, 2 segments = 4 words
        assert results["Transcribe"]["num_words"] == 4

    def test_segmented_punctuation_per_segment(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, punct_model = (
            self._make_mocks()
        )
        vad_model = MagicMock()
        wav = torch.zeros(1, 48000)
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                ["Transcribe"],
                model,
                processor,
                "cpu",
                guardian_model,
                guardian_tokenizer,
                punct_model,
                None,
                vad_model,
                True,
            )

        # punct_model.infer called once per segment
        assert punct_model.infer.call_count == 2

    def test_segmented_safety_on_joined_text_without_timestamps(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, punct_model = (
            self._make_mocks()
        )
        vad_model = MagicMock()
        wav = torch.zeros(1, 48000)
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                ["Transcribe"],
                model,
                processor,
                "cpu",
                guardian_model,
                guardian_tokenizer,
                punct_model,
                None,
                vad_model,
                True,
            )

        # Guardian receives joined punctuated text without timestamps
        guardian_tokenizer.assert_called_once_with(
            ["Decoded text. Decoded text."],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def test_segmented_translation_skips_punctuation_and_safety(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, punct_model = (
            self._make_mocks()
        )
        vad_model = MagicMock()
        wav = torch.zeros(1, 48000)
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                ["French"],
                model,
                processor,
                "cpu",
                guardian_model,
                guardian_tokenizer,
                punct_model,
                None,
                vad_model,
                True,
            )

        punct_model.infer.assert_not_called()
        guardian_tokenizer.assert_not_called()
        assert "is_toxic" not in results["French"]

    def test_pipeline_unchanged_when_segmentation_disabled(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        vad_model = MagicMock()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            processor,
            "cpu",
            guardian_model,
            guardian_tokenizer,
            None,
            None,
            vad_model,
            False,
        )

        # No timestamps in transcript
        assert "[" not in results["Transcribe"]["transcript"]
        assert results["Transcribe"]["transcript"] == "decoded text"

    def test_pipeline_unchanged_when_vad_model_none(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            processor,
            "cpu",
            guardian_model,
            guardian_tokenizer,
            None,
            None,
            None,
            True,
        )

        assert "[" not in results["Transcribe"]["transcript"]
        assert results["Transcribe"]["transcript"] == "decoded text"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestRunPipeline::test_segmented_transcript_has_timestamps -v`
Expected: FAIL — `TypeError` due to unexpected arguments

- [ ] **Step 3: Write implementation**

Replace the `run_pipeline` function in `streamlit_app.py`. The new signature adds `vad_model` and `use_segmentation` after `punct_model`:

```python
@torch.inference_mode()
def run_pipeline(
    wav: torch.Tensor,
    tasks: list[str],
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    device: str,
    guardian_model: AutoModelForSequenceClassification | None = None,
    guardian_tokenizer: AutoTokenizer | None = None,
    punct_model: PunctCapSegModelONNX | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    vad_model: torch.nn.Module | None = None,
    use_segmentation: bool = False,
) -> dict[str, dict[str, object]]:
    segmented = use_segmentation and vad_model is not None
    segments = get_speech_segments(wav, vad_model) if segmented else None
    results: dict[str, dict[str, object]] = {}
    for i, task in enumerate(tasks):
        if on_progress:
            on_progress(i, len(tasks), task)
        prompt = PROMPT_CHOICES[task]
        if segmented and segments:
            segment_texts: list[str] = []
            total_words = 0
            total_duration = 0.0
            for seg in segments:
                start_sample = int(seg["start"] * 16000)
                end_sample = int(seg["end"] * 16000)
                wav_segment = wav[:, start_sample:end_sample]
                seg_transcript, seg_duration = transcribe_audio.__wrapped__(
                    wav_segment, prompt, model, processor, device
                )
                if task in ENGLISH_TASKS and punct_model is not None:
                    seg_transcript = apply_punctuation(seg_transcript, punct_model)
                total_words += len(seg_transcript.split())
                total_duration += seg_duration
                ts_start = format_timestamp(seg["start"])
                ts_end = format_timestamp(seg["end"])
                segment_texts.append(f"[{ts_start} - {ts_end}] {seg_transcript}")
            transcript = "\n".join(segment_texts)
            eval_duration = round(total_duration, 2)
            num_words = total_words
        else:
            transcript, eval_duration = transcribe_audio.__wrapped__(
                wav, prompt, model, processor, device
            )
            if task in ENGLISH_TASKS and punct_model is not None:
                transcript = apply_punctuation(transcript, punct_model)
            num_words = len(transcript.split())
        result: dict[str, object] = {
            "transcript": transcript,
            "num_words": num_words,
            "eval_duration": eval_duration,
        }
        if (
            task in ENGLISH_TASKS
            and guardian_model is not None
            and guardian_tokenizer is not None
        ):
            if segmented and segments:
                safety_text = " ".join(
                    seg_transcript
                    for line in segment_texts
                    for seg_transcript in [line.split("] ", 1)[1]]
                )
            else:
                safety_text = transcript
            is_toxic, toxicity_score = check_safety.__wrapped__(
                safety_text, guardian_model, guardian_tokenizer
            )
            result["is_toxic"] = is_toxic
            result["toxicity_score"] = toxicity_score
        results[task] = result
    return results
```

- [ ] **Step 4: Run new segmentation tests**

Run: `uv run pytest tests/test_streamlit_app.py::TestRunPipeline -k "segmented" -v`
Expected: All 8 new segmentation tests PASS

- [ ] **Step 5: Run all pipeline tests to verify no regressions**

Run: `uv run pytest tests/test_streamlit_app.py::TestRunPipeline -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS

- [ ] **Step 7: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors

- [ ] **Step 8: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: integrate VAD segmentation into pipeline"
```

## Chunk 3: UI Integration and Documentation

### Task 7: Add segmentation UI controls to main()

**Files:**
- Modify: `streamlit_app.py:241-376` (`main` function)

- [ ] **Step 1: Add segmentation checkbox**

In `main()`, add the checkbox after the audio input tabs block (after line 277 `audio_file = recorded or uploaded`):

```python
    use_segmentation = st.checkbox("VAD segmentation", value=True)
```

- [ ] **Step 2: Add use_segmentation to input_key**

Update the `input_key` tuple (line 279) to include the segmentation toggle:

```python
    input_key = (
        (audio_file.name, audio_file.size, tuple(tasks), use_segmentation)
        if audio_file
        else None
    )
```

- [ ] **Step 3: Add VAD model loading in the pipeline block**

Inside the `if st.button(...)` block, after loading the speech model (line 295), add VAD model loading:

```python
            if use_segmentation:
                with st.spinner("Loading VAD model..."):
                    vad_model = load_vad_model()
            else:
                vad_model = None
```

- [ ] **Step 4: Pass new parameters to run_pipeline**

Update the `run_pipeline` call to include the new parameters:

```python
            pipeline_results = run_pipeline(
                wav,
                tasks,
                model,
                processor,
                device,
                guardian_model,
                guardian_tokenizer,
                punct_model,
                on_progress=update_progress,
                vad_model=vad_model,
                use_segmentation=use_segmentation,
            )
```

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run lint, format, and typecheck**

Run: `uv run ruff check . && uv run ruff format . && uv run ty check`
Expected: No errors (ty may show pre-existing warnings — no new ones)

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add VAD segmentation UI controls"
```

### Task 8: Update documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `pyproject.toml:3` (version bump)

- [ ] **Step 1: Bump version**

In `pyproject.toml`, update version from `"0.4.0"` to `"0.5.0"`.

- [ ] **Step 2: Update CLAUDE.md**

Update Project Overview to mention VAD segmentation. Change:
```
Supports multi-task pipeline processing with preset task groups. Includes automatic punctuation/capitalization and English toxicity detection.
```
To:
```
Supports multi-task pipeline processing with preset task groups. Includes VAD-based audio segmentation, automatic punctuation/capitalization, and English toxicity detection.
```

Add to Dependencies section (alphabetical):
```
- `silero-vad` — Voice Activity Detection for audio segmentation
```

Add to Models section:
```
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice Activity Detection for speech segmentation (runs on CPU)
```

Update UI Layout section — add a bullet for segmentation:
```
- **Segmentation** — `st.checkbox` for VAD segmentation toggle (default on)
```

Update Architecture section — add new functions and pipeline note:
```
- `format_timestamp` — formats seconds to `M:SS` or `H:MM:SS`
- `silero_vad` — runs Silero VAD on waveform, returns `(start, end)` tuples in seconds
- `get_speech_segments` — post-processes VAD output with buffering and merging
- `load_vad_model` — cached Silero VAD model loader
- Pipeline supports optional VAD segmentation with per-segment transcription and timestamped output
```

Update Performance section:
```
- Silero VAD model runs on CPU (~3MB)
```

Update Tests section to mention new test classes:
```
- `TestFormatTimestamp`, `TestSileroVad`, `TestGetSpeechSegments`, `TestLoadVadModel`, plus new segmentation cases in `TestRunPipeline`
```

Note: `main()` UI changes (checkbox, model loading, parameter passing) are not unit-tested, consistent with the existing codebase where `main()` has no dedicated unit tests.

- [ ] **Step 3: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml CLAUDE.md
git commit -m "docs: update CLAUDE.md and bump version for VAD segmentation"
```

### Task 9: Manual verification

- [ ] **Step 1: Run the full test suite one final time**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run all quality checks**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: No errors

- [ ] **Step 3: Verify app starts**

Run: `uv run python -c "import streamlit_app; print('imports ok')"`
Expected: `imports ok` (verifies no import errors)
