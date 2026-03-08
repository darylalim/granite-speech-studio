# Pipeline UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the single-task Granite Speech app into a multi-task pipeline with presets, side-by-side results, and combined downloads.

**Architecture:** Add a `TASK_PRESETS` dict mapping preset names to lists of task keys. Replace single-select pills with preset pills + multiselect. Loop over selected tasks calling `transcribe_audio` for each, storing results as a list. Render results in a column grid with individual and combined download options.

**Tech Stack:** Streamlit, PyTorch, Transformers (no new dependencies)

---

### Task 1: Add TASK_PRESETS constant

**Files:**
- Modify: `streamlit_app.py:18-28`
- Test: `tests/test_streamlit_app.py`

**Step 1: Write the failing test**

Add to `tests/test_streamlit_app.py`:

```python
from streamlit_app import (
    MODEL_ID,
    PROMPT_CHOICES,
    SUPPORTED_FORMATS,
    TASK_PRESETS,
    get_device,
    load_and_preprocess_audio,
    load_model,
    transcribe_audio,
)


class TestTaskPresets:
    def test_all_tasks_preset(self) -> None:
        assert set(TASK_PRESETS["All Tasks"]) == set(PROMPT_CHOICES.keys())

    def test_european_preset(self) -> None:
        expected = {"Transcribe", "French", "German", "Spanish", "Portuguese", "Italian"}
        assert set(TASK_PRESETS["European Languages"]) == expected

    def test_asian_preset(self) -> None:
        expected = {"Transcribe", "Japanese", "Mandarin Chinese"}
        assert set(TASK_PRESETS["Asian Languages"]) == expected

    def test_transcribe_only_preset(self) -> None:
        assert TASK_PRESETS["Transcribe Only"] == ["Transcribe"]

    def test_all_preset_values_are_valid_prompt_keys(self) -> None:
        for preset_name, tasks in TASK_PRESETS.items():
            for task in tasks:
                assert task in PROMPT_CHOICES, f"{task} in preset '{preset_name}' not in PROMPT_CHOICES"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestTaskPresets -v`
Expected: FAIL with `ImportError` (TASK_PRESETS not defined)

**Step 3: Write minimal implementation**

Add to `streamlit_app.py` after `SUPPORTED_FORMATS` (line 29):

```python
TASK_PRESETS: dict[str, list[str]] = {
    "All Tasks": list(PROMPT_CHOICES.keys()),
    "European Languages": ["Transcribe", "French", "German", "Spanish", "Portuguese", "Italian"],
    "Asian Languages": ["Transcribe", "Japanese", "Mandarin Chinese"],
    "Transcribe Only": ["Transcribe"],
}
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streamlit_app.py::TestTaskPresets -v`
Expected: PASS

**Step 5: Run all existing tests to verify no regressions**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 6: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`

**Step 7: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add TASK_PRESETS constant for pipeline task groups"
```

---

### Task 2: Add get_selected_tasks helper

**Files:**
- Modify: `streamlit_app.py`
- Test: `tests/test_streamlit_app.py`

**Step 1: Write the failing test**

Add to `tests/test_streamlit_app.py`:

```python
from streamlit_app import (
    MODEL_ID,
    PROMPT_CHOICES,
    SUPPORTED_FORMATS,
    TASK_PRESETS,
    get_device,
    get_selected_tasks,
    load_and_preprocess_audio,
    load_model,
    transcribe_audio,
)


class TestGetSelectedTasks:
    def test_preset_returns_preset_tasks(self) -> None:
        result = get_selected_tasks("European Languages", [])
        assert result == TASK_PRESETS["European Languages"]

    def test_custom_returns_custom_tasks(self) -> None:
        result = get_selected_tasks(None, ["French", "Japanese"])
        assert result == ["French", "Japanese"]

    def test_preset_overrides_custom(self) -> None:
        result = get_selected_tasks("Asian Languages", ["French", "German"])
        assert result == TASK_PRESETS["Asian Languages"]

    def test_none_preset_empty_custom_returns_empty(self) -> None:
        result = get_selected_tasks(None, [])
        assert result == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestGetSelectedTasks -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `streamlit_app.py` after `TASK_PRESETS`:

```python
def get_selected_tasks(
    preset: str | None, custom: list[str]
) -> list[str]:
    if preset is not None:
        return TASK_PRESETS[preset]
    return custom
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streamlit_app.py::TestGetSelectedTasks -v`
Expected: PASS

**Step 5: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`

**Step 6: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add get_selected_tasks helper"
```

---

### Task 3: Add run_pipeline function

**Files:**
- Modify: `streamlit_app.py`
- Test: `tests/test_streamlit_app.py`

**Step 1: Write the failing test**

Add to `tests/test_streamlit_app.py`:

```python
from streamlit_app import (
    MODEL_ID,
    PROMPT_CHOICES,
    SUPPORTED_FORMATS,
    TASK_PRESETS,
    get_device,
    get_selected_tasks,
    load_and_preprocess_audio,
    load_model,
    run_pipeline,
    transcribe_audio,
)


class TestRunPipeline:
    def _make_mocks(self) -> tuple[MagicMock, MagicMock]:
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted"
        tokenizer.batch_decode.return_value = ["decoded text"]
        processor = MagicMock()
        processor.tokenizer = tokenizer
        model = MagicMock()
        return model, processor

    def test_returns_dict_keyed_by_task(self) -> None:
        model, processor = self._make_mocks()
        wav = torch.zeros(1, 16000)
        tasks = ["Transcribe", "French"]

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav, tasks, model, processor, "cpu"
        )

        assert set(results.keys()) == {"Transcribe", "French"}

    def test_each_result_has_transcript_and_duration(self) -> None:
        model, processor = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav, ["Transcribe"], model, processor, "cpu"
        )

        result = results["Transcribe"]
        assert "transcript" in result
        assert "eval_duration" in result
        assert "num_words" in result

    def test_empty_tasks_returns_empty_dict(self) -> None:
        model, processor = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav, [], model, processor, "cpu"
        )

        assert results == {}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestRunPipeline -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `streamlit_app.py` after `transcribe_audio`:

```python
@torch.inference_mode()
def run_pipeline(
    wav: torch.Tensor,
    tasks: list[str],
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    device: str,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for task in tasks:
        prompt = PROMPT_CHOICES[task]
        transcript, eval_duration = transcribe_audio.__wrapped__(
            wav, prompt, model, processor, device
        )
        results[task] = {
            "transcript": transcript,
            "num_words": len(transcript.split()),
            "eval_duration": eval_duration,
        }
    return results
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streamlit_app.py::TestRunPipeline -v`
Expected: PASS

**Step 5: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`

**Step 6: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add run_pipeline function for multi-task processing"
```

---

### Task 4: Replace main() with pipeline UI

**Files:**
- Modify: `streamlit_app.py:92-189`

This task replaces the entire `main()` function. No new unit tests — this is pure Streamlit UI wiring that must be verified manually.

**Step 1: Replace main() in `streamlit_app.py`**

Replace lines 92-189 with:

```python
def main() -> None:
    st.set_page_config(
        page_title="Granite Speech Pipeline",
        page_icon="\U0001f399\ufe0f",
        layout="wide",
    )

    device = get_device()

    st.title("\U0001f399\ufe0f Granite Speech Pipeline")

    with st.spinner(f"Loading model on {device.upper()}..."):
        model, processor = load_model(MODEL_ID, device)

    preset = st.pills(
        "Preset",
        options=list(TASK_PRESETS.keys()),
        default=None,
    )

    default_tasks = TASK_PRESETS[preset] if preset else []
    selected_tasks = st.multiselect(
        "Tasks",
        options=list(PROMPT_CHOICES.keys()),
        default=default_tasks,
    )

    tasks = get_selected_tasks(preset, selected_tasks)

    upload_tab, record_tab = st.tabs(["Upload", "Record"])
    with upload_tab:
        uploaded = st.file_uploader(
            "Upload audio file",
            type=SUPPORTED_FORMATS,
            help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
        )
    with record_tab:
        recorded = st.audio_input("Record audio")

    audio_file = recorded or uploaded

    input_key = (
        (audio_file.name, audio_file.size, tuple(tasks)) if audio_file else None
    )
    if input_key != st.session_state.get("_last_input_key"):
        st.session_state.pop("results", None)
        st.session_state.pop("result_filename", None)
        st.session_state.pop("audio_duration", None)
        st.session_state["_last_input_key"] = input_key

    if audio_file:
        st.audio(audio_file)
        caption = "Recorded audio" if recorded else audio_file.name
        st.caption(caption)

    can_run = audio_file is not None and len(tasks) > 0

    if st.button("Run Pipeline", type="primary", disabled=not can_run) and can_run:
        progress = st.progress(0, text="Starting pipeline...")
        try:
            wav, audio_duration = load_and_preprocess_audio(audio_file)
            pipeline_results: dict[str, dict[str, object]] = {}
            for i, task in enumerate(tasks):
                progress.progress(
                    (i) / len(tasks),
                    text=f"Processing: {task}...",
                )
                prompt = PROMPT_CHOICES[task]
                transcript, eval_duration = transcribe_audio(
                    wav, prompt, model, processor, device
                )
                pipeline_results[task] = {
                    "transcript": transcript,
                    "num_words": len(transcript.split()),
                    "eval_duration": eval_duration,
                }
            progress.progress(1.0, text="Done!")
            st.session_state.results = pipeline_results
            st.session_state.audio_duration = audio_duration
            stem = Path(audio_file.name).stem
            if audio_file.name == "audio.wav":
                stem = datetime.now().strftime("recording_%Y%m%d_%H%M%S")
            st.session_state.result_filename = f"{stem}_pipeline.json"
        except RuntimeError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.exception(e)
            return
        st.toast("Pipeline complete!")

    if "results" in st.session_state:
        results = st.session_state.results
        audio_duration = st.session_state.audio_duration
        task_names = list(results.keys())

        num_cols = min(len(task_names), 3)
        for row_start in range(0, len(task_names), num_cols):
            row_tasks = task_names[row_start : row_start + num_cols]
            cols = st.columns(num_cols)
            for col, task_name in zip(cols, row_tasks):
                result = results[task_name]
                with col:
                    with st.container(border=True):
                        st.subheader(task_name)
                        st.code(result["transcript"], language=None)
                        m_cols = st.columns(3)
                        m_cols[0].metric("Duration", f"{audio_duration:.2f}s")
                        m_cols[1].metric("Words", result["num_words"])
                        m_cols[2].metric("Time", f"{result['eval_duration']}s")
                        dl_cols = st.columns(2)
                        stem = st.session_state.result_filename.replace(
                            "_pipeline.json", ""
                        )
                        dl_cols[0].download_button(
                            "Text",
                            result["transcript"],
                            f"{stem}_{task_name.lower().replace(' ', '_')}.txt",
                            "text/plain",
                            key=f"dl_txt_{task_name}",
                        )
                        dl_cols[1].download_button(
                            "JSON",
                            json.dumps(
                                {
                                    "model": MODEL_ID,
                                    "task": task_name,
                                    "audio_duration": audio_duration,
                                    **result,
                                },
                                indent=2,
                            ),
                            f"{stem}_{task_name.lower().replace(' ', '_')}.json",
                            "application/json",
                            key=f"dl_json_{task_name}",
                        )

        combined = {
            "model": MODEL_ID,
            "audio_duration": audio_duration,
            "results": results,
        }
        st.download_button(
            "Download All (JSON)",
            json.dumps(combined, indent=2),
            st.session_state.result_filename,
            "application/json",
        )

    st.caption(
        f"Model: {MODEL_ID.split('/')[-1]} | Device: {device.upper()} | "
        f"[Model Card](https://huggingface.co/{MODEL_ID})"
    )
```

**Step 2: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS (main() is not unit-tested)

**Step 3: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`

**Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: replace single-task UI with pipeline UI"
```

---

### Task 5: Update existing tests for renamed session state keys

**Files:**
- Test: `tests/test_streamlit_app.py`

The session state keys changed from `result` (singular) to `results` (plural). Existing tests don't test session state directly, so this step is a verification pass.

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 2: Run linter and type checker**

Run: `uv run ruff check . && uv run ruff format . && uv run ty check`

**Step 3: Commit (only if changes were needed)**

```bash
git add tests/test_streamlit_app.py
git commit -m "test: update tests for pipeline UI changes"
```

---

### Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update the Architecture section**

Update the following sections in `CLAUDE.md`:

- **UI Layout**: Replace sidebar description with footer description. Replace single task selection with preset pills + multiselect description. Replace single result display with grid layout description.
- **Downloads**: Update to reflect per-task downloads + combined "Download All (JSON)".

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for pipeline UI"
```

---

### Task 7: Manual verification

**Step 1: Run the app**

Run: `uv run streamlit run streamlit_app.py`

**Step 2: Verify checklist**

- [ ] Preset pills appear and pre-populate multiselect
- [ ] Editing multiselect works independently when no preset selected
- [ ] Upload and Record tabs work
- [ ] "Run Pipeline" button processes all selected tasks
- [ ] Progress bar updates per task
- [ ] Results appear in side-by-side grid (up to 3 columns)
- [ ] Individual Text/JSON download buttons work per result
- [ ] "Download All (JSON)" button exports combined results
- [ ] Footer shows model name, device, and model card link
- [ ] No sidebar visible
