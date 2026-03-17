# Punctuation & Capitalization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic punctuation and capitalization to English transcriptions using `PunctCapSegModelONNX`.

**Architecture:** A new `apply_punctuation` function post-processes English transcripts inside `run_pipeline`, between transcription and safety checking. The ONNX model loads on demand via `@st.cache_resource`, following the same pattern as Guardian.

**Tech Stack:** `punctuators` (PunctCapSegModelONNX, ONNX Runtime)

**Spec:** `docs/specs/2026-03-16-punctuation-capitalization-design.md`

---

## Chunk 1: Full Implementation

### Task 1: Add dependency

**Files:**
- Modify: `pyproject.toml:6-13`

- [ ] **Step 1: Add `punctuators` to dependencies**

In `pyproject.toml`, add `"punctuators"` to the dependencies list (after `"streamlit"`):

```toml
dependencies = [
    "accelerate>=1.12.0",
    "punctuators>=0.0.5",
    "streamlit",
    "torch",
    "torchaudio",
    "torchcodec>=0.10.0",
    "transformers",
]
```

- [ ] **Step 2: Install**

Run: `uv sync`
Expected: resolves and installs `punctuators` and its dependencies

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add punctuators dependency"
```

---

### Task 2: Add `load_punctuation_model` with tests

**Files:**
- Modify: `streamlit_app.py:9-26,92-98`
- Modify: `tests/test_streamlit_app.py:9-24,29-34,198-214`

- [ ] **Step 1: Write failing tests**

In `tests/test_streamlit_app.py`, add imports for `PUNCTUATION_MODEL_ID` and `load_punctuation_model` to the import block (line 9-24). Add a test for the constant and a test class after `TestLoadGuardianModel` (after line 214):

```python
# Add to imports (line 9-24):
#   PUNCTUATION_MODEL_ID,
#   load_punctuation_model,

# Add after TestModelIds (line 29-34):
def test_punctuation_model_id(self) -> None:
    assert PUNCTUATION_MODEL_ID == "pcs_en"

# Add after TestLoadGuardianModel (after line 214):
class TestLoadPunctuationModel:
    @patch("streamlit_app.PunctCapSegModelONNX")
    @patch("streamlit_app.st")
    def test_loads_model(
        self,
        _mock_st: MagicMock,
        mock_model_cls: MagicMock,
    ) -> None:
        result = load_punctuation_model.__wrapped__("pcs_en")  # type: ignore[attr-defined]
        mock_model_cls.from_pretrained.assert_called_once_with("pcs_en")
        assert result == mock_model_cls.from_pretrained.return_value
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestModelIds::test_punctuation_model_id tests/test_streamlit_app.py::TestLoadPunctuationModel -v`
Expected: FAIL — `PUNCTUATION_MODEL_ID` and `load_punctuation_model` not importable

- [ ] **Step 3: Write implementation**

In `streamlit_app.py`:

Add import after line 11 (`import torchaudio`) and before line 12 (`from streamlit.runtime...`). Per ruff isort, bare `import` statements come before `from` statements within the third-party section, then each group is sorted alphabetically — so `from punctuators` goes after the bare imports and before `from streamlit`:

```python
from punctuators.models import PunctCapSegModelONNX
```

Add constant after `GUARDIAN_MODEL_ID` (line 25):

```python
PUNCTUATION_MODEL_ID = "pcs_en"
```

Add function after `load_guardian_model` (after line 98):

```python
@st.cache_resource(show_spinner=False)
def load_punctuation_model(model_id: str) -> PunctCapSegModelONNX:
    return PunctCapSegModelONNX.from_pretrained(model_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestModelIds::test_punctuation_model_id tests/test_streamlit_app.py::TestLoadPunctuationModel -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add punctuation model loading with cache"
```

---

### Task 3: Add `apply_punctuation` with tests

**Files:**
- Modify: `streamlit_app.py` (after `load_punctuation_model`)
- Modify: `tests/test_streamlit_app.py` (after `TestLoadPunctuationModel`)

- [ ] **Step 1: Write failing tests**

In `tests/test_streamlit_app.py`, add `apply_punctuation` to the import block. Add test class after `TestLoadPunctuationModel`:

```python
# Add to imports: apply_punctuation

class TestApplyPunctuation:
    def test_calls_infer_with_list(self) -> None:
        model = MagicMock()
        model.infer.return_value = [["Hello world."]]
        result = apply_punctuation("hello world", model)
        model.infer.assert_called_once_with(["hello world"])
        assert result == "Hello world."

    def test_joins_multiple_sentences(self) -> None:
        model = MagicMock()
        model.infer.return_value = [["Hello.", "How are you?"]]
        result = apply_punctuation("hello how are you", model)
        assert result == "Hello. How are you?"

    def test_cleans_unk_tokens(self) -> None:
        model = MagicMock()
        model.infer.return_value = [["Hello <unk> world <Unk> test."]]
        result = apply_punctuation("hello world test", model)
        assert result == "Hello world test."

    def test_empty_string(self) -> None:
        model = MagicMock()
        model.infer.return_value = [[""]]
        result = apply_punctuation("", model)
        assert result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestApplyPunctuation -v`
Expected: FAIL — `apply_punctuation` not importable

- [ ] **Step 3: Write implementation**

In `streamlit_app.py`, add after `load_punctuation_model`:

```python
def apply_punctuation(text: str, model: PunctCapSegModelONNX) -> str:
    result = model.infer([text])
    output = " ".join(result[0])
    output = output.replace("<unk>", " ").replace("<Unk>", " ")
    return " ".join(output.split()).strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestApplyPunctuation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add apply_punctuation post-processing function"
```

---

### Task 4: Integrate into `run_pipeline` with tests

**Files:**
- Modify: `streamlit_app.py:141-176` (`run_pipeline`)
- Modify: `tests/test_streamlit_app.py` (`TestRunPipeline`)

- [ ] **Step 1: Write failing tests**

In `tests/test_streamlit_app.py`, add new tests to `TestRunPipeline`. Update `_make_mocks` to also return a `punct_model`:

```python
# Update _make_mocks to return 5 items:
def _make_mocks(self) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "formatted"
    tokenizer.batch_decode.return_value = ["decoded text"]
    processor = MagicMock()
    processor.tokenizer = tokenizer
    model = MagicMock()
    guardian_tokenizer = MagicMock()
    guardian_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    guardian_model = MagicMock()
    guardian_model.return_value.logits = torch.tensor([[5.0, -5.0]])
    punct_model = MagicMock()
    punct_model.infer.return_value = [["Decoded text."]]
    return model, processor, guardian_model, guardian_tokenizer, punct_model

# Add new tests:
def test_punctuation_applied_to_english(self) -> None:
    model, processor, guardian_model, guardian_tokenizer, punct_model = self._make_mocks()
    wav = torch.zeros(1, 16000)

    results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
        wav,
        ["Transcribe"],
        model,
        processor,
        "cpu",
        guardian_model,
        guardian_tokenizer,
        punct_model,
    )

    punct_model.infer.assert_called_once_with(["decoded text"])
    assert results["Transcribe"]["transcript"] == "Decoded text."

def test_punctuation_skipped_for_translation(self) -> None:
    model, processor, guardian_model, guardian_tokenizer, punct_model = self._make_mocks()
    wav = torch.zeros(1, 16000)

    results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
        wav,
        ["French"],
        model,
        processor,
        "cpu",
        guardian_model,
        guardian_tokenizer,
        punct_model,
    )

    punct_model.infer.assert_not_called()
    assert results["French"]["transcript"] == "decoded text"

def test_punctuation_before_safety_check(self) -> None:
    model, processor, guardian_model, guardian_tokenizer, punct_model = self._make_mocks()
    wav = torch.zeros(1, 16000)

    run_pipeline.__wrapped__(  # type: ignore[attr-defined]
        wav,
        ["Transcribe"],
        model,
        processor,
        "cpu",
        guardian_model,
        guardian_tokenizer,
        punct_model,
    )

    # Guardian receives punctuated text, not raw
    guardian_tokenizer.assert_called_once_with(
        ["Decoded text."], padding=True, truncation=True, return_tensors="pt"
    )

def test_pipeline_works_without_punct_model(self) -> None:
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
    )

    assert results["Transcribe"]["transcript"] == "decoded text"
```

Also update all existing `TestRunPipeline` tests to unpack 5 values from `_make_mocks`:

```python
model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestRunPipeline -v`
Expected: new punctuation tests FAIL (run_pipeline doesn't accept `punct_model` yet), existing tests PASS

- [ ] **Step 3: Update `run_pipeline` implementation**

In `streamlit_app.py`, modify `run_pipeline` (line 141-176):

Add `punct_model` parameter after `guardian_tokenizer`:

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
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for i, task in enumerate(tasks):
        if on_progress:
            on_progress(i, len(tasks), task)
        prompt = PROMPT_CHOICES[task]
        transcript, eval_duration = transcribe_audio.__wrapped__(
            wav, prompt, model, processor, device
        )
        if task in ENGLISH_TASKS and punct_model is not None:
            transcript = apply_punctuation(transcript, punct_model)
        result: dict[str, object] = {
            "transcript": transcript,
            "num_words": len(transcript.split()),
            "eval_duration": eval_duration,
        }
        if (
            task in ENGLISH_TASKS
            and guardian_model is not None
            and guardian_tokenizer is not None
        ):
            is_toxic, toxicity_score = check_safety.__wrapped__(
                transcript, guardian_model, guardian_tokenizer
            )
            result["is_toxic"] = is_toxic
            result["toxicity_score"] = toxicity_score
        results[task] = result
    return results
```

- [ ] **Step 4: Run all pipeline tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestRunPipeline -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: integrate punctuation into pipeline for English tasks"
```

---

### Task 5: Update `main()` and footer

**Files:**
- Modify: `streamlit_app.py:284-300,347-353` (`main()`)

- [ ] **Step 1: Load punctuation model alongside Guardian**

In `main()`, update the `if ENGLISH_TASKS.intersection(tasks):` block (line 284-290) to also load the punctuation model:

```python
            if ENGLISH_TASKS.intersection(tasks):
                with st.spinner("Loading punctuation model..."):
                    punct_model = load_punctuation_model(PUNCTUATION_MODEL_ID)
                with st.spinner("Loading safety model..."):
                    guardian_model, guardian_tokenizer = load_guardian_model(
                        GUARDIAN_MODEL_ID
                    )
            else:
                punct_model = None
                guardian_model, guardian_tokenizer = None, None
```

- [ ] **Step 2: Pass `punct_model` to `run_pipeline`**

Update the `run_pipeline` call (line 292-301) to include `punct_model`:

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
            )
```

- [ ] **Step 3: Update footer**

Update the footer `st.caption` (line 347-353) to include the punctuation model:

```python
    st.caption(
        f"Model: {MODEL_ID.split('/')[-1]} | "
        f"Punctuation: {PUNCTUATION_MODEL_ID} | "
        f"Safety: {GUARDIAN_MODEL_ID.split('/')[-1]} | "
        f"Device: {device.upper()} | "
        f"[Model Card](https://huggingface.co/{MODEL_ID}) | "
        f"[Safety Model](https://huggingface.co/{GUARDIAN_MODEL_ID})"
    )
```

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 5: Run lint and typecheck**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: no issues

- [ ] **Step 6: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: load punctuation model in main and update footer"
```

---

### Task 6: Update documentation

**Files:**
- Modify: `CLAUDE.md:7,31-40,50-53,74-81,96-98`

- [ ] **Step 1: Update project description (line 7)**

Add mention of punctuation:

```
Streamlit web app for speech-to-text and translation using IBM's [Granite Speech](https://huggingface.co/collections/ibm-granite/granite-speech) models. Supports multi-task pipeline processing with preset task groups. Includes automatic punctuation/capitalization and English toxicity detection.
```

- [ ] **Step 2: Update Dependencies section (line 31-40)**

Add after `streamlit` entry:

```
- `punctuators` — English punctuation and capitalization (ONNX)
```

- [ ] **Step 3: Update Models section (line 50-53)**

Add:

```
- pcs_en (via `punctuators`) — English punctuation and capitalization (runs on CPU, ONNX)
```

- [ ] **Step 4: Update Performance section (line 74-81)**

Add:

```
- Punctuation model runs on CPU via ONNX Runtime (no `@torch.inference_mode()`)
```

- [ ] **Step 5: Update Tests section (line 96-98)**

Update to include punctuation:

```
`tests/test_streamlit_app.py` — unit tests for device detection, prompt choices, supported formats, task presets, task selection, audio loading, model loading, guardian model loading, punctuation model loading, punctuation application, safety checking, transcription, pipeline execution, result card rendering, and error handling.
```

- [ ] **Step 6: Update Footer description (line 66)**

```
- **Footer** — model name, punctuation model name, safety model name, device, links to model cards
```

- [ ] **Step 7: Run full test suite and lint**

Run: `uv run pytest -v && uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: ALL PASS, no lint issues

- [ ] **Step 8: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for punctuation feature"
```
