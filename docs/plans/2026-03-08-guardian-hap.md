# Guardian HAP Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add toxicity detection to the speech pipeline using Granite Guardian HAP 38m model.

**Architecture:** After each transcription/translation, run the output text through a 38M-param binary toxicity classifier. Results are flag-only (always show transcript, display warning if toxic). Guardian model runs on CPU, loaded once at startup with caching.

**Tech Stack:** transformers (AutoModelForSequenceClassification, AutoTokenizer), torch, streamlit

---

### Task 1: Add Guardian model constant and import updates

**Files:**
- Modify: `streamlit_app.py:12,18`
- Test: `tests/test_streamlit_app.py`

**Step 1: Write the failing test**

Add to `tests/test_streamlit_app.py` imports:
```python
from streamlit_app import (
    GUARDIAN_MODEL_ID,
    ...existing imports...
)
```

Add test class after `TestModelId`:
```python
class TestGuardianModelId:
    def test_guardian_model_id(self) -> None:
        assert GUARDIAN_MODEL_ID == "ibm-granite/granite-guardian-hap-38m"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestGuardianModelId -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

In `streamlit_app.py` line 12, update import:
```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
)
```

After line 18, add:
```python
GUARDIAN_MODEL_ID = "ibm-granite/granite-guardian-hap-38m"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streamlit_app.py::TestGuardianModelId -v`
Expected: PASS

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add Guardian HAP model constant and imports"
```

---

### Task 2: Add `load_guardian_model` function

**Files:**
- Modify: `streamlit_app.py` (after `load_and_preprocess_audio`, before `transcribe_audio`)
- Test: `tests/test_streamlit_app.py`

**Step 1: Write the failing test**

Add `load_guardian_model` to the test file imports. Add test class after `TestLoadModel`:

```python
class TestLoadGuardianModel:
    @patch("streamlit_app.AutoModelForSequenceClassification")
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.st")
    def test_loads_model_and_tokenizer(
        self,
        _mock_st: MagicMock,
        mock_tokenizer_cls: MagicMock,
        mock_model_cls: MagicMock,
    ) -> None:
        load_guardian_model.__wrapped__("test-model")  # type: ignore[attr-defined]
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("test-model")
        mock_model_cls.from_pretrained.assert_called_once_with("test-model")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestLoadGuardianModel -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Insert between `load_and_preprocess_audio` and `transcribe_audio` in `streamlit_app.py`:

```python
@st.cache_resource(show_spinner=False)
def load_guardian_model(
    model_id: str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return model, tokenizer
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streamlit_app.py::TestLoadGuardianModel -v`
Expected: PASS

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add load_guardian_model with cache"
```

---

### Task 3: Add `check_safety` function

**Files:**
- Modify: `streamlit_app.py` (after `load_guardian_model`, before `transcribe_audio`)
- Test: `tests/test_streamlit_app.py`

**Step 1: Write the failing tests**

Add `check_safety` to the test file imports. Add test class after `TestLoadGuardianModel`:

```python
class TestCheckSafety:
    def _make_mocks(
        self, logits_values: list[list[float]]
    ) -> tuple[MagicMock, MagicMock]:
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        model = MagicMock()
        model.return_value.logits = torch.tensor(logits_values)
        return model, tokenizer

    def test_safe_content(self) -> None:
        model, tokenizer = self._make_mocks([[5.0, -5.0]])
        is_toxic, score = check_safety.__wrapped__(  # type: ignore[attr-defined]
            "safe text", model, tokenizer
        )
        assert is_toxic is False
        assert score < 0.5

    def test_toxic_content(self) -> None:
        model, tokenizer = self._make_mocks([[-5.0, 5.0]])
        is_toxic, score = check_safety.__wrapped__(  # type: ignore[attr-defined]
            "toxic text", model, tokenizer
        )
        assert is_toxic is True
        assert score > 0.5

    def test_returns_rounded_score(self) -> None:
        model, tokenizer = self._make_mocks([[0.0, 0.0]])
        _, score = check_safety.__wrapped__(  # type: ignore[attr-defined]
            "text", model, tokenizer
        )
        assert score == 0.5
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestCheckSafety -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Insert after `load_guardian_model` in `streamlit_app.py`:

```python
@torch.inference_mode()
def check_safety(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
) -> tuple[bool, float]:
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    logits = model(**inputs).logits
    probability = torch.softmax(logits, dim=1)[0, 1].item()
    is_toxic = probability > 0.5
    return is_toxic, round(probability, 4)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestCheckSafety -v`
Expected: PASS

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add check_safety toxicity classifier"
```

---

### Task 4: Update `run_pipeline` to include safety checks

**Files:**
- Modify: `streamlit_app.py:111-130`
- Test: `tests/test_streamlit_app.py:238-307`

**Step 1: Update the tests**

Update `TestRunPipeline._make_mocks` to return guardian mocks:

```python
def _make_mocks(self) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
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
    return model, processor, guardian_model, guardian_tokenizer
```

Update ALL five test methods to unpack 4 mocks and pass all 4 to `run_pipeline.__wrapped__`. Example for `test_returns_dict_keyed_by_task`:

```python
def test_returns_dict_keyed_by_task(self) -> None:
    model, processor, guardian_model, guardian_tokenizer = self._make_mocks()
    wav = torch.zeros(1, 16000)
    tasks = ["Transcribe", "French"]

    results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
        wav, tasks, model, processor, "cpu", guardian_model, guardian_tokenizer
    )

    assert set(results.keys()) == {"Transcribe", "French"}
```

Add new test for safety fields:

```python
def test_each_result_has_safety_fields(self) -> None:
    model, processor, guardian_model, guardian_tokenizer = self._make_mocks()
    wav = torch.zeros(1, 16000)

    results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
        wav, ["Transcribe"], model, processor, "cpu", guardian_model, guardian_tokenizer
    )

    result = results["Transcribe"]
    assert "is_toxic" in result
    assert "toxicity_score" in result
    assert result["is_toxic"] is False
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestRunPipeline -v`
Expected: FAIL (wrong number of arguments)

**Step 3: Update `run_pipeline` implementation**

```python
@torch.inference_mode()
def run_pipeline(
    wav: torch.Tensor,
    tasks: list[str],
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    device: str,
    guardian_model: AutoModelForSequenceClassification,
    guardian_tokenizer: AutoTokenizer,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for task in tasks:
        prompt = PROMPT_CHOICES[task]
        transcript, eval_duration = transcribe_audio.__wrapped__(
            wav, prompt, model, processor, device
        )
        is_toxic, toxicity_score = check_safety.__wrapped__(
            transcript, guardian_model, guardian_tokenizer
        )
        results[task] = {
            "transcript": transcript,
            "num_words": len(transcript.split()),
            "eval_duration": eval_duration,
            "is_toxic": is_toxic,
            "toxicity_score": toxicity_score,
        }
    return results
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestRunPipeline -v`
Expected: PASS

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: integrate safety checks into run_pipeline"
```

---

### Task 5: Update `main()` UI with Guardian integration

**Files:**
- Modify: `streamlit_app.py:133-287` (main function)

**Step 1: Load guardian model at startup**

After the speech model spinner (line 145), add:

```python
    with st.spinner("Loading safety model..."):
        guardian_model, guardian_tokenizer = load_guardian_model(GUARDIAN_MODEL_ID)
```

**Step 2: Update pipeline execution loop**

Replace the pipeline loop (lines 192-206) to include safety checks with interleaved progress:

```python
            pipeline_results: dict[str, dict[str, object]] = {}
            total_steps = len(tasks) * 2
            for i, task in enumerate(tasks):
                progress.progress(
                    (i * 2) / total_steps,
                    text=f"Processing: {task}...",
                )
                prompt = PROMPT_CHOICES[task]
                transcript, eval_duration = transcribe_audio(
                    wav, prompt, model, processor, device
                )
                progress.progress(
                    (i * 2 + 1) / total_steps,
                    text=f"Safety check: {task}...",
                )
                is_toxic, toxicity_score = check_safety(
                    transcript, guardian_model, guardian_tokenizer
                )
                pipeline_results[task] = {
                    "transcript": transcript,
                    "num_words": len(transcript.split()),
                    "eval_duration": eval_duration,
                    "is_toxic": is_toxic,
                    "toxicity_score": toxicity_score,
                }
```

**Step 3: Add safety banner to result cards**

After the metrics row (line 240), before `dl_cols`, add:

```python
                        if result["is_toxic"]:
                            st.warning(
                                f"Toxic content detected (score: {result['toxicity_score']:.1%})"
                            )
                        else:
                            st.success(
                                f"Content is safe (score: {result['toxicity_score']:.1%})"
                            )
```

**Step 4: Update footer**

Replace the footer caption (lines 280-283):

```python
    st.caption(
        f"Model: {MODEL_ID.split('/')[-1]} | "
        f"Safety: {GUARDIAN_MODEL_ID.split('/')[-1]} | "
        f"Device: {device.upper()} | "
        f"[Model Card](https://huggingface.co/{MODEL_ID}) | "
        f"[Safety Model](https://huggingface.co/{GUARDIAN_MODEL_ID})"
    )
```

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 6: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: clean

**Step 7: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add Guardian HAP safety UI to pipeline"
```
