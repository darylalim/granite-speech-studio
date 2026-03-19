# Max Tokens & English Translation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase `max_new_tokens` from 200 to 512 to prevent truncation on long segments, and add English as a translation target.

**Architecture:** Two independent changes to the single-file app. `max_new_tokens` is a one-line change in `transcribe_audio`. English translation is a new entry in `PROMPT_CHOICES` that flows through existing preset and pipeline logic without touching `ENGLISH_TASKS`.

**Tech Stack:** Python, Streamlit, PyTorch, Transformers, pytest

**Spec:** `docs/superpowers/specs/2026-03-19-max-tokens-and-english-translation-design.md`

---

## File Structure

| Action | File | What changes |
|--------|------|--------------|
| Modify | `streamlit_app.py:194` | `max_new_tokens=200` → `max_new_tokens=512` |
| Modify | `streamlit_app.py:37-38` | Add `"English"` entry to `PROMPT_CHOICES` |
| Modify | `tests/test_streamlit_app.py` | Update and add tests (see tasks below) |
| Modify | `CLAUDE.md:69-70` | Add English to Languages section |

---

## Chunk 1: Implementation

### Task 1: Add `max_new_tokens` test

**Files:**
- Modify: `tests/test_streamlit_app.py` (class `TestTranscribeAudio`, after line 471)

- [ ] **Step 1: Write the failing test**

Add this test to the `TestTranscribeAudio` class:

```python
def test_max_new_tokens(self) -> None:
    model, processor, tokenizer = self._make_mocks()
    wav = torch.zeros(1, 16000)

    transcribe_audio.__wrapped__(  # type: ignore[attr-defined]
        wav, "transcribe", model, processor, "cpu"
    )

    call_kwargs = model.generate.call_args[1]
    assert call_kwargs["max_new_tokens"] == 512
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestTranscribeAudio::test_max_new_tokens -v`
Expected: FAIL — `assert 200 == 512`

- [ ] **Step 3: Update `max_new_tokens` in `transcribe_audio`**

In `streamlit_app.py:194`, change:

```python
outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, num_beams=1)
```

to:

```python
outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streamlit_app.py::TestTranscribeAudio::test_max_new_tokens -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: increase max_new_tokens from 200 to 512"
```

---

### Task 2: Add English translation tests

**Files:**
- Modify: `tests/test_streamlit_app.py` (class `TestPromptChoices`, import block at line 8)
- Import: add `ENGLISH_TASKS` to the import block

- [ ] **Step 1: Add `ENGLISH_TASKS` to test imports**

In `tests/test_streamlit_app.py:8-30`, add `ENGLISH_TASKS` to the import list:

```python
from streamlit_app import (
    ENGLISH_TASKS,
    GUARDIAN_MODEL_ID,
    ...
)
```

- [ ] **Step 2: Write the failing tests**

Update `test_includes_translations` in `TestPromptChoices` (line 71-82) — add `"English"` to the tuple:

```python
def test_includes_translations(self) -> None:
    for lang in (
        "French",
        "German",
        "Spanish",
        "Portuguese",
        "Italian",
        "Japanese",
        "Mandarin Chinese",
        "English",
    ):
        assert lang in PROMPT_CHOICES
        assert lang in PROMPT_CHOICES[lang]
```

Update `test_total_count` in `TestPromptChoices` (line 89-90):

```python
def test_total_count(self) -> None:
    assert len(PROMPT_CHOICES) == 9
```

Add a new test to `TestPromptChoices` after `test_total_count`:

```python
def test_english_translation_not_in_english_tasks(self) -> None:
    assert "English" in PROMPT_CHOICES
    assert "English" not in ENGLISH_TASKS
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestPromptChoices -v`
Expected: FAIL — `"English" not in PROMPT_CHOICES` and `assert 8 == 9`

- [ ] **Step 4: Add English to `PROMPT_CHOICES`**

In `streamlit_app.py:29-38`, add `"English"` after `"Mandarin Chinese"`:

```python
PROMPT_CHOICES = {
    "Transcribe": "can you transcribe the speech into a written format?",
    "French": "translate the speech to French",
    "German": "translate the speech to German",
    "Spanish": "translate the speech to Spanish",
    "Portuguese": "translate the speech to Portuguese",
    "Italian": "translate the speech to Italian",
    "Japanese": "translate the speech to Japanese",
    "Mandarin Chinese": "translate the speech to Mandarin Chinese",
    "English": "translate the speech to English",
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestPromptChoices -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add English as a translation target"
```

---

### Task 3: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md:69-70`

- [ ] **Step 1: Update Languages section**

Change:

```markdown
- English (transcription)
- French, German, Spanish, Portuguese, Italian, Japanese, Mandarin Chinese (translation)
```

to:

```markdown
- English (transcription)
- French, German, Spanish, Portuguese, Italian, Japanese, Mandarin Chinese, English (translation)
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add English translation target to CLAUDE.md"
```

---

### Task 4: Lint and format check

- [ ] **Step 1: Run linter and formatter**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No issues

- [ ] **Step 2: Run type checker**

Run: `uv run ty check`
Expected: No errors related to these changes

- [ ] **Step 3: Run full test suite one final time**

Run: `uv run pytest -v`
Expected: All tests pass
