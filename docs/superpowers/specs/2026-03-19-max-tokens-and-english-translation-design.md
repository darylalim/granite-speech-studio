# Increase max_new_tokens and Add English Translation

**Date:** 2026-03-19
**Status:** Approved

## Context

Comparing with the HF Space at `ibm-granite/granite-speech`, two features are worth adopting for this production-facing Streamlit app:

1. The current `max_new_tokens=200` limit can truncate longer speech segments. The HF Space uses 512.
2. The HF Space supports English as a translation target, enabling non-English audio to be translated to English. This project currently only supports translating from English.

## Changes

### 1. Increase `max_new_tokens` from 200 to 512

- In `transcribe_audio`, change `max_new_tokens=200` to `max_new_tokens=512` in the `model.generate()` call.
- No other code changes required.

### 2. Add English translation to `PROMPT_CHOICES`

- Add `"English": "translate the speech to English"` to `PROMPT_CHOICES`, inserted at the end of the dict (after Mandarin Chinese).
- English translation is NOT an English transcription task, so it must NOT be added to `ENGLISH_TASKS`. This means no punctuation post-processing and no toxicity checking will run on it, consistent with other translation tasks. The `ENGLISH_TASKS.intersection(tasks)` guard in `main()` will correctly skip loading punctuation/safety models for English-translation-only runs.
- The "All Tasks" preset automatically includes all `PROMPT_CHOICES` keys (via `list(PROMPT_CHOICES.keys())`), so it will pick up English translation without changes.
- "European Languages" and "Asian Languages" presets are unchanged — English translation is language-agnostic and doesn't belong in either.

### 3. Update `CLAUDE.md`

- Add English to the Languages section as a translation target.

### 4. Test updates

- Update `test_total_count` to expect 9 prompt choices (was 8).
- `test_all_tasks_preset` needs no change — it dynamically compares against `PROMPT_CHOICES.keys()`.
- Add "English" to the language tuple in `test_includes_translations`.
- Existing `test_translation_prompt_prefixes` already covers all non-"Transcribe" entries, so it will automatically verify the English translation prefix.
- Add a test in `TestTranscribeAudio` verifying `max_new_tokens=512` is passed to `model.generate`.
- Add a test asserting `"English" not in ENGLISH_TASKS` to guard the critical invariant that English translation skips punctuation and safety.
