# Pipeline UI Design

## Context

The Granite Speech app currently processes one audio file with one task at a time. Enterprise users need to process the same audio through multiple tasks simultaneously (e.g., transcribe to English and translate to French and Japanese in one go).

## Design

### Sidebar Removed

Remove `st.sidebar` entirely. Move model name, device, and model card link to a compact caption at the bottom of the main page.

### Task Selection: Presets + Custom

**Presets row** using `st.pills` (single-select):
- "All Tasks" — transcribe + all 7 translations
- "European Languages" — transcribe + French, German, Spanish, Portuguese, Italian
- "Asian Languages" — transcribe + Japanese, Mandarin Chinese
- "Transcribe Only"

**Custom selection** using `st.multiselect` below presets, showing individual tasks. Selecting a preset pre-populates the multiselect. Manual edits to the multiselect deselect the preset pill.

### Audio Input

Upload/Record tabs unchanged.

### Results: Side-by-Side Grid

- 2-3 column grid of bordered `st.container` cells
- Each cell: task name header, transcript in `st.code`, compact metrics row (audio duration, words, processing time), individual download buttons (text + JSON)
- Progress bar updating as each task completes

### Combined Download

- "Download All (JSON)" button below the results grid
- Single JSON file with all results keyed by task name, plus shared metadata (model, audio_duration)

### Footer

Compact caption with model name, device, and model card link.
