# Guardian HAP Integration Design

## Goal

Add toxicity detection to the speech pipeline using IBM Granite Guardian HAP 38m.

## Approach

- Flag-only: always show transcripts, display warning banner when toxic content detected
- Guardian model (38M params) runs on CPU, loaded at startup with `@st.cache_resource`
- Each transcript/translation output is scored for toxicity after generation

## Data Flow

Audio → Speech Model → transcript → Guardian Model → {is_toxic, toxicity_score}

## New Fields Per Result

- `is_toxic` (bool): True if toxicity probability > 0.5
- `toxicity_score` (float): probability 0-1, rounded to 4 decimal places

## UI Changes

- Startup spinner for safety model loading
- Progress bar includes safety check steps
- `st.success`/`st.warning` banner per result card with score
- Footer shows Guardian model name + link
- JSON downloads include safety fields automatically

## Functions

- `load_guardian_model(model_id)` — cached loader returning (model, tokenizer)
- `check_safety(text, model, tokenizer)` — returns (is_toxic, toxicity_score)
- `run_pipeline` updated with guardian_model/tokenizer params

## Model

- ID: `ibm-granite/granite-guardian-hap-38m`
- Architecture: compressed RoBERTa (4 layers, 38M params)
- Uses: `AutoModelForSequenceClassification`, `AutoTokenizer`
