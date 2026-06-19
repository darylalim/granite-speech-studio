import io
import re
import warnings
from collections.abc import Callable
from datetime import datetime
from typing import Any, NotRequired, TypedDict
from pathlib import Path

import streamlit as st
import torch
import torchaudio
from mlx_audio.stt.utils import load_model as _load_stt_model
from silero_vad import get_speech_timestamps, load_silero_vad
from streamlit.runtime.uploaded_file_manager import UploadedFile
from torchcodec.decoders import AudioDecoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings(
    "ignore", message="An output with one or more elements was resized"
)
MODEL_ID = "mlx-community/granite-4.0-1b-speech-8bit"
GUARDIAN_MODEL_ID = "ibm-granite/granite-guardian-hap-125m"
SOURCE_LANGUAGES = [
    "English",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Japanese",
]
EN_TARGETS = [
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Italian",
    "Japanese",
    "Mandarin Chinese",
]
TRANSCRIBE_PROMPT = "can you transcribe the speech into a written format?"
SUPPORTED_FORMATS = [
    "wav",
    "flac",
    "m4a",
    "mp3",
    "ogg",
    "aac",
    "mp4",
    "mov",
    "webm",
    "mkv",
]
VIDEO_FORMATS = {"mp4", "mov", "webm", "mkv"}
SAMPLE_RATE = 16000
MAX_VAD_OFF_DURATION_S = 300
TOXICITY_THRESHOLD = 0.5
TOXICITY_SCORE_PRECISION = 4


class PipelineResult(TypedDict):
    transcript: str
    is_toxic: NotRequired[bool]
    toxicity_score: NotRequired[float]


def is_video(filename: str) -> bool:
    return Path(filename).suffix.lower().lstrip(".") in VIDEO_FORMATS


def build_tasks(source: str) -> dict[str, str]:
    tasks: dict[str, str] = {"Transcribe": TRANSCRIBE_PROMPT}
    targets = EN_TARGETS if source == "English" else ["English"]
    for target in targets:
        tasks[target] = f"translate the speech to {target}"
    return tasks


def apply_keywords(prompt: str, keywords: list[str]) -> str:
    if not keywords:
        return prompt
    return f"{prompt} Keywords: {', '.join(keywords)}"


def produces_english(source: str, task: str) -> bool:
    if task == "Transcribe":
        return source == "English"
    return task == "English"


def compute_safety_tasks(
    selected_tasks: list[str], source: str, use_toxicity_check: bool
) -> set[str]:
    if not use_toxicity_check:
        return set()
    return {name for name in selected_tasks if produces_english(source, name)}


def result_title(source: str, task: str) -> str:
    if task == "Transcribe":
        return f"Transcribe ({source})"
    return task


def result_slug(source: str, task: str) -> str:
    if task == "Transcribe":
        return f"transcribe_{source.lower().replace(' ', '_')}"
    return task.lower().replace(" ", "_")


def format_timestamp(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def _detect_cot_target(tasks: dict[str, str]) -> str | None:
    if "Transcribe" not in tasks:
        return None
    other_tasks = [t for t in tasks if t != "Transcribe"]
    if len(other_tasks) != 1:
        return None
    return other_tasks[0]


def _cot_prompt(target: str) -> str:
    return f"Can you transcribe the speech, and then translate it to {target}?"


def _parse_cot_output(text: str) -> tuple[str, str]:
    transcription_match = re.search(
        r"\[Transcription\](.*?)(?=\[Translation\]|$)", text, re.DOTALL
    )
    translation_match = re.search(r"\[Translation\](.*)$", text, re.DOTALL)
    transcription = transcription_match.group(1).strip() if transcription_match else ""
    translation = translation_match.group(1).strip() if translation_match else ""
    return transcription, translation


def silero_vad(
    wav: torch.Tensor, model: torch.nn.Module, sample_rate: int = SAMPLE_RATE
) -> list[tuple[float, float]]:
    speech_timestamps = get_speech_timestamps(
        wav.squeeze(), model, sampling_rate=sample_rate
    )
    return [
        (ts["start"] / sample_rate, ts["end"] / sample_rate) for ts in speech_timestamps
    ]


def get_speech_segments(
    wav: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: int = SAMPLE_RATE,
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


@st.cache_resource(show_spinner=False)
def load_model(model_id: str) -> Any:
    return _load_stt_model(model_id)


def load_and_preprocess_audio(audio_file: UploadedFile) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(io.BytesIO(audio_file.getvalue()))
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}") from e

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def audio_duration_seconds(audio_file: UploadedFile) -> float | None:
    try:
        decoder = AudioDecoder(audio_file.getvalue())
        duration = decoder.metadata.duration_seconds
    except (RuntimeError, ValueError, OSError):
        return None
    if duration is None or duration <= 0:
        return None
    return float(duration)


@st.cache_resource(show_spinner=False)
def load_guardian_model(model_id: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_vad_model() -> torch.nn.Module:
    return load_silero_vad()


def check_safety(
    text: str,
    model: Any,
    tokenizer: Any,
) -> tuple[bool, float]:
    # Guardian (RoBERTa) caps at 512 tokens. For inputs longer than that,
    # chunk into 510-token windows (reserving 2 slots for CLS/SEP) and take
    # the max — otherwise truncation would silently drop late content.
    max_content_tokens = 510
    encoding = tokenizer(
        text, return_tensors="pt", truncation=False, add_special_tokens=False
    )
    input_ids = encoding["input_ids"][0]
    if len(input_ids) <= max_content_tokens:
        chunks = [text]
    else:
        chunks = [
            tokenizer.decode(
                input_ids[i : i + max_content_tokens], skip_special_tokens=True
            )
            for i in range(0, len(input_ids), max_content_tokens)
        ]

    max_probability = 0.0
    for chunk in chunks:
        inputs = tokenizer([chunk], padding=True, truncation=True, return_tensors="pt")
        logits = model(**inputs).logits
        probability = torch.softmax(logits, dim=1)[0, 1].item()
        if probability > max_probability:
            max_probability = probability
    return (
        max_probability > TOXICITY_THRESHOLD,
        round(max_probability, TOXICITY_SCORE_PRECISION),
    )


def transcribe_audio(
    wav: torch.Tensor,
    prompt: str,
    model: Any,
) -> str:
    audio_np = wav.squeeze().numpy()
    output = model.generate(audio=audio_np, prompt=prompt, max_tokens=512)
    return output.text


def _aggregate_segment_safety(
    texts: list[str], model: Any, tokenizer: Any
) -> tuple[bool, float]:
    max_probability = 0.0
    for text in texts:
        if not text.strip():
            continue
        _, probability = check_safety(text, model, tokenizer)
        if probability > max_probability:
            max_probability = probability
    return (
        max_probability > TOXICITY_THRESHOLD,
        round(max_probability, TOXICITY_SCORE_PRECISION),
    )


@torch.inference_mode()
def run_pipeline(
    wav: torch.Tensor,
    tasks: dict[str, str],
    safety_tasks: set[str],
    model: Any,
    vad_model: torch.nn.Module | None = None,
    guardian_model: Any | None = None,
    guardian_tokenizer: Any | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    use_segmentation: bool = True,
    keywords: list[str] | None = None,
) -> dict[str, PipelineResult]:
    if use_segmentation:
        assert vad_model is not None, "vad_model required when use_segmentation=True"
        segments = get_speech_segments(wav, vad_model)
    else:
        duration = wav.shape[-1] / SAMPLE_RATE
        segments = [{"start": 0.0, "end": duration}]
    if keywords is None:
        keywords = []
    # CoT relies on Transcribe being iterated before the translation target;
    # build_tasks preserves this via insertion order.
    cot_target = _detect_cot_target(tasks)
    cot_translation_cache: dict[int, str] = {}
    results: dict[str, PipelineResult] = {}
    for i, (task, prompt) in enumerate(tasks.items()):
        if on_progress:
            on_progress(i, len(tasks), task)
        use_cot = cot_target is not None and task == "Transcribe"
        read_cot_cache = cot_target is not None and task == cot_target
        if use_cot:
            assert cot_target is not None
            base_prompt = _cot_prompt(cot_target)
        else:
            base_prompt = prompt
        actual_prompt = apply_keywords(base_prompt, keywords)
        raw_texts: list[str] = []
        lines: list[str] = []
        for seg_idx, seg in enumerate(segments):
            start_sample = int(seg["start"] * SAMPLE_RATE)
            end_sample = int(seg["end"] * SAMPLE_RATE)
            if read_cot_cache and seg_idx in cot_translation_cache:
                text = cot_translation_cache[seg_idx]
            else:
                raw = transcribe_audio(
                    wav[:, start_sample:end_sample], actual_prompt, model
                )
                if use_cot:
                    transcription, translation = _parse_cot_output(raw)
                    if not transcription:
                        # CoT didn't yield a usable transcription (no tags, or
                        # only [Translation] was emitted). Re-run with the
                        # direct ASR prompt so untagged/translation output
                        # isn't mistakenly displayed as the transcription.
                        text = transcribe_audio(
                            wav[:, start_sample:end_sample],
                            apply_keywords(prompt, keywords),
                            model,
                        )
                    else:
                        text = transcription
                    # Only cache when the model actually produced a
                    # translation; an empty value would later be served from
                    # the cache instead of triggering a direct AST call.
                    if translation:
                        cot_translation_cache[seg_idx] = translation
                else:
                    text = raw
            raw_texts.append(text)
            ts_start = format_timestamp(seg["start"])
            ts_end = format_timestamp(seg["end"])
            lines.append(f"[{ts_start} - {ts_end}] {text}")
        result: PipelineResult = {"transcript": "\n".join(lines)}
        if (
            task in safety_tasks
            and guardian_model is not None
            and guardian_tokenizer is not None
        ):
            # Per-segment so long transcripts don't get silently truncated by
            # the guardian's 512-token cap; report the worst segment's score.
            is_toxic, score = _aggregate_segment_safety(
                raw_texts, guardian_model, guardian_tokenizer
            )
            result["is_toxic"] = is_toxic
            result["toxicity_score"] = score
        results[task] = result
    return results


def _row_sizes(n: int) -> list[int]:
    """Split n result cards into rows of at most 3, as evenly as possible.

    e.g. 4 → [2, 2] instead of [3, 1]; 7 → [3, 2, 2] instead of [3, 3, 1].
    """
    if n <= 0:
        return []
    rows = -(-n // 3)
    base, extra = divmod(n, rows)
    return [base + 1] * extra + [base] * (rows - extra)


def _labeled_toggle(label: str, help: str, key: str, value: bool = True) -> bool:
    label_col, toggle_col = st.columns([15, 1], vertical_alignment="center")
    with label_col:
        st.markdown(f"**{label}**", help=help)
    with toggle_col:
        return st.toggle(label, value=value, label_visibility="collapsed", key=key)


def _render_result_card(
    source: str,
    task: str,
    result: PipelineResult,
    stem: str,
) -> None:
    transcript = result["transcript"]
    title = result_title(source, task)
    is_transcription = task == "Transcribe"
    slug = result_slug(source, task)
    with st.container(border=True, height="stretch"):
        st.subheader(title)
        st.text(transcript)
        if "is_toxic" in result:
            score = f"score: {result['toxicity_score']:.1%}"
            if result["is_toxic"]:
                st.warning(
                    f"Toxic content detected ({score})", icon=":material/warning:"
                )
            else:
                st.success(f"Content is safe ({score})", icon=":material/check_circle:")
        download_help = (
            "Download transcription" if is_transcription else "Download translation"
        )
        st.download_button(
            "",
            transcript,
            f"{stem}_{slug}.txt",
            "text/plain",
            key=f"dl_txt_{source}_{task}",
            icon=":material/download:",
            help=download_help,
        )


def main() -> None:
    st.set_page_config(
        page_title="Granite Speech Studio",
        page_icon=":material/graphic_eq:",
        layout="centered",
    )

    st.title("Granite Speech Studio", text_alignment="center")
    st.markdown(
        "Transcribe and translate audio and video files with the "
        "[IBM Granite 4.0 1B Speech model]"
        "(https://huggingface.co/ibm-granite/granite-4.0-1b-speech).",
        text_alignment="center",
    )

    upload_tab, record_tab = st.tabs(["Upload", "Record"])
    with upload_tab:
        uploaded = st.file_uploader(
            "Upload audio file",
            type=SUPPORTED_FORMATS,
            help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
            label_visibility="collapsed",
        )
    with record_tab:
        recorded = st.audio_input("Record audio", label_visibility="collapsed")

    audio_file = uploaded or recorded

    if audio_file:
        if is_video(audio_file.name):
            st.video(audio_file)
        else:
            st.audio(audio_file)
        st.caption(audio_file.name if uploaded else "Recorded audio")

    source_value = st.segmented_control(
        "Source language",
        options=SOURCE_LANGUAGES,
        default="English",
        label_visibility="collapsed",
    )
    source: str = source_value if isinstance(source_value, str) else "English"

    available_tasks = build_tasks(source)
    selected_value = st.pills(
        "Tasks",
        options=list(available_tasks.keys()),
        selection_mode="multi",
        default=["Transcribe"],
        label_visibility="collapsed",
        key=f"tasks_{source}",
    )
    selected_tasks: list[str] = [t for t in selected_value if isinstance(t, str)]

    use_segmentation = _labeled_toggle(
        "VAD segmentation",
        help=(
            "Splits audio into speech segments with timestamps using "
            "Silero VAD. Disable for short utterances or to process "
            "the whole audio in one pass."
        ),
        key="use_segmentation",
    )

    vad_off_too_long = False
    if audio_file is not None and not use_segmentation:
        # Single-slot cache: getvalue() copies the full byte buffer each rerun,
        # so memoize the duration and recompute only when the file changes. One
        # slot can't grow, so no eviction is needed.
        cache_id = (audio_file.name, audio_file.size)
        cached = st.session_state.get("_duration")
        if cached is None or cached[0] != cache_id:
            cached = (cache_id, audio_duration_seconds(audio_file))
            st.session_state["_duration"] = cached
        duration = cached[1]
        if duration is not None and duration > MAX_VAD_OFF_DURATION_S:
            vad_off_too_long = True
            st.warning(
                f"Enable VAD segmentation: audio is longer than "
                f"{MAX_VAD_OFF_DURATION_S // 60} minutes, which exceeds the "
                "model's per-call audio limit.",
                icon=":material/warning:",
            )

    st.markdown(
        "**Keywords**",
        help=(
            "Up to 15 keywords to be boosted during transcription. "
            "Boosted terms are more likely to appear in the output."
        ),
    )
    keywords = st.multiselect(
        "Keywords",
        options=[],
        accept_new_options=True,
        max_selections=15,
        placeholder="Add keywords...",
        label_visibility="collapsed",
        key="keywords",
    )

    use_toxicity_check = _labeled_toggle(
        "Toxicity check",
        help=(
            "Checks English transcripts and translations for toxic content "
            "via Granite Guardian. Non-English output is skipped regardless "
            "of this setting."
        ),
        key="use_toxicity_check",
    )

    input_key = (
        (
            audio_file.name,
            audio_file.size,
            source,
            tuple(selected_tasks),
            use_segmentation,
            tuple(sorted(keywords)),
            use_toxicity_check,
        )
        if audio_file
        else None
    )
    if input_key != st.session_state.get("_last_input_key"):
        for key in ("results", "result_stem", "result_source"):
            st.session_state.pop(key, None)
        st.session_state["_last_input_key"] = input_key

    can_run = (
        audio_file is not None and len(selected_tasks) > 0 and not vad_off_too_long
    )

    with st.container(horizontal_alignment="right"):
        run_clicked = st.button(
            "Transcribe",
            type="primary",
            disabled=not can_run,
        )

    if run_clicked and can_run:
        assert audio_file is not None
        progress = st.progress(0, text="Starting pipeline...")
        try:
            with st.spinner("Loading speech model..."):
                model = load_model(MODEL_ID)
            wav = load_and_preprocess_audio(audio_file)

            if use_segmentation:
                with st.spinner("Loading VAD model..."):
                    vad_model = load_vad_model()
            else:
                vad_model = None

            def update_progress(i: int, total: int, task: str) -> None:
                progress.progress(i / total, text=f"Processing: {task}...")

            tasks_to_run = {name: available_tasks[name] for name in selected_tasks}
            safety_tasks = compute_safety_tasks(
                selected_tasks, source, use_toxicity_check
            )

            if safety_tasks:
                with st.spinner("Loading safety model..."):
                    guardian_model, guardian_tokenizer = load_guardian_model(
                        GUARDIAN_MODEL_ID
                    )
            else:
                guardian_model, guardian_tokenizer = None, None

            pipeline_results = run_pipeline(
                wav,
                tasks_to_run,
                safety_tasks,
                model,
                vad_model,
                guardian_model,
                guardian_tokenizer,
                on_progress=update_progress,
                use_segmentation=use_segmentation,
                keywords=keywords,
            )
            progress.empty()
            st.session_state.results = pipeline_results
            if uploaded:
                stem = Path(audio_file.name).stem
            else:
                stem = datetime.now().strftime("recording_%Y%m%d_%H%M%S")
            st.session_state.result_stem = stem
            st.session_state.result_source = source
        except RuntimeError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.exception(e)
            return
        st.toast("Pipeline complete!")

    if "results" in st.session_state:
        results = st.session_state.results
        stem = st.session_state.result_stem
        source_used = st.session_state.result_source
        task_names = list(results.keys())

        idx = 0
        for row_size in _row_sizes(len(task_names)):
            cols = st.columns(row_size)
            for col, task_name in zip(
                cols, task_names[idx : idx + row_size], strict=True
            ):
                with col:
                    _render_result_card(
                        source_used, task_name, results[task_name], stem
                    )
            idx += row_size


if __name__ == "__main__":
    main()
