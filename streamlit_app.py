import io
import warnings
from collections.abc import Callable
from datetime import datetime
from typing import Any
from pathlib import Path

import streamlit as st
import torch
import torchaudio
from mlx_audio.stt.utils import load_model as _load_stt_model
from silero_vad import get_speech_timestamps, load_silero_vad
from streamlit.runtime.uploaded_file_manager import UploadedFile
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings(
    "ignore", message="An output with one or more elements was resized"
)
MODEL_ID = "mlx-community/granite-4.0-1b-speech-8bit"
GUARDIAN_MODEL_ID = "ibm-granite/granite-guardian-hap-38m"
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


def is_video(filename: str) -> bool:
    return Path(filename).suffix.lower().lstrip(".") in VIDEO_FORMATS


def build_tasks(source: str) -> dict[str, str]:
    tasks: dict[str, str] = {"Transcribe": TRANSCRIBE_PROMPT}
    targets = EN_TARGETS if source == "English" else ["English"]
    for target in targets:
        tasks[target] = f"translate the speech to {target}"
    return tasks


def produces_english(source: str, task: str) -> bool:
    if task == "Transcribe":
        return source == "English"
    return task == "English"


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


@st.cache_resource(show_spinner=False)
def load_guardian_model(
    model_id: str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_vad_model() -> torch.nn.Module:
    return load_silero_vad()


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


def transcribe_audio(
    wav: torch.Tensor,
    prompt: str,
    model: Any,
) -> str:
    audio_np = wav.squeeze().numpy()
    output = model.generate(audio=audio_np, prompt=prompt, max_tokens=512)
    return output.text


@torch.inference_mode()
def run_pipeline(
    wav: torch.Tensor,
    tasks: dict[str, str],
    safety_tasks: set[str],
    model: Any,
    vad_model: torch.nn.Module | None = None,
    guardian_model: AutoModelForSequenceClassification | None = None,
    guardian_tokenizer: AutoTokenizer | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    use_segmentation: bool = True,
) -> dict[str, dict[str, object]]:
    if use_segmentation:
        assert vad_model is not None, "vad_model required when use_segmentation=True"
        segments = get_speech_segments(wav, vad_model)
    else:
        duration = wav.shape[-1] / SAMPLE_RATE
        segments = [{"start": 0.0, "end": duration}]
    results: dict[str, dict[str, object]] = {}
    for i, (task, prompt) in enumerate(tasks.items()):
        if on_progress:
            on_progress(i, len(tasks), task)
        raw_texts: list[str] = []
        lines: list[str] = []
        for seg in segments:
            start_sample = int(seg["start"] * SAMPLE_RATE)
            end_sample = int(seg["end"] * SAMPLE_RATE)
            text = transcribe_audio(wav[:, start_sample:end_sample], prompt, model)
            raw_texts.append(text)
            ts_start = format_timestamp(seg["start"])
            ts_end = format_timestamp(seg["end"])
            lines.append(f"[{ts_start} - {ts_end}] {text}")
        result: dict[str, object] = {"transcript": "\n".join(lines)}
        if (
            task in safety_tasks
            and guardian_model is not None
            and guardian_tokenizer is not None
        ):
            is_toxic, toxicity_score = check_safety(
                " ".join(raw_texts), guardian_model, guardian_tokenizer
            )
            result["is_toxic"] = is_toxic
            result["toxicity_score"] = toxicity_score
        results[task] = result
    return results


def _render_result_card(
    source: str,
    task: str,
    result: dict[str, object],
    stem: str,
) -> None:
    title = result_title(source, task)
    is_transcription = task == "Transcribe"
    slug = result_slug(source, task)
    with st.container(border=True):
        st.subheader(title)
        st.text(result["transcript"])
        if "is_toxic" in result:
            score = f"score: {result['toxicity_score']:.1%}"
            if result["is_toxic"]:
                st.warning(f"Toxic content detected ({score})")
            else:
                st.success(f"Content is safe ({score})")
        download_help = (
            "Download transcription" if is_transcription else "Download translation"
        )
        st.download_button(
            "",
            result["transcript"],
            f"{stem}_{slug}.txt",
            "text/plain",
            key=f"dl_txt_{source}_{task}",
            icon=":material/download:",
            help=download_help,
        )


def main() -> None:
    st.set_page_config(page_title="Granite Speech Pipeline")

    st.title("Granite Speech Pipeline")
    st.markdown(
        "Transcribe and translate audio and video files with the "
        "[IBM Granite 4.0 1B Speech model]"
        "(https://huggingface.co/ibm-granite/granite-4.0-1b-speech)."
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
    selected_tasks: list[str] = (
        [t for t in selected_value if isinstance(t, str)]
        if isinstance(selected_value, list)
        else []
    )

    label_col, toggle_col = st.columns([15, 1], vertical_alignment="center")
    with label_col:
        st.markdown(
            "VAD segmentation",
            help=(
                "Splits audio into speech segments with timestamps using "
                "Silero VAD. Disable for short utterances or to process "
                "the whole audio in one pass."
            ),
        )
    with toggle_col:
        use_segmentation = st.toggle(
            "VAD segmentation",
            value=True,
            label_visibility="collapsed",
            key="use_segmentation",
        )

    input_key = (
        (
            audio_file.name,
            audio_file.size,
            source,
            tuple(selected_tasks),
            use_segmentation,
        )
        if audio_file
        else None
    )
    if input_key != st.session_state.get("_last_input_key"):
        for key in ("results", "result_stem", "result_source"):
            st.session_state.pop(key, None)
        st.session_state["_last_input_key"] = input_key

    can_run = audio_file is not None and len(selected_tasks) > 0

    _, btn_col = st.columns([4, 1])
    with btn_col:
        run_clicked = st.button(
            "Transcribe",
            type="primary",
            disabled=not can_run,
            width="stretch",
        )

    if run_clicked and can_run:
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
            safety_tasks = {
                name for name in selected_tasks if produces_english(source, name)
            }

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

        num_cols = min(len(task_names), 3)
        for row_start in range(0, len(task_names), num_cols):
            row_tasks = task_names[row_start : row_start + num_cols]
            cols = st.columns(num_cols)
            for col, task_name in zip(cols, row_tasks):
                with col:
                    _render_result_card(
                        source_used, task_name, results[task_name], stem
                    )


if __name__ == "__main__":
    main()
