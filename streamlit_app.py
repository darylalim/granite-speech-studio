import io
import json
import time
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
import torchaudio
from punctuators.models import PunctCapSegModelONNX
from silero_vad import get_speech_timestamps, load_silero_vad
from streamlit.runtime.uploaded_file_manager import UploadedFile
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
)

warnings.filterwarnings(
    "ignore", message="An output with one or more elements was resized"
)

MODEL_ID = "ibm-granite/granite-4.0-1b-speech"
GUARDIAN_MODEL_ID = "ibm-granite/granite-guardian-hap-38m"
PUNCTUATION_MODEL_ID = "pcs_en"
PROMPT_CHOICES = {
    "Transcribe": "can you transcribe the speech into a written format?",
    "French": "translate the speech to French",
    "German": "translate the speech to German",
    "Spanish": "translate the speech to Spanish",
    "Portuguese": "translate the speech to Portuguese",
    "Italian": "translate the speech to Italian",
    "Japanese": "translate the speech to Japanese",
    "Mandarin Chinese": "translate the speech to Mandarin Chinese",
}
SUPPORTED_FORMATS = ["wav", "mp3", "m4a", "ogg", "flac", "webm", "aac"]
TASK_PRESETS: dict[str, list[str]] = {
    "All Tasks": list(PROMPT_CHOICES.keys()),
    "European Languages": [
        "Transcribe",
        "French",
        "German",
        "Spanish",
        "Portuguese",
        "Italian",
    ],
    "Asian Languages": ["Transcribe", "Japanese", "Mandarin Chinese"],
    "Transcribe Only": ["Transcribe"],
}
ENGLISH_TASKS: set[str] = {"Transcribe"}
SAMPLE_RATE = 16000


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


def get_selected_tasks(preset: str | None, custom: list[str]) -> list[str]:
    if preset is not None:
        return TASK_PRESETS[preset]
    return custom


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner=False)
def load_model(
    model_id: str,
    device: str,
) -> tuple[AutoModelForSpeechSeq2Seq, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, device_map=device, dtype=dtype
    )
    return model, processor


def load_and_preprocess_audio(audio_file: UploadedFile) -> tuple[torch.Tensor, float]:
    try:
        wav, sr = torchaudio.load(io.BytesIO(audio_file.getvalue()))
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}") from e

    duration = wav.shape[1] / sr
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav, duration


@st.cache_resource(show_spinner=False)
def load_guardian_model(
    model_id: str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_punctuation_model(model_id: str) -> PunctCapSegModelONNX:
    return PunctCapSegModelONNX.from_pretrained(model_id)


@st.cache_resource(show_spinner=False)
def load_vad_model() -> torch.nn.Module:
    return load_silero_vad()


def apply_punctuation(text: str, model: PunctCapSegModelONNX) -> str:
    result = model.infer([text])
    output = " ".join(result[0])
    output = output.replace("<unk>", " ").replace("<Unk>", " ")
    return " ".join(output.split()).strip()


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


@torch.inference_mode()
def transcribe_audio(
    wav: torch.Tensor,
    prompt: str,
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    device: str,
) -> tuple[str, float]:
    start = time.perf_counter()
    tokenizer = processor.tokenizer
    chat = [
        {"role": "user", "content": f"<|audio|>{prompt}"},
    ]
    text_prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text_prompt, wav, device=device, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, num_beams=1)
    num_input_tokens = inputs["input_ids"].shape[-1]
    transcript = tokenizer.batch_decode(
        outputs[:, num_input_tokens:],
        add_special_tokens=False,
        skip_special_tokens=True,
    )[0]
    return transcript, round(time.perf_counter() - start, 2)


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
            seg_start_time = time.perf_counter()
            for seg in segments:
                start_sample = int(seg["start"] * SAMPLE_RATE)
                end_sample = int(seg["end"] * SAMPLE_RATE)
                wav_segment = wav[:, start_sample:end_sample]
                seg_transcript, _ = transcribe_audio.__wrapped__(
                    wav_segment, prompt, model, processor, device
                )
                if task in ENGLISH_TASKS and punct_model is not None:
                    seg_transcript = apply_punctuation(seg_transcript, punct_model)
                total_words += len(seg_transcript.split())
                ts_start = format_timestamp(seg["start"])
                ts_end = format_timestamp(seg["end"])
                segment_texts.append(f"[{ts_start} - {ts_end}] {seg_transcript}")
            transcript = "\n".join(segment_texts)
            eval_duration = round(time.perf_counter() - seg_start_time, 2)
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


def _render_result_card(
    task_name: str,
    result: dict[str, object],
    audio_duration: float,
    stem: str,
) -> None:
    with st.container(border=True):
        st.subheader(task_name)
        st.code(result["transcript"], language=None)
        m1, m2, m3 = st.columns(3)
        m1.metric("Duration", f"{audio_duration:.2f}s")
        m2.metric("Words", result["num_words"])
        m3.metric("Time", f"{result['eval_duration']}s")
        if "is_toxic" in result:
            score = f"score: {result['toxicity_score']:.1%}"
            if result["is_toxic"]:
                st.warning(f"Toxic content detected ({score})")
            else:
                st.success(f"Content is safe ({score})")
        slug = task_name.lower().replace(" ", "_")
        d1, d2 = st.columns(2)
        d1.download_button(
            "Text",
            result["transcript"],
            f"{stem}_{slug}.txt",
            "text/plain",
            key=f"dl_txt_{task_name}",
        )
        d2.download_button(
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
            f"{stem}_{slug}.json",
            "application/json",
            key=f"dl_json_{task_name}",
        )


def main() -> None:
    st.set_page_config(
        page_title="Granite Speech Pipeline",
        page_icon="\U0001f399\ufe0f",
        layout="wide",
    )

    device = get_device()

    st.title("\U0001f399\ufe0f Granite Speech Pipeline")

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

    use_segmentation = st.checkbox("VAD segmentation", value=True)

    input_key = (
        (audio_file.name, audio_file.size, tuple(tasks), use_segmentation)
        if audio_file
        else None
    )
    if input_key != st.session_state.get("_last_input_key"):
        for key in ("results", "result_filename", "audio_duration"):
            st.session_state.pop(key, None)
        st.session_state["_last_input_key"] = input_key

    if audio_file:
        st.audio(audio_file)
        st.caption("Recorded audio" if recorded else audio_file.name)

    can_run = audio_file is not None and len(tasks) > 0

    if st.button("Run Pipeline", type="primary", disabled=not can_run) and can_run:
        progress = st.progress(0, text="Starting pipeline...")
        try:
            with st.spinner(f"Loading model on {device.upper()}..."):
                model, processor = load_model(MODEL_ID, device)
            wav, audio_duration = load_and_preprocess_audio(audio_file)

            if use_segmentation:
                with st.spinner("Loading VAD model..."):
                    vad_model = load_vad_model()
            else:
                vad_model = None

            def update_progress(i: int, total: int, task: str) -> None:
                progress.progress(i / total, text=f"Processing: {task}...")

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
        stem = st.session_state.result_filename.replace("_pipeline.json", "")
        task_names = list(results.keys())

        num_cols = min(len(task_names), 3)
        for row_start in range(0, len(task_names), num_cols):
            row_tasks = task_names[row_start : row_start + num_cols]
            cols = st.columns(num_cols)
            for col, task_name in zip(cols, row_tasks):
                with col:
                    _render_result_card(
                        task_name, results[task_name], audio_duration, stem
                    )

        st.download_button(
            "Download All (JSON)",
            json.dumps(
                {
                    "model": MODEL_ID,
                    "audio_duration": audio_duration,
                    "results": results,
                },
                indent=2,
            ),
            st.session_state.result_filename,
            "application/json",
        )

    st.caption(
        f"Model: {MODEL_ID.split('/')[-1]} | "
        f"Punctuation: {PUNCTUATION_MODEL_ID} | "
        f"Safety: {GUARDIAN_MODEL_ID.split('/')[-1]} | "
        f"Device: {device.upper()} | "
        f"[Model Card](https://huggingface.co/{MODEL_ID}) | "
        f"[Safety Model](https://huggingface.co/{GUARDIAN_MODEL_ID})"
    )


if __name__ == "__main__":
    main()
