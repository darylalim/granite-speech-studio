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
        model_id, device_map=device, torch_dtype=dtype
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
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav, duration


@st.cache_resource(show_spinner=False)
def load_guardian_model(
    model_id: str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return model, tokenizer


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
    guardian_model: AutoModelForSequenceClassification,
    guardian_tokenizer: AutoTokenizer,
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

    with st.spinner("Loading safety model..."):
        guardian_model, guardian_tokenizer = load_guardian_model(GUARDIAN_MODEL_ID)

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

    input_key = (audio_file.name, audio_file.size, tuple(tasks)) if audio_file else None
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

            def update_progress(i: int, total: int, task: str) -> None:
                progress.progress(i / total, text=f"Processing: {task}...")

            pipeline_results = run_pipeline(
                wav,
                tasks,
                model,
                processor,
                device,
                guardian_model,
                guardian_tokenizer,
                on_progress=update_progress,
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
                        if result["is_toxic"]:
                            st.warning(
                                f"Toxic content detected (score: {result['toxicity_score']:.1%})"
                            )
                        else:
                            st.success(
                                f"Content is safe (score: {result['toxicity_score']:.1%})"
                            )
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
        f"Model: {MODEL_ID.split('/')[-1]} | "
        f"Safety: {GUARDIAN_MODEL_ID.split('/')[-1]} | "
        f"Device: {device.upper()} | "
        f"[Model Card](https://huggingface.co/{MODEL_ID}) | "
        f"[Safety Model](https://huggingface.co/{GUARDIAN_MODEL_ID})"
    )


if __name__ == "__main__":
    main()
