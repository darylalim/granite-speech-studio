import io
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
import torchaudio
from streamlit.runtime.uploaded_file_manager import UploadedFile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

warnings.filterwarnings(
    "ignore", message="An output with one or more elements was resized"
)

MODEL_ID = "ibm-granite/granite-speech-3.3-2b"
SYSTEM_PROMPT_TEMPLATE = """Knowledge Cutoff Date: April 2024.
Today's Date: {today}.
You are Granite, developed by IBM. You are a helpful AI assistant"""
PROMPT_CHOICES = {
    "Transcribe": "Transcribe the speech to text",
    "French": "Translate the speech to French",
    "German": "Translate the speech to German",
    "Spanish": "Translate the speech to Spanish",
    "Portuguese": "Translate the speech to Portuguese",
}
SUPPORTED_FORMATS = ["wav", "mp3", "m4a", "ogg", "flac", "webm", "aac"]


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
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav, duration


@torch.inference_mode()
def transcribe_audio(
    wav: torch.Tensor,
    prompt: str,
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    device: str,
) -> tuple[str, float]:
    start = time.perf_counter()
    today = datetime.today().strftime("%B %d, %Y")
    tokenizer = processor.tokenizer
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(today=today)},
        {"role": "user", "content": f"<|audio|>{prompt}"},
    ]
    text_prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text_prompt, wav, device=device, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
    transcript = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    return transcript, round(time.perf_counter() - start, 2)


def main() -> None:
    st.set_page_config(page_title="Granite Speech Pipeline", page_icon="🎙️")

    device = get_device()

    with st.sidebar:
        st.caption(f"Model: {MODEL_ID.split('/')[-1]}")
        st.caption(f"Running on {device.upper()}")
        st.link_button("Model Card", f"https://huggingface.co/{MODEL_ID}")

    st.title("🎙️ Granite Speech Pipeline")

    with st.spinner(f"Loading model on {device.upper()}..."):
        model, processor = load_model(MODEL_ID, device)

    task = st.pills("Task", options=list(PROMPT_CHOICES.keys()), default="Transcribe")

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

    input_key = (audio_file.name, audio_file.size, task) if audio_file else None
    if input_key != st.session_state.get("_last_input_key"):
        st.session_state.pop("result", None)
        st.session_state.pop("result_filename", None)
        st.session_state["_last_input_key"] = input_key

    if audio_file:
        st.audio(audio_file)

    is_translate = task is not None and task != "Transcribe"
    button_label = "Translate" if is_translate else "Transcribe"
    can_run = audio_file is not None and task is not None

    if st.button(button_label, type="primary", disabled=not can_run) and can_run:
        user_prompt = PROMPT_CHOICES[task]
        with st.spinner(
            f"{'Translating' if is_translate else 'Transcribing'} audio..."
        ):
            try:
                wav, audio_duration = load_and_preprocess_audio(audio_file)
                transcript, eval_duration = transcribe_audio(
                    wav, user_prompt, model, processor, device
                )
                st.session_state.result = {
                    "model": MODEL_ID,
                    "audio_duration": audio_duration,
                    "transcript": transcript,
                    "num_words": len(transcript.split()),
                    "eval_duration": eval_duration,
                }
                stem = Path(audio_file.name).stem
                if audio_file.name == "audio.wav":
                    stem = datetime.now().strftime("recording_%Y%m%d_%H%M%S")
                st.session_state.result_filename = f"{stem}_transcription.json"
            except RuntimeError as e:
                st.error(str(e))
                return
            except Exception as e:
                st.exception(e)
                return
        st.toast("Done!")

    if "result" in st.session_state:
        result = st.session_state.result
        with st.container(border=True):
            st.code(result["transcript"], language=None)
            cols = st.columns(3)
            cols[0].metric("Audio Duration", f"{result['audio_duration']:.2f}s")
            cols[1].metric("Words", result["num_words"])
            cols[2].metric("Processing Time", f"{result['eval_duration']}s")
            dl_cols = st.columns(2)
            dl_cols[0].download_button(
                "Download Text",
                result["transcript"],
                st.session_state.result_filename.replace(".json", ".txt"),
                "text/plain",
            )
            dl_cols[1].download_button(
                "Download JSON",
                json.dumps(result, indent=2),
                st.session_state.result_filename,
                "application/json",
            )


if __name__ == "__main__":
    main()
