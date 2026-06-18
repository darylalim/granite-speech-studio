"""End-to-end UI tests driving the real main() widget tree via AppTest.

These complement test_streamlit_app.py (which mocks `st` entirely and never
exercises the UI wiring). AppTest re-executes streamlit_app.py in a fresh
namespace on every .run(), so mocks must patch the SHARED upstream imports
(mlx_audio, torchcodec) rather than streamlit_app.* — patching streamlit_app
attributes does not cross AppTest's script-runner boundary, and clicking Run
without an upstream patch would load the real ~2.9GB speech model.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parent.parent / "streamlit_app.py"
AUDIO_DIR = Path(__file__).parent / "data" / "audio"


@pytest.fixture
def audio_bytes() -> bytes:
    return (AUDIO_DIR / "sample_10s.wav").read_bytes()


def _app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=60)


def test_default_state() -> None:
    """App renders without exception in the documented default widget state."""
    at = _app().run()
    assert not at.exception
    assert at.title[0].value == "Granite Speech Studio"
    assert at.segmented_control[0].value == "English"
    assert at.pills[0].value == ["Transcribe"]
    assert at.pills[0].options == [
        "Transcribe",
        "French",
        "German",
        "Spanish",
        "Portuguese",
        "Italian",
        "Japanese",
        "Mandarin Chinese",
    ]
    assert {t.key for t in at.toggle} == {"use_segmentation", "use_toxicity_check"}
    assert all(t.value is True for t in at.toggle)
    # Run is disabled until audio is loaded.
    assert at.button[0].disabled is True


def test_run_button_gating(audio_bytes: bytes) -> None:
    """Run enables once audio + a task are present and re-disables when tasks
    are cleared."""
    at = _app().run()
    at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
    at.run()
    assert at.button[0].disabled is False
    assert at.caption[0].value == "sample_10s.wav"

    at.pills[0].set_value([])
    at.run()
    assert at.button[0].disabled is True


def test_source_change_resets_tasks() -> None:
    """A non-English source narrows tasks to Transcribe + English translation
    and resets the selection to Transcribe."""
    at = _app().run()
    at.segmented_control[0].set_value("French")
    at.run()
    assert at.pills[0].options == ["Transcribe", "English"]
    assert at.pills[0].value == ["Transcribe"]


def test_vad_off_long_audio_disables_run(audio_bytes: bytes) -> None:
    """With VAD off and audio over the 5-minute limit, Run is disabled and a
    warning is shown."""
    with patch("torchcodec.decoders.AudioDecoder") as decoder:
        decoder.return_value.metadata.duration_seconds = 600.0
        at = _app().run()
        at.file_uploader[0].set_value(("long.wav", audio_bytes, "audio/wav"))
        at.run()
        at.toggle(key="use_segmentation").set_value(False)
        at.run()
    assert at.button[0].disabled is True
    assert any("longer than 5 minutes" in w.value for w in at.warning)


def test_run_renders_result_card(audio_bytes: bytes) -> None:
    """Clicking Run with mocked inference renders a result card. Patches the
    upstream MLX loader so no real model loads."""
    fake_model = MagicMock()
    fake_model.generate.return_value.text = "the quick brown fox (mocked)"
    with patch(
        "mlx_audio.stt.utils.load_model", return_value=fake_model
    ) as loader:
        at = _app().run()
        at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
        at.toggle(key="use_segmentation").set_value(False)
        at.toggle(key="use_toxicity_check").set_value(False)
        at.run()
        at.button[0].click()
        at.run()
    assert not at.exception
    loader.assert_called()  # guard: the real speech model must never load
    assert at.subheader[0].value == "Transcribe (English)"
    assert "mocked" in at.text[0].value
