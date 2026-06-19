"""End-to-end UI tests driving the real main() widget tree via AppTest.

These complement test_streamlit_app.py (which mocks `st` entirely and never
exercises the UI wiring). AppTest re-executes streamlit_app.py in a fresh
namespace on every .run(), so mocks must patch the SHARED upstream imports
(mlx_audio, torchcodec) rather than streamlit_app.* — patching streamlit_app
attributes does not cross AppTest's script-runner boundary, and clicking Run
without an upstream patch would load the real ~2.9GB speech model.
"""

import tomllib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit import config
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parent.parent / "streamlit_app.py"
AUDIO_DIR = Path(__file__).parent / "data" / "audio"
CONFIG = Path(__file__).parent.parent / ".streamlit" / "config.toml"


@pytest.fixture
def audio_bytes() -> bytes:
    return (AUDIO_DIR / "sample_10s.wav").read_bytes()


@pytest.fixture
def theme_config() -> dict:
    """The parsed `[theme]` table from .streamlit/config.toml."""
    return tomllib.loads(CONFIG.read_text())["theme"]


def _app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=60)


@pytest.fixture(autouse=True)
def _clear_streamlit_caches() -> Iterator[None]:
    # AppTest does not reset Streamlit's process-global @st.cache_resource store
    # between instances, so a cached (mocked or real) model could leak across
    # tests — which would make the Run-path loader.assert_called() guard
    # order-dependent. Clear it before every test for isolation.
    st.cache_resource.clear()
    yield


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


def test_faux_labels_are_bold() -> None:
    """The VAD / Toxicity / Keywords pseudo-labels render as bold markdown so
    they read as form labels (the real widget labels are collapsed)."""
    at = _app().run()
    values = {m.value for m in at.markdown}
    assert "**VAD segmentation**" in values
    assert "**Toxicity check**" in values
    assert "**Keywords**" in values


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
    vad_warning = next(
        (w for w in at.warning if "longer than 5 minutes" in w.value), None
    )
    assert vad_warning is not None
    assert vad_warning.icon == ":material/warning:"


def test_duration_cache_is_single_slot(audio_bytes: bytes) -> None:
    """With VAD off, the duration cache is a single `_duration` slot holding the
    current file's ((name, size), duration) — swapping files overwrites it rather
    than accumulating an entry per file, so it can't grow unbounded."""
    size = len(audio_bytes)
    at = _app().run()
    at.toggle(key="use_segmentation").set_value(False)
    at.file_uploader[0].set_value(("a.wav", audio_bytes, "audio/wav"))
    at.run()
    assert at.session_state["_duration"][0] == ("a.wav", size)

    at.file_uploader[0].set_value(("b.wav", audio_bytes, "audio/wav"))
    at.run()
    # Slot is overwritten in place, not accumulated.
    assert at.session_state["_duration"][0] == ("b.wav", size)
    duration_slots = [k for k in at.session_state.filtered_state if k == "_duration"]
    assert duration_slots == ["_duration"]


def test_run_renders_result_card(audio_bytes: bytes) -> None:
    """Clicking Run with mocked inference renders a result card. Patches the
    upstream MLX loader so no real model loads."""
    fake_model = MagicMock()
    fake_model.generate.return_value.text = "the quick brown fox (mocked)"
    with patch("mlx_audio.stt.utils.load_model", return_value=fake_model) as loader:
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


def test_run_renders_multiple_result_cards(audio_bytes: bytes) -> None:
    """Transcribe + one translation drives the CoT-AST path and the multi-card
    result grid (the N>1 _row_sizes -> st.columns loop in main()), which the
    single-task Run test does not exercise."""
    fake_model = MagicMock()
    fake_model.generate.return_value.text = (
        "[Transcription] hello world [Translation] bonjour le monde"
    )
    with patch("mlx_audio.stt.utils.load_model", return_value=fake_model):
        at = _app().run()
        at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
        at.pills[0].set_value(["Transcribe", "French"])
        at.toggle(key="use_segmentation").set_value(False)
        at.toggle(key="use_toxicity_check").set_value(False)
        at.run()
        at.button[0].click()
        at.run()
    assert not at.exception
    assert {s.value for s in at.subheader} == {"Transcribe (English)", "French"}
    # CoT output is split: transcription -> Transcribe card, translation -> French.
    texts = " ".join(t.value for t in at.text)
    assert "hello world" in texts
    assert "bonjour le monde" in texts


def test_theme_config_defines_light_and_dark(theme_config: dict) -> None:
    """config.toml defines both [theme.light] and [theme.dark] color palettes,
    which together enable the settings-menu light/dark toggle."""
    theme = theme_config
    assert "light" in theme and "dark" in theme
    for mode in ("light", "dark"):
        for key in (
            "primaryColor",
            "backgroundColor",
            "textColor",
            "greenColor",
            "redColor",
        ):
            assert key in theme[mode], f"theme.{mode}.{key} missing"


def test_theme_config_has_no_invalid_options(theme_config: dict) -> None:
    """Every key under [theme] is a registered Streamlit config option. Guards
    against silently-dropped keys like the invalid `theme.light.base` (only
    `theme.base` exists; sub-themes have no `base`)."""
    valid = set(config._config_options_template)
    assert "theme.primaryColor" in valid  # registry is populated

    def walk(prefix: str, table: dict) -> Iterator[str]:
        for key, value in table.items():
            full = f"{prefix}.{key}"
            if isinstance(value, dict):
                yield from walk(full, value)
            else:
                yield full

    invalid = [k for k in walk("theme", theme_config) if k not in valid]
    assert invalid == [], f"invalid theme config options: {invalid}"
