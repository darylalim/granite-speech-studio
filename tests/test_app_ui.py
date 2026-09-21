"""End-to-end UI tests driving the real main() widget tree via AppTest.

These complement test_streamlit_app.py (which mocks `st` entirely and never
exercises the UI wiring). AppTest re-executes streamlit_app.py in a fresh
namespace on every .run(), so mocks must patch the SHARED upstream imports
(mlx_audio, transformers, av) rather than streamlit_app.* — patching
streamlit_app attributes does not cross AppTest's script-runner boundary, and
clicking Run without an upstream patch would fetch the real ~0.9 GB speech
model, the guardian and the MLX VAD checkpoint from the Hub.

The Run tests are hermetic: the speech model and the guardian are stubbed at
their loaders, and the MLX VAD loader is made to fail so load_vad_model takes
its documented fallback onto the PyTorch build that ships inside the silero-vad
wheel — real VAD spans on the fixture, nothing downloaded.
"""

import tomllib
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import pytest
import streamlit as st
import torch
from streamlit import config
from streamlit.testing.v1 import AppTest

from streamlit_app import GUARDIAN_MODEL_ID, MODEL_ID, MODEL_REVISION

APP = Path(__file__).parent.parent / "streamlit_app.py"
AUDIO_DIR = Path(__file__).parent / "data" / "audio"
CONFIG = Path(__file__).parent.parent / ".streamlit" / "config.toml"


@pytest.fixture
def audio_bytes() -> bytes:
    return (AUDIO_DIR / "sample_10s.wav").read_bytes()


@pytest.fixture
def app_config() -> dict:
    """The parsed .streamlit/config.toml. Read as UTF-8 explicitly, which TOML
    mandates and Streamlit itself passes — `read_text()` would otherwise decode
    the file's comments with whatever the platform locale happens to be."""
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


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


@pytest.fixture
def speech_loader() -> Iterator[MagicMock]:
    """The upstream MLX speech loader, patched to hand back a fake whose
    generate() returns "hello world". spec= keeps the fake honest: the app
    drives the model through generate() alone, so any other attribute access
    would be a reach into internals the real class need not have."""
    fake_model = MagicMock(spec=["generate"])
    fake_model.generate.return_value.text = "hello world"
    with patch("mlx_audio.stt.utils.load_model", return_value=fake_model) as loader:
        yield loader


@contextmanager
def _stub_guardian(toxic: bool) -> Iterator[MagicMock]:
    """Patch the transformers entry points load_guardian_model goes through,
    and yield the model class so a test can assert the load happened (or did
    not). The tokenizer answers both call shapes check_safety uses with one
    short encoding, and the model's logits pin the verdict: [4, -4] softmaxes to
    p(toxic) ~ 3e-4, [-4, 4] to ~0.9997."""
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    model = MagicMock()
    model.return_value.logits = torch.tensor([[-4.0, 4.0] if toxic else [4.0, -4.0]])
    with (
        patch("transformers.AutoTokenizer") as tokenizer_cls,
        patch("transformers.AutoModelForSequenceClassification") as model_cls,
    ):
        tokenizer_cls.from_pretrained.return_value = tokenizer
        model_cls.from_pretrained.return_value = model
        yield model_cls


@contextmanager
def _header_duration(seconds: float) -> Iterator[MagicMock]:
    """Patch the shared `av.open` so audio_duration_seconds reads `seconds`
    from the (fake) stream header — no fixture is half an hour long. The
    stream's duration is in its own time base, which is how real containers
    carry it, so the app's `duration * time_base` arithmetic is exercised
    rather than bypassed."""
    stream = MagicMock(
        spec=av.AudioStream,
        duration=round(seconds * 16000),
        time_base=Fraction(1, 16000),
    )
    container = MagicMock()
    # The app asks for streams.best("audio"), not streams.audio[0], and narrows
    # the result with isinstance — hence the spec.
    container.streams.best.return_value = stream
    with patch("av.open") as opener:
        opener.return_value.__enter__.return_value = container
        yield opener


def _fixture_samples() -> int:
    """The wav fixture's frame count straight from its header, so the Run
    tests can bound the VAD segment against the clip without going through
    the decoder under test."""
    with wave.open(str(AUDIO_DIR / "sample_10s.wav")) as w:
        return w.getnframes()


@contextmanager
def _offline_mlx_vad() -> Iterator[MagicMock]:
    """Make the MLX VAD loader fail as it would with no cache and no network.

    load_vad_model then falls back to the PyTorch build bundled in the
    silero-vad wheel, whose 16 kHz weights are bit-exact with the MLX
    checkpoint — so the spans are the real ones, and nothing is fetched.
    """
    with patch(
        "mlx_audio.vad.utils.load_model", side_effect=OSError("offline")
    ) as loader:
        yield loader


def _upload_and_run(
    audio_bytes: bytes, *, use_segmentation: bool, use_toxicity_check: bool
) -> AppTest:
    at = _app().run()
    at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
    at.toggle(key="use_segmentation").set_value(use_segmentation)
    at.toggle(key="use_toxicity_check").set_value(use_toxicity_check)
    at.run()
    at.button[0].click()
    at.run()
    return at


def _assert_single_result_card(at: AppTest) -> None:
    """Exactly one card: one subheader, one transcript, one download button.
    The download filename is hashed into the served media URL rather than
    carried on the element, so it is pinned by the unit-level card test; here
    the stem it derives from is checked through session state instead."""
    assert not at.exception
    assert [s.value for s in at.subheader] == ["Transcription"]
    assert len(at.text) == 1
    (download,) = at.download_button
    assert download.key == "dl_txt"
    assert download.help == "Download transcription"
    assert download.icon == ":material/download:"
    assert at.session_state["result_stem"] == "sample_10s"


def test_default_state() -> None:
    """App renders without exception in the documented default widget state:
    two toggles on, Run disabled, no results — and none of the prompt-era
    controls (source language, task pills, keywords) anywhere in the tree."""
    at = _app().run()
    assert not at.exception
    assert at.title[0].value == "Granite Speech Studio"
    assert len(at.segmented_control) == 0
    assert len(at.pills) == 0
    assert len(at.multiselect) == 0
    assert {t.key for t in at.toggle} == {"use_segmentation", "use_toxicity_check"}
    assert all(t.value is True for t in at.toggle)
    # Run is disabled until audio is loaded.
    assert at.button[0].disabled is True
    assert len(at.subheader) == 0
    assert "result" not in at.session_state.filtered_state


def test_description_names_the_loaded_model() -> None:
    """The description links the model card of the model the app actually
    loads, and says English — the CTC model transcribes nothing else."""
    at = _app().run()
    description = at.markdown[0].value
    assert f"https://huggingface.co/{MODEL_ID}" in description
    assert "granite-speech-5.0-470m-turboctc" in description
    assert "English" in description


def test_faux_labels_are_bold() -> None:
    """The VAD / Toxicity pseudo-labels render as bold markdown so they read
    as form labels (the real widget labels are collapsed). Keywords went with
    the prompt: CTC has nothing to bias."""
    at = _app().run()
    values = {m.value for m in at.markdown}
    assert "**VAD segmentation**" in values
    assert "**Toxicity check**" in values
    assert "**Keywords**" not in values


def test_run_button_gating(audio_bytes: bytes) -> None:
    """Run enables as soon as audio is present — there is no task selection
    to wait on any more."""
    at = _app().run()
    at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
    at.run()
    assert at.button[0].disabled is False
    assert at.caption[0].value == "sample_10s.wav"


@pytest.mark.parametrize(("duration", "blocked"), [(1801.0, True), (1799.0, False)])
def test_vad_off_long_audio_gates_run(
    audio_bytes: bytes, duration: float, blocked: bool
) -> None:
    """With VAD off, audio over MAX_VAD_OFF_DURATION_S disables Run behind a
    warning; just under it, neither. The boundary is pinned at 30 minutes on
    purpose: the message's "4 GB" is what a single inference measured at
    exactly that length (4.5 GB of MLX memory, before the decoded audio the
    process also holds), so retuning the constant has to retune the text."""
    with _header_duration(duration):
        at = _app().run()
        at.file_uploader[0].set_value(("long.wav", audio_bytes, "audio/wav"))
        at.toggle(key="use_segmentation").set_value(False)
        at.run()
    assert not at.exception
    assert at.button[0].disabled is blocked
    if blocked:
        (vad_warning,) = at.warning
        assert "longer than 30 minutes" in vad_warning.value
        assert "more than 4 GB" in vad_warning.value
        assert vad_warning.icon == ":material/warning:"
    else:
        assert len(at.warning) == 0


def test_vad_off_short_clip_shows_no_warning(audio_bytes: bytes) -> None:
    """VAD off on a 30 s clip renders no warning at all. The prompted model
    fired a translation-passthrough warning here (anything past 20 s); an
    English-only CTC model has nothing to pass through, so the hour-long
    memory gate is the only VAD-off warning left."""
    with _header_duration(30.0):
        at = _app().run()
        at.file_uploader[0].set_value(("clip.wav", audio_bytes, "audio/wav"))
        at.toggle(key="use_segmentation").set_value(False)
        at.run()
    assert not at.exception
    assert len(at.warning) == 0
    assert at.button[0].disabled is False


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


@pytest.mark.parametrize(
    ("toxic", "banner", "text", "icon"),
    [
        (False, "success", "Content is safe", ":material/check_circle:"),
        (True, "warning", "Toxic content detected", ":material/warning:"),
    ],
)
def test_run_with_toxicity_check_renders_banner(
    audio_bytes: bytes,
    speech_loader: MagicMock,
    toxic: bool,
    banner: str,
    text: str,
    icon: str,
) -> None:
    """VAD on, toxicity on: one card with a timestamped transcript and the
    verdict banner the stubbed guardian dictates. The VAD path trims the
    fixture's 0.76 s digital-silence lead-in, so the one segment starts past
    zero (still formatting as 0:00) and is shorter than the clip."""
    with (
        _stub_guardian(toxic) as guardian_cls,
        _offline_mlx_vad() as vad_loader,
        pytest.warns(RuntimeWarning, match="using the PyTorch VAD"),
    ):
        at = _upload_and_run(
            audio_bytes, use_segmentation=True, use_toxicity_check=True
        )
    _assert_single_result_card(at)
    # Guards: every model came from a patch, not the Hub.
    speech_loader.assert_called_once_with(
        MODEL_ID, revision=MODEL_REVISION, strict=True
    )
    guardian_cls.from_pretrained.assert_called_once_with(GUARDIAN_MODEL_ID)
    vad_loader.assert_called_once()

    assert at.text[0].value == "[0:00 - 0:09] hello world"
    generate = speech_loader.return_value.generate
    assert generate.call_count == 1
    assert generate.call_args.kwargs.keys() == {"audio"}
    segment = generate.call_args.kwargs["audio"]
    full_clip = _fixture_samples()
    assert 0 < segment.shape[0] < full_clip

    assert len(at.success) + len(at.warning) == 1
    (element,) = getattr(at, banner)
    assert element.value.startswith(text)
    assert element.icon == icon


def test_run_with_toxicity_check_off_renders_no_banner(
    audio_bytes: bytes, speech_loader: MagicMock
) -> None:
    """Toxicity off: the card renders with neither banner, and the guardian is
    never loaded — main() skips the load rather than passing a flag down."""
    with (
        _stub_guardian(toxic=True) as guardian_cls,
        _offline_mlx_vad(),
        pytest.warns(RuntimeWarning, match="using the PyTorch VAD"),
    ):
        at = _upload_and_run(
            audio_bytes, use_segmentation=True, use_toxicity_check=False
        )
    _assert_single_result_card(at)
    speech_loader.assert_called_once_with(
        MODEL_ID, revision=MODEL_REVISION, strict=True
    )
    guardian_cls.from_pretrained.assert_not_called()
    assert at.text[0].value == "[0:00 - 0:09] hello world"
    assert len(at.success) == 0
    assert len(at.warning) == 0


def test_run_with_vad_off_transcribes_the_whole_clip(
    audio_bytes: bytes, speech_loader: MagicMock
) -> None:
    """VAD off: no VAD load, one inference over every sample of the clip, and
    one timestamped line spanning it."""
    with _stub_guardian(toxic=False), _offline_mlx_vad() as vad_loader:
        at = _upload_and_run(
            audio_bytes, use_segmentation=False, use_toxicity_check=False
        )
    _assert_single_result_card(at)
    vad_loader.assert_not_called()
    speech_loader.assert_called_once_with(
        MODEL_ID, revision=MODEL_REVISION, strict=True
    )

    assert at.text[0].value.splitlines() == ["[0:00 - 0:09] hello world"]
    generate = speech_loader.return_value.generate
    assert generate.call_count == 1
    full_clip = _fixture_samples()
    assert generate.call_args.kwargs["audio"].shape == (full_clip,)


def test_changing_a_toggle_discards_the_result(
    audio_bytes: bytes, speech_loader: MagicMock
) -> None:
    """The toggles are part of the input key, so flipping one after a run
    drops the stale card instead of leaving it beside settings it no longer
    reflects."""
    with _stub_guardian(toxic=False), _offline_mlx_vad():
        at = _upload_and_run(
            audio_bytes, use_segmentation=False, use_toxicity_check=False
        )
        _assert_single_result_card(at)
        speech_loader.assert_called_once_with(
            MODEL_ID, revision=MODEL_REVISION, strict=True
        )
        at.toggle(key="use_toxicity_check").set_value(True)
        at.run()
    assert not at.exception
    assert len(at.subheader) == 0
    assert "result" not in at.session_state.filtered_state
    assert "result_stem" not in at.session_state.filtered_state


def test_config_defines_no_custom_theme(app_config: dict) -> None:
    """No [theme] table, so the app gets Streamlit's built-in themes and the
    settings menu offers System / Light / Dark. See CLAUDE.md for why writing
    the defaults back in is not the same thing.

    The second half is the guard that outlives this config: should a custom
    theme ever be reintroduced, it has to define both sub-palettes or the
    appearance menu disappears entirely and the app locks to one mode
    (verified against 1.61.1 — a [theme] table carrying only primaryColor
    renders the menu with no appearance section at all)."""
    if "theme" in app_config:
        assert {"light", "dark"} <= app_config["theme"].keys()


def _option_paths(table: dict, valid: set[str]) -> Iterator[str]:
    """Flatten a parsed config.toml into the option paths it sets."""

    def walk(path: str, value: object) -> Iterator[str]:
        if path in valid:
            # A registered option, so stop rather than descend: a few options
            # take a table as their value (server.trustedUserHeaders) or a list
            # of them ([[theme.fontFaces]]), and those inner keys are free-form
            # rather than registry names.
            yield path
        elif isinstance(value, dict) and value:
            # An unregistered table with contents is a section, not a key.
            for key, child in value.items():
                yield from walk(f"{path}.{key}", child)
        else:
            # Scalar, list, or empty table under an unregistered path. The empty
            # case is what reports a typo'd section header with no keys of its
            # own, which would otherwise flatten to nothing and pass.
            yield path

    for key, value in table.items():
        yield from walk(key, value)


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ({"server": {"maxUploadSize": 500}}, ["server.maxUploadSize"]),
        # An option whose value is a table is a leaf; its keys are free-form.
        (
            {"server": {"trustedUserHeaders": {"X-Forwarded-User": "email"}}},
            ["server.trustedUserHeaders"],
        ),
        # As is one whose value is an array of tables.
        ({"theme": {"fontFaces": [{"family": "X", "url": "u"}]}}, ["theme.fontFaces"]),
        # Typos still surface at every depth and every shape.
        ({"theme": {"typoColor": "#fff"}}, ["theme.typoColor"]),
        ({"thheme": {"fontFaces": [{"family": "X"}]}}, ["thheme.fontFaces"]),
        ({"browserr": {}}, ["browserr"]),
    ],
)
def test_option_paths_stops_at_registered_options(
    table: dict, expected: list[str]
) -> None:
    """The shipped config exercises one path of _option_paths, so the rest are
    pinned here: descending into an option's own value fails a valid config,
    and not reaching a typo under an unregistered prefix passes an invalid one.
    """
    assert list(_option_paths(table, set(config._config_options_template))) == expected


def test_config_has_no_invalid_options(app_config: dict) -> None:
    """Every key in config.toml is a registered Streamlit config option.
    Unknown keys are dropped silently rather than rejected, so a typo (or an
    option that only ever existed in a sibling table, as `base` does for
    [theme] but not [theme.light]) would otherwise go unnoticed."""
    valid = set(config._config_options_template)
    assert "theme.primaryColor" in valid  # registry is populated
    invalid = [path for path in _option_paths(app_config, valid) if path not in valid]
    assert invalid == [], f"invalid config options: {invalid}"
