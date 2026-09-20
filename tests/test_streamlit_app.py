import inspect
import re
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import MagicMock, patch

import mlx.core as mx  # ty: ignore[unresolved-import]
import numpy as np
import pytest
import torch

from streamlit_app import (
    _MLX_VAD_BRANCH_ATTRS,
    _MLX_VAD_BRANCH_CONFIG_ATTRS,
    _MLX_VAD_CONFIG_ATTRS,
    GUARDIAN_MODEL_ID,
    MAX_SEGMENT_DURATION_S,
    MAX_VAD_OFF_DURATION_S,
    MIN_AUDIO_SAMPLES,
    MIN_SEGMENT_DURATION_S,
    MLX_VAD_REPO,
    MLX_VAD_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    SUPPORTED_FORMATS,
    VAD_CHUNK_SAMPLES,
    VAD_CONTEXT_SAMPLES,
    VIDEO_FORMATS,
    PipelineResult,
    _aggregate_segment_safety,
    _mlx_vad_probabilities,
    _mlx_vad_windows,
    _render_result_card,
    _split_long_segment,
    _supports_mlx_vad,
    audio_duration_seconds,
    check_safety,
    format_timestamp,
    get_speech_segments,
    is_video,
    load_and_preprocess_audio,
    load_guardian_model,
    load_model,
    load_vad_model,
    run_pipeline,
    silero_vad,
    transcribe_audio,
)

# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

AUDIO_DIR = Path(__file__).parent / "data" / "audio"

# Unwrap st.cache_resource / torch.inference_mode wrappers so tests call the
# originals without going through Streamlit/torch machinery.
_load_model = load_model.__wrapped__  # ty: ignore[unresolved-attribute]
_load_guardian_model = load_guardian_model.__wrapped__  # ty: ignore[unresolved-attribute]
_load_vad_model = load_vad_model.__wrapped__  # ty: ignore[unresolved-attribute]
_run_pipeline = run_pipeline.__wrapped__  # ty: ignore[unresolved-attribute]


def make_upload(
    path: Path | None = None, raw: bytes = b"", name: str = "bad.wav"
) -> MagicMock:
    upload = MagicMock()
    if path is not None:
        upload.name = path.name
        upload.getvalue.return_value = path.read_bytes()
    else:
        upload.name = name
        upload.getvalue.return_value = raw
    return upload


def classification_calls(tokenizer: MagicMock) -> list:
    """Filter a guardian-tokenizer mock to the classification-shape calls
    only. check_safety also calls the tokenizer once per check to measure
    input length (truncation=False, add_special_tokens=False); the actual
    classification call uses padding=True."""
    return [c for c in tokenizer.call_args_list if c.kwargs.get("padding") is True]


class PipelineMocks(NamedTuple):
    """The four positional arguments after `wav` in run_pipeline's signature:
    (model, vad_model, guardian_model, guardian_tokenizer), in that order, so
    tests can unpack with `*pipeline_mocks`."""

    model: MagicMock
    vad: MagicMock
    guardian: MagicMock
    guardian_tokenizer: MagicMock


@pytest.fixture
def pipeline_mocks() -> PipelineMocks:
    # spec= so the mock exposes generate() and nothing else: the app drives the
    # speech model through its public generate() alone, and a bare MagicMock
    # would conjure any other attribute the code reached for rather than fail.
    model = MagicMock(spec=["generate"])
    model.generate.return_value = MagicMock(text="decoded text")
    guardian_tokenizer = MagicMock()
    guardian_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    guardian = MagicMock()
    guardian.return_value.logits = torch.tensor([[5.0, -5.0]])
    return PipelineMocks(model, MagicMock(), guardian, guardian_tokenizer)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_model_id() -> None:
    assert MODEL_ID == "ibm-granite/granite-speech-5.0-470m-turboctc"


def test_model_revision_is_pinned() -> None:
    # Pinned for a different reason than MLX_VAD_REVISION. This repo loads with
    # strict=True, but strict only proves every weight key is *present* — not
    # that the weights are the calibrated ones the memory and accuracy
    # constants were measured against. A retrained checkpoint pushed to main
    # would load strictly, decode without complaint, and produce different
    # transcripts on the next cold cache. Only a full commit hash is immutable:
    # a tag or branch name would be a valid `revision` and still move.
    assert MODEL_REVISION == "286456107c8ba1161f5c22dfe85466402c88333b"
    assert re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION)


def test_guardian_model_id() -> None:
    assert GUARDIAN_MODEL_ID == "ibm-granite/granite-guardian-hap-125m"


def test_supported_formats() -> None:
    assert set(SUPPORTED_FORMATS) == {
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
    }
    assert VIDEO_FORMATS.issubset(set(SUPPORTED_FORMATS))


def test_max_vad_off_duration_is_one_hour() -> None:
    # Memory is the only bound on the single-inference path: attention is
    # block-local so there is no context to run out of, and peak MLX memory
    # grows linearly at ~2 MB per second of audio — measured 7.8 GB at one
    # hour. That keeps a 16 GB Mac out of swap; two hours would not.
    assert MAX_VAD_OFF_DURATION_S == 3600


def test_mlx_vad_revision_is_pinned() -> None:
    # mlx_audio's VAD loader cannot use strict=True (the v6 conversion ships no
    # vad_8k keys), so renamed weight keys would load as random layers and pass
    # every other guard. A content hash is what rules that out.
    assert MLX_VAD_REPO == "mlx-community/silero-vad-v6"
    assert MLX_VAD_REVISION == "2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, expected",
    [
        pytest.param(0.0, "0:00", id="zero"),
        pytest.param(15.0, "0:15", id="seconds_only"),
        pytest.param(62.0, "1:02", id="minutes_and_seconds"),
        pytest.param(3661.0, "1:01:01", id="hours"),
        pytest.param(15.7, "0:15", id="fractional_truncated"),
    ],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("clip.mp4", True),
        ("clip.mov", True),
        ("clip.webm", True),
        ("clip.mkv", True),
        ("CLIP.MP4", True),
        ("recordings/2026/clip.mp4", True),
        ("sound.wav", False),
        ("sound.flac", False),
        ("sound.m4a", False),
        ("sound.mp3", False),
        ("sound.ogg", False),
        ("sound.aac", False),
        ("Sound.WAV", False),
        ("filename", False),
        ("recordings/2026/sound.wav", False),
    ],
)
def test_is_video(filename: str, expected: bool) -> None:
    assert is_video(filename) is expected


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------


_MLX_VAD_MODEL_ATTRS = ("vad_16k", "config", "dtype", "_probs_to_timestamps")


def make_mlx_vad_model(drop: str | None = None) -> MagicMock:
    """A model shaped like mlx_audio's, optionally missing one probed name.

    spec= throughout so hasattr answers honestly — a bare MagicMock would
    conjure every attribute the probe asks for and never report drift.
    """
    model_attrs = [a for a in _MLX_VAD_MODEL_ATTRS if a != drop]
    model = MagicMock(spec=model_attrs)
    if "dtype" in model_attrs:
        model.dtype = mx.float32
    if "vad_16k" in model_attrs:
        branch_attrs = [a for a in _MLX_VAD_BRANCH_ATTRS if a != drop]
        model.vad_16k = MagicMock(spec=[*branch_attrs, "config"])
        branch_config_attrs = [a for a in _MLX_VAD_BRANCH_CONFIG_ATTRS if a != drop]
        model.vad_16k.config = MagicMock(spec=branch_config_attrs)
        # The probe checks these two by value, not just by name.
        if "chunk_size" in branch_config_attrs:
            model.vad_16k.config.chunk_size = VAD_CHUNK_SAMPLES
        if "context_size" in branch_config_attrs:
            model.vad_16k.config.context_size = VAD_CONTEXT_SAMPLES
    if "config" in model_attrs:
        model.config = MagicMock(spec=[a for a in _MLX_VAD_CONFIG_ATTRS if a != drop])
    return model


class TestSileroVad:
    """The PyTorch path, which is now the fallback rather than the default."""

    def test_returns_tuples_in_seconds(self) -> None:
        mock_timestamps = [
            {"start": 16000, "end": 48000},
            {"start": 64000, "end": 96000},
        ]
        with patch("streamlit_app.get_speech_timestamps", return_value=mock_timestamps):
            result = silero_vad(torch.zeros(1, 160000), MagicMock(spec=torch.nn.Module))
        assert result == [(1.0, 3.0), (4.0, 6.0)]

    def test_empty_audio_returns_empty_list(self) -> None:
        with patch("streamlit_app.get_speech_timestamps", return_value=[]):
            assert (
                silero_vad(torch.zeros(1, 16000), MagicMock(spec=torch.nn.Module)) == []
            )

    def test_passes_model_and_sample_rate(self) -> None:
        model = MagicMock(spec=torch.nn.Module)
        wav = torch.zeros(1, 16000)
        with patch("streamlit_app.get_speech_timestamps", return_value=[]) as mock_fn:
            silero_vad(wav, model)
        assert torch.equal(mock_fn.call_args[0][0], wav.squeeze())
        assert mock_fn.call_args[0][1] is model
        assert mock_fn.call_args[1]["sampling_rate"] == 16000


class TestSupportsMlxVad:
    def test_accepts_a_complete_model(self) -> None:
        assert _supports_mlx_vad(make_mlx_vad_model())

    @pytest.mark.parametrize(
        "missing",
        [
            *_MLX_VAD_MODEL_ATTRS,
            *_MLX_VAD_BRANCH_ATTRS,
            *_MLX_VAD_BRANCH_CONFIG_ATTRS,
            *_MLX_VAD_CONFIG_ATTRS,
        ],
    )
    def test_rejects_a_model_missing_any_probed_name(self, missing: str) -> None:
        assert not _supports_mlx_vad(make_mlx_vad_model(drop=missing))

    def test_rejects_an_unrelated_object(self) -> None:
        assert not _supports_mlx_vad(object())

    @pytest.mark.parametrize("field", ["chunk_size", "context_size"])
    def test_rejects_framing_this_path_cannot_serve(self, field: str) -> None:
        # The pin freezes the repo's config.json but not mlx_audio's
        # BranchConfig defaults, which fill in omitted fields — so a dependency
        # upgrade can still move the framing. Neither this path nor
        # _probs_to_timestamps (which hardcodes a 512-sample stride) could
        # serve it, so the probe must refuse rather than frame it wrong.
        model = make_mlx_vad_model()
        setattr(model.vad_16k.config, field, 256)
        assert not _supports_mlx_vad(model)


class TestSileroVadDispatch:
    def test_torch_model_takes_the_torch_path(self) -> None:
        with patch("streamlit_app.get_speech_timestamps", return_value=[]) as mock_fn:
            silero_vad(torch.zeros(1, 16000), MagicMock(spec=torch.nn.Module))
        mock_fn.assert_called_once()

    def test_mlx_model_takes_the_batched_path(self) -> None:
        model = make_mlx_vad_model()
        model._probs_to_timestamps.return_value = [{"start": 16000, "end": 48000}]
        with patch(
            "streamlit_app._mlx_vad_probabilities", return_value=mx.zeros((4,))
        ) as mock_probs:
            result = silero_vad(torch.zeros(1, 160000), model)
        mock_probs.assert_called_once()
        assert result == [(1.0, 3.0)]
        assert model._probs_to_timestamps.call_args[1]["audio_len"] == 160000

    def test_threshold_config_reaches_the_timestamp_scan(self) -> None:
        # The four values the probe checks for must be the four actually used.
        # Hardcode any of them and the PyTorch fallback keeps its own defaults,
        # so a repo shipping a non-default threshold makes the backends disagree.
        model = make_mlx_vad_model()
        model._probs_to_timestamps.return_value = []
        sentinels = {
            "threshold": 0.37,
            "min_speech_duration_ms": 111,
            "min_silence_duration_ms": 222,
            "speech_pad_ms": 33,
        }
        for name, value in sentinels.items():
            setattr(model.config, name, value)
        with patch("streamlit_app._mlx_vad_probabilities", return_value=mx.zeros((4,))):
            silero_vad(torch.zeros(1, 16000), model)
        kwargs = model._probs_to_timestamps.call_args[1]
        assert {name: kwargs[name] for name in sentinels} == sentinels

    def test_non_16k_audio_never_reaches_the_mlx_model(self) -> None:
        # The v6 conversion ships no trained 8 kHz weights, but mlx_audio builds
        # that branch anyway and loads non-strictly, so an 8 kHz call would run
        # randomly initialised layers and return confident nonsense.
        model = make_mlx_vad_model()
        with (
            patch("streamlit_app._torch_vad_model") as mock_load,
            patch("streamlit_app.get_speech_timestamps", return_value=[]) as mock_fn,
            patch("streamlit_app._mlx_vad_probabilities") as mock_probs,
        ):
            silero_vad(torch.zeros(1, 8000), model, sample_rate=8000)
        mock_probs.assert_not_called()
        assert mock_fn.call_args[0][1] is mock_load.return_value

    def test_mlx_failure_warns_and_falls_back_to_torch(self) -> None:
        # hasattr only catches renames; a changed signature lands here instead.
        model = make_mlx_vad_model()
        with (
            patch(
                "streamlit_app._mlx_vad_probabilities",
                side_effect=TypeError("unexpected keyword"),
            ),
            patch("streamlit_app._torch_vad_model") as mock_load,
            patch(
                "streamlit_app.get_speech_timestamps",
                return_value=[{"start": 0, "end": 16000}],
            ) as mock_fn,
            pytest.warns(RuntimeWarning, match="MLX VAD failed"),
        ):
            result = silero_vad(torch.zeros(1, 16000), model)
        assert result == [(0.0, 1.0)]
        assert mock_fn.call_args[0][1] is mock_load.return_value


class TestMlxVadInternals:
    """Executes the real _mlx_vad_windows / _mlx_vad_probabilities bodies.

    Every other VAD test mocks them out, so without this class the code driving
    mlx_audio's branch submodules never runs in CI and an upstream change ships
    green. _supports_mlx_vad cannot help: it only checks that names exist.

    The model here is a genuine mlx_audio Model built from a real ModelConfig
    with random weights — no download, but real classes, real shapes, and a real
    streaming implementation to check the batched one against.
    """

    @staticmethod
    def _real_model() -> Any:
        module = pytest.importorskip("mlx_audio.vad.models.silero_vad.silero_vad")
        config = pytest.importorskip("mlx_audio.vad.models.silero_vad.config")
        return module.Model(config.ModelConfig())

    def test_windows_frame_audio_like_the_streaming_loop(self) -> None:
        audio = mx.arange(1000, dtype=mx.float32)
        windows = _mlx_vad_windows(audio)
        # 1000 samples pad up to 1024 = 2 chunks, each carrying 64 of context.
        assert windows.shape == (2, VAD_CONTEXT_SAMPLES + VAD_CHUNK_SAMPLES)
        assert mx.all(windows[0, :VAD_CONTEXT_SAMPLES] == 0).item()
        assert mx.array_equal(windows[0, VAD_CONTEXT_SAMPLES:], mx.arange(512)).item()
        # The second window's context is the first chunk's tail, not zeros.
        assert mx.array_equal(
            windows[1, :VAD_CONTEXT_SAMPLES], mx.arange(448, 512)
        ).item()

    def test_windows_are_empty_for_empty_audio(self) -> None:
        assert _mlx_vad_windows(mx.zeros((0,))).shape[0] == 0

    def test_batched_probabilities_match_the_streaming_reference(self) -> None:
        # The whole premise: batching the encoder and running one LSTM pass must
        # reproduce mlx_audio's chunk-at-a-time loop, not merely approximate it.
        model = self._real_model()
        audio = mx.array(
            np.random.default_rng(0).standard_normal(16000).astype(np.float32)
        )
        mine = _mlx_vad_probabilities(model, audio)
        reference = model._predict_proba_array(audio, 16000).reshape(-1)
        mx.eval(mine, reference)
        assert mine.shape == reference.shape
        assert float(mx.max(mx.abs(mine - reference)).item()) < 1e-5

    def test_lstm_state_carries_across_encoder_batches(self) -> None:
        # With a batch size below the chunk count the (hidden, cell) handoff is
        # what keeps the recurrence intact; drop it and probabilities restart
        # mid-audio. A single batch would never exercise it.
        model = self._real_model()
        audio = mx.array(
            np.random.default_rng(1).standard_normal(16000).astype(np.float32)
        )
        reference = model._predict_proba_array(audio, 16000).reshape(-1)
        with patch("streamlit_app.VAD_ENCODER_BATCH_CHUNKS", 4):
            batched = _mlx_vad_probabilities(model, audio)
        mx.eval(batched, reference)
        assert batched.shape == reference.shape
        assert float(mx.max(mx.abs(batched - reference)).item()) < 1e-5

    def test_real_mlx_audio_vad_model_still_exposes_the_internals(self) -> None:
        # Guards the actual installed mlx_audio, which make_mlx_vad_model cannot.
        assert _supports_mlx_vad(self._real_model())

    def test_refuses_framing_that_leaves_more_than_one_frame_per_chunk(self) -> None:
        """The batched path reuses the batch axis as the LSTM's sequence axis.

        That only holds while the conv stack collapses each window to one frame.
        Measured: hop_length=64 leaves 2, and taking frame 0 would silently drop
        half of them while the reference averages. This must raise so the
        try/except in silero_vad falls back rather than emit plausible garbage.
        """
        module = pytest.importorskip("mlx_audio.vad.models.silero_vad.silero_vad")
        config = pytest.importorskip("mlx_audio.vad.models.silero_vad.config")
        model = module.Model(
            config.ModelConfig(branch_16k=config.BranchConfig(hop_length=64))
        )
        with pytest.raises(ValueError, match="frames per chunk"):
            _mlx_vad_probabilities(model, mx.zeros((16000,)))

    def test_audio_is_cast_to_the_model_dtype(self) -> None:
        # Model.__call__ casts every input; this path bypasses it, and MLX would
        # silently promote float32 audio against float16 weights rather than
        # raise, diverging from the reference it claims to reproduce.
        model = make_mlx_vad_model()
        model.dtype = mx.float16
        model._probs_to_timestamps.return_value = []
        with patch(
            "streamlit_app._mlx_vad_probabilities", return_value=mx.zeros((4,))
        ) as mock_probs:
            silero_vad(torch.zeros(1, 16000), model)
        assert mock_probs.call_args[0][1].dtype == mx.float16

    def test_both_backends_agree_on_real_audio(self) -> None:
        """Real weights, real speech, both paths — spans must match exactly.

        The tests above run random weights, so they prove the batching is
        arithmetically equivalent but say nothing about the two *checkpoints*
        agreeing. Pinning MLX_VAD_REVISION froze one side of that pair, not
        both: `silero-vad` is an ordinary dependency, so a `uv lock` bumping it
        to a future Silero release would leave the backends disagreeing with
        nothing else to notice.

        Skipped rather than failed when the repo cannot be fetched — an offline
        CI run should not turn red over a network absence.
        """
        from silero_vad import load_silero_vad

        from streamlit_app import _load_mlx_vad_model

        try:
            mlx_model = _load_mlx_vad_model(MLX_VAD_REPO)
        except Exception as e:  # noqa: BLE001 - any failure means skip
            pytest.skip(f"{MLX_VAD_REPO} unavailable: {e}")
        if not _supports_mlx_vad(mlx_model):  # pragma: no cover - upstream drift
            pytest.skip("mlx_audio VAD internals have moved")

        wav = load_and_preprocess_audio(make_upload(AUDIO_DIR / "sample_10s.wav"))
        assert silero_vad(wav, mlx_model) == silero_vad(wav, load_silero_vad())


class TestGetSpeechSegments:
    @staticmethod
    def _run(
        wav: torch.Tensor,
        vad_segments: list[tuple[float, float]],
        max_segment_duration: float = MAX_SEGMENT_DURATION_S,
    ) -> list[dict[str, float]]:
        with patch("streamlit_app.silero_vad", return_value=vad_segments):
            return get_speech_segments(
                wav, MagicMock(), max_segment_duration=max_segment_duration
            )

    def test_adds_start_and_end_buffer(self) -> None:
        result = self._run(torch.zeros(1, 160000), [(1.0, 2.0)])
        assert result[0]["start"] == pytest.approx(0.7)
        assert result[0]["end"] == pytest.approx(2.3)

    def test_clamps_start_buffer_to_zero(self) -> None:
        result = self._run(torch.zeros(1, 160000), [(0.1, 1.0)])
        assert result[0]["start"] == 0.0

    def test_clamps_end_buffer_to_duration(self) -> None:
        # 2 seconds at 16 kHz
        result = self._run(torch.zeros(1, 32000), [(0.5, 1.9)])
        assert result[0]["end"] == 2.0

    def test_merges_close_segments(self) -> None:
        result = self._run(torch.zeros(1, 160000), [(1.0, 2.0), (2.3, 3.0)])
        assert len(result) == 1
        assert result[0]["start"] == pytest.approx(0.7)
        assert result[0]["end"] == pytest.approx(3.3)

    def test_keeps_distant_segments_separate(self) -> None:
        result = self._run(torch.zeros(1, 160000), [(1.0, 2.0), (5.0, 6.0)])
        assert len(result) == 2

    def test_no_speech_falls_back_to_full_audio(self) -> None:
        # 10 seconds at 16 kHz, covered end to end but still subject to the cap.
        result = self._run(torch.zeros(1, 160000), [])
        assert result[0]["start"] == 0.0
        assert result[-1]["end"] == pytest.approx(10.0)
        assert all(
            seg["end"] - seg["start"] <= MAX_SEGMENT_DURATION_S + 1e-6 for seg in result
        )

    def test_merge_stops_at_max_duration(self) -> None:
        # Both spans are within min_gap, so the old code merged them into one
        # 9.3s segment. With a 5s cap the merge must not happen.
        result = self._run(
            torch.zeros(1, 160000), [(1.0, 5.0), (5.2, 9.0)], max_segment_duration=5.0
        )
        assert len(result) == 2

    def test_continuous_speech_does_not_collapse_into_one_segment(self) -> None:
        # 60s of speech with sub-min_gap breaks: previously one 60s segment.
        vad = [(float(i), i + 0.9) for i in range(60)]
        result = self._run(torch.zeros(1, 16000 * 60), vad, max_segment_duration=10.0)
        assert len(result) > 1
        assert all(seg["end"] - seg["start"] <= 10.0 + 1e-6 for seg in result)

    def test_single_long_span_is_split_into_equal_parts(self) -> None:
        # One unbroken ~92s span, 30s cap -> 4 equal parts of ~22.9s, not
        # 3 x 30s + a 1.6s sliver. The cap sits above the 20s floor here, so
        # the part count is driven by the cap rather than backed off by the
        # floor (a cap at or below the floor clamps the floor to the cap, and
        # the parts then land at or above the cap instead).
        result = self._run(
            torch.zeros(1, 16000 * 100), [(0.3, 91.3)], max_segment_duration=30.0
        )
        assert len(result) == 4
        durations = [seg["end"] - seg["start"] for seg in result]
        assert all(d == pytest.approx(durations[0]) for d in durations)
        assert all(d <= 30.0 + 1e-6 for d in durations)

    def test_split_parts_are_contiguous_and_cover_the_span(self) -> None:
        result = self._run(
            torch.zeros(1, 16000 * 100), [(0.3, 91.3)], max_segment_duration=30.0
        )
        assert result[0]["start"] == pytest.approx(0.0)
        assert result[-1]["end"] == pytest.approx(91.6)
        for earlier, later in pairwise(result):
            assert earlier["end"] == pytest.approx(later["start"])

    def test_no_speech_fallback_is_also_capped(self) -> None:
        result = self._run(torch.zeros(1, 16000 * 75), [], max_segment_duration=30.0)
        assert len(result) == 3
        assert all(seg["end"] - seg["start"] <= 30.0 + 1e-6 for seg in result)

    def test_short_segments_are_untouched(self) -> None:
        result = self._run(
            torch.zeros(1, 160000), [(1.0, 2.0)], max_segment_duration=10.0
        )
        assert result == [{"start": pytest.approx(0.7), "end": pytest.approx(2.3)}]

    def test_segments_never_overlap_when_the_cap_blocks_a_merge(self) -> None:
        # Buffering pulls each span's start 0.3s back and pushes its end 0.3s
        # on, so consecutive spans overlap by 0.5s. Merging used to absorb that
        # unconditionally; once the cap can refuse the merge, the overlap has to
        # be clamped or the same audio is transcribed twice and the rendered
        # timestamps run backwards.
        vad = [(float(i), i + 0.9) for i in range(60)]
        result = self._run(torch.zeros(1, 16000 * 60), vad, max_segment_duration=8.0)
        assert len(result) > 1
        for earlier, later in pairwise(result):
            assert later["start"] >= earlier["end"] - 1e-6

    def test_segments_are_monotonic_across_cap_values(self) -> None:
        vad = [(float(i), i + 0.9) for i in range(40)]
        for cap in (5.0, 8.0, 10.0, 15.0):
            result = self._run(
                torch.zeros(1, 16000 * 40), vad, max_segment_duration=cap
            )
            for earlier, later in pairwise(result):
                assert later["start"] >= earlier["end"] - 1e-6, f"overlap at cap={cap}"
            assert all(seg["end"] > seg["start"] for seg in result)

    def test_span_just_over_the_cap_is_not_split_below_the_floor(self) -> None:
        # 8.3s at an 8.0s cap would halve into 2 x 4.15s. The floor clamps to
        # the cap, so neither half clears it and the overshoot is kept whole:
        # every forced split lands mid-utterance and costs a little accuracy.
        parts = _split_long_segment(0.0, 8.3, 8.0)
        assert len(parts) == 1
        assert parts[0] == {"start": 0.0, "end": 8.3}

    def test_long_spans_still_split_above_the_floor(self) -> None:
        # The floor is clamped to the cap, so at an 8.0s cap "above the floor"
        # means every part is at least 8.0s — and 24s is exactly 3 x 8s.
        parts = _split_long_segment(0.0, 24.0, 8.0)
        assert len(parts) == 3
        floor = min(MIN_SEGMENT_DURATION_S, 8.0)
        assert all(p["end"] - p["start"] >= floor - 1e-6 for p in parts)

    def test_floor_never_defeats_a_cap_smaller_than_it(self) -> None:
        # A caller passing max_duration below the floor must still get a split.
        parts = _split_long_segment(0.0, 12.0, 4.0)
        assert len(parts) == 3
        assert all(p["end"] - p["start"] <= 4.0 + 1e-6 for p in parts)

    def test_span_modestly_over_the_default_cap_stays_whole(self) -> None:
        # 30.1s at the shipped cap would halve into 2 x 15.05s, both under the
        # 20s floor, so the part count backs off to 1 and the overshoot is
        # tolerated. Greedy CTC has no length cliff to protect against, and
        # forced splits measurably cost WER (6.2% whole vs 6.5% in 10s parts vs
        # 8.2% in 5s parts on LibriSpeech dev-clean), so whole is the better
        # trade.
        parts = _split_long_segment(0.0, 30.1, MAX_SEGMENT_DURATION_S)
        assert parts == [{"start": 0.0, "end": 30.1}]

    def test_span_well_over_the_default_cap_splits_into_equal_halves(self) -> None:
        # 45s clears the floor when halved (2 x 22.5s), so it splits — into
        # equal halves, not 30s + 15s.
        parts = _split_long_segment(0.0, 45.0, MAX_SEGMENT_DURATION_S)
        assert len(parts) == 2
        assert parts[0] == {"start": 0.0, "end": pytest.approx(22.5)}
        assert parts[1] == {"start": pytest.approx(22.5), "end": pytest.approx(45.0)}

    def test_default_cap_and_floor(self) -> None:
        # Greedy CTC has no length cliff (the LLM decoder this replaced did, at
        # ~20s), so 30s is a readability cap for timestamp lines, not an
        # accuracy bound. The floor sits above half the cap on purpose: a span
        # just over the cap is tolerated whole rather than halved, because
        # every forced split lands mid-utterance and measurably costs WER (see
        # the measurements on MAX_SEGMENT_DURATION_S in streamlit_app.py).
        assert MAX_SEGMENT_DURATION_S == 30.0
        assert (
            MAX_SEGMENT_DURATION_S / 2
            < MIN_SEGMENT_DURATION_S
            <= MAX_SEGMENT_DURATION_S
        )


# ---------------------------------------------------------------------------
# Audio IO
# ---------------------------------------------------------------------------


class TestLoadAndPreprocessAudio:
    @pytest.mark.parametrize("filename", ["sample_10s.wav", "sample_10s_video.mp4"])
    def test_loads_real_audio(self, filename: str) -> None:
        wav = load_and_preprocess_audio(make_upload(AUDIO_DIR / filename))
        assert wav.shape[0] == 1
        assert wav.shape[1] > 0

    def test_zero_sample_audio_raises_a_readable_error(self) -> None:
        # Caught at load, not downstream: an empty waveform passes VAD (which
        # falls back to a zero-length segment) and would otherwise die inside
        # mlx_audio's compute_features as a ValueError, which main() shows as
        # an st.exception traceback rather than the one-line st.error.
        with (
            patch(
                "streamlit_app.torchaudio.load", return_value=(torch.zeros(1, 0), 16000)
            ),
            pytest.raises(RuntimeError, match="No audio detected"),
        ):
            load_and_preprocess_audio(make_upload(AUDIO_DIR / "sample_10s.wav"))

    @pytest.mark.parametrize(
        ("samples", "sr"),
        [
            # One under the floor at the model's own rate.
            (MIN_AUDIO_SAMPLES - 1, SAMPLE_RATE),
            # 3000 samples at 48 kHz is 1000 after the resample: the floor
            # must be measured in the model's samples, not the file's.
            (3000, 48000),
        ],
    )
    def test_audio_under_the_floor_raises_a_readable_error(
        self, samples: int, sr: int
    ) -> None:
        # Verified against the real model: 1119 samples dies as an opaque
        # [conv] ValueError inside the encoder, 1120 decodes.
        with (
            patch(
                "streamlit_app.torchaudio.load",
                return_value=(torch.zeros(1, samples), sr),
            ),
            pytest.raises(RuntimeError, match="Audio is too short"),
        ):
            load_and_preprocess_audio(make_upload(AUDIO_DIR / "sample_10s.wav"))

    def test_audio_at_the_floor_loads(self) -> None:
        with patch(
            "streamlit_app.torchaudio.load",
            return_value=(torch.zeros(1, MIN_AUDIO_SAMPLES), SAMPLE_RATE),
        ):
            wav = load_and_preprocess_audio(make_upload(AUDIO_DIR / "sample_10s.wav"))
        assert wav.shape == (1, MIN_AUDIO_SAMPLES)

    def test_invalid_audio_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="Failed to load audio file"):
            load_and_preprocess_audio(make_upload(raw=b"not audio data"))


class TestAudioDurationSeconds:
    @pytest.mark.parametrize("filename", ["sample_10s.wav", "sample_10s_video.mp4"])
    def test_duration_matches_fixture(self, filename: str) -> None:
        duration = audio_duration_seconds(make_upload(AUDIO_DIR / filename))
        assert duration is not None
        assert 9.5 < duration < 10.5

    def test_invalid_audio_returns_none(self) -> None:
        assert audio_duration_seconds(make_upload(raw=b"not audio data")) is None


# ---------------------------------------------------------------------------
# Model loaders (test the unwrapped originals)
# ---------------------------------------------------------------------------


@patch("streamlit_app.st")
class TestLoadModel:
    @patch("streamlit_app._load_stt_model")
    def test_calls_load_stt_model_and_returns_result(
        self, mock_load: MagicMock, _mock_st: MagicMock
    ) -> None:
        result = _load_model("test-model")
        mock_load.assert_called_once_with("test-model", revision=None, strict=True)
        assert result == mock_load.return_value

    @patch("streamlit_app._load_stt_model")
    def test_forwards_revision_pin(
        self, mock_load: MagicMock, _mock_st: MagicMock
    ) -> None:
        _load_model(MODEL_ID, MODEL_REVISION)
        mock_load.assert_called_once_with(
            MODEL_ID, revision=MODEL_REVISION, strict=True
        )

    @patch("streamlit_app._load_stt_model")
    def test_loads_strictly(self, mock_load: MagicMock, _mock_st: MagicMock) -> None:
        # mlx_audio defaults to strict=False, which leaves any weight the
        # checkpoint fails to supply randomly initialised — and greedy CTC would
        # still decode, into confident nonsense. The VAD loader cannot avoid
        # that (its checkpoint legitimately omits keys); this checkpoint ships
        # every key, so a missing one is a renamed key upstream and must raise
        # at load time. Asserted by name: mlx_audio's signature is
        # (model_path, lazy=False, strict=False, **kwargs), so a positional
        # True would land on `lazy` and leave strict at its default.
        _load_model(MODEL_ID, MODEL_REVISION)
        assert mock_load.call_args.kwargs["strict"] is True


@patch("streamlit_app.st")
class TestLoadGuardianModel:
    @patch("streamlit_app.AutoModelForSequenceClassification")
    @patch("streamlit_app.AutoTokenizer")
    def test_loads_model_and_tokenizer(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_model_cls: MagicMock,
        _mock_st: MagicMock,
    ) -> None:
        result = _load_guardian_model("test-model")
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("test-model")
        mock_model_cls.from_pretrained.assert_called_once_with("test-model")
        assert result == (
            mock_model_cls.from_pretrained.return_value,
            mock_tokenizer_cls.from_pretrained.return_value,
        )


@patch("streamlit_app.st")
class TestLoadVadModel:
    @patch("streamlit_app._load_mlx_vad_model")
    def test_loads_the_mlx_model(
        self, mock_load: MagicMock, _mock_st: MagicMock
    ) -> None:
        mock_load.return_value = make_mlx_vad_model()
        result = _load_vad_model()
        mock_load.assert_called_once_with(MLX_VAD_REPO, revision=MLX_VAD_REVISION)
        assert result is mock_load.return_value

    @patch("streamlit_app._torch_vad_model")
    @patch("streamlit_app._load_mlx_vad_model", side_effect=OSError("offline"))
    def test_falls_back_when_the_repo_will_not_load(
        self, _mock_mlx: MagicMock, mock_torch: MagicMock, _mock_st: MagicMock
    ) -> None:
        with pytest.warns(RuntimeWarning, match="using the PyTorch VAD"):
            result = _load_vad_model()
        assert result is mock_torch.return_value

    @patch("streamlit_app._torch_vad_model")
    @patch("streamlit_app._load_mlx_vad_model")
    def test_falls_back_when_the_internals_have_moved(
        self, mock_mlx: MagicMock, mock_torch: MagicMock, _mock_st: MagicMock
    ) -> None:
        # A rename upstream must degrade to the slower-but-correct path, not
        # surface as a crash on the first pipeline run.
        mock_mlx.return_value = make_mlx_vad_model(drop="_probs_to_timestamps")
        with pytest.warns(RuntimeWarning, match="internals have moved"):
            result = _load_vad_model()
        assert result is mock_torch.return_value


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


class TestCheckSafety:
    @staticmethod
    def _mocks(logits: list[list[float]]) -> tuple[MagicMock, MagicMock]:
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        model = MagicMock()
        model.return_value.logits = torch.tensor(logits)
        return model, tokenizer

    def test_safe_content(self) -> None:
        model, tokenizer = self._mocks([[5.0, -5.0]])
        is_toxic, score = check_safety("safe text", model, tokenizer)
        assert is_toxic is False
        assert score < 0.5

    def test_toxic_content(self) -> None:
        model, tokenizer = self._mocks([[-5.0, 5.0]])
        is_toxic, score = check_safety("toxic text", model, tokenizer)
        assert is_toxic is True
        assert score > 0.5

    def test_boundary_is_not_toxic(self) -> None:
        model, tokenizer = self._mocks([[0.0, 0.0]])
        is_toxic, score = check_safety("text", model, tokenizer)
        assert score == 0.5
        assert is_toxic is False

    def test_tokenizer_called_with_classification_args(self) -> None:
        # check_safety now calls the tokenizer twice for short inputs:
        # once with the raw text (length check, no special tokens) and once
        # with [text] in the standard classification shape. Verify the
        # classification call is present.
        model, tokenizer = self._mocks([[5.0, -5.0]])
        check_safety("hello world", model, tokenizer)
        tokenizer.assert_any_call(
            ["hello world"], padding=True, truncation=True, return_tensors="pt"
        )

    def test_long_text_chunks_and_returns_max_score(self) -> None:
        # 1099 tokens → 3 chunks (510, 510, 79). Aggregation must surface
        # the toxic middle chunk, not average it away.
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([list(range(1, 1100))])}
        tokenizer.decode.return_value = "chunk text"
        safe = MagicMock(logits=torch.tensor([[5.0, -5.0]]))
        toxic = MagicMock(logits=torch.tensor([[-5.0, 5.0]]))
        model = MagicMock(side_effect=[safe, toxic, safe])

        is_toxic, score = check_safety("long text", model, tokenizer)
        assert is_toxic is True
        assert score > 0.5
        assert model.call_count == 3


class TestAggregateSegmentSafety:
    @staticmethod
    def _mocks(*score_pairs: tuple[float, float]) -> tuple[MagicMock, MagicMock]:
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        responses = [MagicMock(logits=torch.tensor([list(p)])) for p in score_pairs]
        return MagicMock(side_effect=responses), tokenizer

    def test_empty_list_returns_safe_zero(self) -> None:
        tokenizer = MagicMock()
        model = MagicMock()
        assert _aggregate_segment_safety([], model, tokenizer) == (False, 0.0)
        tokenizer.assert_not_called()
        model.assert_not_called()

    def test_skips_whitespace_only_segments(self) -> None:
        model, tokenizer = self._mocks((5.0, -5.0))
        is_toxic, _ = _aggregate_segment_safety(
            ["", "   ", "real text"], model, tokenizer
        )
        assert is_toxic is False
        assert model.call_count == 1

    def test_returns_max_probability(self) -> None:
        model, tokenizer = self._mocks((5.0, -5.0), (-5.0, 5.0), (5.0, -5.0))
        is_toxic, score = _aggregate_segment_safety(["a", "b", "c"], model, tokenizer)
        assert is_toxic is True
        assert score > 0.5


# ---------------------------------------------------------------------------
# Transcription primitive
# ---------------------------------------------------------------------------


class TestTranscribeAudio:
    def test_returns_the_generated_text(self) -> None:
        model = MagicMock(spec=["generate"])
        model.generate.return_value = MagicMock(text="transcribed text")
        assert transcribe_audio(torch.zeros(1, 16000), model) == "transcribed text"

    def test_audio_is_the_squeezed_waveform_as_numpy(self) -> None:
        model = MagicMock(spec=["generate"])
        model.generate.return_value = MagicMock(text="text")
        wav = torch.randn(1, 16000)
        transcribe_audio(wav, model)
        audio = model.generate.call_args.kwargs["audio"]
        # mlx_audio takes a 1-D numpy array, not a (1, N) torch tensor.
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (16000,)
        assert np.array_equal(audio, wav.squeeze().numpy())

    def test_passes_nothing_but_the_audio(self) -> None:
        # Greedy CTC takes no prompt and has no token budget. generate()'s
        # **kwargs would swallow a stray keyword silently rather than reject it,
        # so the exact kwargs set is pinned here: `audio` and nothing else.
        model = MagicMock(spec=["generate"])
        model.generate.return_value = MagicMock(text="text")
        transcribe_audio(torch.zeros(1, 16000), model)
        model.generate.assert_called_once()
        assert model.generate.call_args.args == ()
        assert set(model.generate.call_args.kwargs) == {"audio"}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

SEGMENT_9S = [{"start": 0.0, "end": 9.0}]
SEGMENTS_3S = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]
CLASSIFICATION_KWARGS = {"padding": True, "truncation": True, "return_tensors": "pt"}


class TestRunPipeline:
    @pytest.fixture(autouse=True)
    def _patch_default_segments(self) -> Iterator[None]:
        with patch("streamlit_app.get_speech_segments", return_value=SEGMENT_9S):
            yield

    def test_returns_one_result_not_a_dict_of_them(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        # A single PipelineResult: with the one-model, one-language pipeline
        # there is nothing to key by, so the exact key set is the TypedDict's.
        result = _run_pipeline(torch.zeros(1, 160000), *pipeline_mocks)
        assert set(result) == {"transcript", "is_toxic", "toxicity_score"}

    def test_single_segment_line_format(self, pipeline_mocks: PipelineMocks) -> None:
        result = _run_pipeline(torch.zeros(1, 160000), *pipeline_mocks)
        assert result["transcript"] == "[0:00 - 0:09] decoded text"

    def test_multi_segment_lines_are_timestamped_and_newline_joined(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        pipeline_mocks.model.generate.side_effect = [
            MagicMock(text="first"),
            MagicMock(text="second"),
        ]
        with patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S):
            result = _run_pipeline(torch.zeros(1, 48000), *pipeline_mocks)
        assert result["transcript"] == "[0:00 - 0:01] first\n[0:01 - 0:03] second"

    def test_one_inference_per_segment_on_that_segment_slice(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        with patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S):
            _run_pipeline(torch.zeros(1, 48000), *pipeline_mocks)
        calls = pipeline_mocks.model.generate.call_args_list
        # 1.5s slices at 16 kHz, one call per segment: the encoder pass is the
        # whole call, so there is nothing to share between segments.
        assert [c.kwargs["audio"].shape for c in calls] == [(24000,), (24000,)]

    def test_segmentation_on_passes_the_vad_model_to_the_segmenter(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        wav = torch.zeros(1, 160000)
        with patch(
            "streamlit_app.get_speech_segments", return_value=SEGMENT_9S
        ) as mock_segments:
            _run_pipeline(wav, *pipeline_mocks)
        mock_segments.assert_called_once_with(wav, pipeline_mocks.vad)

    def test_segmentation_on_requires_a_vad_model(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        with pytest.raises(AssertionError, match="vad_model required"):
            _run_pipeline(
                torch.zeros(1, 160000),
                pipeline_mocks.model,
                None,
                pipeline_mocks.guardian,
                pipeline_mocks.guardian_tokenizer,
                use_segmentation=True,
            )

    def test_segmentation_off_skips_vad(self, pipeline_mocks: PipelineMocks) -> None:
        with patch("streamlit_app.get_speech_segments") as mock_vad:
            _run_pipeline(
                torch.zeros(1, 16000),
                pipeline_mocks.model,
                None,
                pipeline_mocks.guardian,
                pipeline_mocks.guardian_tokenizer,
                use_segmentation=False,
            )
        mock_vad.assert_not_called()

    def test_segmentation_off_uses_full_audio(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        result = _run_pipeline(
            torch.zeros(1, 48000),  # 3 seconds at 16 kHz
            pipeline_mocks.model,
            None,
            pipeline_mocks.guardian,
            pipeline_mocks.guardian_tokenizer,
            use_segmentation=False,
        )
        pipeline_mocks.model.generate.assert_called_once()
        assert pipeline_mocks.model.generate.call_args.kwargs["audio"].shape == (48000,)
        assert result["transcript"] == "[0:00 - 0:03] decoded text"

    def test_progress_sequence_ends_with_the_safety_pass(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        calls: list[tuple[int, int, str]] = []
        with patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S):
            _run_pipeline(
                torch.zeros(1, 48000),
                *pipeline_mocks,
                on_progress=lambda i, total, label: calls.append((i, total, label)),
            )
        # Fires before each unit of work, so the label names what is in flight;
        # the closing (total, total) call is what carries the bar to full while
        # the guardian pass runs, rather than leaving it frozen at (total-1)/total
        # under the segment that already finished.
        assert calls == [
            (0, 2, "segment 1 of 2"),
            (1, 2, "segment 2 of 2"),
            (2, 2, "safety check"),
        ]

    def test_progress_final_label_when_no_safety_pass(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        calls: list[tuple[int, int, str]] = []
        with patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S):
            _run_pipeline(
                torch.zeros(1, 48000),
                pipeline_mocks.model,
                pipeline_mocks.vad,
                None,
                None,
                on_progress=lambda i, total, label: calls.append((i, total, label)),
            )
        assert calls == [
            (0, 2, "segment 1 of 2"),
            (1, 2, "segment 2 of 2"),
            (2, 2, "results"),
        ]

    def test_safety_fields_when_guardian_is_passed(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        result = _run_pipeline(torch.zeros(1, 160000), *pipeline_mocks)
        assert result["is_toxic"] is False
        assert "toxicity_score" in result

    def test_toxic_content_flagged(self, pipeline_mocks: PipelineMocks) -> None:
        pipeline_mocks.guardian.return_value.logits = torch.tensor([[-5.0, 5.0]])
        result = _run_pipeline(torch.zeros(1, 160000), *pipeline_mocks)
        assert result["is_toxic"] is True
        assert result["toxicity_score"] > 0.5

    def test_safety_check_receives_transcript(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        _run_pipeline(torch.zeros(1, 160000), *pipeline_mocks)
        calls = classification_calls(pipeline_mocks.guardian_tokenizer)
        assert len(calls) == 1
        assert calls[0].args == (["decoded text"],)
        assert calls[0].kwargs == CLASSIFICATION_KWARGS

    @pytest.mark.parametrize(
        "with_model, with_tokenizer",
        [
            pytest.param(True, False, id="model_only"),
            pytest.param(False, True, id="tokenizer_only"),
            pytest.param(False, False, id="neither"),
        ],
    )
    def test_guardian_runs_only_with_both_model_and_tokenizer(
        self, pipeline_mocks: PipelineMocks, with_model: bool, with_tokenizer: bool
    ) -> None:
        # There is no use_toxicity_check argument: main() simply does not load
        # the guardian when the toggle is off, and the pipeline treats a missing
        # half as "off" rather than calling into None.
        calls: list[tuple[int, int, str]] = []
        result = _run_pipeline(
            torch.zeros(1, 160000),
            pipeline_mocks.model,
            pipeline_mocks.vad,
            pipeline_mocks.guardian if with_model else None,
            pipeline_mocks.guardian_tokenizer if with_tokenizer else None,
            on_progress=lambda i, total, label: calls.append((i, total, label)),
        )
        assert set(result) == {"transcript"}
        assert calls[-1] == (1, 1, "results")
        pipeline_mocks.guardian.assert_not_called()
        pipeline_mocks.guardian_tokenizer.assert_not_called()

    def test_no_safety_keys_without_guardian(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        result = _run_pipeline(
            torch.zeros(1, 160000), pipeline_mocks.model, pipeline_mocks.vad, None, None
        )
        assert "is_toxic" not in result
        assert "toxicity_score" not in result

    def test_multi_segment_safety_runs_per_segment(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        with patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S):
            _run_pipeline(torch.zeros(1, 48000), *pipeline_mocks)
        calls = classification_calls(pipeline_mocks.guardian_tokenizer)
        assert len(calls) == 2
        for call in calls:
            assert call.args == (["decoded text"],)
            assert call.kwargs == CLASSIFICATION_KWARGS

    def test_multi_segment_safety_reports_max_score(self) -> None:
        # A single toxic segment must flag the whole transcript, even when
        # surrounded by safe segments — i.e., aggregation is max, not mean.
        model = MagicMock(spec=["generate"])
        model.generate.return_value = MagicMock(text="decoded text")
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        safe = MagicMock(logits=torch.tensor([[5.0, -5.0]]))
        toxic = MagicMock(logits=torch.tensor([[-5.0, 5.0]]))
        guardian = MagicMock(side_effect=[safe, toxic, safe])
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 2.0},
            {"start": 2.0, "end": 3.0},
        ]
        with patch("streamlit_app.get_speech_segments", return_value=segments):
            result = _run_pipeline(
                torch.zeros(1, 48000), model, MagicMock(), guardian, tokenizer
            )
        assert result["is_toxic"] is True
        assert result["toxicity_score"] > 0.5

    def test_safety_skips_empty_segment_text(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        # If the model returns "" for a segment, the guardian shouldn't be
        # called for it — saves an inference and avoids spurious scores.
        pipeline_mocks.model.generate.side_effect = [
            MagicMock(text=""),
            MagicMock(text="real transcript"),
        ]
        with patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S):
            _run_pipeline(torch.zeros(1, 48000), *pipeline_mocks)
        calls = classification_calls(pipeline_mocks.guardian_tokenizer)
        assert len(calls) == 1
        assert calls[0].args == (["real transcript"],)


# ---------------------------------------------------------------------------
# Granite Speech 5 contract
# ---------------------------------------------------------------------------


class TestGraniteSpeech5Contract:
    """Exercises the real installed mlx_audio surface the speech path relies on.

    Everywhere else the model is a `spec=["generate"]` mock, so without this
    class nothing checks that the installed mlx_audio still registers the
    model type, that generate() still takes `audio`, or that load_model still
    has a `strict` parameter to receive strict=True. The app no longer reaches
    into any private speech-model internals, so this is the whole contract.
    """

    def test_model_type_is_registered(self) -> None:
        # This is what the mlx-audio>=0.5.1 floor in pyproject.toml promises:
        # the type is absent in 0.5.0 and the 0.4.x line, where the load would
        # fail with an unsupported-model error rather than an ImportError.
        from mlx_audio.stt.utils import MODEL_REMAPPING

        assert "granite_speech5_ctc" in MODEL_REMAPPING

    def test_generate_takes_audio(self) -> None:
        from mlx_audio.stt.models.granite_speech5_ctc import Model

        params = inspect.signature(Model.generate).parameters
        assert "audio" in params
        # Whether generate() has **kwargs is not asserted: the app passes only
        # `audio` (pinned in TestTranscribeAudio), so a stricter upstream
        # signature is harmless.

    def test_load_model_has_a_named_strict_parameter(self) -> None:
        # strict=True is only meaningful if load_model binds it by name. Were
        # it to fall into **kwargs it would be forwarded (or dropped) silently
        # and the non-strict default would quietly come back.
        from mlx_audio.stt.utils import load_model as mlx_load_model

        strict = inspect.signature(mlx_load_model).parameters.get("strict")
        assert strict is not None
        assert strict.kind is not inspect.Parameter.VAR_KEYWORD

    def test_transcribes_the_fixture_with_real_weights(self) -> None:
        """Real weights on real speech, through the app's own primitives.

        Greedy CTC on a pinned revision is deterministic, so the assertions are
        stable. Runs only where the pinned snapshot is already in the local HF
        cache, which CI is not, so it skips there. The gate is a deliberate
        choice rather than a try/except around the load: mlx_audio's loader
        calls snapshot_download on a cache miss and does not raise, so on a
        networked runner the load *succeeds* by fetching 0.95 GB — per matrix
        leg, per run, with no cache step to amortise it. The try/except below
        stays for the other case, a cached snapshot that no longer loads.
        `huggingface_hub` is a transitive import (via mlx_audio/transformers),
        the same category as the undeclared `mlx` import CLAUDE.md documents.

        Fixture quirk, verified against the reference transformers
        implementation (byte-identical): sample_10s.wav opens with 0.76s of
        exact digital-zero silence, and on the WHOLE clip the model returns
        'shapeare on scenery by oscar wilde public' — it drops the second
        sentence ("this is a librivox recording all librivox recordings are in
        the public domain"). Through the VAD path, which trims the lead-in, the
        second sentence comes back. On normal audio word counts match, so this
        is fixture-specific; do not "fix" the assertion below to expect the
        full sentence on the whole clip.
        """
        from huggingface_hub import try_to_load_from_cache

        # A str path means the pinned snapshot holds the file; None or the
        # _CACHED_NO_EXIST sentinel means it does not. With a full commit hash
        # as the revision this lookup never touches the network.
        if not isinstance(
            try_to_load_from_cache(
                MODEL_ID, "model.safetensors", revision=MODEL_REVISION
            ),
            str,
        ):
            pytest.skip(
                f"{MODEL_ID} not in the local HF cache; not downloading 0.95 GB"
            )
        try:
            model = _load_model(MODEL_ID, MODEL_REVISION)
        except Exception as e:  # noqa: BLE001 - any failure means skip
            pytest.skip(f"{MODEL_ID} unavailable: {e}")

        wav = load_and_preprocess_audio(make_upload(AUDIO_DIR / "sample_10s.wav"))
        text = transcribe_audio(wav, model)
        assert text
        # The training transcripts were normalised to lowercase, unpunctuated.
        assert text == text.lower()
        assert "oscar wilde" in text


# ---------------------------------------------------------------------------
# Result card rendering
# ---------------------------------------------------------------------------


@patch("streamlit_app.st")
class TestRenderResultCard:
    def test_renders_text(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {"transcript": "hello"}
        _render_result_card(result, "test")
        mock_st.text.assert_called_once_with("hello")

    def test_card_title_is_transcription(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {"transcript": "hello"}
        _render_result_card(result, "test")
        mock_st.subheader.assert_called_once_with("Transcription")

    def test_shows_safe_banner(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {
            "transcript": "hello",
            "is_toxic": False,
            "toxicity_score": 0.1,
        }
        _render_result_card(result, "test")
        msg = mock_st.success.call_args[0][0]
        assert "safe" in msg.lower()
        assert "10.0%" in msg
        assert mock_st.success.call_args.kwargs["icon"] == ":material/check_circle:"
        mock_st.warning.assert_not_called()

    def test_shows_toxic_banner(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {
            "transcript": "bad",
            "is_toxic": True,
            "toxicity_score": 0.9,
        }
        _render_result_card(result, "test")
        msg = mock_st.warning.call_args[0][0]
        assert "toxic" in msg.lower()
        assert "90.0%" in msg
        assert mock_st.warning.call_args.kwargs["icon"] == ":material/warning:"
        mock_st.success.assert_not_called()

    def test_no_safety_banner_without_toxic_field(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {"transcript": "hello"}
        _render_result_card(result, "test")
        mock_st.success.assert_not_called()
        mock_st.warning.assert_not_called()

    def test_download_button_offers_the_transcript_as_text(
        self, mock_st: MagicMock
    ) -> None:
        result: PipelineResult = {"transcript": "hello world"}
        _render_result_card(result, "audio")
        args, kwargs = mock_st.download_button.call_args
        assert args == ("", "hello world", "audio_transcription.txt", "text/plain")
        assert kwargs["key"] == "dl_txt"
        assert kwargs["help"] == "Download transcription"
        assert kwargs["icon"] == ":material/download:"

    def test_renders_bordered_container_without_a_stretch_height(
        self, mock_st: MagicMock
    ) -> None:
        result: PipelineResult = {"transcript": "hello"}
        _render_result_card(result, "test")
        mock_st.container.assert_called_once_with(border=True)
        # height="stretch" existed to equalise cards across a row of the result
        # grid; with a single card there is no row to equalise.
        assert "height" not in mock_st.container.call_args.kwargs
