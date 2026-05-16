from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from streamlit_app import (
    EN_TARGETS,
    GUARDIAN_MODEL_ID,
    MODEL_ID,
    SOURCE_LANGUAGES,
    SUPPORTED_FORMATS,
    TRANSCRIBE_PROMPT,
    VIDEO_FORMATS,
    _render_result_card,
    build_tasks,
    check_safety,
    format_timestamp,
    get_speech_segments,
    is_video,
    load_and_preprocess_audio,
    load_guardian_model,
    load_model,
    load_vad_model,
    produces_english,
    result_slug,
    result_title,
    run_pipeline,
    silero_vad,
    transcribe_audio,
)

AUDIO_DIR = Path(__file__).parent / "data" / "audio"


class TestModelIds:
    def test_model_id(self) -> None:
        assert MODEL_ID == "mlx-community/granite-4.0-1b-speech-8bit"

    def test_guardian_model_id(self) -> None:
        assert GUARDIAN_MODEL_ID == "ibm-granite/granite-guardian-hap-38m"


class TestSourceLanguages:
    def test_contains_english_first(self) -> None:
        assert SOURCE_LANGUAGES[0] == "English"

    def test_contains_model_asr_languages(self) -> None:
        assert set(SOURCE_LANGUAGES) == {
            "English",
            "French",
            "German",
            "Spanish",
            "Portuguese",
            "Japanese",
        }


class TestEnTargets:
    def test_contains_seven_targets(self) -> None:
        assert len(EN_TARGETS) == 7

    def test_excludes_english(self) -> None:
        assert "English" not in EN_TARGETS

    def test_contains_model_supported_targets(self) -> None:
        assert set(EN_TARGETS) == {
            "French",
            "German",
            "Spanish",
            "Portuguese",
            "Italian",
            "Japanese",
            "Mandarin Chinese",
        }


class TestBuildTasks:
    def test_english_source_includes_transcribe(self) -> None:
        tasks = build_tasks("English")
        assert "Transcribe" in tasks
        assert tasks["Transcribe"] == TRANSCRIBE_PROMPT

    def test_english_source_has_seven_translation_targets(self) -> None:
        tasks = build_tasks("English")
        translation_keys = [k for k in tasks if k != "Transcribe"]
        assert set(translation_keys) == set(EN_TARGETS)

    def test_english_source_excludes_english_target(self) -> None:
        tasks = build_tasks("English")
        assert "English" not in tasks

    def test_non_english_source_has_two_tasks(self) -> None:
        for source in ("French", "German", "Spanish", "Portuguese", "Japanese"):
            tasks = build_tasks(source)
            assert set(tasks.keys()) == {"Transcribe", "English"}

    def test_translation_prompts_have_correct_prefix(self) -> None:
        for source in SOURCE_LANGUAGES:
            for task, prompt in build_tasks(source).items():
                if task != "Transcribe":
                    assert prompt == f"translate the speech to {task}"

    def test_preserves_insertion_order(self) -> None:
        tasks = build_tasks("English")
        assert list(tasks.keys())[0] == "Transcribe"


class TestProducesEnglish:
    def test_english_transcribe_is_english(self) -> None:
        assert produces_english("English", "Transcribe") is True

    def test_non_english_transcribe_is_not_english(self) -> None:
        for source in ("French", "German", "Spanish", "Portuguese", "Japanese"):
            assert produces_english(source, "Transcribe") is False

    def test_translation_to_english_is_english(self) -> None:
        for source in ("French", "German", "Spanish", "Portuguese", "Japanese"):
            assert produces_english(source, "English") is True

    def test_translation_to_non_english_is_not_english(self) -> None:
        for target in EN_TARGETS:
            assert produces_english("English", target) is False


class TestResultTitle:
    def test_english_transcribe(self) -> None:
        assert result_title("English", "Transcribe") == "Transcribe (English)"

    def test_non_english_transcribe_shows_source(self) -> None:
        assert result_title("German", "Transcribe") == "Transcribe (German)"
        assert result_title("Japanese", "Transcribe") == "Transcribe (Japanese)"

    def test_translation_uses_target_as_title(self) -> None:
        assert result_title("English", "French") == "French"
        assert result_title("German", "English") == "English"


class TestResultSlug:
    def test_transcribe_includes_source(self) -> None:
        assert result_slug("English", "Transcribe") == "transcribe_english"
        assert result_slug("German", "Transcribe") == "transcribe_german"

    def test_translation_uses_target_slug(self) -> None:
        assert result_slug("English", "French") == "french"
        assert result_slug("French", "English") == "english"

    def test_multi_word_target_is_snake_cased(self) -> None:
        assert result_slug("English", "Mandarin Chinese") == "mandarin_chinese"


class TestSupportedFormats:
    def test_all_formats_present(self) -> None:
        expected = {
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
        assert set(SUPPORTED_FORMATS) == expected

    def test_video_formats_are_subset_of_supported(self) -> None:
        assert VIDEO_FORMATS.issubset(set(SUPPORTED_FORMATS))


class TestIsVideo:
    def test_video_extensions_return_true(self) -> None:
        for ext in ("mp4", "mov", "webm", "mkv"):
            assert is_video(f"clip.{ext}") is True

    def test_audio_extensions_return_false(self) -> None:
        for ext in ("wav", "flac", "m4a", "mp3", "ogg", "aac"):
            assert is_video(f"sound.{ext}") is False

    def test_case_insensitive(self) -> None:
        assert is_video("CLIP.MP4") is True
        assert is_video("Sound.WAV") is False

    def test_no_extension_returns_false(self) -> None:
        assert is_video("filename") is False

    def test_path_with_directories(self) -> None:
        assert is_video("recordings/2026/clip.mp4") is True
        assert is_video("recordings/2026/sound.wav") is False


class TestFormatTimestamp:
    def test_zero_seconds(self) -> None:
        assert format_timestamp(0.0) == "0:00"

    def test_seconds_only(self) -> None:
        assert format_timestamp(15.0) == "0:15"

    def test_minutes_and_seconds(self) -> None:
        assert format_timestamp(62.0) == "1:02"

    def test_hours(self) -> None:
        assert format_timestamp(3661.0) == "1:01:01"

    def test_fractional_seconds_truncated(self) -> None:
        assert format_timestamp(15.7) == "0:15"


class TestSileroVad:
    def test_returns_tuples_in_seconds(self) -> None:
        model = MagicMock()
        mock_timestamps = [
            {"start": 16000, "end": 48000},
            {"start": 64000, "end": 96000},
        ]
        with patch("streamlit_app.get_speech_timestamps", return_value=mock_timestamps):
            result = silero_vad(torch.zeros(1, 160000), model)
        assert result == [(1.0, 3.0), (4.0, 6.0)]

    def test_empty_audio_returns_empty_list(self) -> None:
        model = MagicMock()
        with patch("streamlit_app.get_speech_timestamps", return_value=[]):
            result = silero_vad(torch.zeros(1, 16000), model)
        assert result == []

    def test_passes_model_and_sample_rate(self) -> None:
        model = MagicMock()
        wav = torch.zeros(1, 16000)
        with patch("streamlit_app.get_speech_timestamps", return_value=[]) as mock_fn:
            silero_vad(wav, model)
        mock_fn.assert_called_once()
        call_args = mock_fn.call_args
        assert torch.equal(call_args[0][0], wav.squeeze())
        assert call_args[0][1] is model
        assert call_args[1]["sampling_rate"] == 16000


class TestGetSpeechSegments:
    def test_adds_start_buffer(self) -> None:
        model = MagicMock()
        with patch("streamlit_app.silero_vad", return_value=[(1.0, 2.0)]):
            result = get_speech_segments(torch.zeros(1, 160000), model)
        assert result[0]["start"] == pytest.approx(0.7)

    def test_adds_end_buffer(self) -> None:
        model = MagicMock()
        with patch("streamlit_app.silero_vad", return_value=[(1.0, 2.0)]):
            result = get_speech_segments(torch.zeros(1, 160000), model)
        assert result[0]["end"] == pytest.approx(2.3)

    def test_clamps_start_buffer_to_zero(self) -> None:
        model = MagicMock()
        with patch("streamlit_app.silero_vad", return_value=[(0.1, 1.0)]):
            result = get_speech_segments(torch.zeros(1, 160000), model)
        assert result[0]["start"] == 0.0

    def test_clamps_end_buffer_to_duration(self) -> None:
        model = MagicMock()
        wav = torch.zeros(1, 32000)  # 2 seconds at 16kHz
        with patch("streamlit_app.silero_vad", return_value=[(0.5, 1.9)]):
            result = get_speech_segments(wav, model)
        assert result[0]["end"] == 2.0

    def test_merges_close_segments(self) -> None:
        model = MagicMock()
        with patch(
            "streamlit_app.silero_vad",
            return_value=[(1.0, 2.0), (2.3, 3.0)],
        ):
            result = get_speech_segments(torch.zeros(1, 160000), model)
        assert len(result) == 1
        assert result[0]["start"] == pytest.approx(0.7)
        assert result[0]["end"] == pytest.approx(3.3)

    def test_keeps_distant_segments_separate(self) -> None:
        model = MagicMock()
        with patch(
            "streamlit_app.silero_vad",
            return_value=[(1.0, 2.0), (5.0, 6.0)],
        ):
            result = get_speech_segments(torch.zeros(1, 160000), model)
        assert len(result) == 2

    def test_no_speech_falls_back_to_full_audio(self) -> None:
        model = MagicMock()
        wav = torch.zeros(1, 160000)  # 10 seconds at 16kHz
        with patch("streamlit_app.silero_vad", return_value=[]):
            result = get_speech_segments(wav, model)
        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 10.0


class TestLoadAndPreprocessAudio:
    def _make_upload(self, path: Path) -> MagicMock:
        upload = MagicMock()
        upload.name = path.name
        upload.getvalue.return_value = path.read_bytes()
        return upload

    def test_loads_wav(self) -> None:
        audio_file = self._make_upload(AUDIO_DIR / "sample_10s.wav")
        wav, duration = load_and_preprocess_audio(audio_file)
        assert isinstance(wav, torch.Tensor)
        assert wav.shape[0] == 1
        assert duration > 0

    def test_loads_mp4_video(self) -> None:
        audio_file = self._make_upload(AUDIO_DIR / "sample_10s_video.mp4")
        wav, duration = load_and_preprocess_audio(audio_file)
        assert isinstance(wav, torch.Tensor)
        assert wav.shape[0] == 1
        assert duration > 0

    def test_invalid_audio_raises_runtime_error(self) -> None:
        upload = MagicMock()
        upload.name = "bad.wav"
        upload.getvalue.return_value = b"not audio data"
        with pytest.raises(RuntimeError, match="Failed to load audio file"):
            load_and_preprocess_audio(upload)


class TestLoadModel:
    @patch("streamlit_app._load_stt_model")
    @patch("streamlit_app.st")
    def test_calls_load_stt_model(
        self,
        _mock_st: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        load_model.__wrapped__("test-model")  # type: ignore[attr-defined]
        mock_load.assert_called_once_with("test-model")

    @patch("streamlit_app._load_stt_model")
    @patch("streamlit_app.st")
    def test_returns_model(
        self,
        _mock_st: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        result = load_model.__wrapped__("test-model")  # type: ignore[attr-defined]
        assert result == mock_load.return_value


class TestLoadGuardianModel:
    @patch("streamlit_app.AutoModelForSequenceClassification")
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.st")
    def test_loads_model_and_tokenizer(
        self,
        _mock_st: MagicMock,
        mock_tokenizer_cls: MagicMock,
        mock_model_cls: MagicMock,
    ) -> None:
        result = load_guardian_model.__wrapped__("test-model")  # type: ignore[attr-defined]
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("test-model")
        mock_model_cls.from_pretrained.assert_called_once_with("test-model")
        assert result == (
            mock_model_cls.from_pretrained.return_value,
            mock_tokenizer_cls.from_pretrained.return_value,
        )


class TestLoadVadModel:
    @patch("streamlit_app.load_silero_vad")
    @patch("streamlit_app.st")
    def test_loads_model(
        self,
        _mock_st: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        result = load_vad_model.__wrapped__()  # type: ignore[attr-defined]
        mock_load.assert_called_once()
        assert result == mock_load.return_value


class TestCheckSafety:
    def _make_mocks(
        self, logits_values: list[list[float]]
    ) -> tuple[MagicMock, MagicMock]:
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        model = MagicMock()
        model.return_value.logits = torch.tensor(logits_values)
        return model, tokenizer

    def test_safe_content(self) -> None:
        model, tokenizer = self._make_mocks([[5.0, -5.0]])
        is_toxic, score = check_safety("safe text", model, tokenizer)
        assert is_toxic is False
        assert score < 0.5

    def test_toxic_content(self) -> None:
        model, tokenizer = self._make_mocks([[-5.0, 5.0]])
        is_toxic, score = check_safety("toxic text", model, tokenizer)
        assert is_toxic is True
        assert score > 0.5

    def test_boundary_is_not_toxic(self) -> None:
        model, tokenizer = self._make_mocks([[0.0, 0.0]])
        is_toxic, score = check_safety("text", model, tokenizer)
        assert score == 0.5
        assert is_toxic is False

    def test_tokenizer_called_with_correct_args(self) -> None:
        model, tokenizer = self._make_mocks([[5.0, -5.0]])
        check_safety("hello world", model, tokenizer)
        tokenizer.assert_called_once_with(
            ["hello world"], padding=True, truncation=True, return_tensors="pt"
        )


class TestTranscribeAudio:
    def test_calls_model_generate(self) -> None:
        model = MagicMock()
        model.generate.return_value = MagicMock(text="transcribed text")
        wav = torch.zeros(1, 16000)

        transcript = transcribe_audio(wav, "test prompt", model)

        model.generate.assert_called_once()
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["prompt"] == "test prompt"
        assert call_kwargs["max_tokens"] == 512
        assert transcript == "transcribed text"

    def test_audio_passed_as_squeezed_numpy(self) -> None:
        model = MagicMock()
        model.generate.return_value = MagicMock(text="text")
        wav = torch.randn(1, 16000)

        transcribe_audio(wav, "prompt", model)

        call_kwargs = model.generate.call_args[1]
        audio = call_kwargs["audio"]
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (16000,)


class TestRunPipeline:
    def setup_method(self) -> None:
        self._segments_patcher = patch(
            "streamlit_app.get_speech_segments",
            return_value=[{"start": 0.0, "end": 1.0}],
        )
        self._segments_patcher.start()

    def teardown_method(self) -> None:
        self._segments_patcher.stop()

    def _make_mocks(
        self,
    ) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
        model = MagicMock()
        model.generate.return_value = MagicMock(text="decoded text")
        guardian_tokenizer = MagicMock()
        guardian_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        guardian_model = MagicMock()
        guardian_model.return_value.logits = torch.tensor([[5.0, -5.0]])
        vad_model = MagicMock()
        return model, vad_model, guardian_model, guardian_tokenizer

    def test_returns_dict_keyed_by_task(self) -> None:
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)
        tasks = {"Transcribe": "transcribe prompt", "French": "translate to French"}

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            tasks,
            {"Transcribe"},
            model,
            vad_model,
            guardian_model,
            guardian_tokenizer,
        )

        assert set(results.keys()) == {"Transcribe", "French"}

    def test_each_result_has_transcript(self) -> None:
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            {"Transcribe": "p"},
            {"Transcribe"},
            model,
            vad_model,
            guardian_model,
            guardian_tokenizer,
        )

        assert "transcript" in results["Transcribe"]

    def test_empty_tasks_returns_empty_dict(self) -> None:
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav, {}, set(), model, vad_model, guardian_model, guardian_tokenizer
        )

        assert results == {}

    def test_uses_correct_prompt_per_task(self) -> None:
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)
        tasks = {
            "Transcribe": "transcribe prompt",
            "French": "translate to French",
        }

        run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            tasks,
            {"Transcribe"},
            model,
            vad_model,
            guardian_model,
            guardian_tokenizer,
        )

        calls = model.generate.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["prompt"] == "transcribe prompt"
        assert calls[1][1]["prompt"] == "translate to French"

    def test_preserves_task_order(self) -> None:
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)
        tasks = {"Japanese": "p1", "Transcribe": "p2", "German": "p3"}

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            tasks,
            {"Transcribe"},
            model,
            vad_model,
            guardian_model,
            guardian_tokenizer,
        )

        assert list(results.keys()) == ["Japanese", "Transcribe", "German"]

    def test_each_result_has_safety_fields_when_in_safety_tasks(self) -> None:
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            {"Transcribe": "p"},
            {"Transcribe"},
            model,
            vad_model,
            guardian_model,
            guardian_tokenizer,
        )

        result = results["Transcribe"]
        assert "is_toxic" in result
        assert "toxicity_score" in result
        assert result["is_toxic"] is False

    def test_toxic_content_flagged(self) -> None:
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        guardian_model.return_value.logits = torch.tensor([[-5.0, 5.0]])
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            {"Transcribe": "p"},
            {"Transcribe"},
            model,
            vad_model,
            guardian_model,
            guardian_tokenizer,
        )

        result = results["Transcribe"]
        assert result["is_toxic"] is True
        assert result["toxicity_score"] > 0.5

    def test_safety_check_receives_transcript(self) -> None:
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)

        run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            {"Transcribe": "p"},
            {"Transcribe"},
            model,
            vad_model,
            guardian_model,
            guardian_tokenizer,
        )

        guardian_tokenizer.assert_called_once_with(
            ["decoded text"], padding=True, truncation=True, return_tensors="pt"
        )

    def test_tasks_not_in_safety_set_skip_safety_check(self) -> None:
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            {"French": "translate to French"},
            set(),
            model,
            vad_model,
            guardian_model,
            guardian_tokenizer,
        )

        result = results["French"]
        assert "is_toxic" not in result
        assert "toxicity_score" not in result
        guardian_tokenizer.assert_not_called()

    def test_mixed_tasks_safety_only_on_listed_tasks(self) -> None:
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            {"Transcribe": "p1", "French": "p2"},
            {"Transcribe"},
            model,
            vad_model,
            guardian_model,
            guardian_tokenizer,
        )

        assert "is_toxic" in results["Transcribe"]
        assert "toxicity_score" in results["Transcribe"]
        assert "is_toxic" not in results["French"]
        assert "toxicity_score" not in results["French"]

    def test_non_english_source_translation_to_english_gets_safety(self) -> None:
        # When source is non-English and task translates to English, the output
        # is English text and should be safety-checked.
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            {"English": "translate to English"},
            {"English"},
            model,
            vad_model,
            guardian_model,
            guardian_tokenizer,
        )

        assert "is_toxic" in results["English"]
        assert "toxicity_score" in results["English"]

    def test_multi_segment_transcript_has_timestamps(self) -> None:
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 48000)  # 3 seconds
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                {"Transcribe": "p"},
                {"Transcribe"},
                model,
                vad_model,
                guardian_model,
                guardian_tokenizer,
            )

        transcript = results["Transcribe"]["transcript"]
        assert "[0:00 - 0:01]" in transcript
        assert "[0:01 - 0:03]" in transcript
        assert "\n" in transcript

    def test_multi_segment_safety_on_joined_text_without_timestamps(self) -> None:
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 48000)
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                {"Transcribe": "p"},
                {"Transcribe"},
                model,
                vad_model,
                guardian_model,
                guardian_tokenizer,
            )

        # Guardian receives joined text without timestamps
        guardian_tokenizer.assert_called_once_with(
            ["decoded text decoded text"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def test_translation_skips_safety_when_not_listed(self) -> None:
        model, vad_model, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 48000)
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                {"French": "translate to French"},
                set(),
                model,
                vad_model,
                guardian_model,
                guardian_tokenizer,
            )

        guardian_tokenizer.assert_not_called()
        assert "is_toxic" not in results["French"]

    def test_segmentation_off_skips_vad(self) -> None:
        model, _, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)

        with patch("streamlit_app.get_speech_segments") as mock_vad:
            run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                {"Transcribe": "p"},
                {"Transcribe"},
                model,
                None,
                guardian_model,
                guardian_tokenizer,
                use_segmentation=False,
            )

        mock_vad.assert_not_called()

    def test_segmentation_off_uses_full_audio_as_single_segment(self) -> None:
        model, _, guardian_model, guardian_tokenizer = self._make_mocks()
        wav = torch.zeros(1, 48000)  # 3 seconds at 16kHz

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            {"Transcribe": "p"},
            {"Transcribe"},
            model,
            None,
            guardian_model,
            guardian_tokenizer,
            use_segmentation=False,
        )

        assert model.generate.call_count == 1
        assert "[0:00 - 0:03]" in results["Transcribe"]["transcript"]


class TestRenderResultCard:
    @patch("streamlit_app.st")
    def test_renders_text(self, mock_st: MagicMock) -> None:
        result = {"transcript": "hello"}
        _render_result_card("English", "Transcribe", result, "test")
        mock_st.text.assert_called_once_with("hello")

    @patch("streamlit_app.st")
    def test_shows_safe_banner(self, mock_st: MagicMock) -> None:
        result = {
            "transcript": "hello",
            "is_toxic": False,
            "toxicity_score": 0.1,
        }
        _render_result_card("English", "Transcribe", result, "test")
        mock_st.success.assert_called_once()
        assert "safe" in mock_st.success.call_args[0][0].lower()
        assert "10.0%" in mock_st.success.call_args[0][0]

    @patch("streamlit_app.st")
    def test_shows_toxic_banner(self, mock_st: MagicMock) -> None:
        result = {
            "transcript": "bad content",
            "is_toxic": True,
            "toxicity_score": 0.9,
        }
        _render_result_card("English", "Transcribe", result, "test")
        mock_st.warning.assert_called_once()
        assert "toxic" in mock_st.warning.call_args[0][0].lower()
        assert "90.0%" in mock_st.warning.call_args[0][0]

    @patch("streamlit_app.st")
    def test_no_safety_banner_for_translation(self, mock_st: MagicMock) -> None:
        result = {"transcript": "bonjour"}
        _render_result_card("English", "French", result, "test")
        mock_st.success.assert_not_called()
        mock_st.warning.assert_not_called()

    @patch("streamlit_app.st")
    def test_download_filename_for_translation(self, mock_st: MagicMock) -> None:
        result = {"transcript": "hello world"}
        _render_result_card("English", "Mandarin Chinese", result, "audio")
        txt_filename = mock_st.download_button.call_args[0][2]
        assert txt_filename == "audio_mandarin_chinese.txt"

    @patch("streamlit_app.st")
    def test_download_filename_for_transcription_includes_source(
        self, mock_st: MagicMock
    ) -> None:
        result = {"transcript": "hallo"}
        _render_result_card("German", "Transcribe", result, "audio")
        txt_filename = mock_st.download_button.call_args[0][2]
        assert txt_filename == "audio_transcribe_german.txt"

    @patch("streamlit_app.st")
    def test_download_tooltip_for_transcription(self, mock_st: MagicMock) -> None:
        result = {"transcript": "hello"}
        _render_result_card("English", "Transcribe", result, "test")
        assert mock_st.download_button.call_args[1]["help"] == "Download transcription"

    @patch("streamlit_app.st")
    def test_download_tooltip_for_translation(self, mock_st: MagicMock) -> None:
        result = {"transcript": "bonjour"}
        _render_result_card("English", "French", result, "test")
        assert mock_st.download_button.call_args[1]["help"] == "Download translation"

    @patch("streamlit_app.st")
    def test_renders_bordered_container(self, mock_st: MagicMock) -> None:
        result = {"transcript": "hello"}
        _render_result_card("English", "Transcribe", result, "test")
        mock_st.container.assert_called_once_with(border=True)

    @patch("streamlit_app.st")
    def test_transcription_card_title_includes_source(
        self, mock_st: MagicMock
    ) -> None:
        result = {"transcript": "hallo"}
        _render_result_card("German", "Transcribe", result, "test")
        mock_st.subheader.assert_called_once_with("Transcribe (German)")

    @patch("streamlit_app.st")
    def test_translation_card_title_is_target_language(
        self, mock_st: MagicMock
    ) -> None:
        result = {"transcript": "bonjour"}
        _render_result_card("English", "French", result, "test")
        mock_st.subheader.assert_called_once_with("French")
