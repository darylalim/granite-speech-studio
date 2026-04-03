import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from streamlit_app import (
    ENGLISH_TASKS,
    GUARDIAN_MODEL_ID,
    MODEL_ID,
    PUNCTUATION_MODEL_ID,
    PROMPT_CHOICES,
    SUPPORTED_FORMATS,
    TASK_PRESETS,
    _render_result_card,
    apply_punctuation,
    check_safety,
    format_timestamp,
    get_selected_tasks,
    get_speech_segments,
    load_and_preprocess_audio,
    load_guardian_model,
    load_model,
    load_punctuation_model,
    load_vad_model,
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

    def test_punctuation_model_id(self) -> None:
        assert PUNCTUATION_MODEL_ID == "pcs_en"


class TestPromptChoices:
    def test_includes_transcription(self) -> None:
        assert "Transcribe" in PROMPT_CHOICES
        assert "transcribe" in PROMPT_CHOICES["Transcribe"]

    def test_includes_translations(self) -> None:
        for lang in (
            "French",
            "German",
            "Spanish",
            "Portuguese",
            "Italian",
            "Japanese",
            "Mandarin Chinese",
            "English",
        ):
            assert lang in PROMPT_CHOICES
            assert lang in PROMPT_CHOICES[lang]

    def test_translation_prompt_prefixes(self) -> None:
        for key, value in PROMPT_CHOICES.items():
            if key != "Transcribe":
                assert value.startswith("translate the speech to ")

    def test_total_count(self) -> None:
        assert len(PROMPT_CHOICES) == 9

    def test_english_translation_not_in_english_tasks(self) -> None:
        assert "English" in PROMPT_CHOICES
        assert "English" not in ENGLISH_TASKS


class TestSupportedFormats:
    def test_all_formats_present(self) -> None:
        expected = {"wav", "mp3", "m4a", "ogg", "flac", "webm", "aac"}
        assert set(SUPPORTED_FORMATS) == expected


class TestTaskPresets:
    def test_all_tasks_preset(self) -> None:
        assert set(TASK_PRESETS["All Tasks"]) == set(PROMPT_CHOICES.keys())

    def test_european_preset(self) -> None:
        expected = {
            "Transcribe",
            "French",
            "German",
            "Spanish",
            "Portuguese",
            "Italian",
        }
        assert set(TASK_PRESETS["European Languages"]) == expected

    def test_asian_preset(self) -> None:
        expected = {"Transcribe", "Japanese", "Mandarin Chinese"}
        assert set(TASK_PRESETS["Asian Languages"]) == expected

    def test_transcribe_only_preset(self) -> None:
        assert TASK_PRESETS["Transcribe Only"] == ["Transcribe"]

    def test_all_preset_values_are_valid_prompt_keys(self) -> None:
        for preset_name, tasks in TASK_PRESETS.items():
            for task in tasks:
                assert task in PROMPT_CHOICES, (
                    f"{task} in preset '{preset_name}' not in PROMPT_CHOICES"
                )

    def test_preset_count(self) -> None:
        assert len(TASK_PRESETS) == 4


class TestGetSelectedTasks:
    def test_preset_returns_preset_tasks(self) -> None:
        result = get_selected_tasks("European Languages", [])
        assert result == TASK_PRESETS["European Languages"]

    def test_custom_returns_custom_tasks(self) -> None:
        result = get_selected_tasks(None, ["French", "Japanese"])
        assert result == ["French", "Japanese"]

    def test_preset_overrides_custom(self) -> None:
        result = get_selected_tasks("Asian Languages", ["French", "German"])
        assert result == TASK_PRESETS["Asian Languages"]

    def test_none_preset_empty_custom_returns_empty(self) -> None:
        result = get_selected_tasks(None, [])
        assert result == []


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

    def test_loads_mp3(self) -> None:
        audio_file = self._make_upload(AUDIO_DIR / "sample_10s.mp3")
        wav, duration = load_and_preprocess_audio(audio_file)
        assert isinstance(wav, torch.Tensor)
        assert wav.shape[0] == 1
        assert duration > 0

    def test_resamples_to_16khz(self) -> None:
        audio_file = self._make_upload(AUDIO_DIR / "sample_10s.mp3")
        wav, _ = load_and_preprocess_audio(audio_file)
        assert wav.shape[1] > 0

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


class TestLoadPunctuationModel:
    @patch("streamlit_app.PunctCapSegModelONNX")
    @patch("streamlit_app.st")
    def test_loads_model(
        self,
        _mock_st: MagicMock,
        mock_model_cls: MagicMock,
    ) -> None:
        result = load_punctuation_model.__wrapped__("pcs_en")  # type: ignore[attr-defined]
        mock_model_cls.from_pretrained.assert_called_once_with("pcs_en")
        assert result == mock_model_cls.from_pretrained.return_value


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


class TestApplyPunctuation:
    def test_calls_infer_with_list(self) -> None:
        model = MagicMock()
        model.infer.return_value = [["Hello world."]]
        result = apply_punctuation("hello world", model)
        model.infer.assert_called_once_with(["hello world"])
        assert result == "Hello world."

    def test_joins_multiple_sentences(self) -> None:
        model = MagicMock()
        model.infer.return_value = [["Hello.", "How are you?"]]
        result = apply_punctuation("hello how are you", model)
        assert result == "Hello. How are you?"

    def test_cleans_unk_tokens(self) -> None:
        model = MagicMock()
        model.infer.return_value = [["Hello <unk> world <Unk> test."]]
        result = apply_punctuation("hello world test", model)
        assert result == "Hello world test."

    def test_empty_string(self) -> None:
        model = MagicMock()
        model.infer.return_value = [[""]]
        result = apply_punctuation("", model)
        assert result == ""


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
        is_toxic, score = check_safety.__wrapped__(  # type: ignore[attr-defined]
            "safe text", model, tokenizer
        )
        assert is_toxic is False
        assert score < 0.5

    def test_toxic_content(self) -> None:
        model, tokenizer = self._make_mocks([[-5.0, 5.0]])
        is_toxic, score = check_safety.__wrapped__(  # type: ignore[attr-defined]
            "toxic text", model, tokenizer
        )
        assert is_toxic is True
        assert score > 0.5

    def test_boundary_is_not_toxic(self) -> None:
        model, tokenizer = self._make_mocks([[0.0, 0.0]])
        is_toxic, score = check_safety.__wrapped__(  # type: ignore[attr-defined]
            "text", model, tokenizer
        )
        assert score == 0.5
        assert is_toxic is False

    def test_tokenizer_called_with_correct_args(self) -> None:
        model, tokenizer = self._make_mocks([[5.0, -5.0]])
        check_safety.__wrapped__(  # type: ignore[attr-defined]
            "hello world", model, tokenizer
        )
        tokenizer.assert_called_once_with(
            ["hello world"], padding=True, truncation=True, return_tensors="pt"
        )


class TestTranscribeAudio:
    def test_calls_model_generate(self) -> None:
        model = MagicMock()
        model.generate.return_value = MagicMock(text="transcribed text")
        wav = torch.zeros(1, 16000)

        transcript, _ = transcribe_audio(wav, "test prompt", model)

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

    def test_returns_elapsed_time(self) -> None:
        model = MagicMock()
        model.generate.return_value = MagicMock(text="text")
        wav = torch.zeros(1, 16000)

        _, elapsed = transcribe_audio(wav, "prompt", model)

        assert isinstance(elapsed, float)
        assert elapsed >= 0


class TestRunPipeline:
    def _make_mocks(
        self,
    ) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
        model = MagicMock()
        model.generate.return_value = MagicMock(text="decoded text")
        guardian_tokenizer = MagicMock()
        guardian_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        guardian_model = MagicMock()
        guardian_model.return_value.logits = torch.tensor([[5.0, -5.0]])
        punct_model = MagicMock()
        punct_model.infer.return_value = [["Decoded text."]]
        return model, guardian_model, guardian_tokenizer, punct_model

    def test_returns_dict_keyed_by_task(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)
        tasks = ["Transcribe", "French"]

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav, tasks, model, guardian_model, guardian_tokenizer
        )

        assert set(results.keys()) == {"Transcribe", "French"}

    def test_each_result_has_transcript_and_duration(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            guardian_model,
            guardian_tokenizer,
        )

        result = results["Transcribe"]
        assert "transcript" in result
        assert "eval_duration" in result
        assert "num_words" in result

    def test_empty_tasks_returns_empty_dict(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav, [], model, guardian_model, guardian_tokenizer
        )

        assert results == {}

    def test_uses_correct_prompt_per_task(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe", "French"],
            model,
            guardian_model,
            guardian_tokenizer,
        )

        calls = model.generate.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["prompt"] == PROMPT_CHOICES["Transcribe"]
        assert calls[1][1]["prompt"] == PROMPT_CHOICES["French"]

    def test_preserves_task_order(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)
        tasks = ["Japanese", "Transcribe", "German"]

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav, tasks, model, guardian_model, guardian_tokenizer
        )

        assert list(results.keys()) == tasks

    def test_each_result_has_safety_fields(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            guardian_model,
            guardian_tokenizer,
        )

        result = results["Transcribe"]
        assert "is_toxic" in result
        assert "toxicity_score" in result
        assert result["is_toxic"] is False

    def test_toxic_content_flagged(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        guardian_model.return_value.logits = torch.tensor([[-5.0, 5.0]])
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            guardian_model,
            guardian_tokenizer,
        )

        result = results["Transcribe"]
        assert result["is_toxic"] is True
        assert result["toxicity_score"] > 0.5

    def test_safety_check_receives_transcript(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            guardian_model,
            guardian_tokenizer,
        )

        guardian_tokenizer.assert_called_once_with(
            ["decoded text"], padding=True, truncation=True, return_tensors="pt"
        )

    def test_translation_tasks_skip_safety_check(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav, ["French"], model, guardian_model, guardian_tokenizer
        )

        result = results["French"]
        assert "is_toxic" not in result
        assert "toxicity_score" not in result
        guardian_tokenizer.assert_not_called()

    def test_mixed_tasks_safety_only_on_transcribe(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe", "French"],
            model,
            guardian_model,
            guardian_tokenizer,
        )

        assert "is_toxic" in results["Transcribe"]
        assert "toxicity_score" in results["Transcribe"]
        assert "is_toxic" not in results["French"]
        assert "toxicity_score" not in results["French"]

    def test_punctuation_applied_to_english(self) -> None:
        model, guardian_model, guardian_tokenizer, punct_model = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            guardian_model,
            guardian_tokenizer,
            punct_model,
        )

        punct_model.infer.assert_called_once_with(["decoded text"])
        assert results["Transcribe"]["transcript"] == "Decoded text."

    def test_punctuation_skipped_for_translation(self) -> None:
        model, guardian_model, guardian_tokenizer, punct_model = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["French"],
            model,
            guardian_model,
            guardian_tokenizer,
            punct_model,
        )

        punct_model.infer.assert_not_called()
        assert results["French"]["transcript"] == "decoded text"

    def test_punctuation_before_safety_check(self) -> None:
        model, guardian_model, guardian_tokenizer, punct_model = self._make_mocks()
        wav = torch.zeros(1, 16000)

        run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            guardian_model,
            guardian_tokenizer,
            punct_model,
        )

        # Guardian receives punctuated text, not raw
        guardian_tokenizer.assert_called_once_with(
            ["Decoded text."], padding=True, truncation=True, return_tensors="pt"
        )

    def test_pipeline_works_without_punct_model(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            guardian_model,
            guardian_tokenizer,
        )

        assert results["Transcribe"]["transcript"] == "decoded text"

    def test_segmented_transcript_has_timestamps(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        vad_model = MagicMock()
        wav = torch.zeros(1, 48000)  # 3 seconds
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                ["Transcribe"],
                model,
                guardian_model,
                guardian_tokenizer,
                None,
                None,
                vad_model,
                True,
            )

        transcript = results["Transcribe"]["transcript"]
        assert "[0:00 - 0:01]" in transcript
        assert "[0:01 - 0:03]" in transcript
        assert "\n" in transcript

    def test_segmented_eval_duration_is_rounded_float(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        vad_model = MagicMock()
        wav = torch.zeros(1, 48000)
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                ["Transcribe"],
                model,
                guardian_model,
                guardian_tokenizer,
                None,
                None,
                vad_model,
                True,
            )

        duration = results["Transcribe"]["eval_duration"]
        assert isinstance(duration, float)
        assert duration >= 0

    def test_segmented_num_words_excludes_timestamps(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        vad_model = MagicMock()
        wav = torch.zeros(1, 48000)
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                ["Transcribe"],
                model,
                guardian_model,
                guardian_tokenizer,
                None,
                None,
                vad_model,
                True,
            )

        # "decoded text" = 2 words per segment, 2 segments = 4 words
        assert results["Transcribe"]["num_words"] == 4

    def test_segmented_punctuation_per_segment(self) -> None:
        model, guardian_model, guardian_tokenizer, punct_model = self._make_mocks()
        vad_model = MagicMock()
        wav = torch.zeros(1, 48000)
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                ["Transcribe"],
                model,
                guardian_model,
                guardian_tokenizer,
                punct_model,
                None,
                vad_model,
                True,
            )

        # punct_model.infer called once per segment
        assert punct_model.infer.call_count == 2

    def test_segmented_safety_on_joined_text_without_timestamps(self) -> None:
        model, guardian_model, guardian_tokenizer, punct_model = self._make_mocks()
        vad_model = MagicMock()
        wav = torch.zeros(1, 48000)
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                ["Transcribe"],
                model,
                guardian_model,
                guardian_tokenizer,
                punct_model,
                None,
                vad_model,
                True,
            )

        # Guardian receives joined punctuated text without timestamps
        guardian_tokenizer.assert_called_once_with(
            ["Decoded text. Decoded text."],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def test_segmented_safety_handles_bracket_in_transcript(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        model.generate.return_value = MagicMock(text="see figure [3] here")
        vad_model = MagicMock()
        wav = torch.zeros(1, 48000)
        segments = [{"start": 0.0, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                ["Transcribe"],
                model,
                guardian_model,
                guardian_tokenizer,
                None,
                None,
                vad_model,
                True,
            )

        guardian_tokenizer.assert_called_once_with(
            ["see figure [3] here"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def test_segmented_translation_skips_punctuation_and_safety(self) -> None:
        model, guardian_model, guardian_tokenizer, punct_model = self._make_mocks()
        vad_model = MagicMock()
        wav = torch.zeros(1, 48000)
        segments = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]

        with patch("streamlit_app.get_speech_segments", return_value=segments):
            results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
                wav,
                ["French"],
                model,
                guardian_model,
                guardian_tokenizer,
                punct_model,
                None,
                vad_model,
                True,
            )

        punct_model.infer.assert_not_called()
        guardian_tokenizer.assert_not_called()
        assert "is_toxic" not in results["French"]

    def test_pipeline_unchanged_when_segmentation_disabled(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        vad_model = MagicMock()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            guardian_model,
            guardian_tokenizer,
            None,
            None,
            vad_model,
            False,
        )

        # No timestamps in transcript
        assert "[" not in results["Transcribe"]["transcript"]
        assert results["Transcribe"]["transcript"] == "decoded text"

    def test_pipeline_unchanged_when_vad_model_none(self) -> None:
        model, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            guardian_model,
            guardian_tokenizer,
            None,
            None,
            None,
            True,
        )

        assert "[" not in results["Transcribe"]["transcript"]
        assert results["Transcribe"]["transcript"] == "decoded text"


class TestRenderResultCard:
    def _make_columns(self, *counts: int) -> list[list[MagicMock]]:
        return [[MagicMock() for _ in range(n)] for n in counts]

    @patch("streamlit_app.st")
    def test_renders_metrics(self, mock_st: MagicMock) -> None:
        m1, m2, m3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.side_effect = [[m1, m2, m3], self._make_columns(2)[0]]
        result = {
            "transcript": "hello",
            "num_words": 1,
            "eval_duration": 0.5,
        }
        _render_result_card("Transcribe", result, 2.5, "test")
        m1.metric.assert_called_once_with("Duration", "2.50s")
        m2.metric.assert_called_once_with("Words", 1)
        m3.metric.assert_called_once_with("Time", "0.5s")

    @patch("streamlit_app.st")
    def test_shows_safe_banner(self, mock_st: MagicMock) -> None:
        mock_st.columns.side_effect = self._make_columns(3, 2)
        result = {
            "transcript": "hello",
            "num_words": 1,
            "eval_duration": 0.5,
            "is_toxic": False,
            "toxicity_score": 0.1,
        }
        _render_result_card("Transcribe", result, 1.0, "test")
        mock_st.success.assert_called_once()
        assert "safe" in mock_st.success.call_args[0][0].lower()
        assert "10.0%" in mock_st.success.call_args[0][0]

    @patch("streamlit_app.st")
    def test_shows_toxic_banner(self, mock_st: MagicMock) -> None:
        mock_st.columns.side_effect = self._make_columns(3, 2)
        result = {
            "transcript": "bad content",
            "num_words": 2,
            "eval_duration": 0.5,
            "is_toxic": True,
            "toxicity_score": 0.9,
        }
        _render_result_card("Transcribe", result, 1.0, "test")
        mock_st.warning.assert_called_once()
        assert "toxic" in mock_st.warning.call_args[0][0].lower()
        assert "90.0%" in mock_st.warning.call_args[0][0]

    @patch("streamlit_app.st")
    def test_no_safety_banner_for_translation(self, mock_st: MagicMock) -> None:
        mock_st.columns.side_effect = self._make_columns(3, 2)
        result = {
            "transcript": "bonjour",
            "num_words": 1,
            "eval_duration": 0.5,
        }
        _render_result_card("French", result, 1.0, "test")
        mock_st.success.assert_not_called()
        mock_st.warning.assert_not_called()

    @patch("streamlit_app.st")
    def test_download_filenames_use_slug(self, mock_st: MagicMock) -> None:
        d1, d2 = MagicMock(), MagicMock()
        mock_st.columns.side_effect = [self._make_columns(3)[0], [d1, d2]]
        result = {
            "transcript": "hello world",
            "num_words": 2,
            "eval_duration": 0.5,
        }
        _render_result_card("Mandarin Chinese", result, 1.0, "audio")
        txt_filename = d1.download_button.call_args[0][2]
        json_filename = d2.download_button.call_args[0][2]
        assert txt_filename == "audio_mandarin_chinese.txt"
        assert json_filename == "audio_mandarin_chinese.json"

    @patch("streamlit_app.st")
    def test_json_download_payload(self, mock_st: MagicMock) -> None:
        d1, d2 = MagicMock(), MagicMock()
        mock_st.columns.side_effect = [self._make_columns(3)[0], [d1, d2]]
        result = {
            "transcript": "hello",
            "num_words": 1,
            "eval_duration": 0.5,
        }
        _render_result_card("French", result, 2.5, "test")
        payload = json.loads(d2.download_button.call_args[0][1])
        assert payload["model"] == MODEL_ID
        assert payload["task"] == "French"
        assert payload["audio_duration"] == 2.5
        assert payload["transcript"] == "hello"
        assert payload["num_words"] == 1
        assert payload["eval_duration"] == 0.5

    @patch("streamlit_app.st")
    def test_renders_bordered_container(self, mock_st: MagicMock) -> None:
        mock_st.columns.side_effect = self._make_columns(3, 2)
        result = {
            "transcript": "hello",
            "num_words": 1,
            "eval_duration": 0.5,
        }
        _render_result_card("Transcribe", result, 1.0, "test")
        mock_st.container.assert_called_once_with(border=True)
