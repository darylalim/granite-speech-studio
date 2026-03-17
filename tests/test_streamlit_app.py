from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import json

from streamlit_app import (
    GUARDIAN_MODEL_ID,
    MODEL_ID,
    PUNCTUATION_MODEL_ID,
    PROMPT_CHOICES,
    SUPPORTED_FORMATS,
    TASK_PRESETS,
    _render_result_card,
    apply_punctuation,
    check_safety,
    get_device,
    get_selected_tasks,
    load_and_preprocess_audio,
    load_guardian_model,
    load_punctuation_model,
    load_model,
    run_pipeline,
    transcribe_audio,
)

AUDIO_DIR = Path(__file__).parent / "data" / "audio"


class TestModelIds:
    def test_model_id(self) -> None:
        assert MODEL_ID == "ibm-granite/granite-4.0-1b-speech"

    def test_guardian_model_id(self) -> None:
        assert GUARDIAN_MODEL_ID == "ibm-granite/granite-guardian-hap-38m"

    def test_punctuation_model_id(self) -> None:
        assert PUNCTUATION_MODEL_ID == "pcs_en"


class TestGetDevice:
    @patch("streamlit_app.torch")
    def test_mps_preferred(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "mps"

    @patch("streamlit_app.torch")
    def test_cuda_fallback(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "cuda"

    @patch("streamlit_app.torch")
    def test_cpu_fallback(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        assert get_device() == "cpu"


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
        ):
            assert lang in PROMPT_CHOICES
            assert lang in PROMPT_CHOICES[lang]

    def test_translation_prompt_prefixes(self) -> None:
        for key, value in PROMPT_CHOICES.items():
            if key != "Transcribe":
                assert value.startswith("translate the speech to ")

    def test_total_count(self) -> None:
        assert len(PROMPT_CHOICES) == 8


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
    @patch("streamlit_app.AutoProcessor")
    @patch("streamlit_app.AutoModelForSpeechSeq2Seq")
    @patch("streamlit_app.st")
    def test_uses_float32_on_cpu(
        self,
        _mock_st: MagicMock,
        mock_model_cls: MagicMock,
        mock_processor_cls: MagicMock,
    ) -> None:
        load_model.__wrapped__("test-model", "cpu")  # type: ignore[attr-defined]
        mock_model_cls.from_pretrained.assert_called_once_with(
            "test-model", device_map="cpu", dtype=torch.float32
        )

    @patch("streamlit_app.AutoProcessor")
    @patch("streamlit_app.AutoModelForSpeechSeq2Seq")
    @patch("streamlit_app.st")
    def test_uses_bfloat16_on_gpu(
        self,
        _mock_st: MagicMock,
        mock_model_cls: MagicMock,
        mock_processor_cls: MagicMock,
    ) -> None:
        load_model.__wrapped__("test-model", "cuda")  # type: ignore[attr-defined]
        mock_model_cls.from_pretrained.assert_called_once_with(
            "test-model", device_map="cuda", dtype=torch.bfloat16
        )


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
    def _make_mocks(self) -> tuple[MagicMock, MagicMock, MagicMock]:
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted"
        tokenizer.batch_decode.return_value = ["decoded text"]
        processor = MagicMock()
        processor.tokenizer = tokenizer
        model = MagicMock()
        return model, processor, tokenizer

    def test_no_system_prompt_in_chat(self) -> None:
        model, processor, tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)

        transcribe_audio.__wrapped__(  # type: ignore[attr-defined]
            wav, "transcribe", model, processor, "cpu"
        )

        chat_arg = tokenizer.apply_chat_template.call_args[0][0]
        roles = [msg["role"] for msg in chat_arg]
        assert "system" not in roles
        assert roles == ["user"]

    def test_user_content_has_audio_tag(self) -> None:
        model, processor, tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)

        transcribe_audio.__wrapped__(  # type: ignore[attr-defined]
            wav, "test prompt", model, processor, "cpu"
        )

        chat_arg = tokenizer.apply_chat_template.call_args[0][0]
        assert chat_arg[0]["content"] == "<|audio|>test prompt"

    def test_uses_batch_decode(self) -> None:
        model, processor, tokenizer = self._make_mocks()
        wav = torch.zeros(1, 16000)

        transcript, _ = transcribe_audio.__wrapped__(  # type: ignore[attr-defined]
            wav, "transcribe", model, processor, "cpu"
        )

        tokenizer.batch_decode.assert_called_once()
        call_kwargs = tokenizer.batch_decode.call_args[1]
        assert call_kwargs["skip_special_tokens"] is True
        assert call_kwargs["add_special_tokens"] is False
        assert transcript == "decoded text"


class TestRunPipeline:
    def _make_mocks(self) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted"
        tokenizer.batch_decode.return_value = ["decoded text"]
        processor = MagicMock()
        processor.tokenizer = tokenizer
        model = MagicMock()
        guardian_tokenizer = MagicMock()
        guardian_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        guardian_model = MagicMock()
        guardian_model.return_value.logits = torch.tensor([[5.0, -5.0]])
        punct_model = MagicMock()
        punct_model.infer.return_value = [["Decoded text."]]
        return model, processor, guardian_model, guardian_tokenizer, punct_model

    def test_returns_dict_keyed_by_task(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)
        tasks = ["Transcribe", "French"]

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav, tasks, model, processor, "cpu", guardian_model, guardian_tokenizer
        )

        assert set(results.keys()) == {"Transcribe", "French"}

    def test_each_result_has_transcript_and_duration(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            processor,
            "cpu",
            guardian_model,
            guardian_tokenizer,
        )

        result = results["Transcribe"]
        assert "transcript" in result
        assert "eval_duration" in result
        assert "num_words" in result

    def test_empty_tasks_returns_empty_dict(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav, [], model, processor, "cpu", guardian_model, guardian_tokenizer
        )

        assert results == {}

    def test_uses_correct_prompt_per_task(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        tokenizer = processor.tokenizer
        wav = torch.zeros(1, 16000)

        run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe", "French"],
            model,
            processor,
            "cpu",
            guardian_model,
            guardian_tokenizer,
        )

        calls = tokenizer.apply_chat_template.call_args_list
        assert len(calls) == 2
        first_chat = calls[0][0][0]
        second_chat = calls[1][0][0]
        assert PROMPT_CHOICES["Transcribe"] in first_chat[0]["content"]
        assert PROMPT_CHOICES["French"] in second_chat[0]["content"]

    def test_preserves_task_order(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)
        tasks = ["Japanese", "Transcribe", "German"]

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav, tasks, model, processor, "cpu", guardian_model, guardian_tokenizer
        )

        assert list(results.keys()) == tasks

    def test_each_result_has_safety_fields(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            processor,
            "cpu",
            guardian_model,
            guardian_tokenizer,
        )

        result = results["Transcribe"]
        assert "is_toxic" in result
        assert "toxicity_score" in result
        assert result["is_toxic"] is False

    def test_toxic_content_flagged(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        guardian_model.return_value.logits = torch.tensor([[-5.0, 5.0]])
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            processor,
            "cpu",
            guardian_model,
            guardian_tokenizer,
        )

        result = results["Transcribe"]
        assert result["is_toxic"] is True
        assert result["toxicity_score"] > 0.5

    def test_safety_check_receives_transcript(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            processor,
            "cpu",
            guardian_model,
            guardian_tokenizer,
        )

        guardian_tokenizer.assert_called_once_with(
            ["decoded text"], padding=True, truncation=True, return_tensors="pt"
        )

    def test_translation_tasks_skip_safety_check(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav, ["French"], model, processor, "cpu", guardian_model, guardian_tokenizer
        )

        result = results["French"]
        assert "is_toxic" not in result
        assert "toxicity_score" not in result
        guardian_tokenizer.assert_not_called()

    def test_mixed_tasks_safety_only_on_transcribe(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe", "French"],
            model,
            processor,
            "cpu",
            guardian_model,
            guardian_tokenizer,
        )

        assert "is_toxic" in results["Transcribe"]
        assert "toxicity_score" in results["Transcribe"]
        assert "is_toxic" not in results["French"]
        assert "toxicity_score" not in results["French"]

    def test_punctuation_applied_to_english(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, punct_model = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            processor,
            "cpu",
            guardian_model,
            guardian_tokenizer,
            punct_model,
        )

        punct_model.infer.assert_called_once_with(["decoded text"])
        assert results["Transcribe"]["transcript"] == "Decoded text."

    def test_punctuation_skipped_for_translation(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, punct_model = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["French"],
            model,
            processor,
            "cpu",
            guardian_model,
            guardian_tokenizer,
            punct_model,
        )

        punct_model.infer.assert_not_called()
        assert results["French"]["transcript"] == "decoded text"

    def test_punctuation_before_safety_check(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, punct_model = self._make_mocks()
        wav = torch.zeros(1, 16000)

        run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            processor,
            "cpu",
            guardian_model,
            guardian_tokenizer,
            punct_model,
        )

        # Guardian receives punctuated text, not raw
        guardian_tokenizer.assert_called_once_with(
            ["Decoded text."], padding=True, truncation=True, return_tensors="pt"
        )

    def test_pipeline_works_without_punct_model(self) -> None:
        model, processor, guardian_model, guardian_tokenizer, _ = self._make_mocks()
        wav = torch.zeros(1, 16000)

        results = run_pipeline.__wrapped__(  # type: ignore[attr-defined]
            wav,
            ["Transcribe"],
            model,
            processor,
            "cpu",
            guardian_model,
            guardian_tokenizer,
        )

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
