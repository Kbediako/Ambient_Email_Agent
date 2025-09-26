"""Tests for configuration helpers and Gemini model routing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from email_assistant.configuration import get_llm, normalize_model_spec


def test_normalize_model_spec_prefixed_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMAIL_ASSISTANT_MODEL_PROVIDER", raising=False)
    spec = normalize_model_spec("google_genai:gemini-2.5-pro")
    assert spec.provider == "google_genai"
    assert spec.model == "gemini-2.5-pro"


def test_normalize_model_spec_strips_models_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMAIL_ASSISTANT_MODEL_PROVIDER", "google_genai")
    spec = normalize_model_spec("models/gemini-1.5-pro")
    assert spec.provider == "google_genai"
    assert spec.model == "gemini-1.5-pro"


def test_get_llm_calls_init_chat_model_with_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake_init(model: str, **kwargs):
        """
        Test helper that records initialization arguments and returns a dummy chat model instance.
        
        Parameters:
            model (str): Model identifier passed to the initializer.
            **kwargs: Additional initialization keyword arguments; all are recorded.
        
        Detailed behavior:
            Stores the provided `model` and all `kwargs` into the surrounding `calls` dict as side effects.
        
        Returns:
            _DummyModel: A minimal stand-in instance representing a chat model.
        """
        calls["model"] = model
        calls.update(kwargs)

        class _DummyModel:  # minimal stand-in for BaseChatModel
            pass

        return _DummyModel()

    monkeypatch.setenv("EMAIL_ASSISTANT_MODEL_PROVIDER", "")
    with patch("email_assistant.configuration.init_chat_model", side_effect=_fake_init):
        get_llm(model="google_genai:gemini-1.5-pro", temperature=0.1, max_output_tokens=128)

    assert calls.get("model") == "gemini-1.5-pro"
    assert calls.get("model_provider") == "google_genai"
    assert calls.get("temperature") == 0.1
    assert calls.get("max_output_tokens") == 128
    assert calls.get("convert_system_message_to_human") is False
