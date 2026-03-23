from __future__ import annotations

import pytest

from config import Settings


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BASE_MODEL_ID", raising=False)
    monkeypatch.delenv("ADAPTER_LOCAL_PATH", raising=False)
    monkeypatch.delenv("MAX_NEW_TOKENS_CAP", raising=False)
    monkeypatch.delenv("DEFAULT_MAX_NEW_TOKENS", raising=False)

    settings = Settings.from_env()

    assert settings.base_model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert settings.adapter_name == "rs_lora"
    assert settings.max_new_tokens_cap == 256
    assert settings.default_max_new_tokens == 128


def test_settings_caps_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_NEW_TOKENS_CAP", "999")
    monkeypatch.setenv("DEFAULT_MAX_NEW_TOKENS", "900")

    settings = Settings.from_env()

    assert settings.max_new_tokens_cap == 256
    assert settings.default_max_new_tokens == 256
