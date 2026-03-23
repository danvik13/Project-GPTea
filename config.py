from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


@dataclass(frozen=True, slots=True)
class Settings:
    base_model_id: str
    adapter_name: str
    adapter_local_path: Path
    adapter_repo_id: str | None
    hf_cache_dir: Path | None
    trust_remote_code: bool
    host: str
    port: int
    ui_mount_path: str
    default_temperature: float
    default_top_p: float
    default_max_new_tokens: int
    max_new_tokens_cap: int

    @classmethod
    def from_env(cls) -> Settings:
        mount_path = os.getenv("UI_MOUNT_PATH", "/ui").strip() or "/ui"
        if not mount_path.startswith("/"):
            mount_path = f"/{mount_path}"

        cache_dir_raw = os.getenv("HF_CACHE_DIR")
        cache_dir = Path(cache_dir_raw).expanduser().resolve() if cache_dir_raw else None

        max_new_tokens_cap = min(max(_read_int("MAX_NEW_TOKENS_CAP", 256), 1), 256)
        default_max_new_tokens = min(
            max(_read_int("DEFAULT_MAX_NEW_TOKENS", 128), 1),
            max_new_tokens_cap,
        )

        return cls(
            base_model_id=os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"),
            adapter_name=os.getenv("ADAPTER_NAME", "rs_lora"),
            adapter_local_path=Path(os.getenv("ADAPTER_LOCAL_PATH", "adapter"))
            .expanduser()
            .resolve(),
            adapter_repo_id=os.getenv("ADAPTER_REPO_ID") or None,
            hf_cache_dir=cache_dir,
            trust_remote_code=_read_bool("TRUST_REMOTE_CODE", True),
            host=os.getenv("HOST", "0.0.0.0"),
            port=_read_int("PORT", 8000),
            ui_mount_path=mount_path,
            default_temperature=min(max(_read_float("DEFAULT_TEMPERATURE", 0.7), 0.0), 2.0),
            default_top_p=min(max(_read_float("DEFAULT_TOP_P", 0.9), 1e-6), 1.0),
            default_max_new_tokens=default_max_new_tokens,
            max_new_tokens_cap=max_new_tokens_cap,
        )

    @property
    def cache_dir(self) -> str | None:
        if self.hf_cache_dir is None:
            return None
        return str(self.hf_cache_dir)
