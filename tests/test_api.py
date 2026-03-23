from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import router
from config import Settings
from model_manager import CompareResult, GenerationParams, GenerationResult, InferenceMode


class FakeModelManager:
    ready = True

    def generate(
        self,
        prompt: str,
        mode: InferenceMode,
        params: GenerationParams,
    ) -> GenerationResult:
        text = f"{mode}:{prompt[:12]}:{params.max_new_tokens}"
        return GenerationResult(
            mode=mode,
            text=text,
            latency_seconds=0.123,
            output_tokens=8,
            tokens_per_second=65.04,
        )

    def compare(self, prompt: str, params: GenerationParams) -> CompareResult:
        base = self.generate(prompt=prompt, mode="base", params=params)
        adapted = self.generate(prompt=prompt, mode="adapted", params=params)
        return CompareResult(base=base, adapted=adapted)


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.state.settings = Settings(
        base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        adapter_name="rs_lora",
        adapter_local_path=Path("adapter").resolve(),
        adapter_repo_id=None,
        hf_cache_dir=None,
        trust_remote_code=True,
        host="0.0.0.0",
        port=8000,
        ui_mount_path="/ui",
        default_temperature=0.7,
        default_top_p=0.9,
        default_max_new_tokens=128,
        max_new_tokens_cap=256,
    )
    app.state.model_manager = FakeModelManager()
    app.include_router(router)
    return app


def test_health_endpoint() -> None:
    client = TestClient(_build_test_app())
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_generate_endpoint() -> None:
    client = TestClient(_build_test_app())
    response = client.post(
        "/generate",
        json={
            "prompt": "Hello world",
            "mode": "base",
            "params": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 32, "seed": 7},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "base"
    assert body["text"].startswith("base:")


def test_compare_endpoint() -> None:
    client = TestClient(_build_test_app())
    response = client.post(
        "/compare",
        json={
            "prompt": "Compare this",
            "params": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 16, "seed": 1},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["base"]["mode"] == "base"
    assert body["adapted"]["mode"] == "adapted"
    assert body["total_latency_seconds"] > 0


def test_blank_prompt_rejected() -> None:
    client = TestClient(_build_test_app())
    response = client.post(
        "/generate",
        json={"prompt": "   ", "mode": "adapted", "params": {"max_new_tokens": 16}},
    )

    assert response.status_code == 422
