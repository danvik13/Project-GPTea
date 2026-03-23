from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api import router as api_router
from config import Settings
from model_manager import (
    CompareResult,
    GenerationParams,
    GenerationResult,
    InferenceMode,
    ModelManager,
)
from ui import build_ui

settings = Settings.from_env()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(api_app: FastAPI) -> AsyncIterator[None]:
        manager = ModelManager(settings=settings)
        api_app.state.model_manager = manager
        api_app.state.settings = settings
        try:
            yield
        finally:
            manager.close()

    api_app = FastAPI(
        title="gptea-service",
        version="0.1.0",
        lifespan=lifespan,
    )
    api_app.state.settings = settings
    api_app.include_router(api_router)

    @api_app.get("/", include_in_schema=False)
    def root_redirect() -> RedirectResponse:
        return RedirectResponse(url=f"{settings.ui_mount_path}/")

    def run_generate(prompt: str, mode: str, params: GenerationParams) -> GenerationResult:
        manager = cast(ModelManager, api_app.state.model_manager)
        if mode not in {"base", "adapted"}:
            raise ValueError("Invalid mode")
        normalized_mode = cast(InferenceMode, mode)
        return manager.generate(prompt=prompt, mode=normalized_mode, params=params)

    def run_compare(prompt: str, params: GenerationParams) -> CompareResult:
        manager = cast(ModelManager, api_app.state.model_manager)
        return manager.compare(prompt=prompt, params=params)

    demo = build_ui(settings=settings, run_generate=run_generate, run_compare=run_compare)
    return cast(FastAPI, gr.mount_gradio_app(api_app, demo, path=settings.ui_mount_path))


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=False,
        workers=1,
    )
