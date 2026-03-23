from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from config import Settings
from model_manager import (
    CompareResult,
    GenerationParams,
    GenerationResult,
    ModelManager,
)

router = APIRouter()


class GenerationParamsPayload(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=128, ge=1, le=256)
    seed: int = Field(default=42, ge=0)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    mode: Literal["base", "adapted"]
    params: GenerationParamsPayload = Field(default_factory=GenerationParamsPayload)

    @field_validator("prompt")
    @classmethod
    def _prompt_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Empty prompt")
        return cleaned


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1)
    params: GenerationParamsPayload = Field(default_factory=GenerationParamsPayload)

    @field_validator("prompt")
    @classmethod
    def _prompt_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Empty prompt")
        return cleaned


class GenerateResponse(BaseModel):
    mode: Literal["base", "adapted"]
    text: str
    latency_seconds: float
    output_tokens: int
    tokens_per_second: float


class CompareResponse(BaseModel):
    base: GenerateResponse
    adapted: GenerateResponse
    total_latency_seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    base_model_id: str
    adapter_name: str


def _get_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings unavailable")
    return cast(Settings, settings)


def _get_model_manager(request: Request) -> ModelManager:
    manager = getattr(request.app.state, "model_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    return cast(ModelManager, manager)


def _to_generation_params(payload: GenerationParamsPayload, settings: Settings) -> GenerationParams:
    return GenerationParams(
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_new_tokens=payload.max_new_tokens,
        seed=payload.seed,
    ).normalized(settings.max_new_tokens_cap)


def _to_response(result: GenerationResult) -> GenerateResponse:
    return GenerateResponse(
        mode=result.mode,
        text=result.text,
        latency_seconds=result.latency_seconds,
        output_tokens=result.output_tokens,
        tokens_per_second=result.tokens_per_second,
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    manager = _get_model_manager(request)
    settings = _get_settings(request)
    return HealthResponse(
        status="ok",
        model_loaded=manager.ready,
        base_model_id=settings.base_model_id,
        adapter_name=settings.adapter_name,
    )


@router.post("/generate", response_model=GenerateResponse)
def generate(request_body: GenerateRequest, request: Request) -> GenerateResponse:
    settings = _get_settings(request)
    manager = _get_model_manager(request)
    params = _to_generation_params(request_body.params, settings)
    try:
        result = manager.generate(
            prompt=request_body.prompt,
            mode=request_body.mode,
            params=params,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_response(result)


@router.post("/compare", response_model=CompareResponse)
def compare(request_body: CompareRequest, request: Request) -> CompareResponse:
    settings = _get_settings(request)
    manager = _get_model_manager(request)
    params = _to_generation_params(request_body.params, settings)
    try:
        result: CompareResult = manager.compare(prompt=request_body.prompt, params=params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CompareResponse(
        base=_to_response(result.base),
        adapted=_to_response(result.adapted),
        total_latency_seconds=result.total_latency_seconds,
    )
