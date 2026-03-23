from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Literal, Protocol, cast

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

from config import Settings

InferenceMode = Literal["base", "adapted"]


class AdapterCapableModel(Protocol):
    def eval(self) -> Any: ...

    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

    def load_adapter(self, model_id: str, adapter_name: str, **kwargs: Any) -> Any: ...

    def set_adapter(self, adapter_name: str) -> Any: ...

    def disable_adapter(self) -> Any: ...

    def disable_adapters(self) -> Any: ...

    def enable_adapters(self) -> Any: ...

    def parameters(self) -> Iterator[torch.nn.Parameter]: ...


@dataclass(frozen=True, slots=True)
class GenerationParams:
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int = 42

    def normalized(self, max_new_tokens_cap: int) -> GenerationParams:
        return GenerationParams(
            temperature=min(max(self.temperature, 0.0), 2.0),
            top_p=min(max(self.top_p, 1e-6), 1.0),
            max_new_tokens=min(max(self.max_new_tokens, 1), max_new_tokens_cap),
            seed=max(self.seed, 0),
        )


@dataclass(frozen=True, slots=True)
class GenerationResult:
    mode: InferenceMode
    text: str
    latency_seconds: float
    output_tokens: int
    tokens_per_second: float


@dataclass(frozen=True, slots=True)
class CompareResult:
    base: GenerationResult
    adapted: GenerationResult

    @property
    def total_latency_seconds(self) -> float:
        return self.base.latency_seconds + self.adapted.latency_seconds


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = Lock()
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: AdapterCapableModel | None = None
        self._input_device = torch.device("cpu")
        self._ready = False
        self._load_assets()

    @property
    def ready(self) -> bool:
        return self._ready

    def generate(
        self,
        prompt: str,
        mode: InferenceMode,
        params: GenerationParams,
    ) -> GenerationResult:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Empty prompt")

        normalized = params.normalized(self._settings.max_new_tokens_cap)
        with self._lock:
            return self._generate_locked(cleaned_prompt, mode, normalized)

    def compare(self, prompt: str, params: GenerationParams) -> CompareResult:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Empty prompt")

        normalized = params.normalized(self._settings.max_new_tokens_cap)
        with self._lock:
            base_result = self._generate_locked(cleaned_prompt, "base", normalized)
            adapted_result = self._generate_locked(cleaned_prompt, "adapted", normalized)
        return CompareResult(base=base_result, adapted=adapted_result)

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._ready = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_assets(self) -> None:
        base_model_path = snapshot_download(
            repo_id=self._settings.base_model_id,
            cache_dir=self._settings.cache_dir,
        )
        adapter_path = self._resolve_adapter_path()

        compute_dtype = self._select_compute_dtype()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=self._settings.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=self._settings.trust_remote_code,
        )

        if not hasattr(model, "load_adapter"):
            raise RuntimeError("Adapter API missing")

        model_with_adapter = cast(AdapterCapableModel, model)
        model_with_adapter.load_adapter(
            str(adapter_path),
            adapter_name=self._settings.adapter_name,
        )
        model_with_adapter.set_adapter(self._settings.adapter_name)
        model_with_adapter.eval()

        self._tokenizer = tokenizer
        self._model = model_with_adapter
        self._input_device = self._infer_input_device(model_with_adapter)
        self._ready = True

    def _resolve_adapter_path(self) -> Path:
        if self._settings.adapter_local_path.exists():
            return self._settings.adapter_local_path

        if not self._settings.adapter_repo_id:
            raise FileNotFoundError("Adapter path missing")

        downloaded = snapshot_download(
            repo_id=self._settings.adapter_repo_id,
            cache_dir=self._settings.cache_dir,
        )
        return Path(downloaded)

    def _generate_locked(
        self,
        prompt: str,
        mode: InferenceMode,
        params: GenerationParams,
    ) -> GenerationResult:
        if mode not in {"base", "adapted"}:
            raise ValueError("Invalid mode")

        tokenizer = self._require_tokenizer()
        model = self._require_model()
        rendered_prompt = self._render_prompt(prompt, tokenizer)
        encoded = tokenizer(rendered_prompt, return_tensors="pt")
        inputs = {key: value.to(self._input_device) for key, value in encoded.items()}

        self._seed_everything(params.seed)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_new_tokens,
            "do_sample": params.temperature > 0.0,
            "temperature": params.temperature if params.temperature > 0.0 else 1.0,
            "top_p": params.top_p,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if not generation_kwargs["do_sample"]:
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_p", None)

        start = perf_counter()
        with torch.inference_mode():
            if mode == "base":
                with self._adapter_disabled(model):
                    generated = model.generate(**inputs, **generation_kwargs)
            else:
                model.set_adapter(self._settings.adapter_name)
                generated = model.generate(**inputs, **generation_kwargs)
        latency_seconds = perf_counter() - start

        input_token_count = int(inputs["input_ids"].shape[-1])
        new_tokens = generated[0, input_token_count:]
        output_tokens = int(new_tokens.shape[-1])
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        tokens_per_second = output_tokens / latency_seconds if latency_seconds > 0.0 else 0.0

        return GenerationResult(
            mode=mode,
            text=text,
            latency_seconds=latency_seconds,
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
        )

    def _render_prompt(self, prompt: str, tokenizer: PreTrainedTokenizerBase) -> str:
        if not hasattr(tokenizer, "apply_chat_template"):
            return prompt

        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

        return str(rendered)

    @contextmanager
    def _adapter_disabled(self, model: AdapterCapableModel) -> Iterator[None]:
        if hasattr(model, "disable_adapter"):
            maybe_context = model.disable_adapter()
            if hasattr(maybe_context, "__enter__") and hasattr(maybe_context, "__exit__"):
                with maybe_context:
                    yield
                return

        if not hasattr(model, "disable_adapters") or not hasattr(model, "enable_adapters"):
            raise RuntimeError("Adapter toggle API missing")

        model.disable_adapters()
        try:
            yield
        finally:
            model.enable_adapters()

    @staticmethod
    def _seed_everything(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _select_compute_dtype() -> torch.dtype:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    @staticmethod
    def _infer_input_device(model: AdapterCapableModel) -> torch.device:
        for parameter in model.parameters():
            return parameter.device
        return torch.device("cpu")

    def _require_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self._tokenizer

    def _require_model(self) -> AdapterCapableModel:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model
