from __future__ import annotations

from collections.abc import Callable
from typing import cast

import gradio as gr

from config import Settings
from model_manager import CompareResult, GenerationParams, GenerationResult


def _latency_line(result: GenerationResult) -> str:
    return (
        f"{result.mode}: {result.latency_seconds:.3f}s | "
        f"{result.output_tokens} tokens | {result.tokens_per_second:.2f} tok/s"
    )


def build_ui(
    settings: Settings,
    run_generate: Callable[[str, str, GenerationParams], GenerationResult],
    run_compare: Callable[[str, GenerationParams], CompareResult],
) -> gr.Blocks:
    def submit(
        prompt: str,
        mode: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[str, str, str]:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            return "", "", "Prompt cannot be empty."

        params = GenerationParams(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=42,
        ).normalized(settings.max_new_tokens_cap)

        try:
            if mode == "compare":
                result = run_compare(cleaned_prompt, params)
                latency_text = "\n".join(
                    [
                        _latency_line(result.base),
                        _latency_line(result.adapted),
                        f"total: {result.total_latency_seconds:.3f}s",
                    ]
                )
                return result.base.text, result.adapted.text, latency_text

            generated = run_generate(cleaned_prompt, mode, params)
            latency_text = _latency_line(generated)
            if mode == "base":
                return generated.text, "", latency_text
            return "", generated.text, latency_text
        except Exception as exc:
            return "", "", f"Error: {exc}"

    with gr.Blocks(title="GPTea") as demo:
        gr.Markdown("## Project: GPTea\nThe most advanced AI model so far.")
        prompt = gr.Textbox(
            label="Prompt",
            placeholder="Enter an instruction or question...",
            lines=8,
        )
        mode = gr.Dropdown(
            label="Mode",
            choices=["base", "adapted", "compare"],
            value="compare",
        )
        with gr.Row():
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.0,
                maximum=2.0,
                value=settings.default_temperature,
                step=0.05,
            )
            top_p = gr.Slider(
                label="Top-p",
                minimum=0.01,
                maximum=1.0,
                value=settings.default_top_p,
                step=0.01,
            )
            max_new_tokens = gr.Slider(
                label="Max new tokens",
                minimum=1,
                maximum=settings.max_new_tokens_cap,
                value=settings.default_max_new_tokens,
                step=1,
            )

        run_button = gr.Button("Run")
        with gr.Row():
            base_output = gr.Textbox(label="Base output", lines=12)
            adapted_output = gr.Textbox(label="Adapted output", lines=12)
        latency = gr.Textbox(label="Latency", lines=4)

        run_button.click(
            submit,
            inputs=[prompt, mode, temperature, top_p, max_new_tokens],
            outputs=[base_output, adapted_output, latency],
            concurrency_limit=1,
        )
        prompt.submit(
            submit,
            inputs=[prompt, mode, temperature, top_p, max_new_tokens],
            outputs=[base_output, adapted_output, latency],
            concurrency_limit=1,
        )

    return cast(gr.Blocks, demo)
