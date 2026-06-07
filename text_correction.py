"""
Transcript correction stage using MLX Python libraries.
"""

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


# Common generation settings
GENERATION_BACKEND = "mlx_vlm"  # "mlx_lm" or "mlx_vlm"
DEBUG_PRINT_COMMAND = True
MAX_TOKENS = "2560"
TEMPERATURE = "1.0"
ENABLE_THINKING = False
THINKING_BUDGET = "256"

# mlx_lm settings
MLX_LM_MODEL = "mlx-community/gemma-4-31B-it-uncensored-heretic-4bit"

# mlx_vlm settings
# MLX_VLM_MODEL = "mlx-community/gemma-4-12B-it-qat-4bit"
MLX_VLM_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
# MLX_VLM_MODEL = "mlx-community/gemma-4-E4B-it-bf16"
MLX_VLM_USE_DRAFT_MODEL = False
MLX_VLM_DRAFT_MODEL = "mlx-community/gemma-4-12B-it-qat-assistant-4bit"
MLX_VLM_DRAFT_KIND = "mtp"


def fail(message: str) -> None:
    """Print an error and exit the pipeline."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def parse_int_setting(name: str, value: str, *, minimum: int = 1) -> int:
    """Parse an integer generation setting."""
    try:
        parsed = int(value)
    except ValueError:
        fail(f"{name} must be an integer, got {value!r}.")

    if parsed < minimum:
        fail(f"{name} must be at least {minimum}, got {parsed}.")

    return parsed


def parse_float_setting(name: str, value: str) -> float:
    """Parse a floating-point generation setting."""
    try:
        return float(value)
    except ValueError:
        fail(f"{name} must be a number, got {value!r}.")


def check_text_correction_dependencies() -> None:
    """Check for text correction Python package dependencies."""
    package = {
        "mlx_lm": ("mlx_lm", "mlx-lm"),
        "mlx_vlm": ("mlx_vlm", "mlx-vlm"),
    }.get(GENERATION_BACKEND)

    if package is None:
        fail("GENERATION_BACKEND must be 'mlx_lm' or 'mlx_vlm'.")

    module_name, package_name = package
    if importlib.util.find_spec(module_name) is None:
        fail(
            f"{package_name} is not installed in this Python environment. "
            "Run: python -m pip install -r requirements.txt"
        )


def build_correction_prompt(transcript: str) -> str:
    """Build the fixed prompt for transcript correction."""
    return (
        "You are an expert multilingual transcript editor.\n"
        "Correct speech-to-text transcription mistakes in the transcript below.\n"
        "You are correcting ASR transcription errors, not merely proofreading.\n"
        "Rules:\n"
        "- Identify the language from context.\n"
        "- Preserve speaker labels, line order, and line breaks.\n"
        "- Keep the original language; do not translate.\n"
        "- Fix malformed, nonexistent, or phonetically corrupted words.\n"
        "- If a word is not plausible in the detected language, replace it with the most likely valid word based on surrounding context and similar pronunciation.\n"
        "- Prefer fluent, idiomatic text in the same language over literal preservation of suspicious tokens.\n"
        "- Do not change valid names, numbers, or technical terms unless they are clearly corrupted.\n"
        "- Return only the corrected transcript.\n\n"
        "Transcript:\n"
        "<<<TRANSCRIPT\n"
        f"{transcript.rstrip()}\n"
        "TRANSCRIPT>>>"
    )


def clean_model_output(output: str) -> str:
    """Normalize generated text before saving."""
    lines = output.replace("\r\n", "\n").replace("\r", "\n").splitlines()

    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        lines = lines[1:-1]

    cleaned = "\n".join(lines).strip()
    return f"{cleaned}\n" if cleaned else ""


def print_debug_call(function_name: str, params: dict[str, Any]) -> None:
    """Print the library call settings when debug output is enabled."""
    if not DEBUG_PRINT_COMMAND:
        return

    print("LLM library call:")
    print(function_name)
    for key, value in params.items():
        print(f"  {key}={value!r}")


@lru_cache(maxsize=1)
def load_mlx_lm_resources(model_name: str) -> tuple[Any, Any]:
    """Load and cache mlx_lm model resources."""
    try:
        from mlx_lm import load
    except Exception as e:
        fail(f"Failed to import mlx_lm: {e}")

    try:
        return load(model_name)
    except Exception as e:
        fail(f"Failed to load mlx_lm model {model_name!r}: {e}")


def apply_mlx_lm_chat_template(tokenizer: Any, prompt: str) -> list[int]:
    """Apply the same chat-template behavior used by mlx_lm.generate."""
    prompt = prompt.replace("\\n", "\n").replace("\\t", "\t")

    if getattr(tokenizer, "has_chat_template", False):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer.encode(formatted_prompt, add_special_tokens=False)

    return tokenizer.encode(prompt)


def run_mlx_lm_correction(prompt: str) -> str:
    """Run mlx_lm library generation for transcript correction."""
    max_tokens = parse_int_setting("MAX_TOKENS", MAX_TOKENS)
    temperature = parse_float_setting("TEMPERATURE", TEMPERATURE)
    model, tokenizer = load_mlx_lm_resources(MLX_LM_MODEL)
    formatted_prompt = apply_mlx_lm_chat_template(tokenizer, prompt)

    try:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(
            temperature,
            top_p=1.0,
            min_p=0.0,
            min_tokens_to_keep=1,
            top_k=0,
            xtc_probability=0.0,
            xtc_threshold=0.0,
            xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
        )
        print_debug_call(
            "mlx_lm.generate",
            {
                "model": MLX_LM_MODEL,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "prompt_chars": len(prompt),
            },
        )
        generated = generate(
            model,
            tokenizer,
            formatted_prompt,
            max_tokens=max_tokens,
            verbose=False,
            sampler=sampler,
        )
    except Exception as e:
        fail(f"mlx_lm generation failed: {e}")

    return clean_model_output(generated)


@lru_cache(maxsize=1)
def load_mlx_vlm_resources(model_name: str) -> tuple[Any, Any]:
    """Load and cache mlx_vlm model resources."""
    try:
        from mlx_vlm import load
    except Exception as e:
        fail(f"Failed to import mlx_vlm: {e}")

    try:
        return load(model_name)
    except Exception as e:
        fail(f"Failed to load mlx_vlm model {model_name!r}: {e}")


@lru_cache(maxsize=1)
def load_mlx_vlm_draft_model(model_name: str, draft_kind: str) -> tuple[Any, str]:
    """Load and cache an mlx_vlm speculative decoding drafter."""
    try:
        from mlx_vlm.speculative.drafters import load_drafter
    except Exception as e:
        fail(f"Failed to import mlx_vlm drafter support: {e}")

    try:
        return load_drafter(model_name, kind=draft_kind or None)
    except Exception as e:
        fail(f"Failed to load mlx_vlm draft model {model_name!r}: {e}")


def add_mlx_vlm_draft_model(model: Any, generation_kwargs: dict[str, Any]) -> None:
    """Add optional mlx_vlm speculative decoding settings."""
    if not MLX_VLM_USE_DRAFT_MODEL:
        return

    draft_model, resolved_kind = load_mlx_vlm_draft_model(
        MLX_VLM_DRAFT_MODEL,
        MLX_VLM_DRAFT_KIND,
    )

    try:
        from mlx_vlm.speculative.drafters import validate_drafter_compatibility

        validate_drafter_compatibility(model, draft_model, resolved_kind)
    except ValueError as e:
        print(
            "WARNING: Speculative drafter is incompatible with the target model; "
            f"falling back to standard generation. {e}",
            file=sys.stderr,
        )
        return
    except Exception as e:
        fail(f"Failed to validate mlx_vlm draft model: {e}")

    generation_kwargs["draft_model"] = draft_model
    generation_kwargs["draft_kind"] = resolved_kind


def run_mlx_vlm_correction(prompt: str) -> str:
    """Run mlx_vlm library generation for transcript correction."""
    max_tokens = parse_int_setting("MAX_TOKENS", MAX_TOKENS)
    temperature = parse_float_setting("TEMPERATURE", TEMPERATURE)
    model, processor = load_mlx_vlm_resources(MLX_VLM_MODEL)

    try:
        from mlx_vlm import apply_chat_template, generate

        formatted_prompt = apply_chat_template(
            processor,
            model.config,
            prompt,
            num_images=0,
            num_audios=0,
            enable_thinking=ENABLE_THINKING,
        )

        generation_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "verbose": False,
            "enable_thinking": ENABLE_THINKING,
        }
        if ENABLE_THINKING:
            generation_kwargs["thinking_budget"] = parse_int_setting(
                "THINKING_BUDGET",
                THINKING_BUDGET,
                minimum=1,
            )

        add_mlx_vlm_draft_model(model, generation_kwargs)

        print_debug_call(
            "mlx_vlm.generate",
            {
                "model": MLX_VLM_MODEL,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "enable_thinking": ENABLE_THINKING,
                "thinking_budget": generation_kwargs.get("thinking_budget"),
                "draft_model": MLX_VLM_DRAFT_MODEL
                if "draft_model" in generation_kwargs
                else None,
                "draft_kind": generation_kwargs.get("draft_kind"),
                "prompt_chars": len(prompt),
            },
        )

        result = generate(model, processor, formatted_prompt, **generation_kwargs)
    except Exception as e:
        fail(f"mlx_vlm generation failed: {e}")

    return clean_model_output(result.text)


def run_text_correction(input_txt: Path, output_txt: Path) -> Path:
    """Run the configured MLX generator to correct transcript text."""
    transcript = input_txt.read_text(encoding="utf-8")
    prompt = build_correction_prompt(transcript)

    print(f"Running transcript correction on {input_txt}...")
    if GENERATION_BACKEND == "mlx_lm":
        corrected = run_mlx_lm_correction(prompt)
    elif GENERATION_BACKEND == "mlx_vlm":
        corrected = run_mlx_vlm_correction(prompt)
    else:
        fail("GENERATION_BACKEND must be 'mlx_lm' or 'mlx_vlm'.")

    if not corrected:
        fail("MLX generator did not produce corrected text.")

    output_txt.write_text(corrected, encoding="utf-8")
    print(f"Corrected transcript saved to {output_txt}")
    return output_txt
