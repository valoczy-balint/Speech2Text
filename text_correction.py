"""
Transcript correction stage using MLX text generation CLIs.
"""

import shutil
import shlex
import subprocess
import sys
from pathlib import Path


# Common generation settings
GENERATION_BACKEND = "mlx_vlm"  # "mlx_lm" or "mlx_vlm"
DEBUG_PRINT_COMMAND = True
MAX_TOKENS = "2560"
TEMPERATURE = "1.0"
ENABLE_THINKING = False
THINKING_BUDGET = "256"

# mlx_lm.generate settings
MLX_LM_COMMAND = "mlx_lm.generate"
MLX_LM_MODEL = "mlx-community/gemma-4-31B-it-uncensored-heretic-4bit"

# mlx_vlm.generate settings
MLX_VLM_COMMAND = "mlx_vlm.generate"
# MLX_VLM_MODEL = "mlx-community/gemma-4-12B-it-qat-4bit"
# MLX_VLM_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
MLX_VLM_MODEL = "mlx-community/gemma-4-E4B-it-bf16"
MLX_VLM_USE_DRAFT_MODEL = False
# MLX_VLM_DRAFT_MODEL = "mlx-community/gemma-4-12B-it-qat-assistant-4bit"
MLX_VLM_DRAFT_MODEL = "mlx-community/gemma-4-12B-it-qat-assistant-4bit"
MLX_VLM_DRAFT_KIND = "mtp"


def check_text_correction_dependencies() -> None:
    """Check for text correction dependencies."""
    command = {
        "mlx_lm": MLX_LM_COMMAND,
        "mlx_vlm": MLX_VLM_COMMAND,
    }.get(GENERATION_BACKEND)

    if command is None:
        print(
            "ERROR: GENERATION_BACKEND must be 'mlx_lm' or 'mlx_vlm'.",
            file=sys.stderr,
        )
        sys.exit(1)

    if shutil.which(command) is None:
        print(
            f"ERROR: {command} not found. Install the matching MLX package first.",
            file=sys.stderr,
        )
        sys.exit(1)


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
    """Remove known MLX generator status lines from captured stdout."""
    lines = output.replace("\r\n", "\n").replace("\r", "\n").splitlines()

    while lines and (
        lines[0].startswith("Loading drafter")
        or lines[0].startswith("  ->")
        or lines[0].startswith("  \u2192")
    ):
        lines.pop(0)

    while lines and lines[-1].startswith("Speculative decoding:"):
        lines.pop()

    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        lines = lines[1:-1]

    cleaned = "\n".join(lines).strip()
    return f"{cleaned}\n" if cleaned else ""


def print_debug_command(cmd: list[str]) -> None:
    """Print the command that will be executed when debug output is enabled."""
    if DEBUG_PRINT_COMMAND:
        print("LLM command:")
        print(shlex.join(cmd))


def run_mlx_lm_correction(prompt: str) -> str:
    """Run mlx_lm.generate for transcript correction."""
    cmd = [
        MLX_LM_COMMAND,
        "--model", MLX_LM_MODEL,
        "--prompt", prompt,
        "--max-tokens", MAX_TOKENS,
        "--temp", TEMPERATURE,
        "--verbose", "False",
    ]
    print_debug_command(cmd)

    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {MLX_LM_COMMAND} failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    return clean_model_output(completed.stdout)


def run_mlx_vlm_correction(prompt: str) -> str:
    """Run mlx_vlm.generate for transcript correction."""
    cmd = [
        MLX_VLM_COMMAND,
        "--model", MLX_VLM_MODEL,
        "--prompt", prompt,
        "--max-tokens", MAX_TOKENS,
        "--temperature", TEMPERATURE,
        "--no-verbose",
    ]
    if ENABLE_THINKING:
        cmd.extend([
            "--enable-thinking",
            "--thinking-budget", THINKING_BUDGET
        ])
    if MLX_VLM_USE_DRAFT_MODEL:
        cmd.extend([
            "--draft-model", MLX_VLM_DRAFT_MODEL,
            "--draft-kind", MLX_VLM_DRAFT_KIND,
        ])
    print_debug_command(cmd)

    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {MLX_VLM_COMMAND} failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    return clean_model_output(completed.stdout)


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
        print(
            "ERROR: GENERATION_BACKEND must be 'mlx_lm' or 'mlx_vlm'.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not corrected:
        print("ERROR: MLX generator did not produce corrected text.", file=sys.stderr)
        sys.exit(1)

    output_txt.write_text(corrected, encoding="utf-8")
    print(f"Corrected transcript saved to {output_txt}")
    return output_txt
