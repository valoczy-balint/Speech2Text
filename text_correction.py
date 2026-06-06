"""
Transcript correction stage using mlx_vlm.generate.
"""

import shutil
import subprocess
import sys
from pathlib import Path


MODEL = "mlx-community/gemma-4-12B-it-qat-4bit"
DRAFT_MODEL = "mlx-community/gemma-4-12B-it-qat-assistant-4bit"
DRAFT_KIND = "mtp"
MAX_TOKENS = "2560"
TEMPERATURE = "0"


def check_text_correction_dependencies() -> None:
    """Check for text correction dependencies."""
    if shutil.which("mlx_vlm.generate") is None:
        print("ERROR: mlx_vlm.generate not found. Install mlx-vlm first.", file=sys.stderr)
        sys.exit(1)


def build_correction_prompt(transcript: str) -> str:
    """Build the fixed prompt for transcript correction."""
    return (
        "You are an expert multilingual transcript editor.\n"
        "Correct speech-to-text transcription mistakes in the transcript below.\n"
        "Rules:\n"
        "- Preserve speaker labels exactly.\n"
        "- Preserve line order and line breaks.\n"
        "- Keep every passage in its original language; do not translate.\n"
        "- Fix obvious ASR word errors in all languages.\n"
        "- Do not summarize, explain, add comments, or wrap the result in Markdown.\n"
        "- Return only the corrected transcript.\n\n"
        "Transcript:\n"
        "<<<TRANSCRIPT\n"
        f"{transcript.rstrip()}\n"
        "TRANSCRIPT>>>"
    )


def clean_model_output(output: str) -> str:
    """Remove known mlx_vlm.generate status lines from captured stdout."""
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


def run_text_correction(input_txt: Path, output_txt: Path) -> Path:
    """Run mlx_vlm.generate to correct transcript text."""
    transcript = input_txt.read_text(encoding="utf-8")
    prompt = build_correction_prompt(transcript)

    print(f"Running transcript correction on {input_txt}...")
    cmd = [
        "mlx_vlm.generate",
        "--model", MODEL,
        "--draft-model", DRAFT_MODEL,
        "--draft-kind", DRAFT_KIND,
        "--prompt", prompt,
        "--max-tokens", MAX_TOKENS,
        "--temperature", TEMPERATURE,
        "--no-verbose",
    ]

    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: mlx_vlm.generate failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    corrected = clean_model_output(completed.stdout)
    if not corrected:
        print("ERROR: mlx_vlm.generate did not produce corrected text.", file=sys.stderr)
        sys.exit(1)

    output_txt.write_text(corrected, encoding="utf-8")
    print(f"Corrected transcript saved to {output_txt}")
    return output_txt
