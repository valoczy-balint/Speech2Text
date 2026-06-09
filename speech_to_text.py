"""
Speech-to-text stage using the parakeet-mlx Python library and SRT parsing helpers.
"""

import importlib.util
import re
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, List


# PARAKEET_MODEL = "mlx-community/parakeet-tdt-1.1b"
PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
PARAKEET_USE_FP32 = False
PARAKEET_CHUNK_DURATION = 60 * 2
PARAKEET_OVERLAP_DURATION = 15
PARAKEET_LOCAL_ATTENTION = False
PARAKEET_LOCAL_ATTENTION_CONTEXT_SIZE = 256
PARAKEET_DECODING = "greedy"  # "greedy" or "beam"
PARAKEET_BEAM_SIZE = 5
PARAKEET_LENGTH_PENALTY = 0.013
PARAKEET_PATIENCE = 3.5
PARAKEET_DURATION_REWARD = 0.67
PARAKEET_SENTENCE_MAX_WORDS = None
PARAKEET_SENTENCE_SILENCE_GAP = None
PARAKEET_SENTENCE_MAX_DURATION = None


@dataclass
class TranscriptSegment:
    """A single transcription segment with timing."""
    start: float
    end: float
    text: str


def fail(message: str) -> None:
    """Print an error and exit the pipeline."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def check_stt_dependencies() -> None:
    """Check for speech-to-text dependencies."""
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg not found. Install with: brew install ffmpeg")

    if importlib.util.find_spec("parakeet_mlx") is None:
        fail(
            "parakeet-mlx is not installed in this Python environment. "
            "Run: python -m pip install -r requirements.txt"
        )


def parse_srt_timestamp(ts: str) -> float:
    """Parse SRT timestamp to seconds.

    Example: '00:00:12,345' -> 12.345
    """
    ts = ts.replace(',', '.')
    parts = ts.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def strip_tags(text: str) -> str:
    """Remove HTML-like tags from text."""
    return re.sub(r'<[^>]+>', '', text)


def parse_srt(srt_path: Path) -> List[TranscriptSegment]:
    """Parse SRT file into transcript segments."""
    segments = []

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into cues
    cues = re.split(r'\n\n+', content.strip())

    for cue in cues:
        lines = cue.strip().split('\n')
        if len(lines) < 3:
            continue

        # Line 0: index, Line 1: timestamps, Line 2+: text
        timestamp_line = lines[1]
        match = re.match(r'([\d:,]+)\s*-->\s*([\d:,]+)', timestamp_line)
        if not match:
            continue

        start = parse_srt_timestamp(match.group(1))
        end = parse_srt_timestamp(match.group(2))

        # Join text lines, strip tags, normalize whitespace
        text = ' '.join(lines[2:])
        text = strip_tags(text)
        text = ' '.join(text.split())

        if text:
            segments.append(TranscriptSegment(start=start, end=end, text=text))

    return segments


@lru_cache(maxsize=1)
def load_parakeet_model(model_name: str) -> Any:
    """Load and cache the parakeet-mlx model."""
    try:
        import mlx.core as mx
        from parakeet_mlx import from_pretrained
    except Exception as e:
        fail(f"Failed to import parakeet-mlx: {e}")

    try:
        model = from_pretrained(
            model_name,
            dtype=mx.float32 if PARAKEET_USE_FP32 else mx.bfloat16,
        )
    except Exception as e:
        fail(f"Failed to load parakeet model {model_name!r}: {e}")

    if PARAKEET_LOCAL_ATTENTION:
        try:
            model.encoder.set_attention_model(
                "rel_pos_local_attn",
                (
                    PARAKEET_LOCAL_ATTENTION_CONTEXT_SIZE,
                    PARAKEET_LOCAL_ATTENTION_CONTEXT_SIZE,
                ),
            )
        except Exception as e:
            fail(f"Failed to configure parakeet local attention: {e}")

    return model


def build_decoding_config() -> Any:
    """Build the default parakeet decoding config used by the CLI."""
    try:
        from parakeet_mlx import Beam, DecodingConfig, Greedy, SentenceConfig
    except Exception as e:
        fail(f"Failed to import parakeet decoding config: {e}")

    if PARAKEET_DECODING == "beam":
        decoding = Beam(
            beam_size=PARAKEET_BEAM_SIZE,
            length_penalty=PARAKEET_LENGTH_PENALTY,
            patience=PARAKEET_PATIENCE,
            duration_reward=PARAKEET_DURATION_REWARD,
        )
    elif PARAKEET_DECODING == "greedy":
        decoding = Greedy()
    else:
        fail("PARAKEET_DECODING must be 'greedy' or 'beam'.")

    return DecodingConfig(
        decoding=decoding,
        sentence=SentenceConfig(
            max_words=PARAKEET_SENTENCE_MAX_WORDS,
            silence_gap=PARAKEET_SENTENCE_SILENCE_GAP,
            max_duration=PARAKEET_SENTENCE_MAX_DURATION,
        ),
    )


def run_parakeet(audio_path: Path, output_dir: Path) -> Path:
    """Run parakeet-mlx library transcription to generate an SRT transcript."""
    print(f"Running parakeet-mlx on {audio_path}...")

    model = load_parakeet_model(PARAKEET_MODEL)
    output_srt = output_dir / audio_path.with_suffix(".srt").name

    try:
        import mlx.core as mx
        from parakeet_mlx.cli import to_srt

        result = model.transcribe(
            audio_path,
            dtype=mx.float32 if PARAKEET_USE_FP32 else mx.bfloat16,
            chunk_duration=PARAKEET_CHUNK_DURATION
            if PARAKEET_CHUNK_DURATION != 0
            else None,
            overlap_duration=PARAKEET_OVERLAP_DURATION,
            decoding_config=build_decoding_config(),
        )
        output_srt.write_text(to_srt(result, highlight_words=False), encoding="utf-8")
    except Exception as e:
        fail(f"parakeet-mlx transcription failed: {e}")

    if not output_srt.exists():
        fail(f"Expected SRT file not found: {output_srt}")

    print(f"Transcript saved to {output_srt}")
    return output_srt
