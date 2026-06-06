"""
Speech-to-text stage using parakeet-mlx and SRT parsing helpers.
"""

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TranscriptSegment:
    """A single transcription segment with timing."""
    start: float
    end: float
    text: str


def check_stt_dependencies() -> None:
    """Check for speech-to-text dependencies."""
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found. Install with: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)

    if shutil.which("parakeet-mlx") is None:
        print("ERROR: parakeet-mlx not found. Install globally first.", file=sys.stderr)
        sys.exit(1)


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


def run_parakeet(audio_path: Path, output_dir: Path) -> Path:
    """Run parakeet-mlx to generate SRT transcript."""
    print(f"Running parakeet-mlx on {audio_path}...")

    # Run from output/ because parakeet-mlx writes SRT files relative to cwd.
    cmd = [
        "parakeet-mlx",
        str(audio_path),
        "--output-format", "srt",
        "--no-highlight-words"
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=output_dir,
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: parakeet-mlx failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    output_srt = output_dir / audio_path.with_suffix(".srt").name

    if not output_srt.exists():
        print(f"ERROR: Expected SRT file not found: {output_srt}", file=sys.stderr)
        sys.exit(1)

    print(f"Transcript saved to {output_srt}")
    return output_srt
