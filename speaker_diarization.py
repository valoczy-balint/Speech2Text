"""
Speaker diarization stage using pyannote.audio.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DiarizationTurn:
    """A single speaker turn from diarization."""
    start: float
    end: float
    speaker: str


def check_diarization_dependencies() -> None:
    """Check for speaker diarization dependencies."""
    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
        print("Export your Hugging Face token: export HF_TOKEN='hf_...'", file=sys.stderr)
        sys.exit(1)


def run_diarization(
    audio_path: Path,
    model: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> List[DiarizationTurn]:
    """Run pyannote speaker diarization."""
    from pyannote.audio import Pipeline

    print(f"Loading diarization model: {model}...")
    pipeline = Pipeline.from_pretrained(model, token=os.environ["HF_TOKEN"])
    if pipeline is None:
        print(
            f"ERROR: Could not load diarization model: {model}. "
            "Check your HF_TOKEN and confirm you accepted the model terms.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build kwargs
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    print(f"Running diarization on {audio_path}...")
    if kwargs:
        print(f"Diarization parameters: {kwargs}")

    output = pipeline(str(audio_path), **kwargs)

    # Handle different output formats
    diarization = getattr(output, "exclusive_speaker_diarization", None)
    if diarization is None:
        diarization = getattr(output, "speaker_diarization", output)

    # Convert to list of turns
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(DiarizationTurn(
            start=turn.start,
            end=turn.end,
            speaker=speaker
        ))

    print(f"Found {len(set(t.speaker for t in turns))} speakers in {len(turns)} turns")
    return turns
