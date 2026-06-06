#!/usr/bin/env python3
"""
Speaker-labeled transcription using parakeet-mlx and pyannote.audio.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"


@dataclass
class TranscriptSegment:
    """A single transcription segment with timing."""
    start: float
    end: float
    text: str


@dataclass
class DiarizationTurn:
    """A single speaker turn from diarization."""
    start: float
    end: float
    speaker: str


@dataclass
class DialogueTurn:
    """A dialogue turn with speaker, text, and timing."""
    speaker: str
    text: str
    start: float
    end: float


def check_dependencies() -> None:
    """Check for required dependencies."""
    # Check ffmpeg
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found. Install with: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)
    
    # Check parakeet-mlx
    if shutil.which("parakeet-mlx") is None:
        print("ERROR: parakeet-mlx not found. Install globally first.", file=sys.stderr)
        sys.exit(1)
    
    # Check HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
        print("Export your Hugging Face token: export HF_TOKEN='hf_...'", file=sys.stderr)
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
    
    run_started_at = time.time() - 5
    try:
        completed = subprocess.run(
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
    candidates = [
        output_srt,
        audio_path.with_suffix(".srt"),
        SCRIPT_DIR / output_srt.name,
        Path.cwd() / output_srt.name,
    ]
    candidates = list(dict.fromkeys(candidates))

    generated_srt = next(
        (
            candidate
            for candidate in candidates
            if candidate.exists() and candidate.stat().st_mtime >= run_started_at
        ),
        None,
    )

    if generated_srt is None:
        checked_paths = "\n  ".join(str(candidate) for candidate in candidates)
        print(
            "ERROR: Expected SRT file not found. Checked:\n"
            f"  {checked_paths}",
            file=sys.stderr,
        )
        if completed.stdout:
            print(f"parakeet-mlx stdout:\n{completed.stdout}", file=sys.stderr)
        if completed.stderr:
            print(f"parakeet-mlx stderr:\n{completed.stderr}", file=sys.stderr)
        sys.exit(1)

    if generated_srt != output_srt:
        output_srt.unlink(missing_ok=True)
        shutil.move(str(generated_srt), output_srt)
    
    print(f"Transcript saved to {output_srt}")
    return output_srt


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


def merge_transcript_with_diarization(
    segments: List[TranscriptSegment],
    turns: List[DiarizationTurn],
) -> List[DialogueTurn]:
    """Merge transcript segments with diarization turns by timestamp overlap."""
    dialogue = []
    
    for segment in segments:
        # Calculate overlap with each speaker
        speaker_scores: Dict[str, float] = {}
        
        for turn in turns:
            overlap = max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))
            if overlap > 0:
                speaker_scores[turn.speaker] = speaker_scores.get(turn.speaker, 0.0) + overlap
        
        # Assign speaker with most overlap
        if speaker_scores:
            assigned_speaker = max(speaker_scores, key=speaker_scores.get)
        else:
            assigned_speaker = "UNKNOWN"
        
        dialogue.append(DialogueTurn(
            speaker=assigned_speaker,
            text=segment.text,
            start=segment.start,
            end=segment.end
        ))
    
    return dialogue


def group_adjacent_turns(dialogue: List[DialogueTurn]) -> List[DialogueTurn]:
    """Group adjacent segments from the same speaker."""
    if not dialogue:
        return []
    
    grouped = []
    current = dialogue[0]
    
    for turn in dialogue[1:]:
        if turn.speaker == current.speaker:
            # Merge with current
            current = DialogueTurn(
                speaker=current.speaker,
                text=current.text + " " + turn.text,
                start=current.start,
                end=turn.end
            )
        else:
            grouped.append(current)
            current = turn
    
    grouped.append(current)
    return grouped


def apply_speaker_map(dialogue: List[DialogueTurn], speaker_map: Dict[str, str]) -> List[DialogueTurn]:
    """Apply speaker name mapping."""
    return [
        DialogueTurn(
            speaker=speaker_map.get(turn.speaker, turn.speaker),
            text=turn.text,
            start=turn.start,
            end=turn.end
        )
        for turn in dialogue
    ]


def write_dialogue_txt(dialogue: List[DialogueTurn], output_path: Path) -> None:
    """Write human-readable dialogue transcript."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for turn in dialogue:
            f.write(f"{turn.speaker}: {turn.text}\n\n")
    
    print(f"Dialogue saved to {output_path}")


def write_dialogue_json(dialogue: List[DialogueTurn], output_path: Path) -> None:
    """Write machine-readable JSON transcript."""
    data = [
        {
            "speaker": turn.speaker,
            "text": turn.text,
            "start": turn.start,
            "end": turn.end
        }
        for turn in dialogue
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON saved to {output_path}")


def resolve_input_path(path: Path) -> Path:
    """Resolve a CLI path as an input-folder path."""
    parts = path.parts
    if parts and parts[0] == INPUT_DIR.name:
        return SCRIPT_DIR / path

    return INPUT_DIR / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Speaker-labeled transcription using parakeet-mlx and pyannote.audio"
    )
    
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Input audio filename inside input/ (.m4a, .wav, .mp3, .flac, etc.)"
    )
    
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Exact number of speakers"
    )
    
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers"
    )
    
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers"
    )
    
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-community-1",
        help="Pyannote model (default: pyannote/speaker-diarization-community-1)"
    )
    
    parser.add_argument(
        "--speaker-map",
        type=Path,
        help="JSON file mapping speaker labels to names"
    )
    
    args = parser.parse_args()

    if args.audio_file.is_absolute():
        parser.error("audio_file must be a filename or relative path inside input/")

    if args.speaker_map is not None and args.speaker_map.is_absolute():
        parser.error("--speaker-map must be a filename or relative path inside input/")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args.audio_file = resolve_input_path(args.audio_file)
    if args.speaker_map is not None:
        args.speaker_map = resolve_input_path(args.speaker_map)
    
    # Validate speaker arguments
    if args.num_speakers is not None and (
        args.min_speakers is not None or args.max_speakers is not None
    ):
        parser.error("Cannot specify --num-speakers with --min-speakers or --max-speakers")

    for name in ("num_speakers", "min_speakers", "max_speakers"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")

    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        parser.error("--min-speakers cannot be greater than --max-speakers")
    
    # Validate audio file exists
    if not args.audio_file.exists():
        parser.error(f"Audio file not found: {args.audio_file}")

    if args.speaker_map is not None and not args.speaker_map.exists():
        parser.error(f"Speaker map not found: {args.speaker_map}")
    
    return args


def main() -> None:
    """Main entry point."""
    # Disable pyannote telemetry
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    
    args = parse_args()
    check_dependencies()
    
    print(f"\n=== Speaker-Labeled Transcription ===")
    print(f"Input: {args.audio_file}")
    print(
        "Outputs: "
        f"{OUTPUT_DIR / args.audio_file.with_suffix('.srt').name}, "
        f"{OUTPUT_DIR / args.audio_file.with_suffix('.txt').name}, "
        f"{OUTPUT_DIR / args.audio_file.with_suffix('.json').name}\n"
    )
    
    # Run parakeet transcription
    srt_path = run_parakeet(args.audio_file, OUTPUT_DIR)
    
    # Parse SRT
    print("Parsing transcript...")
    segments = parse_srt(srt_path)
    print(f"Found {len(segments)} transcript segments")
    
    # Run diarization
    turns = run_diarization(
        args.audio_file,
        args.model,
        args.num_speakers,
        args.min_speakers,
        args.max_speakers
    )
    
    # Merge transcript with diarization
    print("Merging transcript with speaker labels...")
    dialogue = merge_transcript_with_diarization(segments, turns)
    
    # Group adjacent turns
    dialogue = group_adjacent_turns(dialogue)
    print(f"Grouped into {len(dialogue)} dialogue turns")
    
    # Apply speaker mapping if provided
    if args.speaker_map:
        print(f"Applying speaker mapping from {args.speaker_map}...")
        with open(args.speaker_map, 'r', encoding='utf-8') as f:
            speaker_map = json.load(f)
        dialogue = apply_speaker_map(dialogue, speaker_map)
    
    # Write outputs
    dialogue_txt = OUTPUT_DIR / args.audio_file.with_suffix(".txt").name
    dialogue_json = OUTPUT_DIR / args.audio_file.with_suffix(".json").name
    
    write_dialogue_txt(dialogue, dialogue_txt)
    write_dialogue_json(dialogue, dialogue_json)
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
