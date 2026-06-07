#!/usr/bin/env python3
"""
Speaker-labeled transcription using parakeet-mlx and pyannote.audio.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

from diarization_matching import (
    DialogueTurn,
    apply_speaker_map,
    group_adjacent_turns,
    merge_transcript_with_diarization,
)
from speaker_diarization import check_diarization_dependencies, run_diarization
from speech_to_text import check_stt_dependencies, parse_srt, run_parakeet
from text_correction import check_text_correction_dependencies, run_text_correction


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
TimingReport = List[Tuple[str, float]]


def check_dependencies() -> None:
    """Check for required pipeline dependencies."""
    check_stt_dependencies()
    check_diarization_dependencies()
    check_text_correction_dependencies()


def format_duration(seconds: float) -> str:
    """Format elapsed seconds for human-readable terminal output."""
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.2f}s"


def finish_step(timings: TimingReport, step_name: str, started_at: float) -> None:
    """Record and print the elapsed time for a completed step."""
    elapsed = time.perf_counter() - started_at
    timings.append((step_name, elapsed))
    print(f"{step_name} completed in {format_duration(elapsed)}")


def print_timing_report(timings: TimingReport, total_elapsed: float) -> None:
    """Print a final timing report for the run."""
    print("\n=== Timing Report ===")
    for step_name, elapsed in timings:
        print(f"{step_name}: {format_duration(elapsed)}")
    print(f"Total: {format_duration(total_elapsed)}")


def write_dialogue_txt(dialogue: List[DialogueTurn], output_path: Path) -> None:
    """Write human-readable dialogue transcript."""
    lines = [
        f"{turn.speaker}: {turn.text.strip()}"
        for turn in dialogue
        if turn.text.strip()
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    
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


def output_dir_for(audio_path: Path) -> Path:
    """Return the run-specific output directory for an input audio file."""
    return OUTPUT_DIR / audio_path.stem


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Speaker-labeled transcription with STT, diarization, "
            "and text correction"
        )
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
    total_started_at = time.perf_counter()
    timings: TimingReport = []

    # Disable pyannote telemetry
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

    step_started_at = time.perf_counter()
    args = parse_args()
    finish_step(timings, "Argument parsing and validation", step_started_at)

    step_started_at = time.perf_counter()
    check_dependencies()
    finish_step(timings, "Dependency checks", step_started_at)

    step_started_at = time.perf_counter()
    run_output_dir = output_dir_for(args.audio_file)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    finish_step(timings, "Output directory setup", step_started_at)
    
    print(f"\n=== Speaker-Labeled Transcription ===")
    print(f"Input: {args.audio_file}")
    print(
        "Outputs: "
        f"{run_output_dir / args.audio_file.with_suffix('.srt').name}, "
        f"{run_output_dir / args.audio_file.with_suffix('.txt').name}, "
        f"{run_output_dir / args.audio_file.with_suffix('.json').name}, "
        f"{run_output_dir / args.audio_file.with_suffix('.corrected.txt').name}\n"
    )
    
    # Run parakeet transcription
    step_started_at = time.perf_counter()
    srt_path = run_parakeet(args.audio_file, run_output_dir)
    finish_step(timings, "Speech-to-text", step_started_at)
    
    # Parse SRT
    print("Parsing transcript...")
    step_started_at = time.perf_counter()
    segments = parse_srt(srt_path)
    print(f"Found {len(segments)} transcript segments")
    finish_step(timings, "SRT parsing", step_started_at)
    
    # Run diarization
    step_started_at = time.perf_counter()
    turns = run_diarization(
        args.audio_file,
        args.model,
        args.num_speakers,
        args.min_speakers,
        args.max_speakers
    )
    finish_step(timings, "Speaker diarization", step_started_at)
    
    # Merge transcript with diarization
    print("Merging transcript with speaker labels...")
    step_started_at = time.perf_counter()
    dialogue = merge_transcript_with_diarization(segments, turns)
    finish_step(timings, "Transcript and speaker matching", step_started_at)
    
    # Group adjacent turns
    step_started_at = time.perf_counter()
    dialogue = group_adjacent_turns(dialogue)
    print(f"Grouped into {len(dialogue)} dialogue turns")
    finish_step(timings, "Adjacent turn grouping", step_started_at)
    
    # Apply speaker mapping if provided
    if args.speaker_map:
        print(f"Applying speaker mapping from {args.speaker_map}...")
        step_started_at = time.perf_counter()
        with open(args.speaker_map, 'r', encoding='utf-8') as f:
            speaker_map = json.load(f)
        dialogue = apply_speaker_map(dialogue, speaker_map)
        finish_step(timings, "Speaker mapping", step_started_at)
    
    # Write outputs
    dialogue_txt = run_output_dir / args.audio_file.with_suffix(".txt").name
    dialogue_json = run_output_dir / args.audio_file.with_suffix(".json").name
    
    step_started_at = time.perf_counter()
    write_dialogue_txt(dialogue, dialogue_txt)
    write_dialogue_json(dialogue, dialogue_json)
    finish_step(timings, "Output writing", step_started_at)

    corrected_txt = run_output_dir / args.audio_file.with_suffix(".corrected.txt").name
    step_started_at = time.perf_counter()
    run_text_correction(dialogue_txt, corrected_txt)
    finish_step(timings, "Text correction", step_started_at)
    
    print_timing_report(timings, time.perf_counter() - total_started_at)
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
