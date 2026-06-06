#!/usr/bin/env python3
"""
Speaker-labeled transcription using parakeet-mlx and pyannote.audio.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from speaker_diarization import (
    DiarizationTurn,
    check_diarization_dependencies,
    run_diarization,
)
from stt import TranscriptSegment, check_stt_dependencies, parse_srt, run_parakeet
from text_correction import check_text_correction_dependencies, run_text_correction


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"


@dataclass
class DialogueTurn:
    """A dialogue turn with speaker, text, and timing."""
    speaker: str
    text: str
    start: float
    end: float


def check_dependencies() -> None:
    """Check for required pipeline dependencies."""
    check_stt_dependencies()
    check_diarization_dependencies()
    check_text_correction_dependencies()


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
        f"{OUTPUT_DIR / args.audio_file.with_suffix('.json').name}, "
        f"{OUTPUT_DIR / args.audio_file.with_suffix('.corrected.txt').name}\n"
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

    corrected_txt = OUTPUT_DIR / args.audio_file.with_suffix(".corrected.txt").name
    run_text_correction(dialogue_txt, corrected_txt)
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
