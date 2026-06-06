"""
Transcript and diarization matching helpers.
"""

from dataclasses import dataclass
from typing import Dict, List

from speaker_diarization import DiarizationTurn
from speech_to_text import TranscriptSegment


@dataclass
class DialogueTurn:
    """A dialogue turn with speaker, text, and timing."""
    speaker: str
    text: str
    start: float
    end: float


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
