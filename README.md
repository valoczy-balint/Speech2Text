# Speaker-Labeled Transcription

A macOS command-line tool for transcribing audio files with speaker diarization (speaker separation).

Uses:
- **parakeet-mlx** for fast, accurate transcription on Apple Silicon
- **pyannote.audio** for speaker diarization (identifying who spoke when)
- **mlx-vlm** for transcript text correction

## Prerequisites

### Install System Dependencies

```bash
# Install ffmpeg
brew install ffmpeg

# Install parakeet-mlx globally
pip3 install parakeet-mlx

# Install mlx-vlm globally
pip3 install mlx-vlm
```

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

```bash
# Set Hugging Face Token
export HF_TOKEN="hf_..."
```

### Basic Usage

Put audio files in `input/`, then pass the filename to the script. Generated files are written to `output/`.

```bash
# Default (auto-detect speaker count)
python main.py input.m4a

# Specify exact number of speakers
python main.py input.m4a --num-speakers 2

# Specify min/max speaker range
python main.py input.m4a --min-speakers 2 --max-speakers 4
```

### Advanced Usage

```bash
# Use speaker name mapping
echo '{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}' > input/speakers.json
python main.py input.m4a --speaker-map speakers.json

# Use different diarization model
python main.py input.m4a --model pyannote/speaker-diarization-community-1
```

### Output Files

The script writes outputs to `output/`:

1. **`<audio-name>.srt`** - Raw timestamped transcript from parakeet
2. **`<audio-name>.txt`** - Human-readable speaker-labeled dialogue:
   ```
   SPEAKER_00: Hello, this is the first speaker.
   SPEAKER_01: And this is the second speaker.
   SPEAKER_00: Right, exactly.
   ```

3. **`<audio-name>.json`** - Machine-readable JSON with timestamps:
   ```json
   [
     {
       "speaker": "SPEAKER_00",
       "text": "Hello, this is the first speaker.",
       "start": 0.5,
       "end": 3.2
     },
     ...
   ]
   ```

4. **`<audio-name>.corrected.txt`** - Text transcript after MLX-VLM correction

## Command-Line Options

| Option | Description |
|--------|-------------|
| `audio_file` | Input audio filename inside `input/` (required) |
| `--num-speakers N` | Exact number of speakers |
| `--min-speakers N` | Minimum number of speakers |
| `--max-speakers N` | Maximum number of speakers |
| `--model MODEL` | Pyannote model (default: `pyannote/speaker-diarization-community-1`) |
| `--speaker-map FILE` | JSON file inside `input/` mapping speaker labels to names |

**Note:** You can specify either `--num-speakers` OR `--min-speakers`/`--max-speakers`, but not both.

## How It Works

1. **Transcription**: Runs globally-installed `parakeet-mlx` to generate timestamped transcript (SRT format)
2. **Diarization**: Uses `pyannote.audio` to identify speaker segments
3. **Merging**: Assigns speakers to transcript segments by timestamp overlap
4. **Grouping**: Combines adjacent segments from the same speaker
5. **Output**: Moves the SRT and writes readable dialogue plus JSON files to `output/`
6. **Correction**: Runs `mlx_vlm.generate` over the TXT transcript and writes `<audio-name>.corrected.txt`


## Troubleshooting

Install ffmpeg
 `brew install ffmpeg`

Install parakeet-mlx
 `pip3 install parakeet-mlx`

Export your Hugging Face token
Generate an access token at https://huggingface.co/settings/tokens
 `export HF_TOKEN="hf_..."`
Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-community-1

## Limitations

- **Speaker labels are arbitrary** (`SPEAKER_00`, `SPEAKER_01`, etc.) - use `--speaker-map` to assign meaningful names
- **Overlapping speech**: Model may struggle with simultaneous speakers
- **Similar voices**: May confuse speakers with similar vocal characteristics
- **Background noise**: Affects accuracy of both transcription and diarization
- **Short backchannels**: Brief interjections like "mm-hmm" may be misattributed
- **Speaker count**: Better results when you know the exact number of speakers
