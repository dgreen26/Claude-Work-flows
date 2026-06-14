# Workflow: Burn Captions

## Objective
Transcribe a voiceover audio file at the word level and burn synced captions onto a video. Captions are derived from the actual spoken audio — not the script — so timing matches exactly what was recorded.

## Required Inputs
- `video`: Assembled video file (from `assemble_timeline.py` or any MP4)
- `audio`: The voiceover audio file used for transcription (MP3 or WAV)
- `output`: Output path (default: `.tmp/captioned.mp4`)
- `model`: Whisper model size — `base` for speed, `small` or `medium` for accuracy on complex speech

## Steps

### Step 1 — Transcribe Audio
`burn_captions.py` loads the faster-whisper model and transcribes the audio with `word_timestamps=True`. The word-level transcript is saved to `.tmp/transcript.json`.

```bash
venv/bin/python tools/burn_captions.py \
  --video .tmp/assembled.mp4 \
  --audio voiceover.mp3 \
  --output .tmp/captioned.mp4 \
  --model base
```

### Step 2 — Review Transcript
Open `.tmp/transcript.json` and spot-check that word start/end times look correct. If timing drifts on key words (especially the CTA phrase), switch to a larger model (`--model small` or `--model medium`) and re-run.

### Step 3 — Verify Caption Output
Play `.tmp/captioned.mp4` and confirm:
- Captions are readable and synced to speech
- No caption text is clipped at the edges
- Text sits inside the safe zone (see Caption Spec below)

## Caption Spec

| Property | Value |
|---|---|
| Font size | 72px |
| Color | White |
| Outline | Black, 4px border |
| Weight | Bold |
| Max words per segment | 3 |
| Horizontal position | Centered (x = (w - text_w) / 2) |
| Vertical position | 75% of frame height (y = 1440 at 1920px tall) |
| Safe zone | x: 108–972px, y: 192–1728px (10% margin each edge at 1080x1920) |

The default y position (1440px) is within the safe zone. Do not move captions above 192px or below 1728px.

## Output
`.tmp/captioned.mp4` — same resolution and codec as input, with captions burned into the video stream. `.tmp/transcript.json` — word-level transcript with start/end timestamps.

## Edge Cases
- **Silent audio / no words detected**: Script copies the video without captions and logs a warning. Check that the correct audio file was passed, not a music bed or empty file.
- **Poor transcription accuracy**: Upgrade model size. `base` is fast but may struggle with fast speech or background noise. `medium` is significantly more accurate.
- **Captions overrunning into each other**: Whisper word timestamps can occasionally overlap. If this happens, post-process `.tmp/transcript.json` to add a minimum 0.05s gap between segment end and next start.
- **Non-English audio**: Pass `--language` through to faster-whisper if needed (requires script modification). Default auto-detects language.
- **Long pauses in audio**: Words separated by long pauses will produce segments with wide gaps, which is correct — the caption simply will not appear on screen during silence.
