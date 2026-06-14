# Workflow: Assemble Final Ad

## Objective
Produce a single finished MP4 ad from ordered video clips, a voiceover audio track, synced captions, and a CTA overlay. Output must conform to the 9:16 production spec at 1080x1920.

## Required Inputs
- `clips`: Ordered list of video clip paths (generated b-roll, talking head, etc.)
- `audio`: Voiceover audio file (MP3 or WAV) — must be the final rendered voiceover, not the script
- `cta_text`: The approved CTA string (e.g. "Try it free today")
- `output`: Desired output path (default: `output/final_ad.mp4`)
- `cta_start`: Timestamp (seconds) when the CTA appears (default: 8.0)

## Steps

### Step 1 — Assemble Timeline
Run `assemble_timeline.py` to concatenate clips and replace the audio track.

```bash
venv/bin/python tools/assemble_timeline.py \
  --clips clip1.mp4 clip2.mp4 clip3.mp4 \
  --audio voiceover.mp3 \
  --output .tmp/assembled.mp4
```

Verify output exists and duration matches voiceover length.

### Step 2 — Burn Captions
Run `burn_captions.py` on the assembled video. Transcription uses the voiceover audio file directly so timing is accurate.

```bash
venv/bin/python tools/burn_captions.py \
  --video .tmp/assembled.mp4 \
  --audio voiceover.mp3 \
  --output .tmp/captioned.mp4 \
  --model base
```

Review `.tmp/transcript.json` to verify word timing looks correct before proceeding.

### Step 3 — Add CTA Overlay
Run `add_motion_graphics.py` to burn the CTA text element onto the final frame window.

```bash
venv/bin/python tools/add_motion_graphics.py \
  --video .tmp/captioned.mp4 \
  --cta-text "Try it free today" \
  --output output/final_ad.mp4 \
  --cta-start 8.0 \
  --cta-duration 1.5
```

## Output
`output/final_ad.mp4` — H.264 MP4, 1080x1920, 30fps, AAC audio, captions burned in, CTA overlay present.

## Edge Cases
- **Clips shorter than expected**: Check total clip duration before assembly. If clips sum to less than voiceover length, the `--shortest` flag will cut audio — add a black hold clip to pad.
- **No words transcribed**: `burn_captions.py` will copy the video without captions and log a warning. Check that the audio file is not silent.
- **CTA runs past video end**: Ensure `cta_start + cta_duration` is less than the total video length. Extend the last clip if needed.
- **Resolution mismatch in clips**: `assemble_timeline.py` applies scale+pad to normalize all clips to 1080x1920. Source clips of any resolution are safe to pass in.
- **Re-run after partial failure**: Each step writes to `.tmp/` before the final output step. If Step 3 fails, re-run only Step 3 using `.tmp/captioned.mp4` — do not re-run Steps 1 and 2 unless their outputs are missing or corrupt.
