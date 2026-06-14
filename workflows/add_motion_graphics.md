# Workflow: Add Motion Graphics (CTA Overlay)

## Objective
Burn a Call-to-Action (CTA) text element onto the final video at a specified time. The CTA must be clearly legible, high contrast, and on screen for a minimum of 1.5 seconds during the designated CTA window (seconds 8–12 of a standard ad).

## Required Inputs
- `video`: Captioned video (from `burn_captions.py`) or assembled video if skipping captions
- `cta_text`: The approved CTA string (e.g. "Try it free today", "Book your free call")
- `output`: Output path (default: `.tmp/with_cta.mp4`)
- `cta_start`: Seconds when CTA appears (default: 8.0 — start of the CTA window)
- `cta_duration`: Seconds CTA stays on screen (default: 1.5 — minimum required)

## Steps

### Step 1 — Confirm CTA Text is Approved
Verify the CTA string matches the approved copy. Do not improvise CTA wording. If uncertain, ask before running.

### Step 2 — Run the Overlay
```bash
venv/bin/python tools/add_motion_graphics.py \
  --video .tmp/captioned.mp4 \
  --cta-text "Try it free today" \
  --output output/final_ad.mp4 \
  --cta-start 8.0 \
  --cta-duration 1.5
```

### Step 3 — Verify CTA in Output
Play `output/final_ad.mp4` and confirm:
- CTA text appears at the correct time
- CTA text is fully legible (high contrast yellow on black or dark background)
- CTA remains on screen for the full duration specified
- No part of the CTA text is clipped at the edges

## CTA Spec

| Property | Value |
|---|---|
| Font size | 64px |
| Color | Yellow (#FFD700) |
| Outline | Black, 4px border |
| Weight | Bold |
| Minimum on-screen duration | 1.5 seconds |
| Horizontal position | Centered (x = (w - text_w) / 2) |
| Vertical position | 88% of frame height (y = 1689 at 1920px tall) |
| Safe zone | x: 108–972px, y: 192–1728px (10% margin each edge at 1080x1920) |

The CTA y position (1689px) is within the safe zone (max 1728px). Keep it there. Do not raise the CTA above caption text — the two elements occupy different vertical bands (captions at 75%, CTA at 88%).

## Output
`output/final_ad.mp4` (or specified path) — final deliverable with CTA burned in. H.264, AAC, 1080x1920.

## Edge Cases
- **CTA extends past video end**: If `cta_start + cta_duration` exceeds video length, the CTA will appear but may cut off. Verify total video duration with `ffprobe` before running, and adjust `cta_start` if needed.
- **CTA text too long to fit**: Long strings may overflow horizontally. Test with short phrases (3–5 words). If overflow occurs, break the CTA into two lines using `\n` in the text (requires escaping in shell).
- **Low contrast background**: The yellow + black outline combo maintains readability on most backgrounds. If the background at second 8 is very bright (e.g. a white product shot), consider a background box or adjust the clip order so the CTA window falls on a darker frame.
- **Multiple CTA variants**: If A/B testing CTA copy, run `add_motion_graphics.py` twice with different `--cta-text` and `--output` values. Never modify the captioned intermediate — use it as the input for each variant.
- **CTA required earlier than 8s**: Override `--cta-start` as needed. The 8s default reflects the standard three-part ad structure (Hook 0–3s, Value Build 3–8s, CTA 8–12s). A shorter ad may need an earlier CTA start.
