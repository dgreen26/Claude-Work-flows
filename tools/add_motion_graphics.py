"""Overlay a CTA text element on a video using ffmpeg drawtext."""

import argparse
import sys
from pathlib import Path

import ffmpeg

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.utils import log

TMP_DIR = Path(__file__).parent.parent / ".tmp"
TMP_DIR.mkdir(exist_ok=True)

# CTA style spec
FONT_SIZE = 64
FONT_COLOR = "#FFD700"   # yellow
BORDER_COLOR = "black"
BORDER_WIDTH = 4
# At 1080x1920: 88% of 1920 = 1689.6 — inside safe zone (max y safe = 1728)
CTA_Y_RATIO = 0.88


def escape_drawtext(text: str) -> str:
    return text.replace("'", "\\'").replace(":", "\\:").replace("\\", "\\\\")


def add_cta(video_path: str, cta_text: str, output_path: str, cta_start: float, cta_duration: float) -> None:
    cta_end = cta_start + cta_duration

    # Probe video to get height; fall back to 1920 (pipeline default) if ffprobe unavailable
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        frame_h = int(video_stream["height"])
    except Exception:
        log.warning("ffprobe unavailable — defaulting to 1920px height (9:16 pipeline standard)")
        frame_h = 1920
    y_pos = int(frame_h * CTA_Y_RATIO)

    safe_text = escape_drawtext(cta_text)
    drawtext_filter = (
        f"drawtext=text='{safe_text}'"
        f":fontsize={FONT_SIZE}"
        f":fontcolor={FONT_COLOR}"
        f":bordercolor={BORDER_COLOR}"
        f":borderw={BORDER_WIDTH}"
        f":fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        f":x=(w-text_w)/2"
        f":y={y_pos}"
        f":enable='between(t,{cta_start:.3f},{cta_end:.3f})'"
    )

    log.info("Overlaying CTA '%s' from %.1fs to %.1fs", cta_text, cta_start, cta_end)

    (
        ffmpeg
        .input(video_path)
        .output(
            output_path,
            vf=drawtext_filter,
            vcodec="libx264",
            acodec="aac",
            audio_bitrate="192k",
            video_bitrate="4000k",
            pix_fmt="yuv420p",
        )
        .run(overwrite_output=True)
    )

    print(f"\nSUCCESS: CTA overlay written to {output_path}")
    print(f"         CTA '{cta_text}' visible from {cta_start:.1f}s to {cta_end:.1f}s ({cta_duration:.1f}s duration)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay CTA text element on a video.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--cta-text", required=True, help="CTA string to display")
    parser.add_argument("--output", default=str(TMP_DIR / "with_cta.mp4"), help="Output path")
    parser.add_argument("--cta-start", type=float, default=8.0, help="Seconds when CTA appears (default: 8.0)")
    parser.add_argument("--cta-duration", type=float, default=1.5, help="Minimum seconds CTA stays on screen (default: 1.5)")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    add_cta(args.video, args.cta_text, args.output, args.cta_start, args.cta_duration)


if __name__ == "__main__":
    main()
