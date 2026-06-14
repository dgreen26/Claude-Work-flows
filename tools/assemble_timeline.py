"""Concatenate video clips and replace audio track, output H.264 MP4 at 1080x1920."""

import argparse
import sys
import tempfile
from pathlib import Path

import ffmpeg

# Ensure tools package is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.utils import log

TMP_DIR = Path(__file__).parent.parent / ".tmp"
TMP_DIR.mkdir(exist_ok=True)


def build_concat_list(clip_paths: list[str]) -> Path:
    """Write an ffmpeg concat demuxer list file to .tmp/ and return its path."""
    concat_file = TMP_DIR / "concat_list.txt"
    lines = []
    for path in clip_paths:
        abs_path = Path(path).resolve()
        lines.append(f"file '{abs_path}'")
    concat_file.write_text("\n".join(lines) + "\n")
    log.info("Wrote concat list to %s (%d clips)", concat_file, len(clip_paths))
    return concat_file


def assemble(clip_paths: list[str], audio_path: str, output_path: str) -> None:
    concat_file = build_concat_list(clip_paths)

    # Concatenate clips via concat demuxer (no re-encode of video at this stage)
    concat_video = (
        ffmpeg
        .input(str(concat_file), format="concat", safe=0)
        .video
    )

    # Scale/pad to 1080x1920
    scaled = concat_video.filter(
        "scale",
        w=1080,
        h=1920,
        force_original_aspect_ratio="decrease",
        flags="lanczos",
    ).filter(
        "pad",
        w=1080,
        h=1920,
        x="(ow-iw)/2",
        y="(oh-ih)/2",
        color="black",
    ).filter("fps", fps=30)

    # Audio input
    audio = ffmpeg.input(audio_path).audio

    output = ffmpeg.output(
        scaled,
        audio,
        output_path,
        vcodec="libx264",
        acodec="aac",
        audio_bitrate="192k",
        video_bitrate="4000k",
        pix_fmt="yuv420p",
        shortest=None,
    )

    log.info("Running ffmpeg assemble: %s clips + audio -> %s", len(clip_paths), output_path)
    ffmpeg.run(output, overwrite_output=True, quiet=False)
    print(f"\nSUCCESS: Assembled timeline written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate clips and replace audio.")
    parser.add_argument("--clips", nargs="+", required=True, help="Ordered list of clip paths")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--output", default=str(TMP_DIR / "assembled.mp4"), help="Output path")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    assemble(args.clips, args.audio, args.output)


if __name__ == "__main__":
    main()
