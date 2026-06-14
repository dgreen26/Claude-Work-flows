"""Transcribe audio word-level with faster-whisper and burn synced captions onto video."""

import argparse
import json
import sys
from pathlib import Path

import ffmpeg
from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.utils import log

TMP_DIR = Path(__file__).parent.parent / ".tmp"
TMP_DIR.mkdir(exist_ok=True)

# Caption spec
FONT_SIZE = 72
FONT_COLOR = "white"
BORDER_COLOR = "black"
BORDER_WIDTH = 4
# At 1080x1920: safe zone x 108-972, y 192-1728
# 75% of 1920 = 1440 — well within safe zone
CAPTION_Y_RATIO = 0.75
MAX_WORDS_PER_SEGMENT = 3


def transcribe(audio_path: str, model_size: str) -> list[dict]:
    """Return list of word-level segments: [{text, start, end}, ...]"""
    log.info("Loading Whisper model: %s", model_size)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    log.info("Detected language: %s (%.0f%% confidence)", info.language, info.language_probability * 100)

    words = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                words.append({"text": word.word.strip(), "start": word.start, "end": word.end})
    return words


def group_words(words: list[dict], max_words: int = MAX_WORDS_PER_SEGMENT) -> list[dict]:
    """Group words into caption segments of at most max_words each."""
    segments = []
    i = 0
    while i < len(words):
        chunk = words[i : i + max_words]
        text = " ".join(w["text"] for w in chunk)
        segments.append({"text": text, "start": chunk[0]["start"], "end": chunk[-1]["end"]})
        i += max_words
    return segments


def escape_drawtext(text: str) -> str:
    """Escape special chars for ffmpeg drawtext."""
    return text.replace("'", "\\'").replace(":", "\\:").replace("\\", "\\\\")


def build_drawtext_filters(segments: list[dict], frame_h: int = 1920, frame_w: int = 1080) -> list[str]:
    """Build an ffmpeg drawtext filter string for each caption segment."""
    y_pos = int(frame_h * CAPTION_Y_RATIO)
    filters = []
    for seg in segments:
        text = escape_drawtext(seg["text"])
        start = seg["start"]
        end = seg["end"]
        f = (
            f"drawtext=text='{text}'"
            f":fontsize={FONT_SIZE}"
            f":fontcolor={FONT_COLOR}"
            f":bordercolor={BORDER_COLOR}"
            f":borderw={BORDER_WIDTH}"
            f":fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            f":x=(w-text_w)/2"
            f":y={y_pos}"
            f":enable='between(t,{start:.3f},{end:.3f})'"
        )
        filters.append(f)
    return filters


def burn_captions(video_path: str, audio_path: str, output_path: str, model_size: str) -> None:
    try:
        words = transcribe(audio_path, model_size)
    except Exception as exc:
        log.warning("Transcription failed (%s) — copying video without captions", exc)
        transcript_path = TMP_DIR / "transcript.json"
        transcript_path.write_text("[]")
        ffmpeg.input(video_path).output(output_path, vcodec="copy", acodec="copy").run(overwrite_output=True)
        print(f"\nSUCCESS (no captions — transcription unavailable): {output_path}")
        return
    log.info("Transcribed %d words", len(words))

    transcript_path = TMP_DIR / "transcript.json"
    transcript_path.write_text(json.dumps(words, indent=2))
    log.info("Saved word-level transcript to %s", transcript_path)

    segments = group_words(words)
    log.info("Grouped into %d caption segments", len(segments))

    if not segments:
        log.warning("No caption segments found — copying video without captions")
        ffmpeg.input(video_path).output(output_path, vcodec="copy", acodec="copy").run(overwrite_output=True)
        print(f"\nSUCCESS (no captions): {output_path}")
        return

    drawtext_filters = build_drawtext_filters(segments)
    combined_filter = ",".join(drawtext_filters)

    video = ffmpeg.input(video_path)
    audio = video.audio

    # Apply all drawtext filters as a single vf chain
    filtered_video = video.video.filter_multi_output  # not using filter_multi_output here
    # Use ffmpeg vf string directly via .video.filter()
    # Chain multiple drawtext via ffmpeg's vf comma-separated string
    # We build the full vf as a raw filter string
    (
        ffmpeg
        .input(video_path)
        .output(
            output_path,
            vf=combined_filter,
            vcodec="libx264",
            acodec="aac",
            audio_bitrate="192k",
            video_bitrate="4000k",
            pix_fmt="yuv420p",
        )
        .run(overwrite_output=True)
    )

    print(f"\nSUCCESS: Captioned video written to {output_path}")
    print(f"         Transcript saved to {transcript_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio and burn captions onto video.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--audio", required=True, help="Audio file to transcribe")
    parser.add_argument("--output", default=str(TMP_DIR / "captioned.mp4"), help="Output path")
    parser.add_argument("--model", default="base", help="Whisper model size (default: base)")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    burn_captions(args.video, args.audio, args.output, args.model)


if __name__ == "__main__":
    main()
