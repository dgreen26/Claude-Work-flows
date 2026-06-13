# WAT Video Pipeline

AI video production framework built on the WAT architecture (Workflows, Agents, Tools).

## Layout

```
.tmp/           Intermediate files. Regenerated as needed. Not committed.
output/         Final rendered H.264 MP4 deliverables. Nothing else goes here.
assets/         Reusable inputs: fonts, music beds, reference images, brand kits.
tools/          Python scripts for deterministic execution.
workflows/      Markdown SOPs. Read these before running any tool.
CLAUDE.md       Architecture spec and operating rules for the agent.
```

## Setup

**1. Install ffmpeg**

macOS:
```
brew install ffmpeg
```

Ubuntu/Debian:
```
sudo apt update && sudo apt install ffmpeg
```

Windows: Download from https://ffmpeg.org/download.html and add to PATH.

Verify:
```
ffmpeg -version
```

**2. Python environment**

```
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Environment variables**

```
cp .env.example .env
```

Fill in your keys in `.env`. Never commit `.env`.

## Running tools

Each tool in `tools/` is a standalone script. Run with:

```
python tools/<script>.py --help
```

Read the matching workflow in `workflows/` before running any paid tool.

## Generation rules

- No paid API call without explicit confirmation. This is a hard rule.
- Cache lives in `.tmp/` keyed by prompt and settings. Retries never re-bill a successful prior step.
- Every generation is logged to `.tmp/generation_log.jsonl`.
- Validate clips (resolution, duration, safe zones) before final render.
