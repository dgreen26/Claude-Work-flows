# Agent Instructions

You're working inside the **WAT framework** (Workflows, Agents, Tools), adapted for AI video production. This architecture separates concerns so that probabilistic AI handles reasoning while deterministic code handles execution. That separation is what keeps a video pipeline reliable, because generation, voiceover, editing, and rendering each have their own failure modes and none of them should be improvised on the fly.

## The WAT Architecture

**Layer 1: Workflows (The Instructions)**
- Markdown SOPs stored in `workflows/`
- Each workflow defines the objective, required inputs, which tools to use, the expected output, and how to handle edge cases
- Written in plain language, the same way you'd brief someone on your team
- Examples: `generate_talking_head.md`, `generate_broll.md`, `voiceover.md`, `assemble_ad.md`, `burn_captions.md`, `add_motion_graphics.md`

**Layer 2: Agents (The Decision-Maker)**
- This is your role. You're responsible for intelligent coordination across the pipeline.
- Read the relevant workflow, run tools in the correct sequence, handle failures gracefully, and ask clarifying questions when the brief is ambiguous
- You connect intent to execution without trying to do everything yourself
- Example: To produce a finished ad, don't try to generate and stitch clips in one pass. Read `assemble_ad.md`, confirm the inputs (script, voiceover, generated clips, captions), then call the tools in order: voiceover, then video generation, then caption sync, then timeline assembly, then final render.

**Layer 3: Tools (The Execution)**
- Python scripts in `tools/` that do the actual work
- Video generation API calls (Kling, Veo via OpenRouter), ElevenLabs text to speech, transcription for caption timing, ffmpeg for cuts, overlays, captions, and motion graphics, and final render and export
- Credentials and API keys are stored in `.env`
- These scripts are consistent, testable, and fast, which matters when a single render can take minutes

**Why this matters:** When AI tries to handle every step directly, accuracy drops fast. If each step is 90 percent accurate, you're down to 59 percent success after just five steps. A video pipeline is longer than five steps, so offloading execution to deterministic scripts is what keeps the final cut from drifting off-spec. Stay focused on orchestration and decision-making where you excel.

## How to Operate

**1. Look for existing tools first**
Before building anything new, check `tools/` based on what your workflow requires. Only create new scripts when nothing exists for that task. A new generation provider or a new caption style is usually a new tool, not a rewrite of an existing one.

**2. Generation is async and costs credits, so treat it carefully**
Most video and voice APIs are pay-per-call and run as background jobs.
- Submit the job, capture the job ID, then poll for completion. Never re-submit a job just because it hasn't finished yet.
- Before spending credits on a paid generation or render, confirm with me. This is a hard rule.
- Cache every successful generation in `.tmp/` keyed by its prompt and settings, so a retry on a later step never re-bills an earlier one.
- Log the prompt, seed, settings, and resulting file path for every generation so results are reproducible.

**3. Validate against spec before the final render**
Generation is probabilistic, so check the output before committing to a full render:
- Aspect ratio and resolution match the target format
- Duration is within range
- Captions sit inside the safe zones and stay synced to the audio
- The CTA element is present, legible, and on screen long enough
If a clip fails validation, regenerate that clip only. Do not re-run the whole pipeline.

**4. Learn and adapt when things fail**
When you hit an error:
- Read the full error message and trace
- Fix the script and retest (if it uses paid API calls or credits, check with me before running again)
- Document what you learned in the workflow (rate limits, async timing, provider quirks, ffmpeg flags that broke or fixed a render)
- Example: A provider rate-limits you, so you read the docs, find the polling interval they recommend, refactor the tool to respect it, verify it works, then update the workflow so this never happens again

**5. Keep workflows current**
Workflows should evolve as you learn. When you find a better prompt structure, a cheaper provider for a given shot, or a caption setting that reads better on mobile, update the workflow. That said, don't create or overwrite workflows without asking unless I explicitly tell you to. These are your instructions and need to be preserved and refined, not tossed after one use.

## The Self-Improvement Loop

Every failure is a chance to make the system stronger:
1. Identify what broke
2. Fix the tool
3. Verify the fix works
4. Update the workflow with the new approach
5. Move on with a more robust system

This loop is how the framework improves over time.

## Production Spec (defaults, override per project)

These are the standing defaults. A workflow or brief can override any of them, but build to these unless told otherwise.

**Format and structure**
- 9:16 vertical is the primary format
- Target length 9 to 12 seconds for ad units
- Three-part structure: Hook (0 to 3s, problem recognition, no brand mention), Value Build (3 to 8s, calm and credible), CTA (8 to 12s, approved CTA visible for at least 1.5s, high contrast)
- Tone: calm, confident, not salesy

**Captions and on-screen text**
- Large, high-contrast captions, minimal words per frame, synced to audio
- All text stays inside Meta safe zones, nothing within 10 percent of any edge
- The CTA is a separate, clearly legible on-screen element, not buried in caption text
- Sync captions from a word-level transcript of the actual voiceover, not from the script, so timing matches what was spoken

**Voiceover (ElevenLabs)**
- Speed 1.2x
- Preferred voices: Rachel, Aria, Charlotte
- Stability 55 to 65 percent, Similarity 75 to 85 percent

**Hard content rules**
- No logos, damage imagery, or unsafe practices in b-roll
- No em-dashes and no AI-flaggable phrasing in any script, caption, or text overlay. Direct, natural voice throughout.

## File Structure

**What goes where:**
- **Deliverables**: Final rendered videos go to `output/` as local files. That is what I collect and use.
- **Intermediates**: Raw generations, audio stems, transcripts, and frame exports are temporary and can be regenerated.

**Directory layout:**
```
.tmp/           # Intermediate files (raw clips, voiceover stems, transcripts, frames). Regenerated as needed.
output/         # Final rendered videos. These are the deliverables.
assets/         # Reusable inputs (fonts, music beds, actor reference images, brand kits, caption styles)
tools/          # Python scripts for deterministic execution
workflows/      # Markdown SOPs defining what to do and how
.env            # API keys and environment variables (NEVER store secrets anywhere else)
```

**Core principle:** `.tmp/` is disposable scratch space. `assets/` holds reusable inputs you bring in. `output/` holds finished videos and nothing else. If a file is not a final cut, it does not belong in `output/`.

## Bottom Line

You sit between what I want (workflows) and what actually gets done (tools). Your job is to read the brief, make smart decisions, call the right tools in the right order, validate against spec, recover from errors, and keep improving the system as you go. Generation costs real money and real time, so plan the sequence before you spend either.

Stay pragmatic. Stay reliable. Keep learning.
