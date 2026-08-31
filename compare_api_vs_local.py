"""One-off comparison: OpenAI API path vs the local-fallback path this repo
actually ships (see README "API vs local fallback"). Not part of the
runtime pipeline -- run manually, once, to document the tradeoff with real
numbers instead of just asserting it. Spends a small amount of OpenAI API
credit (3 vision-captioning calls + 2 transcription calls).

Run: python compare_api_vs_local.py
Writes: docs/api_vs_local_comparison.md
"""
import base64
import json
import time

from dotenv import load_dotenv
from openai import OpenAI

import config
from caption_image import caption_image
from transcribe_audio import transcribe

load_dotenv()  # reads .env into the process environment (see .env.example)
client = OpenAI()  # reads OPENAI_API_KEY from the environment automatically

CAPTION_PROMPT = (
    "You are assisting a health-intake triage system. Describe what this "
    "photo shows in one short, plain-language sentence, using neutral "
    "clinical-adjacent vocabulary (e.g. rash, bruise, swelling, wound). "
    "Do not diagnose. If the image does not show a skin condition, wound, "
    "or similar intake-relevant subject, say exactly: NOT_APPLICABLE."
)


def _b64_image(path):
    """OpenAI's vision API accepts an inline base64 data URL as an
    alternative to a hosted image URL -- convenient here since our test
    images are local files, not already hosted anywhere."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def api_caption(image_path: str) -> dict:
    """GPT-4o-mini vision equivalent of caption_image.py's local BLIP call:
    send the image + CAPTION_PROMPT to the chat completions endpoint and
    time the round trip. No safety pre-checks here (unlike caption_image.py's
    unreadable/out-of-scope gates) -- CAPTION_PROMPT itself asks the model to
    self-report NOT_APPLICABLE for out-of-scope images instead."""
    t0 = time.time()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{_b64_image(image_path)}"
                }},
            ],
        }],
        max_tokens=60,
    )
    elapsed = time.time() - t0
    return {"caption": resp.choices[0].message.content.strip(), "latency_s": round(elapsed, 3)}


def api_transcribe(audio_path: str) -> dict:
    """Whisper API equivalent of transcribe_audio.py's local faster-whisper
    call. No field extraction here -- this script only compares raw
    transcript text, not the downstream extract_fields() step."""
    t0 = time.time()
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(model="whisper-1", file=f)
    elapsed = time.time() - t0
    return {"text": resp.text.strip(), "latency_s": round(elapsed, 3)}


TABLE_TEMPLATE = """# API vs local-fallback: side-by-side comparison

This repo's shipped pipeline uses the local/open-source fallback path (see
README "API vs local fallback"), with **no OpenAI credits spent in normal
operation**. This document is the one deliberate exception: a small,
one-time comparison run (`python compare_api_vs_local.py`) that spent a few
cents of OpenAI credit to produce real side-by-side numbers instead of an
unverified claim about the tradeoff.

## Captioning: GPT-4o-mini (API) vs BLIP-base (local)

| Test case | GPT-4o-mini caption | BLIP-base caption | API latency | Local latency |
|---|---|---|---|---|
{caption_rows}

## Transcription: Whisper API vs faster-whisper (local)

| Sample | Whisper API transcript | faster-whisper transcript | API latency | Local latency |
|---|---|---|---|---|
{transcript_rows}

## Takeaway

- **Caption quality:** the API path tends to produce a fuller descriptive
  sentence; the local BLIP path tends to produce a short noun-phrase. Both
  should correctly decline the ambiguous/out-of-scope cases -- compare the
  safety *behaviour*, not just prose style.
- **Transcription quality:** compare the two transcripts per sample closely,
  especially on the accented sample -- faster-whisper is known to
  occasionally append hallucinated text past the end of real speech on
  noisier/accented audio (see `docs/evaluation_table.md`); check whether
  that occurred in this run and call it out explicitly if so, since it's
  the clearest case where the API path can outperform the local fallback.
- **Latency caveat:** the *first* local call in this script includes
  one-time model-loading time inside the timed block, which inflates that
  one row -- prefer later rows (or a warm re-run) for a fair per-call
  latency comparison.
- **Why this repo still ships the local path by default:** the capstone
  brief's stated fallback path avoids spending OpenAI credits during
  normal development/grading; this document exists to make that tradeoff a
  documented, evidence-backed comparison rather than an unverified
  assumption.
"""


def render_caption_row(name, api_result, local_result):
    # Escape any literal "|" in the model output so it can't break the
    # Markdown table's column structure.
    api_caption_text = api_result["caption"].replace("|", "\\|")
    local_caption_text = (local_result["caption"] or "(withheld -- see flags)").replace("|", "\\|")
    return f"| {name} | {api_caption_text} | {local_caption_text} | {api_result['latency_s']}s | {local_result['latency_s']}s |"


def render_transcript_row(name, api_result, local_text, local_latency):
    api_text = api_result["text"].replace("|", "\\|")
    local_text_esc = local_text.replace("|", "\\|")
    return f"| {name} | {api_text} | {local_text_esc} | {api_result['latency_s']}s | {local_latency}s |"


def main():
    """For each of the 3 test images and 2 audio samples, run both the API
    call and the local pipeline call back-to-back, time both, and build one
    Markdown table row per sample. Note: the *first* local call in each loop
    pays for lazy model loading inside its timed block (see caption_image.py
    / transcribe_audio.py's module-level model caches), which is why the
    Takeaway section in the output flags the first row's latency as
    unrepresentative."""
    caption_rows = []
    for name in ["test_clear", "test_ambiguous", "test_out_of_scope"]:
        path = config.IMAGES_TEST_DIR / f"{name}.png"
        print(f"Captioning {name} via API ...")
        api_result = api_caption(str(path))
        print(f"Captioning {name} via local BLIP ...")
        t0 = time.time()
        local_result = caption_image(str(path))
        local_result["latency_s"] = round(time.time() - t0, 3)
        caption_rows.append(render_caption_row(name, api_result, local_result))

    transcript_rows = []
    for name, filename in [("clear", "intake_en_clear.mp3"), ("accented", "intake_en_accented.mp3")]:
        path = config.AUDIO_DIR / filename
        print(f"Transcribing {name} via API ...")
        api_result = api_transcribe(str(path))
        print(f"Transcribing {name} via local faster-whisper ...")
        t0 = time.time()
        local_result = transcribe(str(path))
        local_latency = round(time.time() - t0, 3)
        transcript_rows.append(render_transcript_row(name, api_result, local_result["text"], local_latency))

    table_md = TABLE_TEMPLATE.format(
        caption_rows="\n".join(caption_rows),
        transcript_rows="\n".join(transcript_rows),
    )
    out_path = config.ROOT / "docs" / "api_vs_local_comparison.md"
    out_path.write_text(table_md, encoding="utf-8")
    print(f"Wrote comparison -> {out_path}")


if __name__ == "__main__":
    main()
