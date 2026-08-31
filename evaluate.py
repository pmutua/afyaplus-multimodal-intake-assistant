"""Runs all three pipelines against the test fixtures and writes a
before/after evaluation table (Deliverable 5) to docs/evaluation_table.md.

"Before" figures describe the manual, ops-team process AfyaPlus used prior
to this project (unassisted human review) and are stated as documented
assumptions from the case study, not measurements -- there is no automated
system to time in the "before" condition by definition. "After" figures are
measured by actually running this repo's pipelines.

Run: python evaluate.py
"""
import json
import time

import config
from caption_image import caption_image
from search import search
from transcribe_audio import transcribe

# Ground-truth source text for the clear-audio sample -- used only to
# sanity-check transcription, not shipped as part of the runtime pipeline.
_GROUND_TRUTH_EN = (
    "I have a red, itchy rash on my left forearm that started three days ago. "
    "It is getting bigger and it burns a little at night. I have not had a fever."
)


def _word_overlap_ratio(reference: str, hypothesis: str) -> float:
    """Rough transcription-accuracy proxy: fraction of the reference
    script's *unique words* that also appear anywhere in the transcript.
    Deliberately not a real WER (word error rate) metric -- no ordering or
    duplicate-count sensitivity -- just a cheap sanity check that doesn't
    require an external scoring library."""
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    if not ref_words:
        return 0.0
    return len(ref_words & hyp_words) / len(ref_words)


def run_retrieval_eval():
    """Time one representative search() call and capture its top result,
    for the "Similar-case lookup" row of the evaluation table."""
    t0 = time.time()
    results = search("a red itchy rash", top_k=3)
    elapsed = time.time() - t0
    return {
        "top_result": results[0]["filename"] if results else None,
        "top_similarity": round(results[0]["similarity"], 3) if results else None,
        "latency_s": round(elapsed, 3),
    }


def run_captioning_eval():
    """Run caption_image() on all 3 documented test cases (clear/ambiguous/
    out_of_scope) so the evaluation table can show the full safety-flag
    behaviour side by side, not just a single happy-path example."""
    cases = {}
    for name in ["test_clear", "test_ambiguous", "test_out_of_scope"]:
        path = config.IMAGES_TEST_DIR / f"{name}.png"
        t0 = time.time()
        result = caption_image(str(path))
        elapsed = time.time() - t0
        cases[name] = {
            "caption": result["caption"],
            "flags": result["flags"],
            "has_disclaimer": bool(result["disclaimer"]),
            "latency_s": round(elapsed, 3),
        }
    return cases


def run_transcription_eval():
    """Transcribe the clear-voice sample and score it against the known
    source script (_GROUND_TRUTH_EN) via _word_overlap_ratio()."""
    path = config.AUDIO_DIR / "intake_en_clear.mp3"
    t0 = time.time()
    result = transcribe(str(path))
    elapsed = time.time() - t0
    overlap = _word_overlap_ratio(_GROUND_TRUTH_EN, result["text"])
    return {
        "transcript": result["text"],
        "language": result["language"],
        "fields": result["fields"],
        "word_overlap_vs_source": round(overlap, 3),
        "latency_s": round(elapsed, 3),
    }


def run_transcription_eval_accented():
    """Same as run_transcription_eval() but on the accented-voice sample,
    with no ground-truth-overlap score (there's no separately-known script
    for this one) -- just latency/language/fields, to demonstrate robustness
    to speaker variation."""
    path = config.AUDIO_DIR / "intake_en_accented.mp3"
    t0 = time.time()
    result = transcribe(str(path))
    elapsed = time.time() - t0
    return {
        "transcript": result["text"],
        "language": result["language"],
        "fields": result["fields"],
        "latency_s": round(elapsed, 3),
    }


TABLE_TEMPLATE = """# Evaluation: Before (manual) vs After (AI-assisted)

Measured on {measured_note}. "Before" values are the documented manual
baseline from the AfyaPlus case study (unassisted front-desk / ops review);
"After" values come from actually running this repo's pipelines against the
test fixtures in data/.

| Capability | Before (manual) | After (this system) |
|---|---|---|
| **Image triage note** | ~3-5 min per photo; a staff member manually writes a free-text note; wording and thoroughness vary by person and by end-of-shift fatigue. | {caption_latency}s per photo; standardized note wording every time; automatically flags out-of-scope and unreadable photos ({caption_flags_summary}) instead of guessing. |
| **Audio intake** | ~5-8 min per call; a staff member listens live or replays a recording and writes notes by hand; easy to mishear details on accented or non-English calls. | {transcribe_latency}s per clip; full transcript plus structured fields ({field_list}) extracted automatically; also tested on an accented-voice sample to check robustness to speaker variation. |
| **Similar-case lookup** | Not previously possible; staff relied on memory or paper logs to recall "have we seen something like this before." | {retrieval_latency}s text-to-image search over the photo library; top match "{top_result}" at similarity {top_similarity} for the query "a red itchy rash". |

## Captioning safety-flag detail

| Test case | Caption produced | Flags | Disclaimer attached |
|---|---|---|---|
| clear | {clear_caption} | {clear_flags} | {clear_disclaimer} |
| ambiguous | {ambiguous_caption} | {ambiguous_flags} | {ambiguous_disclaimer} |
| out_of_scope | {oos_caption} | {oos_flags} | {oos_disclaimer} |

## Transcription accuracy check

Transcript word-overlap against the known source script used to generate
`intake_en_clear.mp3` (ElevenLabs TTS): **{word_overlap}** (fraction of
source words that also appear in the transcript -- a rough sanity check,
not a formal WER score, since the audio was synthetic TTS rather than a
recorded patient).

## Accented-voice robustness check

`intake_en_accented.mp3` (ElevenLabs TTS, accented voice) transcribed in
{accented_latency}s, detected as language "{accented_language}":

> {accented_transcript}

Note: the tail end of this transcript contains hallucinated non-English
text. This is a known faster-whisper/Whisper failure mode where the model
invents plausible-sounding words past the end of actual speech (e.g. on
trailing silence or background noise) rather than stopping cleanly; it is
not specific to this pipeline's code. It is a real limitation worth
surfacing to reviewers rather than trimming from the report.

## Notes and limitations

- Retrieval ranking is not perfectly reliable across every query category on
  this 17-image synthetic index: `06_burn_hand.png` scores anomalously high
  against several unrelated text queries (e.g. "a bruise on the leg", "an
  insect bite"), so it sometimes outranks the actual matching image. The
  similarity threshold still correctly filters genuinely off-topic queries
  (see `search.py --calibrate`); this is a ranking-*precision* limitation on
  a small synthetic image set, not a broken retrieval mechanism -- a larger
  or more visually distinct image library would be expected to separate
  categories more cleanly.
- All models run locally (CPU) with the fallback stack named in `config.py`:
  CLIP-ViT-B/32, BLIP-base, faster-whisper-base, flan-t5-small. See README
  "API vs local fallback".
- The image library is synthetic (17 generated skin/wound images with
  consistent visual structure) rather than real patient photos, which caps
  how meaningful raw caption text is; the safety *behaviour* (flagging,
  disclaimers, thresholds) is what this evaluation is validating, not
  clinical caption accuracy.
- The audio samples are ElevenLabs TTS voices (clear + accented), not real
  patient recordings, so transcription latency/behaviour is validated but
  accuracy figures should be read as indicative, not a formal WER benchmark.
"""


def render_table(retrieval, captioning, transcription, accented):
    """Fill in TABLE_TEMPLATE's {placeholders} from the 4 result dicts
    collected in main(). Pure string formatting -- all the actual pipeline
    work already happened in the run_*_eval() functions above."""
    c = captioning
    return TABLE_TEMPLATE.format(
        measured_note="synthetic data/ fixtures, local CPU pipeline",
        caption_latency=round((c["test_clear"]["latency_s"] + c["test_ambiguous"]["latency_s"] + c["test_out_of_scope"]["latency_s"]) / 3, 2),
        caption_flags_summary=f"{c['test_ambiguous']['flags']} / {c['test_out_of_scope']['flags']}",
        transcribe_latency=transcription["latency_s"],
        field_list=", ".join(transcription["fields"].keys()),
        retrieval_latency=retrieval["latency_s"],
        top_result=retrieval["top_result"],
        top_similarity=retrieval["top_similarity"],
        clear_caption=c["test_clear"]["caption"],
        clear_flags=", ".join(c["test_clear"]["flags"]) or "none",
        clear_disclaimer=c["test_clear"]["has_disclaimer"],
        ambiguous_caption=c["test_ambiguous"]["caption"] or "(withheld -- see flags)",
        ambiguous_flags=", ".join(c["test_ambiguous"]["flags"]) or "none",
        ambiguous_disclaimer=c["test_ambiguous"]["has_disclaimer"],
        oos_caption=c["test_out_of_scope"]["caption"] or "(withheld -- see flags)",
        oos_flags=", ".join(c["test_out_of_scope"]["flags"]) or "none",
        oos_disclaimer=c["test_out_of_scope"]["has_disclaimer"],
        word_overlap=transcription["word_overlap_vs_source"],
        accented_latency=accented["latency_s"],
        accented_language=accented["language"],
        accented_transcript=accented["transcript"],
    )


def main():
    """Run every pipeline once against the fixed test fixtures in data/,
    save the raw numbers to outputs/evaluation_raw.json (for programmatic
    reuse/debugging), and render the human-readable Markdown table to
    docs/evaluation_table.md (Deliverable 5)."""
    print("Running retrieval evaluation ...")
    retrieval = run_retrieval_eval()
    print("Running captioning evaluation ...")
    captioning = run_captioning_eval()
    print("Running transcription evaluation ...")
    transcription = run_transcription_eval()
    print("Running accented-voice transcription evaluation ...")
    accented = run_transcription_eval_accented()

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = config.OUTPUTS_DIR / "evaluation_raw.json"
    raw_path.write_text(json.dumps(
        {"retrieval": retrieval, "captioning": captioning, "transcription": transcription, "accented": accented},
        indent=2, ensure_ascii=False,
    ))
    print(f"Wrote raw results -> {raw_path}")

    table_md = render_table(retrieval, captioning, transcription, accented)
    docs_path = config.ROOT / "docs" / "evaluation_table.md"
    docs_path.write_text(table_md, encoding="utf-8")
    print(f"Wrote evaluation table -> {docs_path}")


if __name__ == "__main__":
    main()
