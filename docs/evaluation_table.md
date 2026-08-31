# Evaluation: Before (manual) vs After (AI-assisted)

Measured on synthetic data/ fixtures, local CPU pipeline. "Before" values are the documented manual
baseline from the AfyaPlus case study (unassisted front-desk / ops review);
"After" values come from actually running this repo's pipelines against the
test fixtures in data/.

| Capability | Before (manual) | After (this system) |
|---|---|---|
| **Image triage note** | ~3-5 min per photo; a staff member manually writes a free-text note; wording and thoroughness vary by person and by end-of-shift fatigue. | 2.87s per photo; standardized note wording every time; automatically flags out-of-scope and unreadable photos (['HUMAN_REVIEW_REQUIRED'] / ['OUT_OF_SCOPE', 'HUMAN_REVIEW_REQUIRED']) instead of guessing. |
| **Audio intake** | ~5-8 min per call; a staff member listens live or replays a recording and writes notes by hand; easy to mishear details on accented or non-English calls. | 4.784s per clip; full transcript plus structured fields (symptom, location, duration, severity, fever) extracted automatically; also tested on an accented-voice sample to check robustness to speaker variation. |
| **Similar-case lookup** | Not previously possible; staff relied on memory or paper logs to recall "have we seen something like this before." | 5.296s text-to-image search over the photo library; top match "11_allergic_hives.png" at similarity 0.32 for the query "a red itchy rash". |

## Captioning safety-flag detail

| Test case | Caption produced | Flags | Disclaimer attached |
|---|---|---|---|
| clear | a medical photo of a woman with a rash on her arm | HUMAN_REVIEW_REQUIRED | True |
| ambiguous | a medical photo of a hospital hallway | HUMAN_REVIEW_REQUIRED | True |
| out_of_scope | (withheld -- see flags) | OUT_OF_SCOPE, HUMAN_REVIEW_REQUIRED | True |

## Transcription accuracy check

Transcript word-overlap against the known source script used to generate
`intake_en_clear.mp3` (ElevenLabs TTS): **0.889** (fraction of
source words that also appear in the transcript -- a rough sanity check,
not a formal WER score, since the audio was synthetic TTS rather than a
recorded patient).

## Accented-voice robustness check

`intake_en_accented.mp3` (ElevenLabs TTS, accented voice) transcribed in
2.49s, detected as language "en":

> My daughter has a swollen ankle after she fell yesterday evening. She says it hurts when she walks on it and there's some bruising around the joint. No fever, but she's been limping since this morning. Kapastar L.Y. on Die Suitment van Afrika and Werduk Die Modestad van.

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
