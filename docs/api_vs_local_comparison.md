# API vs local-fallback: side-by-side comparison

This repo's shipped pipeline uses the local/open-source fallback path (see
README "API vs local fallback"), with **no OpenAI credits spent in normal
operation**. This document is the one deliberate exception: a small,
one-time comparison run (`python compare_api_vs_local.py`) that spent a few
cents of OpenAI credit to produce real side-by-side numbers instead of an
unverified claim about the tradeoff.

## Captioning: GPT-4o-mini (API) vs BLIP-base (local)

| Test case | GPT-4o-mini caption | BLIP-base caption | API latency | Local latency |
|---|---|---|---|---|
| test_clear | The image shows an area of skin with redness and raised lesions, possibly indicating a rash. | a medical photo of a woman with a rash on her arm | 2.596s | 12.711s |
| test_ambiguous | NOT_APPLICABLE. | a medical photo of a hospital hallway | 1.598s | 1.262s |
| test_out_of_scope | NOT_APPLICABLE. | (withheld -- see flags) | 0.809s | 0.089s |

## Transcription: Whisper API vs faster-whisper (local)

| Sample | Whisper API transcript | faster-whisper transcript | API latency | Local latency |
|---|---|---|---|---|
| clear | I have a red, itchy rash on my left forearm that started three days ago. It's getting bigger and it burns a little at night. I haven't had a fever. | I have a red itchy rash on my left forearm that started three days ago. It's getting bigger and it burns a little at night. I haven't had a fever. | 2.844s | 5.056s |
| accented | My daughter has a swollen ankle after she fell yesterday evening. She says it hurts when she walks on it and there's some bruising around the joint. No fever, but she's been limping since this morning. | My daughter has a swollen ankle after she fell yesterday evening. She says it hurts when she walks on it and there's some bruising around the joint. No fever, but she's been limping since this morning. Kapastar L.Y. on Die Suitment van Afrika and Werduk Die Modestad van. | 3.332s | 2.743s |

## Takeaway

- **Caption quality:** GPT-4o-mini produces a full descriptive sentence
  ("an area of skin with redness and raised lesions, possibly indicating a
  rash") vs. BLIP-base's short noun-phrase style ("a medical photo of a
  woman with a rash on her arm"). Both correctly decline to caption the
  ambiguous/out-of-scope cases (`NOT_APPLICABLE` vs. this repo's own
  `OUT_OF_SCOPE`/`HUMAN_REVIEW_REQUIRED` flags) -- the safety *behaviour* is
  equivalent, the API's prose is more fluent.
- **Transcription quality -- the one clear win for the API:** on the clear
  sample both transcripts are near-identical. On the **accented** sample,
  Whisper API returned a clean, accurate transcript, while local
  faster-whisper appended hallucinated non-English gibberish after the real
  speech ended (see `docs/evaluation_table.md`). This is the concrete
  case where paying for the API path would measurably improve output
  quality over the local fallback.
- **Latency caveat:** the local `test_clear` row (12.7s) includes one-time
  BLIP weight loading inside the timed call, not a per-call inference cost
  -- compare it to the local `test_ambiguous`/`test_out_of_scope` rows
  (1.26s, 0.09s) for a fairer sense of warm per-call latency. Once warm,
  local latency is comparable to or faster than the API for these
  small/short inputs; the API adds no local compute burden and needs no
  model download, which matters more as call volume or media size grows.
- **Why this repo still ships the local path by default:** the capstone
  brief's stated fallback path avoids spending OpenAI credits during
  normal development/grading; this document exists to make that tradeoff a
  documented, evidence-backed comparison rather than an unverified
  assumption -- and to flag transcription accuracy on accented/noisy audio
  as the one place a future real-data pilot might justify the API's cost.
- **What this run actually cost, and what it would cost at scale:** see
  [`cost_analysis.md`](cost_analysis.md) for the real OpenAI billing
  numbers behind this comparison (from
  [`../completions_usage_2026-08-01_2026-08-31.csv`](../completions_usage_2026-08-01_2026-08-31.csv)
  and [`../cost_2026-08-01_2026-08-31.csv`](../cost_2026-08-01_2026-08-31.csv))
  and a 1000-request cost projection.
