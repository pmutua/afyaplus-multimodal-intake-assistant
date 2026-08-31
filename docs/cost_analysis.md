# Cost analysis: real OpenAI billing data + a 1000-request projection

This document uses **actual OpenAI billing exports**, not published list
pricing, to answer two questions: what did today's real API comparison run
(from [`compare_api_vs_local.py`](../compare_api_vs_local.py)) actually
cost, and what would it cost to run this workload at 1000-request scale.
Source files (committed at the repo root):
[`completions_usage_2026-08-01_2026-08-31.csv`](../completions_usage_2026-08-01_2026-08-31.csv)
and [`cost_2026-08-01_2026-08-31.csv`](../cost_2026-08-01_2026-08-31.csv),
exported directly from the OpenAI dashboard's Usage/Costs pages.

## What we actually consumed today (2026-08-31)

Both CSVs cover the full month of August 2026; every day except
2026-08-31 has empty usage/cost rows, because no API calls were made on
any other day -- this project ran on the local fallback path all month,
and only [`compare_api_vs_local.py`](../compare_api_vs_local.py) touched
the API, once, today.

| Metric | Value | Source |
|---|---|---|
| Model | `gpt-4o-mini-2024-07-18` | `completions_usage` CSV, `model` column |
| Chat-completion requests | 4 | `num_model_requests` |
| Input tokens (total) | 31,422 | `input_tokens` |
| Output tokens (total) | 29 | `output_tokens` |
| **Total billed cost, 2026-08-31** | **$0.0034 USD** | `cost` CSV, `amount_value` |

The 4 requests are: the 3 real GPT-4o-mini vision captioning calls in
[`compare_api_vs_local.py`](../compare_api_vs_local.py) (`test_clear`,
`test_ambiguous`, `test_out_of_scope`) plus 1 tiny "say OK" connectivity
check made earlier in the same session while verifying the API key
worked. The 2 Whisper API transcription calls (`intake_en_clear.mp3`,
`intake_en_accented.mp3`) are billed on OpenAI's audio product, which is
priced per-minute of audio rather than per-token and is **not** broken
out in the `completions_usage` export -- and OpenAI's cost data can lag
several hours behind usage, so the audio portion may not be fully
reflected in the $0.0034 figure yet. Treat $0.0034 as a same-order-of-
magnitude floor for the full 5-call run (3 image + 2 audio), not a
guaranteed exact total.

**Local-path cost for the same day:** $0.00. Every other script in this
repo (`build_index.py`, `search.py`, `caption_image.py`,
`transcribe_audio.py`, `extract_fields.py`, `router.py`, `app.py`,
`evaluate.py`) ran entirely on local CPU with no OpenAI API calls, all
month.

## Projecting to 1000 requests

Two ways to scale the same real, measured data -- shown together because
they should (and do) land in the same ballpark, which is a sanity check
on the extrapolation rather than a single fragile estimate:

**Method 1 -- scale the observed $/request (gpt-4o-mini only).**
$0.0034 / 4 requests = **$0.00085/request**. At 1000 image-captioning
requests: 1000 x $0.00085 = **≈ $0.85**.

**Method 2 -- scale the whole 5-call bundle (3 image + 2 audio) as one unit,
matching this project's actual image:audio ratio.**
$0.0034 / 5 calls = $0.00068/call -> 1000 calls = 200 bundles x $0.0034 =
**≈ $0.68**.

| Scale | Requests | Projected cost (API path) | Projected cost (local path) |
|---|---|---|---|
| Today's real run | 5 calls (3 img + 2 audio) | $0.0034 (measured) | $0.00 |
| 1000 requests | 1000 calls, same mix | **≈ $0.68 - $0.85** | **$0.00** (marginal) |
| 10,000 requests | 10,000 calls, same mix | **≈ $6.80 - $8.50** | **$0.00** (marginal) |

**Caveats on this projection** (stated plainly so it isn't read as more
precise than it is):
- This linearly scales *today's actual measured usage* on 3 small
  synthetic PNGs and 2 short TTS audio clips. Real patient photos or
  longer voice notes would consume more tokens/minutes per call and cost
  more per request than this figure.
- It is **not** OpenAI's published list price -- it is what this account
  was actually billed, which already reflects this org's pricing tier.
  Published per-token/per-minute rates may differ.
- The Whisper/audio portion of the $0.0034 may still be posting (see
  above), so the true per-bundle cost could be somewhat higher than
  measured; the 1000-request range above should be read as a
  lower-bound-to-likely estimate, not a ceiling.
- "Local path = $0.00" means $0 in **marginal per-request API fees**. It
  is not literally free: it costs the electricity to run CPU inference
  and the one-time model download/disk space. Both are small enough
  relative to even sub-cent API costs that they don't change the
  conclusion below.

## Takeaway

At this project's actual measured usage, **API cost is not the reason to
prefer the local path** -- even at 1000 requests/month, GPT-4o-mini +
Whisper API would cost under $1. The real reasons this repo ships the
local fallback by default are the ones already named in
[`../README.md`](../README.md#api-vs-local-fallback) and
[`business_memo.md`](business_memo.md): avoiding OpenAI credit spend
during development/grading, and not depending on network access or an
API key to run the app at all. The one place cost *does* start to matter
is much higher volume or much larger media (long-form audio, high-
resolution photos), where the near-zero marginal cost of local CPU
inference becomes the more meaningful advantage over a per-call API bill
-- at the cost of the transcription-quality gap already documented in
[`evaluation_table.md`](evaluation_table.md) and
[`api_vs_local_comparison.md`](api_vs_local_comparison.md).
