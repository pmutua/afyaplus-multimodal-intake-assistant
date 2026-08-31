# Memo: Should AfyaPlus pilot the Multimodal Intake Assistant?

**To:** Clinical Director
**From:** Data/AI team (Olu's project)
**Re:** Go/no-go on piloting AI-assisted photo and voice intake
**Date:** 2026-08-31

## The short answer

**Go, but only as a paperwork assistant, not a decision-maker.** Pilot it in one
clinic for one month, front-desk use only, with a human checking every AI
note before it reaches a chart. Do not let it suggest a diagnosis or a next
step to a patient directly.

## What this system does

A patient (or a front-desk worker on their behalf) uploads a photo of a
skin concern or leaves a short voice note describing their symptoms. The
system:

- Writes a short, plain-language description of the photo, or
- Turns the voice note into a written transcript and pulls out the key
  facts (what, where on the body, how long, how bad, fever or not).

Every single note it produces carries a fixed warning that it is not a
diagnosis, and anything it isn't confident about (a blurry photo, a photo of
something that isn't a medical concern at all) gets flagged for a person to
look at instead of guessing.

## What we measured (see `docs/evaluation_table.md` for full numbers)

| | Today (manual) | With this tool |
|---|---|---|
| Writing a note from a photo | A few minutes, wording varies by staff member | Seconds, same wording style every time, and it says clearly when it can't tell what a photo shows |
| Writing a note from a voice message | Several minutes, easy to mishear details, especially in a second language | Seconds, plus the same five facts pulled out every time (symptom, location, duration, severity, fever) |
| "Have we seen something like this before?" | Not really possible — staff rely on memory | Searchable in under a second across the photo library |

The tool is consistently faster and more consistent than manual note-taking.
It does **not** make the notes more medically accurate than what a trained
person would write — it was tested here with safe, synthetic stand-in
images and voice clips (see the README's safety note on why we didn't use
real patient photos yet), and it should be treated as a documentation
speed-up, not a quality upgrade, until it has been checked against real
cases under supervision.

**What it costs:** the version we're proposing runs on our own computer, so
it has no per-use fee. We also tested the paid alternative (OpenAI's API)
to see how it compares — real numbers, not a guess: about $0.0034 for a
handful of test requests today, which projects to roughly **$0.68–$0.85 a
month even at 1000 patient submissions** (see
`docs/cost_analysis.md` for the full numbers). That's cheap either way, so
cost is not the reason we're recommending our own computer's version —
staff review time is a bigger cost than the software either way. The one
place the paid version pulled ahead was making cleaner transcripts on a
heavily-accented voice sample; worth another look if voice quality becomes
a real problem in the pilot.

## The main safety risk, and how we're handling it

**Risk: a confidently-worded AI note gets mistaken for a medical opinion**,
either by an overworked staff member who stops double-checking it, or by a
patient who sees it directly. This is the most likely way this tool could
cause harm — not by being wrong, but by being *trusted too much*.

**Mitigation, already built in:**
1. Every note carries a fixed, unremovable disclaimer that it is AI-generated
   and not a diagnosis.
2. Low-confidence cases (blurry photos, photos that don't look like a medical
   photo at all) are refused a caption entirely and routed to a person,
   instead of the system guessing anyway.
3. For the pilot, we recommend the AI note is **never shown to the patient
   directly** — only to staff, as a drafting aid they can edit or discard.

A second, smaller risk worth naming: **using real patient photos to build or
test this system before proper consent and data-handling review is in
place.** We avoided this entirely during development by using synthetic
placeholder images (see README) — the same discipline should carry into the
pilot, i.e., don't expand the photo library with real patient images until
Legal/Compliance has signed off on storage and consent.

## What we're asking for

Approval to run a one-month pilot in a single clinic, front-desk use only,
with:
- A named staff member reviewing every AI note before it's filed
- No patient-facing display of AI notes
- A simple log of cases where the AI flagged "can't tell" or "out of scope,"
  reviewed weekly to see if the flag rate is reasonable

We'll bring back real usage numbers (not synthetic ones) after four weeks to
decide whether to expand.
