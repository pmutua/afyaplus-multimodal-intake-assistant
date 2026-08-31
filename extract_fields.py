"""Structured JSON field extraction from an intake transcript.

Pulls the domain fields a triage nurse would want at a glance: symptom,
body location, duration, severity/red-flags, and fever status. Uses a small
local instruction model (flan-t5-small) to normalize free text into fields,
then a regex safety-net fills in anything the model missed or hallucinated
away, so the output is always valid JSON with every key present.

    from extract_fields import extract_fields
    extract_fields("I have a red itchy rash on my left forearm ...")
"""
import json
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config

FIELDS = ["symptom", "location", "duration", "severity", "fever"]

_model = None
_tokenizer = None

_LOCATIONS = [
    "forearm", "arm", "leg", "shin", "ankle", "hand", "foot", "torso",
    "chest", "back", "face", "eye", "neck", "hip", "knee", "wrist",
]
_SYMPTOM_ALIASES = {
    "rash": ["rash"],
    "wound": ["wound"],
    "cut": ["cut", "laceration", "lacerated"],
    "bruise": ["bruise", "bruising", "bruised"],
    "burn": ["burn", "burning", "burned", "burnt"],
    "swelling": ["swelling", "swollen", "swell"],
    "bite": ["bite", "bitten"],
    "hives": ["hive", "hives"],
    "welt": ["welt"],
    "redness": ["redness"],
    "mole": ["mole"],
    "lesion": ["lesion"],
    "infection": ["infection", "infected"],
    "pus": ["pus"],
    "ulcer": ["ulcer"],
    "itch": ["itch", "itchy", "itching"],
    "pain": ["pain", "painful", "hurts", "hurt"],
}
_DURATION_RE = re.compile(
    r"(\d+|one|two|three|four|five|six|seven)\s+(day|days|week|weeks|hour|hours|month|months)",
    re.IGNORECASE,
)
_SEVERITY_WORDS = {
    "high": ["severe", "unbearable", "worse", "spreading", "getting bigger", "burns"],
    "low": ["mild", "slight", "small", "minor"],
}


def _load_model():
    """Load flan-t5-small into the module-level cache on first use only."""
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.EXTRACTION_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(config.EXTRACTION_MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


_FEVER_NEGATION_CUES = ("no ", "not ", "n't", "without", "denies", "denied")


def _detect_fever(text_l: str) -> str:
    """"fever" alone isn't enough -- check for a negation cue in the words
    immediately before it (covers "no fever", "not had a fever", and
    contractions like "haven't/hasn't had a fever" that a plain substring
    check on "not had a fever" would miss)."""
    idx = text_l.find("fever")
    if idx == -1:
        # word "fever" never mentioned at all -> treat as not reported
        return "no"
    # Look only at the ~40 chars immediately before "fever" for a negation
    # cue -- e.g. "...i haven't had a [fever]" -- rather than scanning the
    # whole transcript, so a negation earlier about an unrelated symptom
    # doesn't accidentally flip this field.
    window = text_l[max(0, idx - 40):idx]
    if any(cue in window for cue in _FEVER_NEGATION_CUES):
        return "no"
    return "yes"


def _regex_fallback(text: str) -> dict:
    """Deterministic, non-ML extraction used both as the LLM's safety net
    and, if the LLM path fails/hallucinates entirely, as the sole source of
    truth. Every field here is a simple keyword/regex lookup against the
    lower-cased transcript -- see the _LOCATIONS / _SYMPTOM_ALIASES /
    _DURATION_RE / _SEVERITY_WORDS constants above for exactly what each
    field matches on."""
    text_l = text.lower()
    # symptom: first canonical category whose alias list has a match, in
    # dict-insertion order (see _SYMPTOM_ALIASES) -- only the first match
    # wins, so a transcript mentioning multiple symptoms reports just one.
    symptom = next(
        (canonical for canonical, aliases in _SYMPTOM_ALIASES.items() if any(a in text_l for a in aliases)),
        None,
    )
    location = next((l for l in _LOCATIONS if l in text_l), None)
    duration_match = _DURATION_RE.search(text_l)
    duration = duration_match.group(0) if duration_match else None
    severity = "unspecified"
    for level, words in _SEVERITY_WORDS.items():
        if any(w in text_l for w in words):
            severity = level
            break
    fever = _detect_fever(text_l)
    return {
        "symptom": symptom,
        "location": location,
        "duration": duration,
        "severity": severity,
        "fever": fever,
    }


def _llm_extract(text: str) -> dict:
    """Ask flan-t5-small to read the transcript and emit the 5 fields as a
    JSON object directly. This is the "smart" path -- it can, in principle,
    understand phrasing the regex fallback can't -- but flan-t5-small is a
    small model and frequently returns malformed JSON or omits fields
    entirely, which is why extract_fields() below never trusts this alone."""
    model, tokenizer = _load_model()
    prompt = (
        "Extract these fields from the patient's spoken intake note as a JSON object "
        "with exactly these keys: symptom, location, duration, severity, fever. "
        "Use null for any field not mentioned.\n\n"
        f"Note: {text}\n\nJSON:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    try:
        # The model doesn't always emit *only* JSON (it may add stray text
        # around it), so pull out the first {...} block rather than parsing
        # the raw decoded string directly.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
    except (json.JSONDecodeError, AttributeError):
        # Malformed JSON, or no {...} found at all -- fall through to an
        # all-None result; extract_fields() will backfill from regex.
        parsed = {}
    return {k: parsed.get(k) for k in FIELDS}


def extract_fields(transcript: str) -> dict:
    """Return a dict with keys symptom/location/duration/severity/fever.

    flan-t5-small is small enough to run on CPU but is not always reliable
    at emitting well-formed JSON; any field it leaves null is backfilled by
    the regex extractor so the output always has usable values where the
    transcript actually states them.
    """
    llm_result = _llm_extract(transcript)
    fallback = _regex_fallback(transcript)
    # Per-field merge: LLM value wins if it's truthy (not None/empty/"null"),
    # otherwise fall back to the regex result. Running both unconditionally
    # (rather than only falling back on total LLM failure) means a single
    # hallucinated/missing field doesn't force ignoring every other field the
    # LLM got right.
    merged = {k: (llm_result.get(k) or fallback.get(k)) for k in FIELDS}
    return merged


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else input("Transcript: ")
    print(json.dumps(extract_fields(text), indent=2))
