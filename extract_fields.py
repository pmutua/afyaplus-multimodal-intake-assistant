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
_SYMPTOMS = [
    "rash", "wound", "cut", "laceration", "bruise", "burn", "swelling",
    "bite", "hive", "welt", "redness", "mole", "lesion", "infection",
    "pus", "ulcer", "itch", "pain",
]
_DURATION_RE = re.compile(
    r"(\d+|one|two|three|four|five|six|seven)\s+(day|days|week|weeks|hour|hours|month|months)",
    re.IGNORECASE,
)
_SEVERITY_WORDS = {
    "high": ["severe", "unbearable", "worse", "spreading", "getting bigger", "burns"],
    "low": ["mild", "slight", "small", "minor"],
}


def _load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.EXTRACTION_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(config.EXTRACTION_MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


def _regex_fallback(text: str) -> dict:
    text_l = text.lower()
    symptom = next((s for s in _SYMPTOMS if s in text_l), None)
    location = next((l for l in _LOCATIONS if l in text_l), None)
    duration_match = _DURATION_RE.search(text_l)
    duration = duration_match.group(0) if duration_match else None
    severity = "unspecified"
    for level, words in _SEVERITY_WORDS.items():
        if any(w in text_l for w in words):
            severity = level
            break
    fever = "yes" if "fever" in text_l and "no fever" not in text_l and "not had a fever" not in text_l else "no"
    return {
        "symptom": symptom,
        "location": location,
        "duration": duration,
        "severity": severity,
        "fever": fever,
    }


def _llm_extract(text: str) -> dict:
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
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
    except (json.JSONDecodeError, AttributeError):
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
    merged = {k: (llm_result.get(k) or fallback.get(k)) for k in FIELDS}
    return merged


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else input("Transcript: ")
    print(json.dumps(extract_fields(text), indent=2))
