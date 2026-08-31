"""Speech-to-text transcription pipeline for patient-submitted audio (Deliverable 3).

Primary path (per the capstone brief) is the OpenAI Whisper API. This
project runs the local-fallback path instead: faster-whisper (CTranslate2
port of Whisper), "base" size, CPU-only. Handles long recordings via
faster-whisper's built-in VAD-based chunking (see transcribe_long below).

    from transcribe_audio import transcribe
    result = transcribe("data/audio/intake_en_clear.wav")
    # -> {"text": ..., "language": "en", "fields": {...}, "disclaimer": ...}

CLI:
    python transcribe_audio.py data/audio/intake_sw_multilang.mp3
"""
import sys

from faster_whisper import WhisperModel

import config
from extract_fields import extract_fields

_model = None


def _load_model():
    """Load faster-whisper into the module-level cache on first use only.
    compute_type="int8" quantizes the model for faster CPU inference at a
    small accuracy cost -- appropriate here since there's no GPU."""
    global _model
    if _model is None:
        _model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe_long(audio_path: str, chunk_length_s: int = 30):
    """Transcribe audio of any length. faster-whisper internally splits audio
    on voice-activity boundaries (vad_filter) so long recordings are handled
    as a sequence of speech chunks rather than one oversized pass, avoiding
    the truncation/timeout issues a naive single-shot call would hit."""
    model = _load_model()
    # vad_filter=True runs voice-activity detection first and only transcribes
    # the speech segments it finds, skipping silence -- this is what lets one
    # call handle audio of any length without a fixed chunking loop here.
    segments, info = model.transcribe(
        audio_path,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    segments = list(segments)  # `segments` is a lazy generator; materialize it once
    text = " ".join(seg.text.strip() for seg in segments)
    return text, info, segments


def transcribe(audio_path: str) -> dict:
    """Main entry point: transcribe audio_path, then immediately pass the
    transcript into extract_fields() (extract_fields.py) so callers (the
    router/app) get both the raw transcript and the structured intake
    fields in one call."""
    text, info, segments = transcribe_long(audio_path)
    fields = extract_fields(text)
    return {
        "text": text,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "num_segments": len(segments),
        "fields": fields,
        "disclaimer": config.TRANSCRIPT_DISCLAIMER,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python transcribe_audio.py <audio_path>")
    import json
    print(json.dumps(transcribe(sys.argv[1]), indent=2, ensure_ascii=False))
