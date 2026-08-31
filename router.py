"""Input router: dispatches a submitted file to the image or audio pipeline
(Deliverable 4). This is the single decision point app.py calls into so the
UI layer never needs to know about captioning or transcription internals.
"""
from pathlib import Path

from caption_image import caption_image
from transcribe_audio import transcribe

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def classify_input(file_path: str) -> str:
    """Decide "image" vs "audio" purely from the file extension -- no
    content sniffing. Raises on anything else so an unsupported upload
    fails loudly here rather than deeper inside a pipeline that assumes
    one modality or the other."""
    ext = Path(file_path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    raise ValueError(f"Unsupported file type '{ext}'. Expected an image ({IMAGE_EXTS}) or audio file ({AUDIO_EXTS}).")


def route(file_path: str) -> dict:
    """Run the appropriate pipeline and return a uniform result envelope:
    {"modality": "image"|"audio", "result": {...pipeline-specific...}}
    """
    modality = classify_input(file_path)
    if modality == "image":
        return {"modality": "image", "result": caption_image(file_path)}
    # classify_input() only ever returns "image" or "audio" (or raises), so
    # anything that isn't "image" here is "audio" -- no third branch needed.
    return {"modality": "audio", "result": transcribe(file_path)}


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) != 2:
        raise SystemExit("usage: python router.py <file_path>")
    print(json.dumps(route(sys.argv[1]), indent=2, ensure_ascii=False))
