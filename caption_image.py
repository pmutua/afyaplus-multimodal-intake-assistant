"""Image captioning pipeline for patient-submitted intake photos (Deliverable 2).

Primary path (per the capstone brief) is GPT-4o vision with a constrained
system prompt. This project runs the local-fallback path instead (no
OpenAI credits spent): Salesforce/blip-image-captioning-base, wrapped with
the same constrained-prompt *behaviour* -- domain vocabulary steering,
out-of-scope detection, and a mandatory safety disclaimer on every response
-- reproduced here as pre/post-processing since BLIP has no system-prompt
mechanism of its own.

    from caption_image import caption_image
    result = caption_image("data/images/test_cases/test_clear.png")
    # -> {"caption": ..., "flags": [...], "disclaimer": ...}

CLI:
    python caption_image.py data/images/test_cases/test_clear.png
"""
import sys

import numpy as np
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

import config
from search import domain_similarity

_caption_model = None
_caption_processor = None


def _load_caption_model():
    """Load BLIP into the module-level cache on first use only, so repeated
    caption_image() calls in the same process (e.g. from evaluate.py or the
    Gradio app) don't reload the ~470MB model every time."""
    global _caption_model, _caption_processor
    if _caption_model is None:
        _caption_processor = BlipProcessor.from_pretrained(config.CAPTION_MODEL_NAME)
        _caption_model = BlipForConditionalGeneration.from_pretrained(config.CAPTION_MODEL_NAME)
        _caption_model.eval()
    return _caption_model, _caption_processor


def _is_out_of_scope(image: Image.Image) -> bool:
    """Zero-shot CLIP scope classifier: if the image doesn't resemble any of
    config.DOMAIN_LABELS, it is probably not a skin/wound-type photo at all.
    See config.py for how the threshold was calibrated."""
    best = domain_similarity(image)
    return best < config.OUT_OF_SCOPE_SIMILARITY_CEILING


def _looks_unreadable(image: Image.Image) -> bool:
    """Very low contrast / near-uniform images are flagged unreadable rather
    than given a confident-sounding caption."""
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    return float(arr.std()) < 12.0


def caption_image(image_path: str) -> dict:
    """Main entry point for the captioning pipeline. Runs two safety checks
    *before* ever calling the caption model, in this order:
      1. _looks_unreadable()  -- is the image too blurry/low-contrast to
         describe at all?
      2. _is_out_of_scope()   -- does the image even look like a medical
         photo (rash/wound/etc.), or something unrelated entirely?
    Only if both checks pass does BLIP actually generate a caption. Every
    return path -- unreadable, out-of-scope, or a real caption -- always
    includes config.CAPTION_DISCLAIMER and HUMAN_REVIEW_REQUIRED, so nothing
    ever reaches the UI without both."""
    image = Image.open(image_path).convert("RGB")
    flags = []

    if _looks_unreadable(image):
        return {
            "caption": None,
            "flags": [config.UNREADABLE_FLAG, config.HUMAN_REVIEW_FLAG],
            "disclaimer": config.CAPTION_DISCLAIMER,
            "note": "Image quality too low (blur/low contrast) for a reliable description. "
                    "Please resubmit a clearer, well-lit photo.",
        }

    if _is_out_of_scope(image):
        return {
            "caption": None,
            "flags": [config.OUT_OF_SCOPE_FLAG, config.HUMAN_REVIEW_FLAG],
            "disclaimer": config.CAPTION_DISCLAIMER,
            "note": "This image does not appear to show a skin condition, wound, or "
                    "other intake-relevant subject. It has been routed to a human "
                    "reviewer instead of an automated description.",
        }

    # Both safety checks passed -- generate the actual caption with BLIP.
    model, processor = _load_caption_model()
    prompt = "a medical photo of"  # constrained prompt: steers BLIP toward clinical framing
    inputs = processor(image, prompt, return_tensors="pt")
    with torch.no_grad():  # inference only, no gradient tracking needed
        out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)

    flags.append(config.HUMAN_REVIEW_FLAG)  # every safety-sensitive caption needs sign-off
    return {
        "caption": caption,
        "flags": flags,
        "disclaimer": config.CAPTION_DISCLAIMER,
        "note": None,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python caption_image.py <image_path>")
    result = caption_image(sys.argv[1])
    print(result)
