"""Build a CLIP embedding index over the domain image library (Deliverable 1).

Usage:
    python build_index.py

Produces outputs/clip_index.npz containing:
    filenames : array of image filenames (relative to data/images/index/)
    embeddings: L2-normalized CLIP image embeddings, one row per file

Runs entirely locally on CPU using openai/clip-vit-base-patch32 (~600MB,
downloaded once from Hugging Face and cached under %USERPROFILE%\\.cache).
"""
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import config


def load_clip():
    """Load the CLIP model + its matching preprocessor (image resize/normalize,
    text tokenizer) from Hugging Face. model.eval() disables dropout etc. since
    we only ever run inference here, never training."""
    model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    model.eval()
    return model, processor


def embed_images(model, processor, image_paths):
    """Turn a list of image file paths into a matrix of CLIP embeddings, one
    row per image, each row L2-normalized to unit length. Normalizing here
    means a plain dot product at search time equals cosine similarity --
    see search.py's search() -- so we don't need a separate norm step per
    query."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():  # inference only, no need to track gradients
        output = model.get_image_features(**inputs)
    # transformers>=5 returns a BaseModelOutputWithPooling; the projected
    # embedding is in .pooler_output. Older versions returned a bare tensor.
    features = output.pooler_output if hasattr(output, "pooler_output") else output
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.numpy()


def build():
    """Entry point: embed every PNG in data/images/index/ and persist the
    result to outputs/clip_index.npz. This is meant to be run once (or again
    whenever the index images change) -- search.py just loads the saved file
    rather than re-embedding images on every query."""
    image_paths = sorted(config.IMAGES_INDEX_DIR.glob("*.png"))
    if len(image_paths) < 15:
        # Rubric requires >=15 index images for the retrieval deliverable;
        # fail loudly here rather than silently building a too-small index.
        raise SystemExit(
            f"Found only {len(image_paths)} images in {config.IMAGES_INDEX_DIR}; "
            "the rubric requires at least 15. Add more domain images to "
            "data/images/index/ first."
        )

    print(f"Loading CLIP model {config.CLIP_MODEL_NAME} ...")
    model, processor = load_clip()

    print(f"Embedding {len(image_paths)} images ...")
    embeddings = embed_images(model, processor, image_paths)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    # np.savez keeps filenames and embeddings together in one .npz archive so
    # search.py can load both with a single np.load() call and match each
    # embedding row back to the image it came from by position.
    np.savez(
        config.INDEX_EMBEDDINGS_PATH,
        filenames=np.array([p.name for p in image_paths]),
        embeddings=embeddings,
    )
    print(f"Saved index -> {config.INDEX_EMBEDDINGS_PATH} ({embeddings.shape[0]} vectors, dim={embeddings.shape[1]})")
    print(f"Minimum similarity threshold in use: {config.MIN_SIMILARITY_THRESHOLD} "
          "(see config.py docstring for justification; re-derive it with "
          "`python search.py --calibrate`).")


if __name__ == "__main__":
    build()
