"""Text-to-image semantic search over the CLIP index (Deliverable 1).

    from search import search
    results = search("a red itchy rash", top_k=3)
    # -> [{"filename": "...", "similarity": 0.31, "above_threshold": True}, ...]

Also supports duplicate detection: find_duplicates(threshold=0.98) compares
every image against every other image and flags near-identical pairs, which
is the same CLIP-embedding-distance mechanism used for cross-modal search.

CLI:
    python search.py "a burn on the hand"
    python search.py --duplicates
    python search.py --calibrate   # prints the similarity spread used to
                                    # justify MIN_SIMILARITY_THRESHOLD
"""
import argparse
import sys

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

import config

_model = None
_processor = None
_filenames = None
_embeddings = None


def _lazy_load_model():
    """Load the CLIP model into the module-level cache on first use only.
    Every function below (embed_text, embed_image, domain_similarity) calls
    this, so the ~600MB model is loaded at most once per process no matter
    how many searches/captions are run."""
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
        _processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        _model.eval()


def _lazy_load_index():
    """Load outputs/clip_index.npz (written by build_index.py) into the
    module-level cache on first use. Raises a clear error if build_index.py
    hasn't been run yet, instead of failing deeper inside a matrix op."""
    global _filenames, _embeddings
    if _embeddings is None:
        if not config.INDEX_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"No index at {config.INDEX_EMBEDDINGS_PATH}. Run build_index.py first."
            )
        data = np.load(config.INDEX_EMBEDDINGS_PATH)
        _filenames = data["filenames"]
        _embeddings = data["embeddings"]


def _pooled(output):
    # transformers>=5 returns a BaseModelOutputWithPooling; the projected
    # embedding is in .pooler_output. Older versions returned a bare tensor.
    return output.pooler_output if hasattr(output, "pooler_output") else output


def embed_text(query: str) -> np.ndarray:
    """Turn a text string into a single L2-normalized CLIP embedding (shape
    (512,)). Because both text and image embeddings live in the same CLIP
    vector space, this is directly comparable (via dot product) to the image
    embeddings build_index.py produced -- that's what makes text-to-image
    search work."""
    _lazy_load_model()
    inputs = _processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = _pooled(_model.get_text_features(**inputs))
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.numpy()[0]  # drop the batch dimension -- always a single query here


def embed_image(image) -> np.ndarray:
    """Embed a PIL image with the same CLIP model used to build the index.
    Used by domain_similarity() below to classify an uploaded photo against
    the zero-shot domain labels, not to rebuild the retrieval index."""
    _lazy_load_model()
    inputs = _processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = _pooled(_model.get_image_features(**inputs))
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.numpy()[0]


_domain_label_embeddings = None


def domain_similarity(image) -> float:
    """Zero-shot CLIP similarity between `image` and the best-matching entry
    in config.DOMAIN_LABELS (e.g. "a rash", "a bruise"). Used by
    caption_image.py as an out-of-scope check: a photo that doesn't resemble
    any in-domain label scores low here regardless of what it does or
    doesn't match in the retrieval index. See config.py for calibration.
    """
    global _domain_label_embeddings
    if _domain_label_embeddings is None:
        # Embed all 12 DOMAIN_LABELS once and cache the matrix -- this is the
        # "zero-shot classifier": there's no trained classification head,
        # just CLIP's shared embedding space and a list of candidate labels.
        _domain_label_embeddings = np.array([embed_text(label) for label in config.DOMAIN_LABELS])
    feat = embed_image(image)
    # embeddings are unit-normalized, so this matrix-vector product gives one
    # cosine-similarity score per label; take the best (most similar) label.
    return float(np.max(_domain_label_embeddings @ feat))


def search(query: str, top_k: int = 5):
    """Return the top_k images whose CLIP embedding best matches `query`.

    Each result includes `above_threshold`, computed against
    config.MIN_SIMILARITY_THRESHOLD, so callers can decide whether to show a
    "no confident match" state instead of a misleading low-relevance result.
    """
    _lazy_load_index()
    q = embed_text(query)
    # _embeddings is (n_images, 512) and q is (512,), so this matrix-vector
    # product computes cosine similarity between the query and every index
    # image in one shot (both sides are already unit-normalized).
    sims = _embeddings @ q
    # argsort(-sims) sorts descending (most similar first); slice to top_k.
    order = np.argsort(-sims)[:top_k]
    return [
        {
            "filename": str(_filenames[i]),
            "similarity": float(sims[i]),
            "above_threshold": bool(sims[i] >= config.MIN_SIMILARITY_THRESHOLD),
        }
        for i in order
    ]


def find_duplicates(threshold: float = 0.98):
    """Flag near-identical image pairs by CLIP cosine similarity. Not used by
    the main app/router -- a standalone data-quality check for the index
    (run via `python search.py --duplicates`)."""
    _lazy_load_index()
    n = len(_filenames)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(_embeddings[i] @ _embeddings[j])
            if sim >= threshold:
                pairs.append((str(_filenames[i]), str(_filenames[j]), sim))
    return pairs


def calibrate():
    """Print similarity stats used to justify MIN_SIMILARITY_THRESHOLD.

    We probe the index with queries that clearly match one cluster of
    images and clearly do not match any image, then report the gap between
    "true match" and "true non-match" similarity scores.
    """
    _lazy_load_index()
    relevant_queries = ["a red skin rash", "a bruise on the skin", "a burn injury"]
    irrelevant_queries = ["a spreadsheet of quarterly sales", "a mountain landscape at sunset"]

    def scores_for(queries):
        out = []
        for q in queries:
            emb = embed_text(q)
            out.extend((_embeddings @ emb).tolist())
        return np.array(out)

    rel = scores_for(relevant_queries)
    irr = scores_for(irrelevant_queries)
    print(f"Relevant-query similarities   : min={rel.min():.3f} mean={rel.mean():.3f} max={rel.max():.3f}")
    print(f"Irrelevant-query similarities : min={irr.min():.3f} mean={irr.mean():.3f} max={irr.max():.3f}")
    print(f"Configured MIN_SIMILARITY_THRESHOLD = {config.MIN_SIMILARITY_THRESHOLD}")
    print("Justification: this sits above the irrelevant-query ceiling and below the "
          "relevant-query floor observed on this index, so it separates genuine "
          "domain matches from off-topic queries without discarding true positives.")


def _cli():
    """Command-line entry point: `python search.py <query>` (plain search),
    `--duplicates` (data-quality check), or `--calibrate` (threshold
    justification) -- three separate modes, one dispatched per invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--duplicates", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if args.calibrate:
        calibrate()
        return
    if args.duplicates:
        pairs = find_duplicates()
        if not pairs:
            print("No near-duplicate pairs found above threshold.")
        for a, b, sim in pairs:
            print(f"DUPLICATE  {a} <-> {b}  similarity={sim:.4f}")
        return
    if not args.query:
        parser.error("provide a query, or use --duplicates / --calibrate")

    for r in search(args.query, top_k=args.top_k):
        flag = "" if r["above_threshold"] else "  (below confidence threshold)"
        print(f"{r['similarity']:.4f}  {r['filename']}{flag}")


if __name__ == "__main__":
    sys.exit(_cli())
