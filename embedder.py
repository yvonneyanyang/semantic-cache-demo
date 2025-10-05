# embedder.py
# Thin wrapper around Gemini embeddings, with batch + consistent 1D vectors.

import os
from typing import List
import numpy as np
import google.generativeai as genai

EMBED_MODEL = "text-embedding-004"

class Embedder:
    def __init__(self, api_key: str | None = None):
        # Prefer an explicit key; otherwise read from environment variables.
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Missing GOOGLE_API_KEY / GEMINI_API_KEY in environment.")
        genai.configure(api_key=key)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts; return an array of shape (n, d), dtype float32.
        We call the API per-item to avoid SDK shape surprises, then stack.
        """
        vecs: List[np.ndarray] = []
        for t in texts:
            out = genai.embed_content(model=EMBED_MODEL, content=t)
            v = np.asarray(out["embedding"], dtype=np.float32).reshape(-1)  # <-- force 1D
            vecs.append(v)
        return np.vstack(vecs)  # (n, d)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text; return (d,) float32 for convenience."""
        return self.embed_batch([text])[0]

    def dim(self) -> int:
        """Probe the embedding dimensionality once using batch path."""
        return int(self.embed_batch(["probe"])[0].shape[0])
