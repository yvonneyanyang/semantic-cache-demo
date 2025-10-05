# cache_index.py
import numpy as np

class CacheIndex:
    """Cosine similarity index (NumPy).
    Always accept vectors in any of (d,), (1,d), (d,1) and convert to 1-D float32.
    """
    def __init__(self, dim: int, dtype=np.float32, normalize=True):
        self.dim = int(dim)
        self.dtype = dtype
        self.normalize = normalize
        self._vecs = []       # list of 1-D vectors (d,)
        self._payloads = []   # list of dict

    def _as_vec(self, v):
        """Convert input array-like to 1-D float32 of length dim (optionally L2-normalized)."""
        a = np.asarray(v, dtype=self.dtype)
        # squeeze to 1-D: (1,d) -> (d,), (d,1) -> (d,), already (d,) stays (d,)
        a = a.reshape(-1)
        if a.shape[0] != self.dim:
            raise ValueError(f"vector dim {a.shape[0]} != index dim {self.dim}")
        if self.normalize:
            n = np.linalg.norm(a) + 1e-9
            a = (a / n).astype(self.dtype)
        return a

    def add(self, vec, payload: dict):
        v = self._as_vec(vec)
        self._vecs.append(v)
        self._payloads.append(payload)

    def search(self, q, topk=3):
        if not self._vecs:
            return np.zeros((0,), dtype=self.dtype), []
        qv = self._as_vec(q)
        M = np.vstack(self._vecs)     # (N, d)
        sims = M @ qv                 # cosine if vectors are L2-normalized
        order = np.argsort(-sims)[:topk]
        return sims[order], [self._payloads[i] for i in order]
