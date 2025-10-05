# semantic_cache_demo.py
# End-to-end demo: semantic caching for LLM calls.
# - Flash-only model picker (avoid hitting pro; no test call at import time)
# - Backoff for 429/5xx
# - Canonicalization + normalization
# - Two vector indices (context & canonical) with NumPy cosine similarity
# - Topic tagging gate (Jaccard)
# - Warm start with small FAQ seeds
# - CLI: --scenario --k --topk --alpha --threshold --tag-thr --warmstart


from __future__ import annotations
"""End-to-end demo: semantic caching for LLM calls."""

import json, random
from collections import deque
import os, time, argparse, re
from typing import List, Dict, Tuple, Optional
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError

# ---------- Setup ----------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("No API key found in .env. Set GOOGLE_API_KEY=...")

genai.configure(api_key=API_KEY)

# --------- global rate limiter (RPM) ----------
class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self.win = 60.0
        self.ts = deque()

    def wait(self):
        now = time.monotonic()
        # roll off old timestamps
        while self.ts and (now - self.ts[0]) > self.win:
            self.ts.popleft()
        if len(self.ts) >= self.rpm:
            sleep = self.win - (now - self.ts[0]) + random.uniform(0, 0.25)
            time.sleep(max(0.0, sleep))
        # record this call
        self.ts.append(time.monotonic())

GEN_RPM = int(os.getenv("GEN_RPM", "2"))   # free tier is 2 RPM
EMB_RPM = int(os.getenv("EMB_RPM", "10"))  # embedding is 10 RPM
rl_gen = RateLimiter(GEN_RPM)
rl_emb = RateLimiter(EMB_RPM)


def _normalize_name(n: str) -> str:
    n = n.strip()
    return n if n.startswith("models/") else "models/" + n

# ---------- Model picker (flash only; no test call here to save quota) ----------
def pick_working_text_model(require_cap: str = "generateContent") -> Tuple[str, genai.GenerativeModel]:
    """Prefer flash models; allow pin via GEN_MODEL_NAME or TEXT_MODEL."""
    pinned = os.getenv("GEN_MODEL_NAME") or os.getenv("TEXT_MODEL")
    if pinned:
        name = _normalize_name(pinned)
        return name, genai.GenerativeModel(name)

    preferred = ["models/gemini-2.5-flash", "models/gemini-2.0-flash"]
    names = []
    try:
        for mm in genai.list_models():
            caps = set(getattr(mm, "supported_generation_methods", []) or [])
            if require_cap in caps and "flash" in mm.name:
                names.append(mm.name)
    except Exception:
        names = preferred[:]

    ordered = [n for n in preferred if n in names] + [n for n in names if n not in preferred]
    last = None
    for name in ordered:
        try:
            return name, genai.GenerativeModel(name)
        except Exception as e:
            last = e
    raise RuntimeError(f"No working flash model. Last error: {last}")

GEN_MODEL_NAME, TEXT_MODEL = pick_working_text_model()
print(f"[Model] Using text model: {GEN_MODEL_NAME}")

# IMPORTANT: embedding model id must NOT be prefixed with "models/"
EMB_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

# ---------- Backoff wrappers ----------
def call_with_backoff(model: genai.GenerativeModel, prompt: str, max_retries=3, base_wait=5):
    """Retry on 429/5xx with exponential backoff; use server-suggested delay if present."""
    wait = base_wait
    for _ in range(max_retries):
        try:
            rl_gen.wait()   
            return model.generate_content(prompt)
        except ResourceExhausted as e:
            secs = getattr(getattr(e, "retry_delay", None), "seconds", None)
            time.sleep(secs or wait); wait = min(wait * 2, 60)
        except (ServiceUnavailable, InternalServerError):
            time.sleep(wait); wait = min(wait * 2, 60)
    # final attempt (may still raise)
    rl_gen.wait()
    return model.generate_content(prompt)

def _parse_embed_result(out) -> List[List[float]]:
    """Support multiple sdk return shapes for embed_content."""
    # dict single
    if isinstance(out, dict) and "embedding" in out:
        return [out["embedding"]]
    # dict batch
    if isinstance(out, dict) and "embeddings" in out:
        rows = []
        for e in out["embeddings"]:
            if isinstance(e, dict) and "values" in e:
                rows.append(e["values"])
            elif isinstance(e, dict) and "embedding" in e:
                rows.append(e["embedding"])
        if rows:
            return rows
    # object style (newer sdk)
    if hasattr(out, "embedding"):
        return [list(out.embedding)]
    if hasattr(out, "embeddings"):
        return [list(e.values) if hasattr(e, "values") else list(e.embedding) for e in out.embeddings]
    raise RuntimeError("Unknown embedding return format")

def _normalize_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M / norms

def embed_with_backoff(texts: List[str]) -> np.ndarray:
    """
    Robust embedding:
    - Accepts str or List[str]
    - Tries batch; if SDK returns only 1 row (no batch support), falls back to per-item calls
    - Always returns (n, d) float32 L2-normalized
    """
    if isinstance(texts, str):
        texts = [texts]

    def _embed_once(batch: List[str]) -> List[List[float]]:
        rl_emb.wait()
        out = genai.embed_content(model=EMB_MODEL_NAME, content=batch if len(batch) > 1 else batch[0])
        return _parse_embed_result(out)

    wait, max_retries = 5, 3
    for _ in range(max_retries):
        try:
            vecs = _embed_once(texts)
            # If SDK ignored our batch and gave only 1 row, do per-item fallback
            if len(vecs) != len(texts):
                rows: List[List[float]] = []
                for t in texts:
                    rows.extend(_embed_once([t]))
                    time.sleep(0.05)  # tiny spacing to be gentle on quota
                vecs = rows
            M = np.asarray(vecs, dtype=np.float32)
            return _normalize_rows(M)
        except ResourceExhausted as e:
            secs = getattr(getattr(e, "retry_delay", None), "seconds", None)
            time.sleep(secs or wait); wait = min(wait * 2, 60)
        except (ServiceUnavailable, InternalServerError):
            time.sleep(wait); wait = min(wait * 2, 60)

    # last attempt (may still raise)
    vecs = _embed_once(texts)
    if len(vecs) != len(texts):
        rows: List[List[float]] = []
        for t in texts:
            rows.extend(_embed_once([t]))
            time.sleep(0.05)
        vecs = rows
    M = np.asarray(vecs, dtype=np.float32)
    return _normalize_rows(M)

# ---------- Light-weight in-memory vector index ----------
class CacheIndex:
    """Simple cosine (inner product on L2-normalized vectors) index using NumPy."""
    def __init__(self, dim: int):
        self.dim = dim
        self._vecs: List[np.ndarray] = []
        self._payloads: List[dict] = []

    def add(self, vec: np.ndarray, payload: dict):
        v = np.asarray(vec, dtype=np.float32).reshape(-1)  # force (d,)
        assert v.shape[0] == self.dim
        self._vecs.append(v)
        self._payloads.append(payload)

    def search(self, q: np.ndarray, topk=3) -> Tuple[np.ndarray, List[dict]]:
        if not self._vecs:
            return np.zeros((0,), dtype=np.float32), []
        M = np.vstack(self._vecs)           # (N, d)
        qv = np.asarray(q, dtype=np.float32).reshape(-1)  # (d,)
        sims = M @ qv                        # (N,) inner product (cosine because vectors are L2-normalized)
        idx = np.argsort(-sims)[:topk]
        return sims[idx], [self._payloads[i] for i in idx]

# ---------- Session & context ----------
class SessionStore:
    def __init__(self):
        self._data: Dict[str, List[Dict[str, str]]] = {}

    def history(self, sid: str, k=4) -> List[Dict[str, str]]:
        return (self._data.get(sid) or [])[-k:]

    def add_turn(self, sid: str, role: str, text: str):
        self._data.setdefault(sid, []).append({"role": role, "text": text})

def build_context_text(history: List[Dict[str, str]], user_query: str) -> str:
    """Minimal context builder for the LLM answer call."""
    lines = ["You are a helpful assistant. Answer succinctly."]
    for t in history[-4:]:
        lines.append(f"[{t['role']}] {t['text']}")
    lines.append(f"[user] {user_query}")
    return "\n".join(lines)

# ---------- Canonicalization & tagging ----------
CANON_PROMPT = """Rewrite the user's question into one canonical FAQ-style question.
- Keep only the core intent in <= 15 words.
- Use common vocabulary (e.g., 'corn' not 'maize').
- Remove fillers, hedges, and personal context.
Return ONLY the rewritten question."""

def _normalize_terms(s: str) -> str:
    t = (s or "").lower().strip()
    t = re.sub(r"(\d+)\s*%", r"\1 percent", t)
    t = t.replace("10-year", "10 year").replace("10yrs", "10 years")
    rep = {
        "maize": "corn",
        "global warming": "climate change",
        "productivity": "yield",
        "shoplifting losses": "retail shrink",
        "supermarkets": "grocery stores",
        "invest monthly": "monthly investment",
        "$": " dollars ",
    }
    for a, b in rep.items():
        t = t.replace(a, b)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def canonicalize(model: genai.GenerativeModel, history: List[Dict[str,str]], user_query: str) -> str:
    # Free-tier friendly: DO NOT call LLM; just normalize deterministically.
    return _normalize_terms(user_query)

def make_topic_tag(text: str) -> str:
    """Ultra-simple tag from canonical text: keep nouns-ish keywords."""
    t = _normalize_terms(text)
    tokens = [w for w in re.findall(r"[a-z]+", t) if w not in {"the","a","an","of","in","to","is","are"}]
    return " ".join(tokens[:6])

def tag_similarity(tag1: str, tag2: str) -> float:
    """Jaccard similarity on token sets; fast & cheap for a gate."""
    s1, s2 = set(tag1.split()), set(tag2.split())
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2); union = len(s1 | s2)
    return inter / max(1, union)

# ---------- LLM answer ----------
def llm_answer(context_text: str) -> Tuple[str, float]:
    t0 = time.time()
    resp = call_with_backoff(TEXT_MODEL, context_text)
    return (resp.text or "").strip(), time.time() - t0

# ---------- Demo queries & warm seeds ----------
SCENARIOS = {
    "agri": [
        "What is the impact of climate change on corn yields in the US?",
        "How does global warming affect the productivity of maize crops?",
        "What about wheat?",
    ],
    "finance": [
        "Explain compound interest for a 10-year horizon at 5%.",
        "Could you show the same for 7%?",
        "What if I invest monthly 200 dollars?",
    ],
    "retail": [
        "Give me 3 ways to reduce retail shrink.",
        "How to cut shoplifting losses in supermarkets?",
        "Any low-cost prevention ideas?",
    ],
}

FAQ_SEEDS = {
    "agri": [
        ("climate change effect on corn yield",
         "Corn yields drop under heat/drought; adapt with heat-tolerant hybrids, irrigation, and shifting planting dates."),
        ("climate change effect on wheat yield",
         "Wheat tolerates heat a bit better; timing and variety choice matter; irrigation helps."),
    ],
    "finance": [
        ("compound interest example 10 year at 5 percent",
         "At 5% annually for 10 years: FV = PV*(1.05)^10 (≈1.629x)."),
        ("compound interest example 10 years at 7 percent",
         "At 7% annually for 10 years: FV = PV*(1.07)^10 (≈1.967x)."),
        ("compound interest with monthly 200 dollars contribution",
         "Monthly $200 at 5%/yr compounded monthly: FV ≈ 200*((1+0.05/12)^(12*Y)-1)/(0.05/12)."),
    ],
    "retail": [
        ("reduce retail shrink practical tips",
         "Do EAS tags on high-loss SKUs, face-up/zone checks, and staff training at exits."),
        ("cut shoplifting losses in grocery stores",
         "Move razor-blades to staff-assisted areas, locked cases for hot SKUs, and CCTV at blind spots."),
    ],
}

# ---------- Pipeline ----------
def run_demo(
    queries: List[str],
    k_turns=4,
    topk=3,
    alpha=0.5,
    threshold=0.84,
    tag_thr=0.60,
    do_warmstart: Optional[str]=None
):
    """
    alpha: mix context score and canonical score (used as simple source-weights)
    threshold: min score to treat as HIT
    tag_thr: minimal topic-tag similarity to accept a HIT
    """
    sessions = SessionStore()
    session_id = "demo"

    # Probe embedding dimension once
    dim = embed_with_backoff(["ping"]).shape[1]
    index_ctx, index_can = CacheIndex(dim), CacheIndex(dim)

    def add_item(can_text: str, ctx_text: str, answer: str):
        # One batch call -> (2, d)
        V = embed_with_backoff([can_text, ctx_text])
        v_can, v_ctx = V[0], V[1]
        payload = {
            "answer": answer,
            "can_text": can_text,
            "ctx_text": ctx_text,
            "tag": make_topic_tag(can_text),
        }
        index_can.add(v_can, payload)
        index_ctx.add(v_ctx, payload)

    # Warm start (seed a few FAQs without LLM calls)
    if do_warmstart:
        which = [do_warmstart] if do_warmstart in FAQ_SEEDS else list(FAQ_SEEDS.keys())
        print(f"[WarmStart] seeding indices for: {', '.join(which)}")
        for cat in which:
            for can_q, ans in FAQ_SEEDS[cat]:
                ctx = f"[system] FAQ seed\n[user] {can_q}"
                add_item(can_q, ctx, ans)
                time.sleep(0.6)  # gentle throttling

    hits = misses = 0
    lat_hit, lat_miss = [], []

    for q in queries:
        hist = sessions.history(session_id, k=k_turns)
        ctx_text = build_context_text(hist, q)
        can_text = canonicalize(TEXT_MODEL, hist, q)

        # Embed both query variants in one call
        Vq = embed_with_backoff([can_text, ctx_text])
        v_can, v_ctx = Vq[0], Vq[1]

        # Search both indices
        s_can, cand_can = index_can.search(v_can, topk=topk)
        s_ctx, cand_ctx = index_ctx.search(v_ctx, topk=topk)

        # Combine rankings with simple source-weights:
        # candidates from canonical side get (1-alpha)*score; context side get alpha*score
        best_s, best_payload = 0.0, None
        for s, p in zip(s_ctx, cand_ctx):
            score = float(alpha * s)
            if score > best_s:
                best_s, best_payload = score, p
        for s, p in zip(s_can, cand_can):
            score = float((1.0 - alpha) * s)
            if score > best_s:
                best_s, best_payload = score, p

        if best_payload and best_s >= threshold:
            # Topic gate
            tag_q = make_topic_tag(can_text)
            tag_p = best_payload.get("tag", "")
            sim_tag = tag_similarity(tag_q, tag_p)
            if sim_tag >= tag_thr:
                hits += 1
                lat_hit.append(0.01)  # pretend cache read
                print(f"[HIT  s={best_s:.3f}  tag={sim_tag:.2f}] {q}")
                ans = best_payload["answer"]
            else:
                # gate rejected -> miss
                ans, latency = llm_answer(ctx_text)
                misses += 1
                lat_miss.append(latency)
                add_item(can_text, ctx_text, ans)
                print(f"[MISS s={best_s:.3f} tag={sim_tag:.2f} {latency:.2f}s] {q}")
        else:
            # miss
            ans, latency = llm_answer(ctx_text)
            misses += 1
            lat_miss.append(latency)
            add_item(can_text, ctx_text, ans)
            print(f"[MISS s={best_s:.3f}  {latency:.2f}s] {q}")

        sessions.add_turn(session_id, "user", q)
        sessions.add_turn(session_id, "assistant", ans)

    # Metrics
    reqs = hits + misses
    print("\n=== Metrics ===")
    print(f"Using model: {GEN_MODEL_NAME}")
    print(f"Requests: {reqs}")
    print(f"Hit rate: {hits/reqs if reqs else 0.0:.2f}")
    if lat_hit:  print(f"Avg latency (hit):  {np.mean(lat_hit):.3f}s")
    if lat_miss: print(f"Avg latency (miss): {np.mean(lat_miss):.3f}s")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="agri", choices=["agri","finance","retail","all"])
    ap.add_argument("--k", type=int, default=4, help="history turns")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight for ctx vs can")
    ap.add_argument("--threshold", type=float, default=0.84)
    ap.add_argument("--tag-thr", type=float, default=0.60)
    ap.add_argument("--warmstart", default=None, help="None | agri | finance | retail | all")
    args = ap.parse_args()

    if args.scenario == "all":
        for s in ["agri","finance","retail"]:
            print(f"\n=== Scenario: {s} ===")
            run_demo(
                SCENARIOS[s],
                k_turns=args.k, topk=args.topk,
                alpha=args.alpha, threshold=args.threshold,
                tag_thr=args.tag_thr,                 # <-- use attribute, not dict
                do_warmstart=args.warmstart
            )
    else:
        print(f"\n=== Scenario: {args.scenario} ===")
        run_demo(
            SCENARIOS[args.scenario],
            k_turns=args.k, topk=args.topk,
            alpha=args.alpha, threshold=args.threshold,
            tag_thr=args.tag_thr,
            do_warmstart=args.warmstart
        )

if __name__ == "__main__":
    main()
