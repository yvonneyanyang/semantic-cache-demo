# Semantic Cache Demo (DSRS Screening Test)

End-to-end demo to intercept LLM queries, detect semantic duplicates, and serve cached answers to reduce **latency** and **cost**.

## TL;DR
- **Dual index**: canonical question & context-aware text  
- **Topic gate**: Jaccard similarity to avoid wrong reuse  
- **Metrics**: Hit rate, latency (hit/miss), avoided LLM calls  
- **Robustness**: rate limiting + backoff; warm-start seeds

---

# Setup

```bash
pip install -r requirements.txt
```
## Create your `.env` from the example

```bash
# Windows
copy .env.example .env
```
Then edit `.env` and paste your key.

## Required environment variables
```ini
GOOGLE_API_KEY=PUT_YOUR_KEY_HERE
GEN_RPM=2
EMB_RPM=10
GEN_MODEL_NAME=models/gemini-2.5-flash
EMBEDDING_MODEL=text-embedding-004
# Optional: NO_LLM=0  # set 1 to avoid LLM calls when quota is tight
```
Security: `.env` is git-ignored; only `.env.example` is public.

# Run
## One-line recommended run

```bash
python semantic_cache_demo.py --scenario retail --warmstart all --alpha=0.2 --threshold=0.40 --tag-thr=0.30 --topk=5
```

## What you’ll see

- `[HIT ...]` → served from cache (≈0.01s)

- `[MISS ... 3.xx s]` → one real LLM call + write-back

## Run other scenarios

```bash
python semantic_cache_demo.py --scenario agri --warmstart all --alpha=0.2 --threshold=0.40 --tag-thr=0.30 --topk=5
python semantic_cache_demo.py --scenario finance --warmstart all --alpha=0.2 --threshold=0.40 --tag-thr=0.30 --topk=5
````

## Results (measured)

**How to reproduce (all scenarios)**

### Windows PowerShell
```powershell
foreach ($s in 'retail','agri','finance') {
  python semantic_cache_demo.py --scenario $s --warmstart all --alpha=0.2 --threshold=0.40 --tag-thr=0.30 --topk=5
}
```

### Scenario: retail

Params: `alpha=0.2, threshold=0.40, tag-thr=0.30, topk=5`

Round 1:
- Requests: **3**
- Hit rate: **0.67**
- Avg latency (hit): **0.010 s**
- Avg latency (miss): **4.277 s**
- Avoided LLM calls (hits): **2**
- Average latency: `0.67×0.010 + 0.33×4.277 ≈ 1.42 s` → vs. 4.277 s baseline ⇒ ~66.8% reduction

### Scenario: agri

Params: `alpha=0.2, threshold=0.40, tag-thr=0.30, topk=5`

Round 1:
- Requests: **3**
- Hit rate: **0.67**
- Avg latency (hit): **0.010 s**
- Avg latency (miss): **1.663 s**
- Avoided LLM calls (hits): **2**
- Average latency: `0.67×0.010 + 0.33×1.663 ≈ 0.56 s` → vs. 1.663 s baseline ⇒ ~66.6% reduction

### Scenario: finance

Params: `alpha=0.2, threshold=0.40, tag-thr=0.30, topk=5`

Round 1:
- Requests: **3**
- Hit rate: **0.33**
- Avg latency (hit): **0.010 s**
- Avg latency (miss): **4.557 s**
- Avoided LLM calls (hits): **1**
- Average latency: `0.33×0.010 + 0.67×4.557 ≈ 3.06 s` → vs. 4.557 s baseline ⇒ ~33.0% reduction

Optional: run each scenario a second time and append “Round 2” here — the hit rate typically increases because newly written entries are reused.

# How it works (design choices)
## Pipeline

1. Canonicalize (deterministic normalize) current query
2. Build minimal context from last k turns
3. Embed both (canonical & context) → two vector indices
4. Search both; blend scores by `alpha`
5. Gate by similarity `threshold` + topic Jaccard `tag-thr`
6. HIT → return cached answer; MISS → call LLM, write-back

## Why dual indices?

Canonical captures intent invariants; context captures conversational reference. Blending prevents one side from dominating.

## Cache content

Store: canonical text, context text, both embeddings, answer, topic tag.

# Metrics to report (repro steps)

For each scenario, run two rounds with the same params:
Round 1 (with warmstart) → Round 2 (benefits from new writes).

Record:

- Requests, Hit rate, Avg latency (hit/miss)
- Avoided LLM calls = number of hits(≈ `Hit rate × Requests`)
- (Optional) Estimated cost saved = avoided_calls × per-call unit cost (or tokens × unit price)

**Example calculation (replace with your measured values)** 
Assume: `hit_rate ≈ 0.67`, `hit_latency ≈ 0.01s`, `miss_latency ≈ 10–16s`

Average latency
`avg_latency ≈ (hit_rate × hit_latency) + (1 - hit_rate) × miss_latency`

If `miss_latency = 10s`: `0.67×0.01 + 0.33×10 ≈ 3.31s` → from 10s down to ~3.3s (↓ ~66%)

If `miss_latency = 16s`: `0.67×0.01 + 0.33×16 ≈ 5.29s` → from 16s down to ~5.3s (↓ ~67%)

One-line summary (example): HIT ≈ 0.01s, MISS ≈ 10–16s, Hit rate ≈ 0.67 → average latency ↓ ~50–70%, avoided LLM calls ≈ ~2/3.

# Sensitivity / Trade-offs (optional)

These two presets illustrate the precision–reuse trade-off.  
Pick **one** to test; the default remains the **One-line recommended run** above.

**Strict (precision first; lower hit rate)**
```bash
python semantic_cache_demo.py --scenario retail --warmstart all --alpha=0.3 --threshold=0.60 --tag-thr=0.50 --topk=5
```

**Looser (higher hit rate; rely more on topic gate)**
```bash
python semantic_cache_demo.py --scenario retail --warmstart all --alpha=0.2 --threshold=0.35 --tag-thr=0.25 --topk=7
```

Notes

- Stricter thresholds → fewer hits, lower risk of wrong reuse.

- Looser thresholds → more hits; use the Jaccard topic gate (`--tag-thr`) to control false reuse.

- To try other domains, change `--scenario` to `agri` or `finance`.

<!-- Optional, if you want a one-liner for deeper tuning --> <!-- Advanced tuning (optional): `--threshold` 0.30–0.80, `--tag-thr` 0.25–0.60, `--alpha` 0.2–0.5, `--topk` 5–7. -->

# Scalability & Eviction (proposal)

- Index: swap NumPy search for FAISS (IVF/HNSW) or Redis Vector for million-scale.
- Eviction: LRU + TTL + heat (recent hits) hybrid score; short TTL for one-off queries, long TTL for FAQ.
- Freshness: store source/versions; check before reuse.

# Optional / Future work (not implemented)
## Limitations of simple context
- **Limitations of simple context**: topic shifts (semantic drift), missing task state (goal/subgoal/progress), freshness/version not encoded.
  Mitigations: conversation segmentation + per-segment summaries; state schema; metadata/TTL checks.
- **Agent caching proposal**: cache tool I/O, subgoal summaries, and final answers; key = embedding(goal+subgoal+tool+doc_id) + hashes(schema/version); TTL/heat eviction; dependency invalidation on dataset/code version changes.
- **Embedding dimension trade-off**: simulate `768 → 384 → 256 → 128` via random projection/PCA; report Recall@k & RAM (see `scripts/dim_sweep.py` if added).

# Troubleshooting

- 429 quota: use warmstart; lower calls; set `NO_LLM=1` as fallback.
- Auth: make sure `.env` exists and the API key is valid.
- Security: never commit your real `.env` (keep only `.env.example` in repo).

# Repo structure
```text
semantic_cache_demo.py
embedder.py / cache_index.py / session_store.py / context_builder.py
topic_tagger.py
.env.example
requirements.txt
scripts/ (optional)  └─ dim_sweep.py
````
# License
MIT
