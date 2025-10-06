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
- Average latency: `0.67×0.010 + 0.33×4.277 ≈ 1.42 s` → vs. 4.277 s baseline ⇒ **~66.8% reduction**

### Scenario: agri

Params: `alpha=0.2, threshold=0.40, tag-thr=0.30, topk=5`

Round 1:
- Requests: **3**
- Hit rate: **0.67**
- Avg latency (hit): **0.010 s**
- Avg latency (miss): **1.663 s**
- Avoided LLM calls (hits): **2**
- Average latency: `0.67×0.010 + 0.33×1.663 ≈ 0.56 s` → vs. 1.663 s baseline ⇒ **~66.6% reduction**

### Scenario: finance

Params: `alpha=0.2, threshold=0.40, tag-thr=0.30, topk=5`

Round 1:
- Requests: **3**
- Hit rate: **0.67** 
- Avg latency (hit): **0.010 s**
- Avg latency (miss): **5.544 s**
- Avoided LLM calls (hits): **2**
- Average latency: `0.67×0.010 + 0.33×5.544 ≈ 1.836 s` → vs. 5.544 s baseline ⇒ **~66.9% reduction**

_Sensitivity note_: The elliptical follow-up “Could you show the same for 7%?” already hits with the current tagger (tag=0.33).  
The query “What if I invest monthly 200 dollars?” has tag≈0.22; either (a) run with `--tag-thr=0.20` or (b) shorten topic tags to ~4 tokens (digits and
keywords first) so it becomes a hit without loosening the gate.

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

The canonical index captures intent-level invariants, while the context index preserves conversational references.

Blending them prevents over-reliance on either the literal phrasing or the surrounding context, achieving more balanced reuse.

## Cache content

Each cache entry stores:

- canonical text
- context text
- both embeddings
- generated answer
- topic tag

# Metrics and Reproducibility

To reproduce results, run two rounds per scenario:

**Round 1** (with warmstart) → **Round 2** (benefits from newly cached entries).

Record the following metrics:

- Requests count
- Hit rate
- Average latency (hit/miss)
- Avoided LLM calls = `Hit rate × Requests`
- _(Optional)_ Estimated cost saved = avoided_calls × per-call cost

**Example Calculation** 
If `hit_rate ≈ 0.67`, `hit_latency ≈ 0.01s`, and `miss_latency ≈ 10–16s`:

    avg_latency=(hit_rate×hit_latency)+(1−hit_rate)×miss_latency 
    
For `miss_latency = 10s`: average latency ≈ 3.3s → ~66% faster than baseline.
For `miss_latency = 16s`: average latency ≈ 5.3s → ~67% faster.

Summary: `HIT ≈ 0.01s`, `MISS ≈ 10–16s`, `Hit rate ≈ 0.67` → average latency reduced by ~50–70%, avoiding ~⅔ of LLM calls.
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

# Scalability & Eviction

To scale beyond this prototype, the NumPy-based search could be replaced with FAISS (IVF/HNSW) or a Redis Vector index to handle millions of cached embeddings efficiently.
For eviction, a hybrid strategy combining LRU, TTL, and recent-hit “heat” scores could balance recency and importance — for instance, using shorter TTLs for one-off queries and longer TTLs for frequently reused FAQs.
To ensure freshness, each cache entry could store the source or data version and be revalidated before reuse.

# Optional / Future work

- **Limitations of simple context**: The current context strategy works well for short conversations but struggles when topics drift or when the task has multiple subgoals.
It also lacks awareness of freshness or versioning (e.g., when data sources change).
Possible mitigations include segmenting long conversations, summarizing each segment, defining explicit state schemas, and using metadata or TTL checks to ensure validity.
- **Agent caching proposal**: In future work, the cache could also store intermediate agent steps such as tool inputs/outputs or subgoal summaries, not just final answers.
A possible cache key could combine the user’s goal, subgoal, and the tool name, while eviction could depend on recent usage (heat) or a fixed TTL. When datasets or code versions change, cached items could be invalidated automatically.
- **Embedding dimension trade-off**: Another potential extension would be testing lower-dimension embeddings (e.g., 768 → 384 → 256 → 128) using random projection or PCA, and comparing Recall@k and memory usage.

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
