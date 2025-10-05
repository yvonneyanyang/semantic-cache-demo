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

# mac/linux
cp .env.example .env
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
## Warm-start (embeddings only)

```bash
python semantic_cache_demo.py --scenario retail --warmstart all
```
## Recommended showcase (balanced)
```bash
python semantic_cache_demo.py --scenario retail --warmstart all --alpha=0.2 --threshold=0.40 --tag-thr=0.30 --topk=5
````
## What you’ll see

- `[HIT ...]` → served from cache (≈0.01s)

- `[MISS ... 3.xx s]` → one real LLM call + write-back

## Run other scenarios

```bash
python semantic_cache_demo.py --scenario agri --warmstart all --alpha=0.2 --threshold=0.40 --tag-thr=0.30 --topk=5
python semantic_cache_demo.py --scenario finance --warmstart all --alpha=0.2 --threshold=0.40 --tag-thr=0.30 --topk=5
````

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
- Avoided LLM calls = number of hits
- (Optional) Estimated cost saved = avoided_calls × per-call unit cost

# Sensitivity / Trade-offs

Try:

- `threshold`: `0.30 → 0.60 → 0.80`
- `tag-thr`: `0.0 → 0.3 → 0.6`
- `alpha`: `0.2–0.5`

Observation:

- Stricter thresholds → fewer hits, lower risk of wrong reuse
- Looser thresholds → more hits, higher risk; Jaccard gate helps control errors

# Scalability & Eviction (proposal)

- Index: swap NumPy search for FAISS (IVF/HNSW) or Redis Vector for million-scale.
- Eviction: LRU + TTL + heat (recent hits) hybrid score; short TTL for one-off queries, long TTL for FAQ.
- Freshness: store source/versions; check before reuse.

# Bonus
## 7.1 Limitations of simple context

- Topic shifts in long chats → semantic drift
- Task state not captured (goal/subgoal/progress)
- Data freshness/version not encoded

Mitigations: conversation segmentation with per-segment summaries; state schema; metadata/TTL checks.

## 7.2 Agent caching proposal

Cache tool I/O (inputs→outputs), subgoal summaries, and final answers.

Cache key = embedding(goal + subgoal + tool + doc_id) plus hashes (schema/version).

Evict by TTL/heat; invalidate along dependency graph when datasets/code versions change.

## 7.3 Embedding dimension trade-off

Use random projection/PCA to simulate `768 → 384 → 256 → 128`; report Recall@k & RAM.
(If included, see `scripts/dim_sweep.py`.)

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
