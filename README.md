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
````
## Create your `.env` from the example

```bash
# Windows
copy .env.example .env

# mac/linux
cp .env.example .env
````
Then edit `.env` and paste your key.

## Required environment variables
```ini
GOOGLE_API_KEY=PUT_YOUR_KEY_HERE
GEN_RPM=2
EMB_RPM=10
GEN_MODEL_NAME=models/gemini-2.5-flash
EMBEDDING_MODEL=text-embedding-004
# Optional: NO_LLM=0  # set 1 to avoid LLM calls when quota is tight
````
Security: `.env` is git-ignored; only `.env.example` is public.
