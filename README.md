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
