# canonicalizer.py
from typing import List, Dict
import re
import google.generativeai as genai

CANON_PROMPT = """Rewrite the user's question into a single canonical FAQ-style question.
- Keep only the core intent.
- Use common vocabulary (e.g., 'corn' not 'maize').
- Remove fillers, hedges, politeness, and personal context.
- Be concise (<=15 words).
Return ONLY the rewritten question."""

def normalize_terms(s: str) -> str:
    t = (s or "").lower().strip()
    t = re.sub(r"(\d+)\s*%", r"\1 percent", t)     # 7% -> 7 percent
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
    lines = []
    for turn in history[-2:]:
        lines.append(f"[{turn['role']}] {turn['text']}")
    lines.append(f"[user] {user_query}")
    resp = model.generate_content(f"{CANON_PROMPT}\n\n" + "\n".join(lines))
    return normalize_terms(resp.text or "")
