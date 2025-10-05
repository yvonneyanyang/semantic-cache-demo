# topic_tagger.py
# Generate a short, stable topic tag for a query (+tiny context).

from typing import List, Dict
import google.generativeai as genai

TAG_PROMPT = """Generate a compact topic tag for retrieval.
Rules:
- 2~5 nouns, slash-separated (e.g., "climate change / corn yields").
- Use common vocabulary (e.g., 'corn' not 'maize').
- Lowercase, no verbs, no punctuation except slashes.
- No extra text, return ONLY the tag."""

def make_topic_tag(model: genai.GenerativeModel,
                   history: List[Dict[str, str]],
                   user_query: str) -> str:
    lines = []
    for turn in history[-2:]:
        lines.append(f"[{turn['role']}] {turn['text']}")
    lines.append(f"[user] {user_query}")
    text = "\n".join(lines)
    resp = model.generate_content(f"{TAG_PROMPT}\n\n{text}")
    return (resp.text or "").strip().lower()
