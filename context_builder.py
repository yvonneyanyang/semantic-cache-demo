#!/usr/bin/env python
# coding: utf-8

# In[1]:


# context_builder.py
# Builds a context-aware text block from recent turns + the current user query.

from typing import List, Dict

def build_context_text(history: List[Dict[str, str]], user_query: str) -> str:
    """
    Format the last few turns + current user query into a single text block.
    Using role tags helps the model and embeddings keep roles distinct.
    """
    lines = []
    for turn in history:
        lines.append(f"[{turn['role'].upper()}] {turn['text']}")
    lines.append(f"[USER] {user_query}")
    return "\n".join(lines)


# In[ ]:




