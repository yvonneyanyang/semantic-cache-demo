#!/usr/bin/env python
# coding: utf-8

# In[1]:


# session_store.py
# Stores recent conversational turns per session to enable context-aware caching.

from collections import defaultdict
from typing import List, Dict

class SessionStore:
    def __init__(self):
        # Map: session_id -> list of {"role": "user"/"assistant", "text": "..."}
        self._hist = defaultdict(list)

    def add_turn(self, session_id: str, role: str, text: str) -> None:
        """Append a new turn to the session history."""
        self._hist[session_id].append({"role": role, "text": text})

    def history(self, session_id: str, k: int = 4) -> List[Dict[str, str]]:
        """Return the last k turns for this session (empty list if not found)."""
        return self._hist[session_id][-k:]

# In[ ]:




