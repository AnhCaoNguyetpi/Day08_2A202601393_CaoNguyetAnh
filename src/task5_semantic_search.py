"""Task 5: cosine semantic search over the persisted local TF-IDF vectors."""
from collections import Counter
import math
from functools import lru_cache
from .task4_chunking_indexing import load_index, query_tokens, tokenize

@lru_cache(maxsize=1)
def _cached_index():
    idx = load_index()
    chunks = idx.get("chunks", [])
    df = Counter(token for chunk in chunks for token in set(tokenize(chunk["content"])))
    return idx, df

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    if top_k <= 0 or not query.strip(): return []
    idx, df = _cached_index(); vocab = idx.get("vocabulary", []); chunks = idx.get("chunks", [])
    if not chunks: return []
    tf = Counter(query_tokens(query)); n = len(chunks)
    q = [tf[t] * (math.log((n+1)/(df[t]+1))+1) for t in vocab]
    norm = math.sqrt(sum(x*x for x in q)) or 1.0; q = [x/norm for x in q]
    out = [{"content": c["content"], "score": max(0.0, min(1.0, sum(a*b for a,b in zip(q,c["embedding"])))), "metadata": c.get("metadata", {})} for c in chunks]
    return sorted(out, key=lambda x: x["score"], reverse=True)[:top_k]
