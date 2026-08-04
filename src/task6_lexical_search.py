"""Task 6: dependency-free BM25 lexical retrieval."""
from collections import Counter
import math
from .task4_chunking_indexing import chunk_documents, load_documents, query_tokens, tokenize

CORPUS: list[dict] = []
def build_bm25_index(corpus: list[dict]):
    tokens = [tokenize(d["content"]) for d in corpus]; n = len(tokens)
    df = Counter(t for row in tokens for t in set(row))
    return {"tokens": tokens, "df": df, "avgdl": sum(map(len,tokens))/max(1,n), "n": n}

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    global CORPUS
    if top_k <= 0 or not query.strip(): return []
    if not CORPUS: CORPUS = chunk_documents(load_documents())
    if not CORPUS: return []
    index = build_bm25_index(CORPUS); q = query_tokens(query); k1,b = 1.5,0.75
    results=[]
    for doc,tokens in zip(CORPUS,index["tokens"]):
        tf=Counter(tokens); dl=len(tokens); score=0.0
        for term in q:
            idf=math.log(1+(index["n"]-index["df"][term]+0.5)/(index["df"][term]+0.5))
            f=tf[term]; score += idf*f*(k1+1)/(f+k1*(1-b+b*dl/max(index["avgdl"],1))) if f else 0
        results.append({"content":doc["content"],"score":float(score),"metadata":doc.get("metadata",{})})
    return sorted(results,key=lambda x:x["score"],reverse=True)[:top_k]
