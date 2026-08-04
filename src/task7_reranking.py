"""Task 7: deterministic reranking (RRF, MMR and lexical cross-encoder fallback)."""
import math
from collections import Counter
from .task4_chunking_indexing import query_tokens, tokenize

def rerank_cross_encoder(query, candidates, top_k=5):
    q=set(query_tokens(query)); out=[]
    for c in candidates:
        words=set(tokenize(c.get("content",""))); lexical=len(q&words)/max(1,len(q))
        item=dict(c); item["score"]=0.75*lexical+0.25*float(c.get("score",0)); out.append(item)
    return sorted(out,key=lambda x:x["score"],reverse=True)[:top_k]

def _cos(a,b):
    den=math.sqrt(sum(x*x for x in a))*math.sqrt(sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b))/den if den else 0.0

def rerank_mmr(query_embedding, candidates, top_k=5, lambda_param=0.7):
    selected=[]; remaining=list(range(len(candidates)))
    while remaining and len(selected)<top_k:
        def score(i):
            emb=candidates[i].get("embedding",[]); rel=_cos(query_embedding,emb)
            diversity=max((_cos(emb,candidates[j].get("embedding",[])) for j in selected),default=0)
            return lambda_param*rel-(1-lambda_param)*diversity
        best=max(remaining,key=score); item=dict(candidates[best]); item["score"]=score(best)
        selected.append(best); remaining.remove(best)
    return [({**candidates[i], "score": _cos(query_embedding,candidates[i].get("embedding",[]))}) for i in selected]

def rerank_rrf(ranked_lists, top_k=5, k=60):
    scores={}; items={}
    for ranking in ranked_lists:
        for rank,item in enumerate(ranking,1):
            key=(item.get("metadata",{}).get("source"),item.get("metadata",{}).get("start"),item.get("content",""))
            scores[key]=scores.get(key,0)+1/(k+rank); items[key]=item
    out=[]
    for key,score in sorted(scores.items(),key=lambda p:p[1],reverse=True)[:top_k]:
        out.append({**items[key],"score":float(score)})
    return out

def rerank(query, candidates, top_k=5, method="rrf"):
    if not candidates:return []
    if method=="cross_encoder": return rerank_cross_encoder(query,candidates,top_k)
    if method=="rrf": return rerank_cross_encoder(query,candidates,top_k)
    raise ValueError(f"Unsupported rerank method: {method}")
