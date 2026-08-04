"""Task 9: hybrid retrieval with RRF and cosine-based fallback decision."""
from concurrent.futures import ThreadPoolExecutor
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank,rerank_rrf
from .task8_pageindex_vectorless import pageindex_search
SCORE_THRESHOLD=0.12; DEFAULT_TOP_K=5

def retrieve(query,top_k=DEFAULT_TOP_K,score_threshold=SCORE_THRESHOLD,use_reranking=True):
    if top_k<=0 or not query.strip(): return []
    with ThreadPoolExecutor(max_workers=2) as pool:
        d=pool.submit(semantic_search,query,top_k*2); s=pool.submit(lexical_search,query,top_k*2)
        dense,sparse=d.result(),s.result()
    best=dense[0]["score"] if dense else 0.0
    if best < score_threshold:
        fallback=pageindex_search(query,top_k)
        if fallback:return fallback
    dense_scores={item["content"]:item["score"] for item in dense}
    merged=rerank_rrf([dense,sparse],top_k*2)
    for item in merged:
        item["source"]="hybrid"
        item["cosine_score"]=float(dense_scores.get(item["content"],0.0))
    final=rerank(query,merged,top_k,"cross_encoder") if use_reranking else merged[:top_k]
    for item in final:item["source"]="hybrid"
    return final[:top_k]
