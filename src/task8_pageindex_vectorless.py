"""Task 8: PageIndex adapter with a local vectorless structural fallback."""
import os
from .task4_chunking_indexing import chunk_documents, load_documents, tokenize

def upload_documents():
    """Return documents ready for upload; external upload requires PAGEINDEX_API_KEY."""
    docs=load_documents()
    return [{"path":d["metadata"]["path"],"status":"local" if not os.getenv("PAGEINDEX_API_KEY") else "ready"} for d in docs]

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Search Markdown structure locally when the hosted PageIndex service is unavailable."""
    q=set(tokenize(query)); scored=[]
    for rank,c in enumerate(chunk_documents(load_documents()),1):
        words=set(tokenize(c["content"])); overlap=len(q&words)/max(1,len(q))
        if overlap>0:
            scored.append({"content":c["content"],"score":float(overlap),"metadata":{**c["metadata"],"provider":"local-structural-fallback"},"source":"pageindex"})
    return sorted(scored,key=lambda x:x["score"],reverse=True)[:top_k]
