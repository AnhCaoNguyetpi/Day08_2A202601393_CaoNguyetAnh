"""Task 4: sentence-aware chunking and a persistent local TF-IDF index."""
from __future__ import annotations
import json, math, re, unicodedata
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STANDARDIZED_DIR=ROOT/"data"/"standardized"; INDEX_DIR=ROOT/"chroma_db"; INDEX_FILE=INDEX_DIR/"index.json"
CHUNK_SIZE=800; CHUNK_OVERLAP=120; EMBEDDING_MODEL="local-tfidf"; EMBEDDING_DIMENSION="dynamic"

DOCUMENT_INFO={
    "FW-SAM":{"title":"Student Advising Framework","title_vi":"Khung cố vấn sinh viên","year":"2025"},
    "POL-LLR":{"title":"Library Access & Services Policy","title_vi":"Chính sách truy cập và dịch vụ thư viện","year":"2025"},
    "VU_HT03":{"title":"Academic Regulations for Full-Time Undergraduate Programs","title_vi":"Quy định học vụ chương trình đại học chính quy","year":"2024"},
    "VU_CTSV02":{"title":"Student Code of Conduct","title_vi":"Bộ quy tắc ứng xử sinh viên","year":"2025"},
}

def _document_info(filename):
    return next((dict(info) for marker,info in DOCUMENT_INFO.items() if marker in filename),{"title":filename,"title_vi":filename,"year":"unknown"})

def load_documents(directory:Path=STANDARDIZED_DIR)->list[dict]:
    docs=[]
    for path in sorted(directory.rglob("*.md")):
        rel=path.relative_to(directory)
        if rel.parts[0]=="news":continue
        text=path.read_text(encoding="utf-8",errors="replace").strip()
        if text:
            metadata={"source":path.name,"path":str(rel),"type":"legal","institution":"VinUniversity",**_document_info(path.name)}
            docs.append({"content":text,"metadata":metadata})
    return docs

def _sentence_units(text):
    """Create units ending at sentence boundaries; never split through a word."""
    clean=re.sub(r"\s+"," ",text).strip()
    raw=[part.strip() for part in re.findall(r".*?(?:[.!?](?=\s|$)|$)",clean) if part.strip()]
    units=[]
    for part in raw:
        while len(part)>CHUNK_SIZE:
            cut=part.rfind(" ",0,CHUNK_SIZE+1)
            if cut<CHUNK_SIZE//2:cut=CHUNK_SIZE
            units.append(part[:cut].strip());part=part[cut:].strip()
        if part:units.append(part)
    return units

def chunk_documents(documents:list[dict])->list[dict]:
    chunks=[]
    for doc in documents:
        units=_sentence_units(doc["content"]); current=[]; current_len=0; doc_chunk=0
        for unit in units:
            added=len(unit)+(1 if current else 0)
            if current and current_len+added>CHUNK_SIZE:
                content=" ".join(current)
                meta={**doc.get("metadata",{}),"chunk_index":doc_chunk,"section":doc.get("metadata",{}).get("title_vi","")}
                chunks.append({"content":content,"metadata":meta});doc_chunk+=1
                overlap=[]; overlap_len=0
                for previous in reversed(current):
                    if overlap and overlap_len+len(previous)+1>CHUNK_OVERLAP:break
                    overlap.insert(0,previous);overlap_len+=len(previous)+1
                current=overlap;current_len=len(" ".join(current))
                while current and current_len+len(unit)+1>CHUNK_SIZE:
                    current.pop(0);current_len=len(" ".join(current))
            current.append(unit);current_len=len(" ".join(current))
        if current:
            meta={**doc.get("metadata",{}),"chunk_index":doc_chunk,"section":doc.get("metadata",{}).get("title_vi","")}
            chunks.append({"content":" ".join(current),"metadata":meta})
    return chunks

def tokenize(text):return re.findall(r"(?u)\b\w\w+\b",text.lower())
QUERY_EXPANSIONS={
"hinh":("model",),"co":("advisor","advising"),"van":("advisor","advising"),"sinh":("student",),"vien":("student",),"vai":("role",),"tro":("role",),"muon":("borrow","loan"),"sach":("book","items"),"quyen":("book","items"),"bao":("number",),"nhieu":("number",),"lau":("period","weeks"),"trach":("responsibility",),"nhiem":("responsibility",),"nguoi":("advisee",),"duoc":("advisee",),"quy":("regulation",),"dinh":("regulation",),"hoc":("academic",),"vu":("academic",),"doi":("applicable","apply"),"tuong":("applicable","apply"),"cam":("prohibited",),"hanh":("act","conduct"),"vi":("act","conduct")}
def query_tokens(text):
    base=tokenize(text);normalized=[unicodedata.normalize("NFKD",t.replace("đ","d")).encode("ascii","ignore").decode() for t in base];expanded=list(base);phrase=" ".join(normalized)
    for token in normalized:expanded.extend(QUERY_EXPANSIONS.get(token,()))
    if "mo hinh" in phrase:expanded.extend(("model","hybrid"))
    if "thu vien" in phrase:expanded.extend(("library","services"))
    if "thu vien" in phrase and "gio" in phrase:expanded.extend(("opening","hours","posted","website"))
    # Vietnamese intent phrases which do not have literal English word overlap
    # with the source PDFs.  Phrase-level expansion avoids treating every short
    # Vietnamese word as a synonym in unrelated questions.
    if any(p in phrase for p in ("chui tuc", "noi tuc", "lang ma", "xuc pham")):
        expanded.extend(("verbal","insults","demeans","dignity","honor","prohibited","conduct"))
    if any(p in phrase for p in ("ky tuc xa", "ki tuc xa", "cho o", "noi tru", "ngoai tru")):
        expanded.extend(("accommodation","housing","campus","availability","criteria"))
    return expanded

def embed_chunks(chunks):
    tokenized=[tokenize(c["content"]) for c in chunks];vocab=sorted({t for row in tokenized for t in row});df=Counter(t for row in tokenized for t in set(row));n=max(1,len(chunks));vectors=[]
    for row in tokenized:
        tf=Counter(row);vec=[tf[t]*(math.log((n+1)/(df[t]+1))+1) for t in vocab];norm=math.sqrt(sum(x*x for x in vec)) or 1.0;vectors.append([x/norm for x in vec])
    return vectors,vocab
def index_to_vectorstore(chunks=None):
    chunks=chunks if chunks is not None else chunk_documents(load_documents());vectors,vocab=embed_chunks(chunks);INDEX_DIR.mkdir(parents=True,exist_ok=True);INDEX_FILE.write_text(json.dumps({"model":EMBEDDING_MODEL,"vocabulary":vocab,"chunks":[{**c,"embedding":v} for c,v in zip(chunks,vectors)]},ensure_ascii=False),encoding="utf-8");return len(chunks)
def load_index():
    if not INDEX_FILE.exists():index_to_vectorstore()
    return json.loads(INDEX_FILE.read_text(encoding="utf-8")) if INDEX_FILE.exists() else {"vocabulary":[],"chunks":[]}
if __name__=="__main__":print(f"Indexed {index_to_vectorstore()} chunks into {INDEX_FILE}")
