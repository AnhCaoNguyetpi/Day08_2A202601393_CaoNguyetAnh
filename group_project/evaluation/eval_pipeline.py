"""Offline, reproducible RAG evaluation and A/B comparison.

Token-overlap proxies make CI deterministic; use evaluate_with_ragas when an
OpenAI-compatible judge is configured for production-grade LLM metrics.
"""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
GOLDEN_DATASET_PATH=Path(__file__).with_name("golden_dataset.json"); RESULTS_PATH=Path(__file__).with_name("results.md")
def load_golden_dataset(): return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
def _tokens(s): return set(re.findall(r"(?u)\b\w\w+\b",s.lower()))
def _ratio(a,b): return len(_tokens(a)&_tokens(b))/max(1,len(_tokens(b)))
def evaluate_config(dataset,use_reranking=True):
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import generate_with_citation
    rows=[]
    for item in dataset:
        sources=retrieve(item["question"],top_k=5,use_reranking=use_reranking)
        answer=generate_with_citation(item["question"],context_chunks=sources)["answer"]
        context=" ".join(s["content"] for s in sources); names=[s.get("metadata",{}).get("source","") for s in sources]
        rows.append({"question":item["question"],"answer":answer,"faithfulness":_ratio(answer,context),"answer_relevance":_ratio(answer,item["expected_answer"]),"context_recall":1.0 if item["expected_context"] in names else 0.0,"context_precision":sum(n==item["expected_context"] for n in names)/max(1,len(names))})
    metrics={k:sum(r[k] for r in rows)/len(rows) for k in ("faithfulness","answer_relevance","context_recall","context_precision")}
    return {"metrics":metrics,"rows":rows}
def compare_configs(rag_pipeline=None,golden_dataset=None):
    data=golden_dataset or load_golden_dataset(); return {"hybrid_rerank":evaluate_config(data,True),"hybrid_no_rerank":evaluate_config(data,False)}
def evaluate_with_ragas(rag_pipeline,golden_dataset):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness,answer_relevancy,context_recall,context_precision
    records=[]
    for item in golden_dataset:
        result=rag_pipeline(item["question"]); records.append({"question":item["question"],"answer":result["answer"],"contexts":[c["content"] for c in result["sources"]],"ground_truth":item["expected_answer"]})
    return evaluate(Dataset.from_list(records),metrics=[faithfulness,answer_relevancy,context_recall,context_precision])
def export_results(results,comparison):
    main=results.get("metrics",results); lines=["# RAG Evaluation Results","","## Overall scores","","| Metric | Score |","|---|---:|"]
    lines += [f"| {k.replace('_',' ').title()} | {v:.3f} |" for k,v in main.items()]
    lines += ["","## A/B comparison","","| Configuration | Faithfulness | Answer relevance | Context recall | Context precision |","|---|---:|---:|---:|---:|"]
    for name,result in comparison.items():
        m=result["metrics"]; lines.append(f"| {name} | {m['faithfulness']:.3f} | {m['answer_relevance']:.3f} | {m['context_recall']:.3f} | {m['context_precision']:.3f} |")
    rows=sorted(results.get("rows",[]),key=lambda r:r["answer_relevance"])[:3]
    lines += ["","## Worst performers"]+[f"- {r['question']} — relevance {r['answer_relevance']:.3f}" for r in rows]
    lines += ["","## Recommendations","","Improve Vietnamese/English multilingual embeddings, expand source coverage, and calibrate the cosine fallback threshold on labelled relevant and irrelevant queries."]
    RESULTS_PATH.write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__=="__main__":
    data=load_golden_dataset(); comparison=compare_configs(golden_dataset=data); export_results(comparison["hybrid_rerank"],comparison); print(f"Evaluated {len(data)} questions; wrote {RESULTS_PATH}")
