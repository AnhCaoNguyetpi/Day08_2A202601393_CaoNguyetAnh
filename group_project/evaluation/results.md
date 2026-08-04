# RAG Evaluation Results

## Overall scores

| Metric | Score |
|---|---:|
| Faithfulness | 0.001 |
| Answer Relevance | 0.007 |
| Context Recall | 1.000 |
| Context Precision | 0.973 |

## A/B comparison

| Configuration | Faithfulness | Answer relevance | Context recall | Context precision |
|---|---:|---:|---:|---:|
| hybrid_rerank | 0.001 | 0.007 | 1.000 | 0.973 |
| hybrid_no_rerank | 0.001 | 0.007 | 1.000 | 0.933 |

## Worst performers
- What are the five principles of excellent advising at VinUniversity? — relevance 0.000
- Which three roles form the hybrid advising model? — relevance 0.000
- What is the typical Faculty Advisor advisee ratio? — relevance 0.000

## Recommendations

Improve Vietnamese/English multilingual embeddings, expand source coverage, and calibrate the cosine fallback threshold on labelled relevant and irrelevant queries.
