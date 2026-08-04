# Bài Tập Nhóm — University Services RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về dịch vụ và chính sách đại học liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
[Vẽ diagram kiến trúc ở đây]
```

---

## Phân Công Công Việc

### Cơ cấu nhóm 3 người

| Thành viên | MSSV | Vai trò | Phạm vi phụ trách | File/đầu ra chính | Trạng thái |
| --- | --- | --- | --- | --- | --- |
| **Cao Nguyệt Ánh** | 2A202601393 | **Team Leader, AI & RAG Integration Engineer** | Task 7–10; RRF/reranking, fallback, pipeline hợp nhất, grounded generation, intent router, guardrail và tích hợp backend | `src/task7_reranking.py` → `src/task10_generation.py`, `src/intent_router.py`, `src/policy_case_reasoning.py` | Hoàn thành |
| **Bùi Trung Hiếu** | 2A202601281 | **Data & Retrieval Engineer** | Task 1–6; xác thực tài liệu, chuẩn hóa dữ liệu, chunk/index, semantic search, lexical search và kiểm tra metadata nguồn | `data/`, `src/task1_collect_legal_docs.py` → `src/task6_lexical_search.py` | Hoàn thành |
| **Trần Thị Vân Anh** | 2A202601411 | **Frontend, Evaluation & QA Engineer** | Streamlit UI; citation/confidence UX; golden dataset, evaluation, acceptance/guardrail/adversarial tests, báo cáo và tài liệu demo | `app.py`, `group_project/evaluation/`, `tests/`, `PRODUCTION_RAG_DESIGN.md` | Hoàn thành |

### Phân công theo checkpoint

| Checkpoint | Cao Nguyệt Ánh | Bùi Trung Hiếu | Trần Thị Vân Anh | Tiêu chí chung |
| --- | --- | --- | --- | --- |
| **CP0 – Môi trường** | Quản lý repo, `.env.example`, thống nhất cấu hình | Tạo `.venv`, cài dependencies, kiểm tra module dữ liệu | Kiểm tra Streamlit và pytest | Cả ba chạy được project |
| **CP1 – Dữ liệu** | Kiểm tra phạm vi 4 tài liệu và quy tắc citation | Thu thập, xác thực PDF, convert Markdown | Kiểm tra tên file, metadata và khả năng truy vết | 4 PDF và Markdown hợp lệ |
| **CP2 – Index/Search** | Review tham số và response schema | Chunking, indexing, semantic và lexical search | Viết test truy hồi và kiểm tra dữ liệu nhiễu | Search trả đúng source |
| **CP3 – Reranking/Fallback** | Xây RRF, reranking và fallback | Cung cấp kết quả dense/sparse, kiểm tra cosine score | Test query khó và query ngoài phạm vi | Fallback kích hoạt đúng |
| **CP4 – Generation** | Nối Task 9–10, citation, intent và guardrail | Kiểm tra evidence/section của kết quả | Kiểm thử answer format và hallucination | Pipeline end-to-end đạt test |
| **CP5 – UI/Evaluation** | Tích hợp backend vào app và review bảo mật | Kiểm tra nguồn hiển thị khớp tài liệu | Hoàn thiện UI, golden dataset, evaluation và báo cáo | UI + evaluation chạy được |
| **CP6 – Demo/Nộp bài** | Trình bày kiến trúc và luồng AI Safety | Trình bày data, hybrid retrieval và reranking | Live demo UI, trình bày test/evaluation | Cả ba review và nộp repo |

### Cơ chế review

- Mỗi đầu ra phải được ít nhất một thành viên khác kiểm tra trước khi đánh dấu hoàn thành.
- Backend do Cao Nguyệt Ánh phụ trách được Bùi Trung Hiếu kiểm tra evidence và Trần Thị Vân Anh chạy regression test.
- Data/retrieval do Bùi Trung Hiếu phụ trách được Cao Nguyệt Ánh kiểm tra tích hợp và Trần Thị Vân Anh kiểm tra citation trên UI.
- UI/evaluation do Trần Thị Vân Anh phụ trách được Cao Nguyệt Ánh kiểm tra response schema và Bùi Trung Hiếu kiểm tra nguồn.

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
