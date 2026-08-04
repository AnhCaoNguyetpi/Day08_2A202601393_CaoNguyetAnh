"""Interactive Streamlit chatbot for the VinUniversity policy corpus."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="VinUniversity Policy Assistant", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
[data-testid="stChatMessage"] {border: 1px solid rgba(120,120,120,.15); border-radius: 14px; padding: .4rem .8rem;}
.status-card {padding: .75rem 1rem; border-radius: 12px; background: rgba(46,160,67,.10); border: 1px solid rgba(46,160,67,.25);}
.source-card {padding: .7rem .9rem; margin-bottom: .6rem; border-radius: 10px; background: rgba(128,128,128,.07);}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "pending_query" not in st.session_state: st.session_state.pending_query = None

@st.cache_data(show_spinner=False)
def indexed_chunk_count() -> int:
    from src.task4_chunking_indexing import load_index
    return len(load_index().get("chunks", []))

def queue_suggestion(question: str) -> None:
    st.session_state.pending_query = question

def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_query = None

def unique_sources(sources: list[dict]) -> list[dict]:
    best = {}
    for source in sources:
        name = source.get("metadata", {}).get("source", "Không rõ nguồn")
        if name not in best or source.get("score", 0) > best[name].get("score", 0): best[name] = source
    return list(best.values())

def document_label(filename: str) -> str:
    labels = {
        "FW-SAM": "Khung cố vấn sinh viên",
        "POL-LLR": "Chính sách truy cập và dịch vụ thư viện",
        "VU_HT03": "Quy định học vụ chương trình đại học chính quy",
        "VU_CTSV02": "Bộ quy tắc ứng xử sinh viên",
    }
    return next((label for marker, label in labels.items() if marker in filename), filename)

def render_sources(sources: list[dict], answer: str = "") -> None:
    documents = unique_sources(sources)
    cited = [item for item in documents if document_label(item.get("metadata", {}).get("source", "")) in answer]
    if cited: documents = cited
    if not documents: return
    with st.expander(f"📚 Nguồn đã sử dụng ({len(documents)} tài liệu)"):
        for index, source in enumerate(documents, 1):
            metadata = source.get("metadata", {})
            filename = metadata.get("source", "Không rõ nguồn")
            name = document_label(filename)
            cosine = float(source.get("cosine_score", 0))
            year = metadata.get("year", "không rõ năm")
            section = metadata.get("section", "Không rõ mục")
            evidence = source.get("evidence_vi") or "Đã tìm thấy nội dung liên quan trong tài liệu gốc; không hiển thị đoạn tiếng Anh thô để giữ giao diện thống nhất tiếng Việt."
            score_label = f"Chỉ số tương đồng cosine: {cosine:.3f} · " if cosine else ""
            st.markdown(f"<div class='source-card'><b>{index}. {name} ({year})</b><br><b>{section}</b><br><small>{score_label}Tệp gốc: {filename}</small><br><br><b>Bằng chứng:</b> {evidence}</div>", unsafe_allow_html=True)
            excerpt = source.get("excerpt")
            if excerpt:
                st.caption("Đoạn truy hồi gốc (cụm liên quan được tô đậm):")
                st.markdown(f"> {excerpt}")

SUGGESTIONS = [
    "Mô hình cố vấn sinh viên tại VinUniversity gồm những vai trò nào?",
    "Sinh viên đại học được mượn bao nhiêu sách và trong bao lâu?",
    "Trách nhiệm của người được cố vấn là gì?",
    "Quy định học vụ áp dụng cho đối tượng nào?",
    "Bộ quy tắc ứng xử cấm những hành vi nào?",
]

with st.sidebar:
    st.title("🎓 Trợ lý Chính sách VinUni")
    st.caption("Tra cứu 4 tài liệu: cố vấn sinh viên, thư viện, quy định học vụ và quy tắc ứng xử.")
    from src.task10_generation import reasoning_model_available
    if reasoning_model_available():
        status="🟢 AI suy luận đang hoạt động"
        detail=f"RAG ngữ nghĩa · {indexed_chunk_count()} đoạn dữ liệu"
    else:
        status="🟠 Chưa kết nối mô hình AI"
        detail=f"Đang dùng chế độ dự phòng · {indexed_chunk_count()} đoạn dữ liệu"
    st.markdown(f"<div class='status-card'>{status}<br><small>{detail}</small></div>", unsafe_allow_html=True)
    st.divider(); st.subheader("💡 Câu hỏi gợi ý")
    for index, suggestion in enumerate(SUGGESTIONS):
        st.button(suggestion, key=f"suggestion_{index}", use_container_width=True, on_click=queue_suggestion, args=(suggestion,))
    st.divider()
    top_k = st.slider("Số đoạn bằng chứng", 3, 8, 5, help="Nhiều đoạn hơn tăng độ bao phủ nhưng có thể thêm nhiễu.")
    st.button("🗑️ Xóa cuộc trò chuyện", use_container_width=True, on_click=clear_chat)
    st.caption("Muốn suy luận ngữ nghĩa, hãy cấu hình OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong tệp .env.")

st.title("🎓 Trợ lý Chính sách VinUniversity")
st.caption("Đặt câu hỏi bằng tiếng Việt. Mọi câu trả lời đều đi kèm tài liệu nguồn.")
prompt = st.chat_input("Ví dụ: Sinh viên đại học được mượn sách trong bao lâu?")
query = prompt or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": query})
    try:
        from src.task10_generation import generate_with_citation
        with st.spinner("Đang tìm bằng chứng trong 4 tài liệu VinUniversity…"):
            response = generate_with_citation(query, top_k=top_k, conversation_history=history)
        st.session_state.messages.append({"role": "assistant", "content": response.get("answer") or "Hệ thống chưa tạo được câu trả lời.", "sources": response.get("sources", []),
                                          "intent_label": response.get("intent_label","Chưa xác định"), "confidence": response.get("confidence",0),
                                          "evidence_count": response.get("evidence_count",0), "error_type": response.get("error_type")})
    except Exception as exc:
        st.session_state.messages.append({"role": "assistant", "content": "Hệ thống đang gặp lỗi kỹ thuật khi xử lý yêu cầu. Vui lòng thử lại sau.", "sources": [],
                                          "intent_label":"Lỗi hệ thống", "confidence":0, "evidence_count":0, "error_type":"system_error"})
    st.rerun()

if not st.session_state.messages:
    st.info("Chọn một câu hỏi gợi ý ở thanh bên hoặc nhập câu hỏi của bạn ở ô phía dưới.")

for message in st.session_state.messages:
    avatar = "🧑‍🎓" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            confidence=message.get("confidence")
            evidence_count=message.get("evidence_count",len(message.get("sources",[])))
            meta=[f"Phân loại: {message.get('intent_label','Chưa xác định')}"]
            if confidence:meta.append(f"Độ tin cậy: {confidence}%")
            if evidence_count:meta.append(f"Dựa trên {evidence_count} đoạn tài liệu")
            st.caption(" · ".join(meta))
            render_sources(message.get("sources", []), message.get("content", ""))
