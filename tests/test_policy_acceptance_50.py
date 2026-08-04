"""Acceptance tests for the 50 official VinUniversity policy questions."""
from src.policy_case_reasoning import CASES, find_policy_case
from src.task10_generation import NOT_UPDATED, generate_with_citation


def test_catalog_contains_all_50_acceptance_questions():
    assert len(CASES) == 50


def test_all_50_questions_have_grounded_answers(monkeypatch):
    # Acceptance behavior must remain available even when the remote LLM is down.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for number, (question, _, _) in enumerate(CASES, 1):
        match = find_policy_case(question)
        assert match and match["question"] == question, f"Câu {number} nhận diện sai ý định"
        result = generate_with_citation(question)
        assert result["answer"] != NOT_UPDATED, f"Câu {number} chưa có câu trả lời"
        assert result["sources"], f"Câu {number} chưa có nguồn"
        assert "[" in result["answer"] and "]" in result["answer"], f"Câu {number} chưa có trích dẫn"


def test_natural_paraphrases_are_understood():
    paraphrases = {
        "Một kỳ chính phải học ít nhất mấy credit?": 2,
        "Thứ bảy chủ nhật vào thư viện được chứ?": 17,
        "Tôi làm thất lạc sách đang mượn thì sao?": 48,
        "Ai giúp tôi về hồ sơ trao đổi?": 32,
        "Nhờ bạn vào phòng thi làm bài thay có sao không?": 38,
        "Tôi bệnh nên muốn tạm dừng một kỳ rồi quay lại": 49,
        "Mang cà phê có nắp vào library được không?": 23,
    }
    for question, expected_number in paraphrases.items():
        match = find_policy_case(question)
        assert match, question
        assert match["question"] == CASES[expected_number - 1][0], question


def test_out_of_scope_questions_are_not_forced_into_policy_cases():
    assert find_policy_case("Thời tiết hôm nay bao nhiêu độ?") is None
    assert find_policy_case("Nhà ăn hôm nay có món gì?") is None

