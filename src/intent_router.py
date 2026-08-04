"""Deterministic safety and UX router that runs before retrieval."""
import re, unicodedata
from .policy_case_reasoning import CASES, find_policy_case

INTENT_LABELS={
    "policy_question":"Câu hỏi chính sách",
    "academic_regulation":"Quy chế học tập",
    "library":"Thư viện",
    "advising":"Cố vấn học tập",
    "student_conduct":"Quy tắc ứng xử",
    "prompt_injection":"Prompt Injection",
    "system_information_request":"Yêu cầu thông tin hệ thống",
    "greeting":"Chào hỏi",
    "small_talk":"Trò chuyện thông thường",
    "ambiguous_question":"Cần làm rõ",
    "out_of_scope":"Ngoài phạm vi",
    "unknown":"Chưa xác định",
}

def _plain(text):
    return unicodedata.normalize("NFKD",text.lower().replace("đ","d")).encode("ascii","ignore").decode()

def _case_number(case):
    if not case:return None
    return next((i for i,item in enumerate(CASES,1) if item[0]==case["question"]),None)

def route_intent(query):
    q=_plain(query).strip(); words=re.findall(r"\b\w+\b",q)
    injection_patterns=(
        r"\b(xem|hien thi|liet ke|xuat|tai|truy cap|doc|mo|xoa|show|display|list|dump|reveal|access)\b.*\b(vector|embedding|database|co so du lieu|system prompt|prompt he thong|api key|khoa api|ma nguon|noi bo)\b",
        r"\b(ignore|bo qua|quen)\b.*\b(instruction|instructions|chi dan|quy tac|system)\b",
        r"\b(tiet lo|hien thi|dua toi|reveal|show)\b.*\b(prompt|secret|secrets|bi mat|api key|embedding|embeddings|vector)\b",
        r"\b(developer mode|jailbreak|dan|do anything now)\b",
        r"\b(pretend|dong vai|gia vo|hay la|act as)\b.*\b(ai|assistant|system|developer|admin|khong bi gioi han|unrestricted)\b",
        r"\b(ignore|bo qua)\b.*\b(document|documents|tai lieu|knowledge base|nguon)\b",
    )
    if any(re.search(pattern,q) for pattern in injection_patterns):
        return {"intent":"prompt_injection","label":INTENT_LABELS["prompt_injection"],"confidence":.99,"case":None}

    if re.fullmatch(r"\s*(xin chao|chao|hello|hi|hey|good morning|good afternoon|cam on|thanks|thank you)[!. ]*",q):
        return {"intent":"greeting","label":INTENT_LABELS["greeting"],"confidence":.99,"case":None}

    if any(phrase in q for phrase in ("ban khoe khong","how are you","hom nay ban the nao","rat vui duoc gap ban","nice to meet you")):
        return {"intent":"small_talk","label":INTENT_LABELS["small_talk"],"confidence":.96,"case":None}

    system_terms=("mo hinh gi","model nao","kien truc he thong","he thong hoat dong the nao","file noi bo","duong dan file","environment variable","bien moi truong","internal log","log noi bo","developer message","thong diep developer")
    if any(term in q for term in system_terms):
        return {"intent":"system_information_request","label":INTENT_LABELS["system_information_request"],"confidence":.97,"case":None}

    vague=("gap van de", "co chuyen roi", "giup em voi", "khong biet lam sao", "em can giup", "tu van cho em", "co van de")
    if len(words)<=8 and any(phrase in q for phrase in vague):
        return {"intent":"ambiguous_question","label":INTENT_LABELS["ambiguous_question"],"confidence":.95,"case":None}

    case=find_policy_case(query)
    number=_case_number(case)
    # These topics belong to the policy domain, but the four documents do not
    # state the requested amounts.  Case 43 is the documented duty to pay fees.
    if any(term in q for term in ("hoc phi","muc hoc bong","so tien hoc bong","phi ky tuc xa")) and number!=43:
        return {"intent":"academic_regulation","label":INTENT_LABELS["academic_regulation"],"confidence":.9,"case":None}
    if number:
        intent="academic_regulation" if number<=15 or number in (46,47,49,50) else "library" if number<=25 or number==48 else "advising" if number<=35 else "student_conduct"
        return {"intent":intent,"label":INTENT_LABELS[intent],"confidence":min(.99,.70+.29*case["score"]),"case":case,"case_number":number}

    out_patterns=("thoi tiet","nha hang","quan an","nha an hom nay","bong da","the thao","gia vang","chung khoan","tin tuc hom nay","du lich","nau an","phim nao")
    if any(pattern in q for pattern in out_patterns):
        return {"intent":"out_of_scope","label":INTENT_LABELS["out_of_scope"],"confidence":.97,"case":None}

    domains={
        "library":("thu vien","muon sach","quyen sach","tai lieu","phong hoc nhom","qua han","laptop","borrow","book","loan","renew"),
        "academic_regulation":("hoc phan","tin chi","gpa","diem","tot nghiep","hoc ky","chuyen nganh","bao luu","phuc khao","song bang","hoc phi","hoc bong"),
        "advising":("co van","advisor","i-rise","i rise","tu van nghe nghiep","nghien cuu khoa hoc","trao doi quoc te"),
        "student_conduct":("ky luat","ung xu","gian lan","thi ho","logo","fanpage","xuc pham","quyen sinh vien","trach nhiem sinh vien","ky tuc xa"),
    }
    for intent,terms in domains.items():
        if any(term in q for term in terms):
            return {"intent":intent,"label":INTENT_LABELS[intent],"confidence":.75,"case":None}
    if any(term in q for term in ("chinh sach","quy dinh","tai lieu nao","ho tro noi dung gi","pham vi ho tro")):
        return {"intent":"policy_question","label":INTENT_LABELS["policy_question"],"confidence":.8,"case":None}
    if len(words)<=3 or not any(len(word)>=3 for word in words):
        return {"intent":"unknown","label":INTENT_LABELS["unknown"],"confidence":.55,"case":None}
    return {"intent":"out_of_scope","label":INTENT_LABELS["out_of_scope"],"confidence":.7,"case":None}
