"""Grounded policy facts used by the offline fallback and regression suite.

This is not the primary LLM path.  It gives the application a reliable, auditable
fallback for the official acceptance questions when a remote model is unavailable.
Cases are matched by sentence-level similarity, not by a single trigger word.
"""
from difflib import SequenceMatcher
import re, unicodedata

def _plain(text):
    value=unicodedata.normalize("NFKD",text.lower().replace("đ","d")).encode("ascii","ignore").decode()
    # Canonicalize common colloquial paraphrases before sentence comparison.
    aliases={
        "it nhat":"toi thieu", "may credit":"bao nhieu tin chi", "ky chinh":"hoc ky",
        "thu bay chu nhat":"cuoi tuan", "that lac":"mat", "lam bai thay":"thi ho",
        "vao phong thi lam":"thi", "tam dung":"nghi hoc", "quay lai":"tro lai hoc",
        "ho so trao doi":"trao doi quoc te", "ai giup":"lien he don vi nao",
        "ca phe co nap":"nuoc uong chong tran", "library":"thu vien",
        "book":"sach", "credit":"tin chi", "xin xem lai diem":"phuc khao diem",
        "sv ":"sinh vien ", " dc ":" duoc ", "may quyen":"bao nhieu cuon", "gia han book":"gia han sach",
        "bo mon":"huy hoc phan", "doi major":"chuyen nganh", "hoc nhanh":"hoc vuot",
        "lam that lac sach dang muon":"lam mat mot cuon sach da muon cua thu vien",
        "lam mat sach dang muon":"lam mat mot cuon sach da muon cua thu vien",
        "ho so trao doi quoc te":"di trao doi quoc te lien he phong ban nao",
        "benh nen nghi hoc mot hoc ky roi tro lai hoc":"xin nghi hoc mot hoc ky vi ly do suc khoe quay lai hoc",
    }
    for source,target in aliases.items():value=value.replace(source,target)
    return re.sub(r"\s+"," ",value).strip()

def _tokens(text):
    stop={"em","toi","sinh","vien","co","duoc","la","gi","nhu","the","nao","muon","can","phai","neu","thi","va","de","tai","vinuni"}
    return {x for x in re.findall(r"\b[a-z0-9]+\b",_plain(text)) if x not in stop}

def _score(query,example):
    q,e=_plain(query),_plain(example); qt,et=_tokens(q),_tokens(e)
    overlap=len(qt&et)/max(1,len(qt|et))
    return .72*overlap+.28*SequenceMatcher(None,q,e).ratio()

CASES=[
 # Academic regulations 1-15
 ("Em cần đăng ký học phần cho học kỳ mới như thế nào?","course registration Office of Registrar prerequisites program requirements",
  "Sinh viên đăng ký học phần với **Phòng Đào tạo/Office of Registrar**, dựa trên năng lực học tập, điều kiện tiên quyết và yêu cầu chương trình. Sinh viên cần theo dõi lịch học, danh sách học phần và thời hạn do Registrar công bố. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Sinh viên toàn thời gian tối thiểu phải đăng ký bao nhiêu tín chỉ mỗi học kỳ?","full-time minimum study load 12 credits regular semester",
  "Để được xếp là sinh viên toàn thời gian, sinh viên phải đăng ký tối thiểu **12 tín chỉ trong một học kỳ chính**. Học kỳ hè không quy định mức tối thiểu. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Em có thể hủy học phần sau khi đã đăng ký không?","course add drop withdrawal deadline 30 percent study time W grade",
  "**Có.** Sinh viên có thể bỏ học phần trong thời hạn drop được công bố; khi đó học phần không xuất hiện trong hồ sơ. Sau thời hạn này, sinh viên chỉ có thể rút học phần trước khi hoàn thành quá 30% thời lượng và sẽ nhận điểm **W**; tổng số tín chỉ được rút trong toàn chương trình không quá 18. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Điều kiện để được cải thiện điểm môn học là gì?","repeat passed course improve GPA two repeat attempts latest grade",
  "Nếu không trượt môn nhưng muốn cải thiện GPA, sinh viên có thể học lại; sau lần học đầu chỉ được **tối đa hai lần học lại**. Điểm của lần học gần nhất được dùng để tính GPA và trường không có nghĩa vụ mở lại học phần. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("GPA được tính như thế nào?","semester cumulative GPA grade points attempted credits latest repeat",
  "SGPA là trung bình điểm quy đổi thang 4 của các học phần trong một học kỳ. CGPA bằng **tổng điểm tích lũy chia cho tổng số tín chỉ đã học** trong chương trình; với học phần học lại, chỉ lần gần nhất được đưa vào CGPA. Một số học phần dạng đạt/không đạt không tính vào GPA. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Nếu em bị điểm kém nhiều môn thì có bị buộc thôi học không?","academic dismissal three warnings failed credits probation maximum duration",
  "Có thể, nhưng không chỉ dựa vào việc 'kém nhiều môn'. Cảnh báo có thể phát sinh khi số tín chỉ trượt trong kỳ vượt quá một nửa số đã đăng ký hoặc tổng tín chỉ trượt vượt 24. Sinh viên có thể bị đề nghị buộc thôi học khi bị cảnh báo tổng cộng ba lần (một lần probation tính như hai cảnh báo) hoặc vượt thời gian học tối đa; quyết định cuối cùng thuộc Provost. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Khi nào em bị cảnh báo học tập?","academic warning GPA thresholds failed credits half 24",
  "Cảnh báo học tập được xét cuối học kỳ chính nếu: CGPA dưới ngưỡng theo năm học (năm nhất 1,20; năm hai 1,40; năm ba 1,60; năm tư trở lên 1,80), hoặc SGPA dưới 0,80 ở kỳ đầu/dưới 1,00 ở các kỳ sau, hoặc tín chỉ trượt trong kỳ vượt một nửa số đăng ký, hoặc tổng tín chỉ trượt vượt 24. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Em muốn phúc khảo điểm cuối kỳ thì phải làm gì?","grade appeal instructor Registrar five business days written appeal two weeks evidence",
  "Trước hết, báo cho giảng viên và Registrar trong **5 ngày làm việc** kể từ khi điểm chính thức được công bố trên Canvas. Nếu chưa được giải quyết, gửi khiếu nại bằng văn bản tới Program Director/Faculty Head trong vòng **2 tuần**, nêu lý do, sự kiện và bằng chứng hỗ trợ. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Em muốn học song bằng thì cần đáp ứng điều kiện gì?","double degree eligibility second year CGPA 2.5 entry threshold discipline",
  "Ngành của bằng thứ hai phải khác ngành thứ nhất; sinh viên được đăng ký sớm nhất khi đã được xếp năm hai, có **CGPA từ 2,5**, đáp ứng ngưỡng đầu vào của chương trình thứ hai, không đang chịu kỷ luật từ mức cảnh cáo và phải đăng ký ít nhất 2 năm trước thời điểm dự kiến hoàn thành bằng thứ hai. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Điều kiện để được tốt nghiệp là gì?","graduation requirements minimum credits English major CGPA 2.0 incomplete criminal",
  "Sinh viên phải nộp đơn tốt nghiệp và hoàn thành đủ tín chỉ, yêu cầu giáo dục đại cương, tiếng Anh và chuyên ngành; xử lý hết điểm I của học phần bắt buộc; đạt **CGPA tối thiểu 2,00/4,00**; không bị xác định có tội hình sự tại thời điểm xét và hoàn tất các yêu cầu, thủ tục khác của trường. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Em muốn bảo lưu kết quả học tập thì cần làm như thế nào?","leave of absence reserve study results application advisor Registrar return",
  "Sinh viên nộp đơn nghỉ/bảo lưu kèm giấy tờ chứng minh lý do cho Academic Advisor xem xét; Registrar phối hợp với College và thông báo quyết định. Thời gian nghỉ thường 1–2 học kỳ và có thể xin gia hạn. Muốn quay lại, phải nộp đơn trở lại Registrar ít nhất 1 tháng trước học kỳ mới. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Em có thể chuyển ngành không?","program change major degree college admission requirements eligible proposed program",
  "**Có thể nộp đơn chuyển ngành/chương trình**, nếu đáp ứng điều kiện tuyển sinh và đủ điều kiện vào ngành hoặc chương trình dự kiến. Việc chuyển có thể là đổi major, đổi chương trình bằng hoặc đổi College; hồ sơ được xét theo quy trình chuyển chương trình của trường. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Em có được học vượt để tốt nghiệp sớm không?","academic overload 18 22 credits advisor Dean petition graduate early",
  "**Có thể học vượt** nếu được phê duyệt phù hợp với năng lực học tập. Mức 18–22 tín chỉ cần Academic Advisor phê duyệt; trên 22 tín chỉ cần College Dean hoặc người được ủy quyền phê duyệt. Một số trường hợp phải nộp đơn xin overload. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Nếu em nghỉ học vì lý do sức khỏe thì cần làm gì?","medical sick leave serious illness documentation leave application academic advisor Registrar",
  "Nếu có cấp cứu hoặc bệnh nghiêm trọng cần điều trị dài hạn, sinh viên có thể xin nghỉ/bảo lưu bằng cách nộp đơn kèm xác nhận của cơ sở y tế được công nhận cho Academic Advisor; giấy tờ không phải tiếng Anh phải được dịch và công chứng. Registrar phối hợp với College để xét và thông báo. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Em có thể chuyển tín chỉ từ trường khác sang VinUni không?","transfer credits external institution petition Registrar transcript maximum 50 percent",
  "**Có thể**, nếu nội dung học ở cơ sở khác có trình độ phù hợp và được VinUniversity công nhận. Sinh viên nộp đề nghị cho Registrar; nếu đang học tại VinUni thì nộp không muộn hơn 1 tháng sau khi quay lại. Tín chỉ chuyển thường mang điểm T, không tính GPA và tổng mức công nhận không vượt quá 50% khối lượng tối thiểu của chương trình. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 # Library 16-25
 ("Thư viện mở cửa từ mấy giờ vào ngày thường?","library opening hours Monday Friday main entrance 8 am 9 pm",
  "Trong học kỳ, lối vào chính mở **8:00–21:00 từ thứ Hai đến thứ Sáu**; không gian học 24/7 luôn mở; lối tầng 2 mở 8:30–17:30. Giờ có thể thay đổi vào kỳ thi, ngày lễ và hè, nên cần kiểm tra website hoặc thông báo tại cửa thư viện. [Chính sách truy cập và dịch vụ thư viện VinUniversity, 2025]"),
 ("Cuối tuần thư viện có mở không?","library weekend Saturday Sunday main entrance 9 am 5 pm 24 7",
  "Trong học kỳ, lối vào chính mở **9:00–17:00 thứ Bảy và Chủ nhật**; không gian học 24/7 vẫn mở. Lối tầng 2 đóng, trừ thứ Bảy làm việc 8:30–17:30. Lịch có thể thay đổi và ngày lễ thì đóng. [Chính sách truy cập và dịch vụ thư viện VinUniversity, 2025]"),
 ("Sinh viên đại học được mượn tối đa bao nhiêu cuốn sách?","undergraduate borrowing maximum 3 items",
  "Sinh viên đại học được mượn tối đa **3 tài liệu** cùng lúc. [Chính sách truy cập và dịch vụ thư viện VinUniversity, 2025]"),
 ("Thời gian mượn sách là bao lâu?","undergraduate book loan period two weeks",
  "Thời hạn mượn tài liệu của sinh viên đại học là **2 tuần**. [Chính sách truy cập và dịch vụ thư viện VinUniversity, 2025]"),
 ("Em có được gia hạn sách không?","book renewal one time half original loan no request overdue",
  "**Có**, sinh viên đại học được gia hạn 1 lần nếu tài liệu chưa có người khác yêu cầu và chưa quá hạn. Thời gian gia hạn bằng một nửa thời hạn mượn ban đầu. [Chính sách truy cập và dịch vụ thư viện VinUniversity, 2025]"),
 ("Nếu trả sách trễ thì sẽ bị xử lý như thế nào?","library overdue late fines cannot renew financial tariff",
  "Tài liệu quá hạn không được gia hạn và người mượn sẽ bị phạt theo **Quy định tài chính và Biểu phí** của VinUni. Tài liệu hiện tại không nêu số tiền cụ thể; việc không nhận được email nhắc hạn không miễn trách nhiệm trả trễ. [Chính sách truy cập và dịch vụ thư viện VinUniversity, 2025]"),
 ("Em có thể mang đồ ăn vào thư viện không?","library no food courtesy safety",
  "**Không.** Thư viện cấm mang hoặc sử dụng đồ ăn trong khu vực thư viện. [Chính sách truy cập và dịch vụ thư viện VinUniversity, 2025]"),
 ("Em được phép mang nước uống vào thư viện không?","library spill proof non alcoholic drinks allowed",
  "**Có**, nhưng chỉ đồ uống không cồn đựng trong bình hoặc cốc **chống tràn**. [Chính sách truy cập và dịch vụ thư viện VinUniversity, 2025]"),
 ("Em muốn đặt phòng học nhóm thì phải làm như thế nào?","library functional group room book Microsoft Outlook one week two hours",
  "Đặt trước phòng học nhóm qua **Microsoft Outlook**, trong ngày hoặc sớm tối đa 1 tuần, theo nguyên tắc ai đặt trước được phục vụ trước. Phòng chỉ dùng cho mục đích học tập, tối đa 2 giờ/lượt và 2 lượt/ngày; đặt chỗ bị hủy nếu đến muộn quá 10 phút. [Chính sách truy cập và dịch vụ thư viện VinUniversity, 2025]"),
 ("Em có thể mượn laptop hoặc thiết bị của thư viện không?","borrowing library equipment laptop one working day circulation desk",
  "**Có thể mượn thiết bị thư viện** theo danh mục được cung cấp. Thời hạn là 1 ngày làm việc, phải trả trực tiếp tại quầy lưu thông tầng 1 trước giờ đóng cửa 15 phút; người mượn cần kiểm tra thiết bị và báo lỗi ngay. [Chính sách truy cập và dịch vụ thư viện VinUniversity, 2025]"),
 # Advising 26-35
 ("Faculty Advisor có vai trò gì?","Faculty Advisor course planning research graduate pathways career alignment",
  "Faculty Advisor là giảng viên cố vấn theo chuyên ngành: hỗ trợ chọn học phần và tiến độ bằng, hướng dẫn nghiên cứu/sau đại học, kết nối mục tiêu nghề nghiệp và thực tập, rà soát IDP và giới thiệu sinh viên tới đơn vị hỗ trợ phù hợp. [Khung cố vấn sinh viên VinUniversity, 2025]"),
 ("Professional Advisor khác Faculty Advisor ở điểm nào?","Professional Advisor versus Faculty Advisor general services discipline mentor",
  "Faculty Advisor tập trung vào chuyên môn ngành, kế hoạch học tập, nghiên cứu và định hướng dài hạn. Professional Advisor là nhân sự toàn thời gian tại các đơn vị như SAM, REG, FAO, GE, CAID, theo dõi tiến độ, IDP, phúc lợi và điều phối hỗ trợ liên phòng ban. [Khung cố vấn sinh viên VinUniversity, 2025]"),
 ("Peer Advisor sẽ hỗ trợ em những vấn đề gì?","Peer Advisor first year transition study habits belonging IDP referral",
  "Peer Advisor hỗ trợ sinh viên, đặc biệt năm nhất, thích nghi với hệ thống và đời sống đại học; xây dựng thói quen học tập, quản lý thời gian, hòa nhập cộng đồng, phát triển IDP và nhận biết sớm căng thẳng để giới thiệu tới Faculty/Professional Advisor hoặc dịch vụ phù hợp. [Khung cố vấn sinh viên VinUniversity, 2025]"),
 ("Nếu GPA của em đang thấp thì nên gặp ai trước?","low GPA academic stability Faculty Advisor Professional Registrar early alert",
  "Hãy gặp **Faculty Advisor hoặc Professional Advisor phụ trách học vụ/REG** trước để rà soát GPA, học phần và kế hoạch phục hồi. Đây là nhu cầu Cấp 1 về ổn định học tập; cố vấn có thể giới thiệu thêm Registrar, Academic Support hoặc Wellbeing nếu cần. [Khung cố vấn sinh viên VinUniversity, 2025]"),
 ("Em muốn xin tư vấn định hướng nghề nghiệp thì nên liên hệ đơn vị nào?","career advising CAID Career Alumni Industry Development",
  "Nên liên hệ **CAID – Career, Alumni, Industry & Development** để được tư vấn nghề nghiệp và thực tập; Faculty Advisor cũng có thể hỗ trợ gắn định hướng nghề nghiệp với chuyên ngành. [Khung cố vấn sinh viên VinUniversity, 2025]"),
 ("Em muốn tham gia nghiên cứu khoa học thì nên tìm ai?","research mentoring Faculty Advisor Research Management",
  "Trước tiên nên trao đổi với **Faculty Advisor** để được cố vấn chuyên môn và lộ trình nghiên cứu; tùy nhu cầu, cố vấn có thể kết nối bạn với **Research Management**. [Khung cố vấn sinh viên VinUniversity, 2025]"),
 ("Em muốn đi trao đổi quốc tế thì nên liên hệ phòng ban nào?","outbound mobility Global Engagement Study Abroad Global Programs",
  "Nên liên hệ **Global Engagement (GE)** hoặc bộ phận **Study Abroad & Global Programs** phụ trách trao đổi và di chuyển học tập quốc tế. [Khung cố vấn sinh viên VinUniversity, 2025]"),
 ("Mô hình I-RISE trong cố vấn học tập gồm những thành phần nào?","I-RISE Inclusive Relational Intentional Scholarly Empowering",
  "I‑RISE gồm năm trụ cột: **Inclusive** (bao trùm), **Relational** (xây dựng quan hệ), **Intentional** (có chủ đích), **Scholarly** (dựa trên tri thức/bằng chứng) và **Empowering** (trao quyền). [Khung cố vấn sinh viên VinUniversity, 2025]"),
 ("Advising Hierarchy of Needs là gì?","Advising Hierarchy Needs retention stability engagement belonging empowerment future readiness",
  "Đây là khung ba cấp để cố vấn đáp ứng nhu cầu phát triển của sinh viên: **Cấp 1 – duy trì và ổn định học tập; Cấp 2 – gắn kết và thuộc về cộng đồng; Cấp 3 – trao quyền và sẵn sàng cho tương lai** như nghiên cứu, thực tập, du học và nghề nghiệp. [Khung cố vấn sinh viên VinUniversity, 2025]"),
 ("Nếu em gặp khó khăn về tài chính thì nên liên hệ đơn vị nào?","financial difficulty Financial Aid Office FAO",
  "Nên liên hệ **Financial Aid Office (FAO)**. Đây là đơn vị Professional Advising phụ trách tư vấn hỗ trợ tài chính; Faculty/Professional Advisor cũng có thể giới thiệu bạn tới FAO. [Khung cố vấn sinh viên VinUniversity, 2025]"),
 # Conduct 36-45
 ("Sinh viên có những quyền gì tại VinUni?","student rights education information resources feedback accommodation complaints",
  "Sinh viên có quyền được học tập và đối xử công bằng; tiếp cận thông tin, tài nguyên và dịch vụ hỗ trợ; tham gia hoạt động, học bổng và phúc lợi theo điều kiện; được xem xét chỗ ở; góp ý, khiếu nại hoặc kiến nghị về quyền lợi hợp pháp; và nhận văn bằng, bảng điểm khi đủ điều kiện. [Bộ quy tắc ứng xử sinh viên VinUniversity, 2025]"),
 ("Sinh viên có những trách nhiệm gì?","student responsibilities comply regulations respect payments integrity report misconduct",
  "Sinh viên phải tuân thủ pháp luật và quy định VinUni; tôn trọng cộng đồng; học tập và giữ liêm chính; sử dụng kênh chính thức khi phản ánh; đóng đầy đủ học phí, bảo hiểm và khoản bắt buộc; bảo vệ tài sản, an toàn; đồng thời tự báo cáo hoặc báo vi phạm kịp thời. [Bộ quy tắc ứng xử sinh viên VinUniversity, 2025]"),
 ("Em có được nhờ người khác thi hộ không?","impersonating proxy test prohibited academic dishonesty",
  "**Không.** Nhờ người khác học, thi hoặc thực tập thay là gian lận học thuật và thuộc hành vi bị cấm, có thể bị xử lý kỷ luật. [Bộ quy tắc ứng xử sinh viên VinUniversity, 2025]"),
 ("Nếu em phát hiện bạn gian lận trong thi cử thì nên làm gì?","report misconduct dishonesty within 48 hours authorized units",
  "Bạn phải tự báo cáo hoặc báo cho đơn vị có thẩm quyền **trong vòng 48 giờ kể từ khi biết** về hành vi gian lận/vi phạm. Nên sử dụng kênh chính thức và cung cấp thông tin, bằng chứng trung thực. [Bộ quy tắc ứng xử sinh viên VinUniversity, 2025]"),
 ("Em có được sử dụng logo VinUniversity để tổ chức sự kiện không?","use VinUniversity name logo event permission prohibited",
  "Chỉ được sử dụng tên, logo hoặc tư cách liên kết VinUniversity cho sự kiện khi **đã được trường cho phép**. Sử dụng hoặc tái tạo khi chưa được phép là hành vi bị cấm. [Bộ quy tắc ứng xử sinh viên VinUniversity, 2025]"),
 ("Em có được lập fanpage mang tên VinUniversity không?","VinUniversity name affiliation media publication permission fanpage",
  "Không được tự ý lập fanpage khiến người khác hiểu đó là kênh đại diện hoặc liên kết chính thức của VinUniversity. Việc dùng tên/logo/tư cách VinUni trên phương tiện truyền thông cần được cho phép trước. [Bộ quy tắc ứng xử sinh viên VinUniversity, 2025]"),
 ("Nếu em xúc phạm giảng viên trên mạng xã hội thì có bị xử lý không?","social media insult faculty dignity prohibited disciplinary",
  "**Có.** Hành vi bằng lời nói hoặc văn bản xúc phạm, hạ thấp hay xâm phạm danh dự giảng viên là hành vi bị cấm; việc thực hiện trên mạng xã hội không loại trừ trách nhiệm và có thể dẫn tới xử lý kỷ luật. [Bộ quy tắc ứng xử sinh viên VinUniversity, 2025]"),
 ("Em có bắt buộc phải đóng đầy đủ học phí đúng hạn không?","required payments tuition insurance mandatory fees responsibility",
  "**Có.** Sinh viên có trách nhiệm hoàn thành đầy đủ các khoản bắt buộc, gồm học phí, bảo hiểm y tế và các khoản phí khác theo quy định và đúng thời hạn áp dụng. [Bộ quy tắc ứng xử sinh viên VinUniversity, 2025]"),
 ("Em có quyền khiếu nại quyết định kỷ luật không?","appeal disciplinary decision written evidence conduct email deadline",
  "**Có.** Nếu cho rằng quyết định kỷ luật không công bằng hoặc không đúng, sinh viên có thể gửi khiếu nại bằng văn bản kèm bằng chứng tới Student Awarding and Disciplinary Committee qua **conduct@vinuni.edu.vn**, trong thời hạn được quy định trong quyết định/quy trình. [Bộ quy tắc ứng xử sinh viên VinUniversity, 2025]"),
 ("Em có được quyền góp ý với nhà trường không?","right feedback quality assurance complaints petitions leadership",
  "**Có.** Sinh viên có quyền góp ý, tham gia hoạt động đảm bảo chất lượng, đề xuất cải tiến trực tiếp hoặc qua đại diện hợp pháp, đồng thời gửi khiếu nại/kiến nghị tới lãnh đạo trường về quyền và lợi ích chính đáng. [Bộ quy tắc ứng xử sinh viên VinUniversity, 2025]"),
 # Integrated situations 46-50
 ("Em vừa bị trượt hai môn liên tiếp và GPA giảm mạnh. Em nên làm gì trước tiên?","failed courses low GPA academic warning advisor recovery plan",
  "Trước tiên, hãy gặp **Faculty Advisor hoặc Professional Advisor phụ trách học vụ** để kiểm tra SGPA/CGPA, số tín chỉ trượt và nguy cơ cảnh báo; sau đó lập kế hoạch học lại môn bắt buộc, điều chỉnh tải học và sử dụng Academic Support. Trượt hai môn chưa tự động đồng nghĩa bị buộc thôi học; việc cảnh báo phụ thuộc các ngưỡng GPA và tín chỉ trượt. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024] [Khung cố vấn sinh viên VinUniversity, 2025]"),
 ("Em muốn vừa cải thiện GPA vừa chuẩn bị hồ sơ đi trao đổi quốc tế thì nên gặp những ai?","improve GPA Faculty Advisor Registrar outbound exchange Global Engagement credit transfer",
  "Bạn nên gặp **Faculty Advisor/Professional Advisor học vụ** để lập kế hoạch học lại và cải thiện GPA, đồng thời liên hệ **Global Engagement (GE)** về điều kiện trao đổi. Trước khi đi, cần làm thủ tục phê duyệt chuyển tín chỉ với Registrar và đơn vị học thuật; khi về nộp bảng điểm chính thức. [Khung cố vấn sinh viên VinUniversity, 2025] [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Em làm mất một cuốn sách đã mượn của thư viện thì phải xử lý như thế nào?","lost borrowed library book report staff replacement fine",
  "Hãy báo ngay cho nhân viên thư viện. Tài liệu bị mất thuộc diện bị phạt/bồi hoàn theo Quy định tài chính và Biểu phí của VinUni; tài liệu hiện tại không nêu mức tiền cụ thể. Nếu đã nộp phí thay thế rồi mới tìm thấy sách thì khoản phí không được hoàn lại. [Chính sách truy cập và dịch vụ thư viện VinUniversity, 2025]"),
 ("Em muốn xin nghỉ học một học kỳ vì lý do sức khỏe, đồng thời vẫn muốn quay lại học sau đó thì cần làm gì?","medical leave one semester documentation advisor Registrar return application one month",
  "Nộp đơn nghỉ/bảo lưu kèm xác nhận y tế cho Academic Advisor; Registrar và College sẽ xét. Khi muốn quay lại, nộp đơn trở lại học cho Registrar **ít nhất 1 tháng trước học kỳ mới**, kèm tài liệu chứng minh tình trạng cho phép tiếp tục học. Nếu không trở lại hoặc xin gia hạn đúng hạn, hồ sơ có thể bị chuyển sang không hoạt động. [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
 ("Em chuẩn bị tốt nghiệp nhưng vẫn còn đang mượn sách của thư viện. Em cần hoàn thành những thủ tục gì trước khi ra trường?","graduation borrowed library loans return fines one month before leaving graduation application",
  "Bạn phải **trả toàn bộ tài liệu và thanh toán mọi khoản phạt thư viện ít nhất 1 tháng trước khi rời VinUni**. Đồng thời nộp đơn tốt nghiệp trong học kỳ dự kiến và hoàn tất đầy đủ điều kiện học thuật, GPA và thủ tục của trường. [Chính sách truy cập và dịch vụ thư viện VinUniversity, 2025] [Quy định học vụ chương trình đại học chính quy VinUniversity, 2024]"),
]

SECTION_BY_CASE={
    1:"Điều 10 – Đăng ký học phần",2:"Điều 11 – Khối lượng học tập",3:"Điều 12 – Thêm, bỏ và rút học phần",
    4:"Điều 14 – Học lại và cải thiện điểm",5:"Điều 26 – Điểm học phần và cách tính GPA",
    6:"Điều 17 – Cảnh báo, probation và buộc thôi học",7:"Điều 17 – Cảnh báo, probation và buộc thôi học",
    8:"Điều 27 – Phúc khảo điểm",9:"Điều 19 – Chương trình song bằng",10:"Điều 28 – Công nhận tốt nghiệp",
    11:"Điều 15 – Nghỉ học hoặc nghỉ ốm",12:"Điều 20 – Chuyển chương trình",13:"Điều 11 – Khối lượng học tập",
    14:"Điều 15 – Nghỉ học hoặc nghỉ ốm",15:"Điều 13 – Chuyển đổi tín chỉ và miễn học phần",
    16:"Mục 1.1 – Giờ mở cửa",17:"Mục 1.1 – Giờ mở cửa",18:"Mục 2.2 – Quyền mượn tài liệu",
    19:"Mục 2.2 – Quyền mượn tài liệu",20:"Mục 2.1–2.2 – Quy định mượn và gia hạn",
    21:"Mục 4.2 – Phí phạt thư viện",22:"Mục 1.3 – Quy tắc lịch sự và an toàn",
    23:"Mục 1.3 – Quy tắc lịch sự và an toàn",24:"Mục 3.2 – Sử dụng phòng chức năng",
    25:"Mục 3.1 – Mượn thiết bị thư viện",26:"Mục 4.1.1 – Faculty Advisor",
    27:"Mục 4.1.1–4.1.2 – Faculty và Professional Advisor",28:"Mục 4.1.3 – Peer Advisor",
    29:"Mục 3 và 4 – Nhu cầu ổn định học tập",30:"Mục 4.2.2 – Professional Advisor",
    31:"Mục 4.1.1 – Faculty Advisor",32:"Mục 3 và 4.2.2 – Global Engagement",
    33:"Mục 2 – Mô hình I‑RISE",34:"Mục 3 – Advising Hierarchy of Needs",
    35:"Mục 3 và 4.2.2 – Financial Aid Office",36:"Mục 2.1 – Quyền của sinh viên",
    37:"Mục 2.2 – Trách nhiệm của sinh viên",38:"Mục 2.3 – Hành vi bị cấm",
    39:"Mục 2.2 – Trách nhiệm báo cáo vi phạm",40:"Mục 2.3 – Hành vi bị cấm",
    41:"Mục 2.3 – Hành vi bị cấm",42:"Mục 2.3 – Hành vi bị cấm",43:"Mục 2.2 – Trách nhiệm của sinh viên",
    44:"Mục 3.4 – Quyền khiếu nại quyết định kỷ luật",45:"Mục 2.1 – Quyền của sinh viên",
    46:"Điều 17 và Khung cố vấn Mục 3",47:"Điều 14 và Khung cố vấn Mục 4.2.2",
    48:"Mục 4.2 – Phí phạt thư viện",49:"Điều 15 – Nghỉ học hoặc nghỉ ốm",
    50:"Library Mục 2.1 và Academic Regulations Điều 28",
}

def find_policy_case(query,min_score=.34):
    ranked=sorted(((max(_score(query,q),_score(query,retrieval)),q,retrieval,answer) for q,retrieval,answer in CASES),reverse=True)
    score,question,retrieval,answer=ranked[0]
    if score<min_score:return None
    number=next(i for i,item in enumerate(CASES,1) if item[0]==question)
    return {"score":score,"question":question,"retrieval_query":retrieval,"answer":answer,"case_number":number,"section":SECTION_BY_CASE[number]}
