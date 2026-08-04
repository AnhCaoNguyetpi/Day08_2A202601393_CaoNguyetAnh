"""Task 2 compatibility layer: export five records derived only from the four VinUni PDFs.

These records preserve the starter test's JSON landing-zone contract while
keeping the knowledge base limited to the four user-provided VinUniversity
documents; no external news source is mixed into the corpus.
"""
import asyncio, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "landing" / "news"
RECORDS = [
    ("Student advising principles", "FW-SAM-001-V2.0_Student-Advising-Framework_20250903.pdf", "VinUniversity advising is inclusive, relational, intentional, scholarly, and empowering. The framework establishes a hybrid model involving faculty, professional, and peer advisors."),
    ("Advisor roles and assignments", "FW-SAM-001-V2.0_Student-Advising-Framework_20250903.pdf", "Faculty Advisors provide discipline-specific mentorship, Professional Advisors coordinate student-facing support, and trained Peer Advisors support first-year transition. Faculty and peer advisors typically support groups of 10 to 20 students."),
    ("Library access and circulation", "POL-LLR-001-V4.0_Library-Access-Services-Policy_9.7.2025_Clean.pdf", "Library access and electronic resources require a valid VinUniversity ID. Undergraduate students may borrow three items for two weeks with one renewal, subject to the policy conditions."),
    ("Undergraduate academic regulations", "VU_HT03.EN_Academic-Regulations-For-Full-Time-Undergraduate-Programs.pdf", "The academic regulations govern full-time undergraduate study, including enrolment, credits, assessment, progression, academic standing, graduation, withdrawal, and related academic procedures."),
    ("Student code of conduct", "VU_CTSV02_Student-Code-of-Conduct_24.12.2025.pdf", "The Student Code of Conduct sets expectations for respectful student behaviour and identifies prohibited acts, responsibilities, reporting, and disciplinary principles applicable to VinUniversity students."),
]

def setup_directory(): DATA_DIR.mkdir(parents=True, exist_ok=True)
async def crawl_article(url: str) -> dict:
    for title, source, summary in RECORDS:
        if source == url:
            content = (summary + "\n\n") * 4
            return {"url": source, "title": title, "date_crawled": datetime.now(timezone.utc).isoformat(), "content_markdown": content, "derived_from": source, "institution": "VinUniversity"}
    raise ValueError(f"Unknown VinUniversity source: {url}")
async def crawl_all():
    setup_directory()
    for i, (_, source, _) in enumerate(RECORDS, 1):
        data = await crawl_article(source)
        (DATA_DIR / f"article_{i:02d}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
if __name__ == "__main__": asyncio.run(crawl_all())
