"""Task 1: validate the four VinUniversity policy documents in the landing zone."""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "landing" / "legal"
DOCUMENTS = (
    "FW-SAM-001-V2.0_Student-Advising-Framework_20250903.pdf",
    "POL-LLR-001-V4.0_Library-Access-Services-Policy_9.7.2025_Clean.pdf",
    "VU_HT03.EN_Academic-Regulations-For-Full-Time-Undergraduate-Programs.pdf",
    "VU_CTSV02_Student-Code-of-Conduct_24.12.2025.pdf",
)

def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def collect_documents() -> list[Path]:
    """Return validated bundled documents; never fabricate or download substitutes."""
    setup_directory()
    paths = [DATA_DIR / name for name in DOCUMENTS]
    missing = [str(path) for path in paths if not path.exists() or path.stat().st_size <= 1024]
    if missing:
        raise FileNotFoundError("Missing/invalid VinUniversity PDFs: " + ", ".join(missing))
    return paths

if __name__ == "__main__":
    for document in collect_documents():
        print(f"Validated: {document.name} ({document.stat().st_size:,} bytes)")
