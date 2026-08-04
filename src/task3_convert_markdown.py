"""Task 3: convert landing-zone files to Markdown while preserving folders."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; LANDING_DIR=ROOT/"data"/"landing"; OUTPUT_DIR=ROOT/"data"/"standardized"

def _markitdown(path):
    try:
        from markitdown import MarkItDown
        return MarkItDown().convert(str(path)).text_content
    except Exception:
        try:
            from pypdf import PdfReader
            return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        except Exception:
            return ""

def convert_legal_docs():
    out=[]; target=OUTPUT_DIR/"legal"; target.mkdir(parents=True,exist_ok=True)
    for path in sorted((LANDING_DIR/"legal").glob("*")):
        if path.suffix.lower() not in {".pdf",".doc",".docx"}: continue
        destination=target/f"{path.stem}.md"
        text=_markitdown(path)
        if not text.strip():
            raise RuntimeError(f"Could not extract text from {path}")
        destination.write_text(
            f"# {path.stem.replace('-', ' ').replace('_', ' ')}\n\n"
            f"Source: {path.name}\nInstitution: VinUniversity\n\n{text}",
            encoding="utf-8",
        )
        if destination.exists(): out.append(destination)
    return out

def convert_news_articles():
    out=[]; target=OUTPUT_DIR/"news"; target.mkdir(parents=True,exist_ok=True)
    for path in sorted((LANDING_DIR/"news").glob("*.json")):
        data=json.loads(path.read_text(encoding="utf-8")); destination=target/f"{path.stem}.md"
        content=data.get("content_markdown") or data.get("content") or ""
        destination.write_text(f"# {data.get('title',path.stem)}\n\nSource: {data.get('url','unknown')}\nCrawled: {data.get('date_crawled','unknown')}\n\n{content}",encoding="utf-8"); out.append(destination)
    return out

def convert_all(): return convert_legal_docs()+convert_news_articles()
if __name__=="__main__": print(f"Converted/verified {len(convert_all())} files")
