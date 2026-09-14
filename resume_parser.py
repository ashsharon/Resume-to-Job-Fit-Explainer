"""
resume_parser.py
-----------------
Extracts plain text from a resume file. Supports PDF (.pdf) and plain text (.txt).

PDF extraction uses pdfplumber, which handles most standard resume layouts
(single and two-column) far more reliably than raw PyPDF2 text extraction.
"""

import os
import pdfplumber


def parse_resume(file_path: str) -> str:
    """
    Returns the extracted plain text of a resume file.
    Raises ValueError for unsupported file types.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext == ".txt":
        return _parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")


def _parse_pdf(file_path: str) -> str:
    text_chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    text = "\n".join(text_chunks).strip()

    if not text:
        raise ValueError(
            "No extractable text found in PDF. It may be a scanned image - "
            "OCR is not supported in this version."
        )
    return text


def _parse_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(here, "data", "sample_resume.txt")
    text = parse_resume(sample_path)
    print(text[:500])
    print("...")
    print(f"\nTotal characters extracted: {len(text)}")
