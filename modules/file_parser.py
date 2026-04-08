"""
File parser — extracts text from non-image formats (Word, Excel, digital PDF).
"""

from pathlib import Path

import PyPDF2
from docx import Document as DocxDocument
from openpyxl import load_workbook


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a digital (non-scanned) PDF.

    Returns list of dicts with 'page' and 'text' keys.
    Returns empty text for pages that have no extractable text
    (indicating they may need OCR instead).
    """
    results = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            results.append({"page": i, "text": text.strip()})
    return results


def extract_text_from_docx(docx_path: str) -> str:
    """Extract all text from a Word (.docx) file."""
    doc = DocxDocument(docx_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_xlsx(xlsx_path: str) -> str:
    """Extract text from all sheets in an Excel (.xlsx) file."""
    wb = load_workbook(xlsx_path, data_only=True)
    lines = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        lines.append(f"=== Sheet: {sheet_name} ===")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            line = " | ".join(cells).strip()
            if line and line != "|".join([""] * len(cells)):
                lines.append(line)

    return "\n".join(lines)


def is_scanned_pdf(pdf_path: str, threshold: float = 0.5) -> bool:
    """
    Heuristic: if more than `threshold` fraction of pages have
    very little text (< 50 chars), consider it a scanned PDF.
    """
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        return True
    empty_count = sum(1 for p in pages if len(p["text"]) < 50)
    return (empty_count / len(pages)) >= threshold


def extract_text(file_path: str) -> dict:
    """
    Auto-detect file type and extract text.

    Returns:
        {
            "source": filename,
            "type": "pdf" | "docx" | "xlsx" | "image",
            "pages": [{"page": int, "text": str}]  # for PDF
            or
            "text": str  # for docx/xlsx
            "needs_ocr": bool  # for scanned PDFs
        }
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    result = {"source": path.name, "type": ext.lstrip(".")}

    if ext == ".pdf":
        pages = extract_text_from_pdf(file_path)
        needs_ocr = is_scanned_pdf(file_path)
        result["pages"] = pages
        result["needs_ocr"] = needs_ocr

    elif ext == ".docx":
        result["text"] = extract_text_from_docx(file_path)
        result["needs_ocr"] = False

    elif ext == ".xlsx":
        result["text"] = extract_text_from_xlsx(file_path)
        result["needs_ocr"] = False

    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"):
        result["type"] = "image"
        result["text"] = ""
        result["needs_ocr"] = True

    else:
        result["text"] = ""
        result["needs_ocr"] = False

    return result
