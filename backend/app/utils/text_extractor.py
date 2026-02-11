import os
from pathlib import Path

# Safety caps (bytes)
_MAX_JSON_SIZE = 100 * 1024 * 1024  # 100 MB
_MAX_CSV_SIZE = 200 * 1024 * 1024   # 200 MB


def extract_text(file_path: Path, file_type: str) -> dict:
    """Extract text from various file types."""
    file_type = file_type.lower()

    if file_type in (".txt", ".md"):
        return _extract_plaintext(file_path)
    elif file_type == ".pdf":
        return _extract_pdf(file_path)
    elif file_type == ".docx":
        return _extract_docx(file_path)
    elif file_type == ".csv":
        return _extract_csv(file_path)
    elif file_type == ".json":
        return _extract_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def _extract_plaintext(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return {"text": text, "pages": 1}


def _extract_pdf(file_path: Path) -> dict:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(file_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return {"text": "\n\n".join(text_parts), "pages": len(text_parts)}
    except ImportError:
        # Fallback: try reading as text
        return _extract_plaintext(file_path)


def _extract_docx(file_path: Path) -> dict:
    try:
        from docx import Document
        doc = Document(str(file_path))
        text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
        return {"text": "\n\n".join(text_parts), "pages": 1}
    except ImportError:
        raise ValueError("python-docx not installed. Cannot extract DOCX files.")


def _extract_csv(file_path: Path) -> dict:
    try:
        import pandas as pd
        # Safety cap: reject very large CSV files
        if file_path.stat().st_size > _MAX_CSV_SIZE:
            raise ValueError(f"CSV file too large (>{_MAX_CSV_SIZE // (1024*1024)}MB). Split into smaller files.")
        df = pd.read_csv(file_path, nrows=50000)
        # Convert each row to text
        text_parts = []
        for _, row in df.iterrows():
            row_text = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            text_parts.append(row_text)
        return {"text": "\n".join(text_parts), "pages": 1}
    except ImportError:
        return _extract_plaintext(file_path)


def _extract_json(file_path: Path) -> dict:
    import json
    # Safety cap: reject very large JSON files
    if file_path.stat().st_size > _MAX_JSON_SIZE:
        raise ValueError(f"JSON file too large (>{_MAX_JSON_SIZE // (1024*1024)}MB). Split into smaller files.")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        text_parts = [json.dumps(item, ensure_ascii=False) for item in data]
        return {"text": "\n".join(text_parts), "pages": 1}
    elif isinstance(data, dict):
        return {"text": json.dumps(data, indent=2, ensure_ascii=False), "pages": 1}
    else:
        return {"text": str(data), "pages": 1}
