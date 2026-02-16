import os
import re
import logging
from pathlib import Path

logger = logging.getLogger("meowllm.text_extractor")

# ── Safety caps ──────────────────────────────────────────────────
_MAX_TEXT_SIZE = 50 * 1024 * 1024     # 50 MB — plaintext, HTML, YAML, XML
_MAX_JSON_SIZE = 100 * 1024 * 1024    # 100 MB
_MAX_CSV_SIZE = 200 * 1024 * 1024     # 200 MB — CSV/TSV
_MAX_EXCEL_SIZE = 100 * 1024 * 1024   # 100 MB
_MAX_IMAGE_SIZE = 50 * 1024 * 1024    # 50 MB
_MAX_PARQUET_SIZE = 500 * 1024 * 1024 # 500 MB
_MAX_XML_DEPTH = 100                  # Recursion depth limit for XML
_MAX_ROWS = 50_000                    # Row cap for tabular data
_OCR_TIMEOUT = 60                     # Seconds — Tesseract timeout

# Compiled regex for collapsing blank lines (used by HTML extractor)
_BLANK_LINE_RE = re.compile(r"\n{3,}")

# Image extensions for OCR routing
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}


def _check_file_size(file_path: Path, max_bytes: int, label: str) -> None:
    """Gate: raise ValueError if file exceeds max_bytes."""
    try:
        size = file_path.stat().st_size
    except OSError as e:
        raise ValueError(f"Cannot read file: {e}")
    if size > max_bytes:
        max_mb = max_bytes // (1024 * 1024)
        actual_mb = round(size / (1024 * 1024), 1)
        raise ValueError(
            f"{label} file too large ({actual_mb} MB > {max_mb} MB limit). "
            f"Split into smaller files or increase MEOWLLM_MAX_UPLOAD_MB."
        )
    if size == 0:
        raise ValueError(f"{label} file is empty (0 bytes). Please upload a file with content.")


def extract_text(file_path: Path, file_type: str) -> dict:
    """Extract text from various file types.

    Returns dict with keys: text (str), pages (int).
    Raises ValueError for unsupported types or extraction failures.
    """
    file_type = file_type.lower()

    if file_type in (".txt", ".md"):
        return _extract_plaintext(file_path)
    elif file_type == ".pdf":
        return _extract_pdf(file_path)
    elif file_type == ".docx":
        return _extract_docx(file_path)
    elif file_type == ".csv":
        return _extract_csv(file_path)
    elif file_type == ".tsv":
        return _extract_tsv(file_path)
    elif file_type in (".json", ".jsonl"):
        return _extract_json(file_path)
    elif file_type in (".xlsx", ".xls"):
        return _extract_excel(file_path)
    elif file_type in (".html", ".htm"):
        return _extract_html(file_path)
    elif file_type == ".xml":
        return _extract_xml(file_path)
    elif file_type in (".yaml", ".yml"):
        return _extract_yaml(file_path)
    elif file_type == ".parquet":
        return _extract_parquet(file_path)
    elif file_type in _IMAGE_EXTENSIONS:
        return _extract_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


# ── Plain text / Markdown ────────────────────────────────────────

def _extract_plaintext(file_path: Path) -> dict:
    _check_file_size(file_path, _MAX_TEXT_SIZE, "Text")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return {"text": text, "pages": 1}


# ── PDF ────────────────────────────────────────────────────────

def _extract_pdf(file_path: Path) -> dict:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ValueError(
            "PyMuPDF (fitz) not installed. Cannot extract PDF files. "
            "Install with: pip install PyMuPDF"
        )

    doc = None
    try:
        doc = fitz.open(str(file_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        return {"text": "\n\n".join(text_parts), "pages": len(text_parts)}
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")
    finally:
        if doc is not None:
            doc.close()


# ── DOCX ────────────────────────────────────────────────────────

def _extract_docx(file_path: Path) -> dict:
    try:
        from docx import Document
    except ImportError:
        raise ValueError("python-docx not installed. Cannot extract DOCX files.")

    try:
        doc = Document(str(file_path))
        text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
        return {"text": "\n\n".join(text_parts), "pages": 1}
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX: {e}")


# ── CSV ────────────────────────────────────────────────────────

def _extract_csv(file_path: Path) -> dict:
    _check_file_size(file_path, _MAX_CSV_SIZE, "CSV")
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed — falling back to raw text for CSV")
        return _extract_plaintext(file_path)

    try:
        df = pd.read_csv(file_path, nrows=_MAX_ROWS, on_bad_lines="skip")
        return _tabular_to_text(df)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")


# ── TSV ────────────────────────────────────────────────────────

def _extract_tsv(file_path: Path) -> dict:
    _check_file_size(file_path, _MAX_CSV_SIZE, "TSV")
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed — falling back to raw text for TSV")
        return _extract_plaintext(file_path)

    try:
        df = pd.read_csv(file_path, sep="\t", nrows=_MAX_ROWS, on_bad_lines="skip")
        return _tabular_to_text(df)
    except Exception as e:
        raise ValueError(f"Failed to parse TSV: {e}")


def _tabular_to_text(df) -> dict:
    """Convert a pandas DataFrame to training-ready text."""
    import pandas as pd
    text_parts = []
    for _, row in df.iterrows():
        row_text = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
        if row_text.strip():
            text_parts.append(row_text)
    return {"text": "\n".join(text_parts), "pages": 1}


# ── JSON ────────────────────────────────────────────────────────

def _extract_json(file_path: Path) -> dict:
    _check_file_size(file_path, _MAX_JSON_SIZE, "JSON")
    import json

    # Handle JSONL (line-delimited JSON)
    if file_path.suffix.lower() == ".jsonl":
        text_parts = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text_parts.append(json.dumps(item, ensure_ascii=False))
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSONL line %d in %s", i + 1, file_path.name)
        if not text_parts:
            raise ValueError("JSONL file contains no valid JSON lines.")
        return {"text": "\n".join(text_parts), "pages": 1}

    # Standard JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if isinstance(data, list):
        text_parts = [json.dumps(item, ensure_ascii=False) for item in data]
        return {"text": "\n".join(text_parts), "pages": 1}
    elif isinstance(data, dict):
        return {"text": json.dumps(data, indent=2, ensure_ascii=False), "pages": 1}
    else:
        return {"text": str(data), "pages": 1}


# ── Excel (.xlsx / .xls) ─────────────────────────────────────

def _extract_excel(file_path: Path) -> dict:
    _check_file_size(file_path, _MAX_EXCEL_SIZE, "Excel")
    try:
        import pandas as pd
    except ImportError:
        raise ValueError("pandas/openpyxl not installed. Cannot extract Excel files.")

    xls = None
    try:
        xls = pd.ExcelFile(file_path)
        all_text_parts = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, nrows=_MAX_ROWS)
            if df.empty:
                continue
            all_text_parts.append(f"--- Sheet: {sheet_name} ---")
            for _, row in df.iterrows():
                row_text = " | ".join(
                    f"{col}: {val}" for col, val in row.items() if pd.notna(val)
                )
                if row_text.strip():
                    all_text_parts.append(row_text)
        return {"text": "\n".join(all_text_parts), "pages": len(xls.sheet_names)}
    except Exception as e:
        raise ValueError(f"Failed to parse Excel file: {e}")
    finally:
        if xls is not None:
            xls.close()


# ── HTML ────────────────────────────────────────────────────────

def _extract_html(file_path: Path) -> dict:
    _check_file_size(file_path, _MAX_TEXT_SIZE, "HTML")
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4/lxml not installed — using raw text extraction for HTML")
        return _extract_plaintext(file_path)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")
    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    text = _BLANK_LINE_RE.sub("\n\n", text)
    return {"text": text, "pages": 1}


# ── XML (safe — defusedxml) ───────────────────────────────────

def _extract_xml(file_path: Path) -> dict:
    _check_file_size(file_path, _MAX_TEXT_SIZE, "XML")
    try:
        import defusedxml.ElementTree as ET
    except ImportError:
        raise ValueError(
            "defusedxml not installed. Cannot safely parse XML files. "
            "Install with: pip install defusedxml"
        )

    try:
        tree = ET.parse(str(file_path))
    except Exception as e:
        raise ValueError(f"Failed to parse XML: {e}")

    root = tree.getroot()

    def _node_text(node, depth=0):
        if depth > _MAX_XML_DEPTH:
            return ["[XML tree too deep — truncated]"]
        parts = []
        tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag
        text = (node.text or "").strip()
        if text:
            parts.append(f"{tag}: {text}")
        for child in node:
            parts.extend(_node_text(child, depth + 1))
        tail = (node.tail or "").strip()
        if tail:
            parts.append(tail)
        return parts

    text_parts = _node_text(root)
    return {"text": "\n".join(text_parts), "pages": 1}


# ── YAML ────────────────────────────────────────────────────────

def _extract_yaml(file_path: Path) -> dict:
    _check_file_size(file_path, _MAX_TEXT_SIZE, "YAML")
    try:
        import yaml
    except ImportError:
        raise ValueError("PyYAML not installed. Cannot extract YAML files.")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML: {e}")

    if data is None:
        raise ValueError("YAML file is empty or contains only comments.")

    import json
    text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    return {"text": text, "pages": 1}


# ── Parquet ────────────────────────────────────────────────────────

def _extract_parquet(file_path: Path) -> dict:
    _check_file_size(file_path, _MAX_PARQUET_SIZE, "Parquet")
    try:
        import pandas as pd
    except ImportError:
        raise ValueError("pandas/pyarrow not installed. Cannot extract Parquet files.")

    try:
        df = pd.read_parquet(file_path)
        if len(df) > _MAX_ROWS:
            logger.info("Parquet file has %d rows — truncating to %d", len(df), _MAX_ROWS)
            df = df.head(_MAX_ROWS)
        return _tabular_to_text(df)
    except Exception as e:
        raise ValueError(f"Failed to parse Parquet file: {e}")


# ── Image (OCR) ────────────────────────────────────────────────

def _extract_image(file_path: Path) -> dict:
    _check_file_size(file_path, _MAX_IMAGE_SIZE, "Image")
    try:
        from PIL import Image
        import pytesseract
    except ImportError as e:
        missing = []
        try:
            import PIL  # noqa: F401
        except ImportError:
            missing.append("Pillow")
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            missing.append("pytesseract")
        raise ValueError(
            f"Missing dependencies for image OCR: {', '.join(missing or ['unknown'])}. "
            "Install with: pip install Pillow pytesseract  "
            "Also install system package: sudo apt install tesseract-ocr"
        )

    img = None
    try:
        img = Image.open(file_path)
        # Convert to RGB if necessary (handles RGBA, palette, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img, timeout=_OCR_TIMEOUT)
        if not text.strip():
            raise ValueError(
                "No text detected in image. Ensure the image contains readable text. "
                "For best results, use clear, high-contrast images with printed text."
            )
        return {"text": text, "pages": 1}
    except pytesseract.pytesseract.TesseractError as e:
        raise ValueError(f"OCR failed: {e}")
    except ValueError:
        raise  # Re-raise our own ValueErrors
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")
    finally:
        if img is not None:
            img.close()
