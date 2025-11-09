#extract text from pdf bytes


# app/services/pdf.py
import io

# Primary: PyMuPDF (fast)
try:
    import fitz  # PyMuPDF
    _HAVE_FITZ = True
except Exception:
    _HAVE_FITZ = False

# Fallback: pdfminer.six (robust)
from pdfminer.high_level import extract_text as pdfminer_extract_text

def _extract_with_fitz(data: bytes) -> str:
    # PyMuPDF can segfault on malformed PDFs (native code).
    # We keep it wrapped to catch Python-side exceptions,
    # but a native crash will still kill the process.
    with fitz.open(stream=data, filetype="pdf") as doc:
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        return "\n".join(texts)

def _extract_with_pdfminer(data: bytes) -> str:
    return pdfminer_extract_text(io.BytesIO(data)) or ""

def extract_text(data: bytes) -> str:
    # Try fast path first; then fallback to robust parser
    if _HAVE_FITZ:
        try:
            txt = _extract_with_fitz(data)
            if txt and txt.strip():
                return txt
        except Exception:
            # Swallow Python exceptions and try pdfminer
            pass
    # Fallback path (handles lots of malformed PDFs)
    return _extract_with_pdfminer(data)
