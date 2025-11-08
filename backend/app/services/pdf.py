#extract text from pdf bytes


import fitz  # pymupdf

def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    for p in doc:
        out.append(p.get_text("text"))
    return "\n".join(out)
