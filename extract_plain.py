"""Baseline: the repo's approach — PyMuPDF plain text layer straight to the LLM.

    python extract_plain.py            # every PDF in sample_invoice_pdf/
    python extract_plain.py some.pdf   # just one
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

from invoice_llm.io import PDF_DIR, write_result
from invoice_llm.llm import extract_invoice

METHOD = "plain_text"


def pdf_to_plain_text(path: Path) -> str:
    """Same as the repo's ``pdf_reader.read_pdf``: each page's ``get_text()`` joined by
    newlines, no layout information."""
    with pymupdf.open(str(path)) as doc:
        return "\n".join(page.get_text() for page in doc)


def extract_plain(path: Path) -> Path:
    text = pdf_to_plain_text(path)
    llm_input = f"Invoice text:\n\n{text}"  # same user message as the repo
    result = extract_invoice(llm_input)
    return write_result(METHOD, path, llm_input, result)


if __name__ == "__main__":
    pdfs = [Path(p) for p in sys.argv[1:]] or sorted(PDF_DIR.glob("*.pdf"))
    for pdf in pdfs:
        print(f"[{METHOD}] {pdf.name} -> {extract_plain(pdf)}")
