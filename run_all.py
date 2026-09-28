"""Run both extractors over every PDF in sample_invoice_pdf/, then compare them."""

from __future__ import annotations

import logging

from compare import compare_all
from extract_plain import extract_plain
from extract_structured import extract_structured
from invoice_llm.io import PDF_DIR

logging.getLogger("pymupdf").setLevel(logging.ERROR)

if __name__ == "__main__":
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        print(f"{pdf.name}")
        print(f"  plain_text         -> {extract_plain(pdf)}")
        print(f"  document_structure -> {extract_structured(pdf)}")
    for r in compare_all():
        print(f"{r['source_pdf']}: {r['fields_same']}/{r['fields_total']} fields agree; "
              f"differs: {r['fields_different'] or 'none'}")
