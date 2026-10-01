"""Run all three extractors over every PDF in sample_invoice_pdf/, then both comparisons:

  sample_invoice_comparison/plain_vs_structure/      LLM plain text vs LLM document structure
  sample_invoice_comparison/structure_llm_vs_regex/  LLM document structure vs regex (no LLM)
"""

from __future__ import annotations

import logging

from compare import compare_all, print_results
from compare_regex_vs_llm import NAME as REGEX_VS_LLM, compare_regex_vs_llm
from extract_plain import extract_plain
from extract_rule_based import extract_rule_based
from extract_structured import extract_structured
from invoice_llm.io import PDF_DIR

logging.getLogger("pymupdf").setLevel(logging.ERROR)

if __name__ == "__main__":
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        print(f"{pdf.name}")
        print(f"  plain_text         -> {extract_plain(pdf)}")
        print(f"  document_structure -> {extract_structured(pdf)}")
        print(f"  rule_based         -> {extract_rule_based(pdf)}")
    print("\nPlain text vs document structure (LLM):")
    print_results(compare_all(), "plain_vs_structure")
    print("\nDocument structure: LLM vs regex:")
    print_results(compare_regex_vs_llm(), REGEX_VS_LLM)
