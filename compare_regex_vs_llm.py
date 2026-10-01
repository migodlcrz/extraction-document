"""Compare the LLM with document structure against regex rules over the same document
structure (no LLM) — i.e. "can we skip the LLM for a known invoice layout?"

    python compare_regex_vs_llm.py

Needs sample_invoice_json/document_structure/ and sample_invoice_json/rule_based/ to exist
(run extract_structured.py and extract_rule_based.py, or run_all.py). Writes to
sample_invoice_comparison/structure_llm_vs_regex/. The per-invoice .md also lists the
rule-based checks (line items sum, total + VAT = due, remittance slip matches).
"""

from __future__ import annotations

from compare import RULE, STRUCT, compare_methods, print_results

NAME = "structure_llm_vs_regex"


def compare_regex_vs_llm() -> list[dict]:
    return compare_methods(STRUCT, RULE, NAME)


if __name__ == "__main__":
    print_results(compare_regex_vs_llm(), NAME)
