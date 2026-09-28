"""Shared helpers for writing each run's output JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from invoice_llm.llm import LlmResult, model_name

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "sample_invoice_pdf"
JSON_DIR = ROOT / "sample_invoice_json"
COMPARISON_DIR = ROOT / "sample_invoice_comparison"


def write_result(method: str, pdf_path: Path, llm_input: str, result: LlmResult) -> Path:
    """Write ``<JSON_DIR>/<method>/<pdf stem>.json`` plus the exact text the LLM saw
    (``.llm_input.txt``) so the two methods' inputs can be inspected side by side."""
    out_dir = JSON_DIR / method
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{pdf_path.stem}.llm_input.txt").write_text(llm_input, encoding="utf-8")
    payload = {
        "source_pdf": pdf_path.name,
        "method": method,
        "model": model_name(),
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_input_chars": len(llm_input),
        "token_usage": result.token_usage,
        "latency_s": result.latency_s,
        "extraction": result.extraction.model_dump(mode="json"),
    }
    out_path = out_dir / f"{pdf_path.stem}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
