"""Rule-based (regex) variant: no LLM. Reads the same document structure the LLM variant
builds (tables as cell grids, text blocks as lines) and pulls each field out with label
lookups and regexes written for ONE known invoice layout — Broadridge Financial Solutions.

    python extract_rule_based.py            # every PDF in sample_invoice_pdf/
    python extract_rule_based.py some.pdf   # just one

Rules only apply to a layout they were written for, so each invoice is first matched
against the template's fingerprint. An unrecognised layout is not guessed at: it comes back
with every field null and ``status: "no_template"``. After extraction, arithmetic and
cross-reference checks run; any failure sets ``status: "needs_llm_fallback"`` — in a real
pipeline that invoice would be sent to the LLM instead.
"""

from __future__ import annotations

import calendar
import json
import re
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from extract_structured import Element, pdf_elements
from invoice_llm.io import JSON_DIR, PDF_DIR
from invoice_llm.schema import InvoiceExtraction, InvoiceLineItemTable

METHOD = "rule_based"

# ── Generic lookups over the document structure ────────────────────────────────────────

# Every label the key/value grid uses. A cell "below" a label only counts as its value if
# it isn't itself one of these — an empty value row is dropped from the grid, so without
# this guard a blank "PO Number" would pick up the next label ("ACCOUNT#") as its value.
_GRID_LABELS = re.compile(
    r"^(invoice|invoice date|activity period|po number|account#|client vat#|terms|due date)$",
    re.IGNORECASE,
)


def _tables(elements: list[Element]) -> list[Element]:
    return [e for e in elements if e.kind == "table"]


def _texts(elements: list[Element]) -> list[Element]:
    return [e for e in elements if e.kind == "text"]


def grid_value(elements: list[Element], label: str) -> str | None:
    """Value of a key/value grid where the label cell sits directly above its value."""
    pattern = re.compile(rf"^{label}$", re.IGNORECASE)
    for table in _tables(elements):
        for r, row in enumerate(table.rows[:-1]):
            for c, cell in enumerate(row):
                if pattern.match(cell.strip()):
                    below = table.rows[r + 1]
                    value = below[c].strip() if c < len(below) else ""
                    if value and not _GRID_LABELS.match(value):
                        return value
                    return None
    return None


def line_after(elements: list[Element], label: str) -> str | None:
    """The line following a text-block line that matches ``label``."""
    pattern = re.compile(rf"^{label}$", re.IGNORECASE)
    for block in _texts(elements):
        for i, line in enumerate(block.lines[:-1]):
            if pattern.match(line.strip()):
                return block.lines[i + 1].strip()
    return None


def search_text(elements: list[Element], pattern: str) -> re.Match | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for e in elements:
        if e.kind in ("text", "table") and (m := regex.search(e.text)):
            return m
    return None


def money(value: str | None) -> Decimal | None:
    if not value:
        return None
    try:
        return Decimal(re.sub(r"[^\d.\-]", "", value))
    except InvalidOperation:
        return None


# ── The Broadridge template ────────────────────────────────────────────────────────────

TEMPLATE = "broadridge_v1"


def matches_template(elements: list[Element]) -> bool:
    return (search_text(elements, r"Broadridge Financial Solutions") is not None
            and search_text(elements, r"REMITTANCE INSTRUCTIONS") is not None
            and search_text(elements, r"Invoice Total due") is not None)


def invoice_pages(elements: list[Element]) -> list[Element]:
    """Only the pages carrying the invoice. The cover email (page 2 here) repeats invoice
    numbers and POs for OTHER invoices, so it must not be searched."""
    pages = {e.page for e in elements if re.search(r"REMITTANCE INSTRUCTIONS", e.text)}
    return [e for e in elements if e.page in pages]


def _line_items(elements: list[Element]) -> InvoiceLineItemTable | None:
    for table in _tables(elements):
        header = table.rows[0]
        if any(re.fullmatch(r"description", h, re.I) for h in header) and \
                any(re.search(r"price", h, re.I) for h in header):
            rows = [
                [cell.replace(" / ", " ") or None for cell in row]
                for row in table.rows[1:]
                if not re.search(r"\btotal\b", row[0], re.IGNORECASE)  # "FEE Total" summary row
            ]
            return InvoiceLineItemTable(columns=header, rows=rows)
    return None


_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}


def _service_period(items: InvoiceLineItemTable | None) -> tuple[str | None, str | None]:
    """Broadridge descriptions end "... for <Month> <Year>"; the period spans every month
    mentioned across the line items."""
    if items is None:
        return None, None
    months = []
    for row in items.rows:
        for m in re.finditer(r"\bfor\s+([A-Za-z]+)\s+(\d{4})\b", " ".join(c or "" for c in row)):
            if m.group(1).lower() in _MONTHS:
                months.append((int(m.group(2)), _MONTHS[m.group(1).lower()]))
    if not months:
        return None, None
    (y0, m0), (y1, m1) = min(months), max(months)
    last_day = calendar.monthrange(y1, m1)[1]
    return f"{y0:04d}-{m0:02d}-01", f"{y1:04d}-{m1:02d}-{last_day:02d}"


def _bill_to(elements: list[Element]) -> tuple[str | None, str | None]:
    for table in _tables(elements):
        for row in table.rows:
            for cell in row:
                if cell.startswith("Bill To"):
                    lines = [l.strip() for l in cell.split(" / ")[1:]]
                    address = [l for l in lines if not l.upper().startswith("ATTN")]
                    entity_lines = []
                    for line in address:
                        if re.match(r"^\d", line):  # street address starts the next part
                            break
                        entity_lines.append(line)
                    return "\n".join(lines), " ".join(entity_lines) or None
    return None, None


def _vendor(elements: list[Element]) -> str | None:
    """The vendor name is the cell directly above the one holding its VAT/company reg."""
    for table in _tables(elements):
        for r in range(1, len(table.rows)):
            for c, cell in enumerate(table.rows[r]):
                if "VAT Reg #" in cell and c < len(table.rows[r - 1]):
                    return table.rows[r - 1][c].strip() or None
    return None


def _payment_description(elements: list[Element], terms, due_date) -> str | None:
    parts = []
    if terms:
        parts.append(f"Terms: {terms}")
    if due_date:
        parts.append(f"Due Date: {due_date}")
    for block in _texts(elements):
        if block.lines and block.lines[0].strip().upper() == "REMITTANCE:":
            parts.append("; ".join(l.strip() for l in block.lines[1:]))
            break
    note = search_text(elements, r"Please reference your INVOICE[^\n]*")
    if note:
        parts.append(note.group(0).strip())
    return ". ".join(parts) or None


def extract_fields(elements: list[Element]) -> dict:
    page = invoice_pages(elements)
    items = _line_items(page)
    start, end = _service_period(items)
    bill_to, bill_to_entity = _bill_to(page)
    terms = grid_value(page, "Terms")
    vat = search_text(page, r"\bat\s+(\d+(?:\.\d+)?%)\s*\n?\s*(£[\d,]+\.\d{2})")
    tax_code = search_text(page, r"VAT Reg #:\s*([A-Z]{2}[\d ]+\d)")
    return {
        "invoice_number": grid_value(page, "Invoice"),
        "bill_to": bill_to,
        "ship_to": None,  # this layout has no Ship To section
        "invoice_date": grid_value(page, "Invoice Date"),
        "vat": vat.group(1) if vat else None,
        "vat_amount": vat.group(2) if vat else None,
        "invoice_total": line_after(page, "Invoice Total"),
        "invoice_total_due": line_after(page, "Invoice Total due"),
        "invoice_description": items,
        "payment_description": _payment_description(page, terms, grid_value(page, "Due Date")),
        "payment_terms": terms,
        "purchase_order": grid_value(page, "PO Number"),
        "vendor": _vendor(page),
        "bill_to_entity": bill_to_entity,
        "service_start": start,
        "service_end": end,
        "tax_code": re.sub(r"\s", "", tax_code.group(1)) if tax_code else None,
    }


# ── Checks: does the extraction hang together? ─────────────────────────────────────────

def run_checks(fields: dict, elements: list[Element]) -> list[dict]:
    page = invoice_pages(elements)
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for name in ("invoice_number", "invoice_date", "invoice_total", "invoice_total_due",
                 "vendor", "bill_to_entity"):
        check(f"required:{name}", fields[name] is not None, f"{name} = {fields[name]!r}")

    total, due, vat_amt = (money(fields[k]) for k in
                           ("invoice_total", "invoice_total_due", "vat_amount"))
    items: InvoiceLineItemTable | None = fields["invoice_description"]
    if items is not None and total is not None:
        price_col = len(items.columns) - 1
        line_sum = sum((money(r[price_col]) or Decimal(0)) for r in items.rows)
        check("line_items_sum_to_invoice_total", line_sum == total,
              f"sum of line items {line_sum} vs invoice total {total}")
    if None not in (total, due, vat_amt):
        check("total_plus_vat_equals_total_due", total + vat_amt == due,
              f"{total} + {vat_amt} = {total + vat_amt} vs total due {due}")
    if fields["vat"] and None not in (total, vat_amt):
        rate = Decimal(fields["vat"].rstrip("%")) / 100
        expected = (total * rate).quantize(Decimal("0.01"))
        check("vat_amount_matches_rate", abs(expected - vat_amt) <= Decimal("0.01"),
              f"{total} x {fields['vat']} = {expected} vs VAT {vat_amt}")

    # The tear-off remittance slip repeats the invoice number and the amount due.
    slip = next((t for t in _tables(page) if t.rows and "INVOICE #" in t.rows[0]), None)
    if slip is not None and len(slip.rows) > 1:
        slip_no = slip.rows[1][slip.rows[0].index("INVOICE #")]
        check("remittance_slip_invoice_number", slip_no == fields["invoice_number"],
              f"slip {slip_no!r} vs invoice {fields['invoice_number']!r}")
    slip_due = line_after(page, r"Total Due This Invoice.*")
    if slip_due and due is not None:
        check("remittance_slip_amount_due", money(slip_due) == due,
              f"slip {slip_due} vs invoice total due {due}")
    return checks


# ── Entry point ────────────────────────────────────────────────────────────────────────

def extract_rule_based(path: Path) -> Path:
    started = time.perf_counter()
    elements = pdf_elements(path)
    if matches_template(elements):
        fields = extract_fields(elements)
        checks = run_checks(fields, elements)
        template = TEMPLATE
        status = "ok" if all(c["passed"] for c in checks) else "needs_llm_fallback"
    else:
        fields = {name: None for name in InvoiceExtraction.model_fields}
        checks, template, status = [], None, "no_template"
    latency = time.perf_counter() - started

    extraction = InvoiceExtraction(**fields).model_dump(mode="json")
    payload = {
        "source_pdf": path.name,
        "method": METHOD,
        "model": None,
        "template": template,
        "status": status,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "token_usage": {"prompt": 0, "completion": 0, "total": 0},
        "latency_s": round(latency, 3),
        "checks": checks,
        "extraction": extraction,
    }
    out_dir = JSON_DIR / METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    pdfs = [Path(p) for p in sys.argv[1:]] or sorted(PDF_DIR.glob("*.pdf"))
    for pdf in pdfs:
        out = extract_rule_based(pdf)
        status = json.loads(out.read_text(encoding="utf-8"))["status"]
        print(f"[{METHOD}] {pdf.name} -> {out} ({status})")
