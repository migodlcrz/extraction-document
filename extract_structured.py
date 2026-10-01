"""Document-structure variant: same LLM, same schema, but the text is handed over with its
layout — pages, reading-order blocks tagged with their region and bounding box, bold/heading
cues, and tables rendered as markdown grids so a label stays attached to its value.

    python extract_structured.py            # every PDF in sample_invoice_pdf/
    python extract_structured.py some.pdf   # just one
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from invoice_llm.io import PDF_DIR, write_result
from invoice_llm.llm import SYSTEM_PROMPT, extract_invoice

METHOD = "document_structure"

STRUCTURE_PROMPT = SYSTEM_PROMPT + """
Input format: the invoice is given as a structured layout, not raw text.
- "## Page N of M" starts each page. A page may be a cover email or a remittance slip
  rather than the invoice itself.
- "[TEXT ...]" is a block of text that sits together on the page. Its tag gives the page
  region (e.g. top-right), the bounding box in points (x0,y0,x1,y1; origin top-left) and
  styling (bold / large) where present.
- "[TABLE ...]" is a table detected from the page's ruling lines, rendered as a markdown grid.
  Multi-line cells are joined with " / ". Some tables are key/value grids where a label
  cell sits directly ABOVE its value cell (e.g. a row "Invoice Date | Activity Period"
  followed by "03-FEB-26 | JAN-26" means Invoice Date = 03-FEB-26).
- "[HORIZONTAL RULE]" marks a separator line, often the tear-off line above a remittance
  slip. Text below it usually repeats values from the invoice.
Use the layout to associate labels with values; the extraction rules above still apply.
"""

_BOLD_FLAG = 16
_DASH_LINE = re.compile(r"^[\s\-_=—–]{10,}$")


@dataclass
class Element:
    """One piece of page layout. ``kind`` is "table", "text" or "rule". Tables carry
    ``rows`` (cells as strings, multi-line cells joined with " / "); text blocks carry
    ``lines``. Shared by the LLM renderer below and the rule-based extractor."""

    kind: str
    page: int  # 1-based
    bbox: tuple[float, float, float, float]
    region: str
    rows: list[list[str]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    style: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        if self.kind == "table":
            return "\n".join(" | ".join(r) for r in self.rows)
        return "\n".join(self.lines)


def _region(bbox: tuple[float, float, float, float], page: pymupdf.Page) -> str:
    cx = (bbox[0] + bbox[2]) / 2 / page.rect.width
    cy = (bbox[1] + bbox[3]) / 2 / page.rect.height
    v = "top" if cy < 1 / 3 else "middle" if cy < 2 / 3 else "bottom"
    h = "left" if cx < 1 / 3 else "center" if cx < 2 / 3 else "right"
    return f"{v}-{h}"


def _bbox_str(bbox) -> str:
    return ",".join(str(round(v)) for v in bbox)


def _clean_cell(cell: str | None) -> str:
    if not cell:
        return ""
    parts = [p.strip() for p in cell.splitlines() if p.strip()]
    return " / ".join(parts).replace("|", "\\|")


def _table_rows(table) -> list[list[str]]:
    rows = [[_clean_cell(c) for c in row] for row in table.extract()]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    keep = [i for i in range(len(rows[0])) if any(r[i] for r in rows)]
    return [[r[i] for i in keep] for r in rows]


def _merge_tables(tables) -> list[tuple[tuple, list[list[str]]]]:
    """PyMuPDF often splits a header row from its body (separate ruling boxes). Glue a
    one-row table onto the table directly below it when the column counts agree."""
    items = [(tuple(t.bbox), _table_rows(t)) for t in tables]
    items = [(b, rows) for b, rows in items if rows]
    items.sort(key=lambda it: (it[0][1], it[0][0]))
    merged: list[tuple[tuple, list[list[str]]]] = []
    for bbox, rows in items:
        if merged:
            pb, prows = merged[-1]
            if (len(prows) == 1 and len(prows[0]) == len(rows[0])
                    and 0 <= bbox[1] - pb[3] < 40):
                union = (min(pb[0], bbox[0]), pb[1], max(pb[2], bbox[2]), bbox[3])
                merged[-1] = (union, prows + rows)
                continue
        merged.append((bbox, rows))
    return merged


def _table_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _inside(point: tuple[float, float], bbox: tuple) -> bool:
    return bbox[0] - 1 <= point[0] <= bbox[2] + 1 and bbox[1] - 1 <= point[1] <= bbox[3] + 1


def page_elements(page: pymupdf.Page) -> list[Element]:
    """The page's layout as data: tables and text blocks in reading order."""
    elements: list[Element] = []
    page_no = page.number + 1

    tables = _merge_tables(page.find_tables().tables)
    table_boxes = [bbox for bbox, _ in tables]
    for bbox, rows in tables:
        elements.append(Element("table", page_no, bbox, _region(bbox, page), rows=rows))

    body_size = _body_font_size(page)
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block["type"] != 0:
            continue
        lines, bold, max_size = [], False, 0.0
        for line in block["lines"]:
            spans = [
                s for s in line["spans"]
                if s["text"].strip() and not any(
                    _inside(((s["bbox"][0] + s["bbox"][2]) / 2, (s["bbox"][1] + s["bbox"][3]) / 2), tb)
                    for tb in table_boxes)
            ]
            if not spans:
                continue
            text = " ".join(" ".join(s["text"].split()) for s in spans)
            bold |= any(s["flags"] & _BOLD_FLAG for s in spans)
            max_size = max(max_size, *(s["size"] for s in spans))
            lines.append(text)
        if not lines:
            continue
        bbox = tuple(block["bbox"])
        if all(_DASH_LINE.match(l) for l in lines):
            elements.append(Element("rule", page_no, bbox, _region(bbox, page)))
            continue
        style = []
        if bold:
            style.append("bold")
        if body_size and max_size >= body_size * 1.3:
            style.append(f"large {max_size:.0f}pt")
        elements.append(Element("text", page_no, bbox, _region(bbox, page),
                                lines=lines, style=style))

    # Reading order: top-to-bottom, then left-to-right for items on roughly the same band.
    elements.sort(key=lambda e: (round(e.bbox[1] / 6), e.bbox[0]))
    return elements


def page_structure(page: pymupdf.Page) -> list[str]:
    """Render :func:`page_elements` as the tagged text the LLM receives."""
    out, table_no = [], 0
    for e in page_elements(page):
        if e.kind == "rule":
            out.append("[HORIZONTAL RULE]")
        elif e.kind == "table":
            table_no += 1
            out.append(f"[TABLE T{e.page}.{table_no} | {e.region} | bbox {_bbox_str(e.bbox)}"
                       f" | {len(e.rows)} rows x {len(e.rows[0])} cols]\n"
                       f"{_table_markdown(e.rows)}")
        else:
            tag = f"[TEXT | {e.region} | bbox {_bbox_str(e.bbox)}"
            tag += f" | {', '.join(e.style)}]" if e.style else "]"
            out.append(tag + "\n" + "\n".join(e.lines))
    return out


def _body_font_size(page: pymupdf.Page) -> float:
    """Most common span size by character count — the page's body text size."""
    counts: dict[float, int] = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for s in line["spans"]:
                size = round(s["size"], 1)
                counts[size] = counts.get(size, 0) + len(s["text"].strip())
    return max(counts, key=counts.get) if counts else 0.0


def pdf_elements(path: Path) -> list[Element]:
    with pymupdf.open(str(path)) as doc:
        return [e for page in doc for e in page_elements(page)]


def pdf_to_structured_text(path: Path) -> str:
    out = []
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            w, h = round(page.rect.width), round(page.rect.height)
            out.append(f"## Page {page.number + 1} of {doc.page_count} ({w} x {h} pt)")
            out.extend(page_structure(page))
    return "\n\n".join(out)


def extract_structured(path: Path) -> Path:
    text = pdf_to_structured_text(path)
    llm_input = f"Invoice document (structured layout):\n\n{text}"
    result = extract_invoice(llm_input, system_prompt=STRUCTURE_PROMPT)
    return write_result(METHOD, path, llm_input, result)


if __name__ == "__main__":
    pdfs = [Path(p) for p in sys.argv[1:]] or sorted(PDF_DIR.glob("*.pdf"))
    for pdf in pdfs:
        print(f"[{METHOD}] {pdf.name} -> {extract_structured(pdf)}")
