"""Compare the plain-text and document-structure extractions field by field.

Reads sample_invoice_json/{plain_text,document_structure}/<stem>.json and writes, into
sample_invoice_comparison/:
  <stem>.json   machine-readable per-field diff + cost/latency
  <stem>.md     the same as a readable table
  summary.md    one row per invoice across both methods

There is no ground truth here, so "differs" means the two methods disagree — look at the
.md and the source PDF to judge which one is right.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from invoice_llm.io import COMPARISON_DIR, JSON_DIR
from invoice_llm.schema import InvoiceExtraction

PLAIN, STRUCT = "plain_text", "document_structure"


def _norm(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return re.sub(r"\s+", " ", value).strip().lower()


def _status(a, b) -> str:
    if _norm(a) == _norm(b):
        return "same"
    if a is None:
        return "only structured"
    if b is None:
        return "only plain"
    return "differs"


def _cell(value) -> str:
    if value is None:
        return "*null*"
    if isinstance(value, list):
        return f"{len(value)} row(s)"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def compare_one(stem: str) -> dict:
    plain = json.loads((JSON_DIR / PLAIN / f"{stem}.json").read_text(encoding="utf-8"))
    struct = json.loads((JSON_DIR / STRUCT / f"{stem}.json").read_text(encoding="utf-8"))
    pe, se = plain["extraction"], struct["extraction"]

    fields = []
    for name in InvoiceExtraction.model_fields:
        fields.append({"field": name, PLAIN: pe.get(name), STRUCT: se.get(name),
                       "status": _status(pe.get(name), se.get(name))})

    def stats(run: dict) -> dict:
        ext = run["extraction"]
        return {
            "llm_input_chars": run["llm_input_chars"],
            "token_usage": run["token_usage"],
            "latency_s": run["latency_s"],
            "fields_filled": sum(v is not None for v in ext.values()),
            "line_item_rows": len(ext.get("invoice_description") or []),
        }

    result = {
        "source_pdf": plain["source_pdf"],
        "model": plain["model"],
        "fields_total": len(fields),
        "fields_same": sum(f["status"] == "same" for f in fields),
        "fields_different": [f["field"] for f in fields if f["status"] != "same"],
        "stats": {PLAIN: stats(plain), STRUCT: stats(struct)},
        "fields": fields,
    }
    (COMPARISON_DIR / f"{stem}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (COMPARISON_DIR / f"{stem}.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _markdown(r: dict) -> str:
    p, s = r["stats"][PLAIN], r["stats"][STRUCT]
    out = [
        f"# {r['source_pdf']}",
        "",
        f"Model: `{r['model']}` · fields agreeing: **{r['fields_same']}/{r['fields_total']}**",
        "",
        "| Metric | Plain text | Document structure |",
        "|---|---|---|",
        f"| LLM input chars | {p['llm_input_chars']:,} | {s['llm_input_chars']:,} |",
        f"| Prompt tokens | {p['token_usage']['prompt']:,} | {s['token_usage']['prompt']:,} |",
        f"| Completion tokens | {p['token_usage']['completion']:,} | {s['token_usage']['completion']:,} |",
        f"| Latency (s) | {p['latency_s']} | {s['latency_s']} |",
        f"| Fields filled | {p['fields_filled']} | {s['fields_filled']} |",
        f"| Line-item rows | {p['line_item_rows']} | {s['line_item_rows']} |",
        "",
        "## Fields",
        "",
        "| Field | Status | Plain text | Document structure |",
        "|---|---|---|---|",
    ]
    for f in r["fields"]:
        mark = "✅" if f["status"] == "same" else "⚠️"
        out.append(f"| `{f['field']}` | {mark} {f['status']} | {_cell(f[PLAIN])} | {_cell(f[STRUCT])} |")

    for f in r["fields"]:
        if f["field"] == "invoice_description" and f["status"] != "same":
            out += ["", "## Line items (differ)", ""]
            for label in (PLAIN, STRUCT):
                out += [f"**{label}**", "", "```json",
                        json.dumps(f[label], indent=2, ensure_ascii=False), "```", ""]
    return "\n".join(out) + "\n"


def compare_all() -> list[dict]:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in (JSON_DIR / PLAIN).glob("*.json")
                   if (JSON_DIR / STRUCT / p.name).exists())
    results = [compare_one(stem) for stem in stems]

    lines = [
        "# Plain text vs document structure — summary",
        "",
        "| Invoice | Fields agreeing | Differing fields | Prompt tokens (plain → struct) | Latency s (plain → struct) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        p, s = r["stats"][PLAIN], r["stats"][STRUCT]
        diff = ", ".join(f"`{f}`" for f in r["fields_different"]) or "—"
        lines.append(
            f"| [{r['source_pdf']}]({Path(r['source_pdf']).stem.replace(' ', '%20')}.md) "
            f"| {r['fields_same']}/{r['fields_total']} | {diff} "
            f"| {p['token_usage']['prompt']:,} → {s['token_usage']['prompt']:,} "
            f"| {p['latency_s']} → {s['latency_s']} |")
    (COMPARISON_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    for r in compare_all():
        print(f"{r['source_pdf']}: {r['fields_same']}/{r['fields_total']} same; "
              f"differs: {r['fields_different'] or 'none'}")
    print(f"Written to {COMPARISON_DIR}")
