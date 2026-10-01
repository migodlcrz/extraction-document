"""Compare two extraction methods field by field.

Reads sample_invoice_json/<method>/<stem>.json for both methods and writes, into
sample_invoice_comparison/<comparison name>/:
  <stem>.json   machine-readable per-field diff + cost/latency
  <stem>.md     the same as a readable table
  summary.md    one row per invoice

    python compare.py    # plain text vs document structure (both LLM)

See compare_regex_vs_llm.py for the rule-based vs LLM comparison. There is no ground
truth here, so "differs" means the two methods disagree — look at the .md and the source
PDF to judge which one is right.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from invoice_llm.io import COMPARISON_DIR, JSON_DIR
from invoice_llm.schema import InvoiceExtraction

PLAIN, STRUCT, RULE = "plain_text", "document_structure", "rule_based"

LABELS = {
    PLAIN: "LLM · plain text",
    STRUCT: "LLM · document structure",
    RULE: "Regex · document structure (no LLM)",
}


def _norm(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return re.sub(r"\s+", " ", value).strip().lower()


def _status(a, b, name_a: str, name_b: str) -> str:
    if _norm(a) == _norm(b):
        return "same"
    if a is None:
        return f"only {name_b}"
    if b is None:
        return f"only {name_a}"
    return "differs"


def _cell(value) -> str:
    if value is None:
        return "*null*"
    if isinstance(value, list):
        return f"{len(value)} row(s)"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _stats(run: dict) -> dict:
    ext = run["extraction"]
    return {
        "llm_input_chars": run.get("llm_input_chars"),
        "token_usage": run["token_usage"],
        "latency_s": run["latency_s"],
        "fields_filled": sum(v is not None for v in ext.values()),
        "line_item_rows": len(ext.get("invoice_description") or []),
        "status": run.get("status"),
    }


def compare_one(stem: str, a: str, b: str, out_dir: Path) -> dict:
    run_a = json.loads((JSON_DIR / a / f"{stem}.json").read_text(encoding="utf-8"))
    run_b = json.loads((JSON_DIR / b / f"{stem}.json").read_text(encoding="utf-8"))
    ea, eb = run_a["extraction"], run_b["extraction"]

    fields = [
        {"field": name, a: ea.get(name), b: eb.get(name),
         "status": _status(ea.get(name), eb.get(name), a, b)}
        for name in InvoiceExtraction.model_fields
    ]
    result = {
        "source_pdf": run_a["source_pdf"],
        "methods": [a, b],
        "models": {a: run_a.get("model"), b: run_b.get("model")},
        "fields_total": len(fields),
        "fields_same": sum(f["status"] == "same" for f in fields),
        "fields_different": [f["field"] for f in fields if f["status"] != "same"],
        "stats": {a: _stats(run_a), b: _stats(run_b)},
        "checks": {m: r["checks"] for m, r in ((a, run_a), (b, run_b)) if "checks" in r},
        "fields": fields,
    }
    (out_dir / f"{stem}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:,}" if isinstance(value, int) else str(value)


def _markdown(r: dict) -> str:
    a, b = r["methods"]
    la, lb = LABELS.get(a, a), LABELS.get(b, b)
    sa, sb = r["stats"][a], r["stats"][b]
    models = ", ".join(f"{LABELS.get(m, m)}: `{v}`" for m, v in r["models"].items() if v)
    out = [
        f"# {r['source_pdf']}",
        "",
        f"{la} vs {lb} · fields agreeing: **{r['fields_same']}/{r['fields_total']}**",
        "",
        f"Models — {models or 'none'}",
        "",
        f"| Metric | {la} | {lb} |",
        "|---|---|---|",
    ]
    rows = [
        ("LLM input chars", sa["llm_input_chars"], sb["llm_input_chars"]),
        ("Prompt tokens", sa["token_usage"]["prompt"], sb["token_usage"]["prompt"]),
        ("Completion tokens", sa["token_usage"]["completion"], sb["token_usage"]["completion"]),
        ("Total tokens", sa["token_usage"]["total"], sb["token_usage"]["total"]),
        ("Latency (s)", sa["latency_s"], sb["latency_s"]),
        ("Fields filled", sa["fields_filled"], sb["fields_filled"]),
        ("Line-item rows", sa["line_item_rows"], sb["line_item_rows"]),
    ]
    if sa["status"] or sb["status"]:
        rows.append(("Status", sa["status"], sb["status"]))
    out += [f"| {name} | {_fmt(x)} | {_fmt(y)} |" for name, x, y in rows]

    out += ["", "## Fields", "", f"| Field | Status | {la} | {lb} |", "|---|---|---|---|"]
    for f in r["fields"]:
        mark = "✅" if f["status"] == "same" else "⚠️"
        out.append(f"| `{f['field']}` | {mark} {f['status']} | {_cell(f[a])} | {_cell(f[b])} |")

    for f in r["fields"]:
        if f["field"] == "invoice_description" and f["status"] != "same":
            out += ["", "## Line items (differ)", ""]
            for m in (a, b):
                out += [f"**{LABELS.get(m, m)}**", "", "```json",
                        json.dumps(f[m], indent=2, ensure_ascii=False), "```", ""]

    for m, checks in r["checks"].items():
        out += ["", f"## Checks — {LABELS.get(m, m)}", "", "| Check | Result | Detail |",
                "|---|---|---|"]
        out += [f"| `{c['check']}` | {'✅ pass' if c['passed'] else '❌ fail'} | {c['detail']} |"
                for c in checks]
    return "\n".join(out) + "\n"


def compare_methods(a: str, b: str, name: str) -> list[dict]:
    """Compare every invoice both methods have output for; write to
    sample_invoice_comparison/<name>/."""
    out_dir = COMPARISON_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in (JSON_DIR / a).glob("*.json")
                   if (JSON_DIR / b / p.name).exists())
    results = [compare_one(stem, a, b, out_dir) for stem in stems]

    la, lb = LABELS.get(a, a), LABELS.get(b, b)
    lines = [
        f"# {la} vs {lb} — summary",
        "",
        f"| Invoice | Fields agreeing | Differing fields | Total tokens ({a} → {b}) "
        f"| Latency s ({a} → {b}) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        sa, sb = r["stats"][a], r["stats"][b]
        diff = ", ".join(f"`{f}`" for f in r["fields_different"]) or "—"
        lines.append(
            f"| [{r['source_pdf']}]({Path(r['source_pdf']).stem.replace(' ', '%20')}.md) "
            f"| {r['fields_same']}/{r['fields_total']} | {diff} "
            f"| {sa['token_usage']['total']:,} → {sb['token_usage']['total']:,} "
            f"| {sa['latency_s']} → {sb['latency_s']} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def compare_all() -> list[dict]:
    """Plain text vs document structure (both LLM)."""
    return compare_methods(PLAIN, STRUCT, "plain_vs_structure")


def print_results(results: list[dict], name: str) -> None:
    for r in results:
        print(f"{r['source_pdf']}: {r['fields_same']}/{r['fields_total']} fields agree; "
              f"differs: {r['fields_different'] or 'none'}")
    print(f"Written to {COMPARISON_DIR / name}")


if __name__ == "__main__":
    print_results(compare_all(), "plain_vs_structure")
