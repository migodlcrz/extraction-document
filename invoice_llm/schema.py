"""The output contract — one Pydantic model, mirroring the ``components.py`` convention of a
typed props class as the single source of truth for a shape, rather than a hand-rolled dict.

Every field is optional. A field the invoice doesn't have, or that the LLM can't read with
confidence, comes back ``None`` — never a guess. ``field_descriptions()`` feeds the same
wording into the extraction prompt, so the schema and the instructions can't drift apart.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, PrivateAttr, field_serializer


class InvoiceLineItemTable(BaseModel):
    """The invoice's line-item table as the model reads it: headers once, then rows of cells.

    Column headers differ per vendor, so the row keys can't be a fixed schema — and Structured
    Outputs forbids free-form object keys. The model therefore returns ``columns`` plus
    positional ``rows``, and :meth:`as_records` turns that into ``{header: value}`` objects
    for the JSON output.
    """

    columns: list[str] = Field(
        ...,
        description="The line-item table's column headers, left to right, exactly as printed "
                    "(e.g. 'Description', 'Quantity', 'Unit Price', 'Extended Price'). Keep "
                    "any marker that is part of the header text, such as a trailing '*'.",
    )
    rows: list[list[str | None]] = Field(
        ...,
        description="One inner list per line-item row, in table order. Each inner list holds "
                    "one value per column, in the same order as 'columns', exactly as printed "
                    "(currency symbols included). A cell whose text wraps onto several lines "
                    "is ONE value: join the lines with single spaces. Use null for an empty "
                    "cell. Do not include subtotal, total or tax summary rows that sit below "
                    "the table.",
    )

    def as_records(self) -> list[dict[str, str | None]]:
        headers: list[str] = []
        for column in self.columns:
            # Two columns can share a header; suffix the repeat so no value is overwritten.
            header, n = column, 2
            while header in headers:
                header, n = f"{column} ({n})", n + 1
            headers.append(header)
        return [
            dict(zip(headers, [*row, *[None] * (len(headers) - len(row))]))
            for row in self.rows
        ]


class InvoiceExtraction(BaseModel):
    """The governed fields pulled from one invoice.

    The first group records the document faithfully — what it says, as it says it. The
    "rule inputs" group below is normalised instead, because the ap_rules decision tree
    reads it.
    """

    invoice_number: str | None = Field(
        None, description="The invoice's own identifying number, e.g. 'INV-29878' or '9853'."
    )
    bill_to: str | None = Field(
        None,
        description="The billed party's name and address, as shown in the invoice's "
                    "'Bill To' section.",
    )
    ship_to: str | None = Field(
        None,
        description="The recipient's name and address from the invoice's 'Ship To' section. "
                    "Null if the invoice has no separate Ship To (common when it's the same "
                    "as Bill To or the invoice has no shipping component).",
    )
    invoice_date: str | None = Field(
        None, description="The date the invoice was issued, exactly as printed on the invoice."
    )
    vat: str | None = Field(
        None,
        description="The VAT rate or VAT identifier tied to the invoice's tax line, e.g. "
                    "'20%' or a VAT registration number quoted next to the tax total. Null if "
                    "the invoice charges no VAT/tax at all (e.g. a zero tax total with no VAT "
                    "line) — do not confuse this with a party's own VAT registration number "
                    "printed in the Bill To/Ship To address block, which is not this field.",
    )
    vat_amount: str | None = Field(
        None,
        description="The monetary VAT/tax amount charged, with its currency symbol as "
                    "printed. Null if the invoice charges no VAT/tax.",
    )
    invoice_total: str | None = Field(
        None,
        description="The invoice's subtotal/total before any prior payments or credits are "
                    "applied, with its currency symbol as printed.",
    )
    invoice_total_due: str | None = Field(
        None,
        description="The final amount the recipient must pay — after any prior payments or "
                    "credits are netted against the invoice total. Equal to invoice_total "
                    "when the invoice shows no separate due amount.",
    )
    invoice_description: InvoiceLineItemTable | None = Field(
        None,
        description="The billed line items: what was charged for. Copy the invoice's "
                    "line-item / service table faithfully — its column headers exactly as "
                    "printed, and every data row under them. Null if the invoice has no "
                    "line-item table.",
    )
    payment_description: str | None = Field(
        None,
        description="Payment terms and instructions: due date, payment terms code (e.g. "
                    "'N30'), remittance instructions, or bank/reference details for paying "
                    "the invoice. Null if the invoice states none of this.",
    )
    payment_terms: str | None = Field(
        None,
        description="Just the payment terms, short, as printed: 'Net 30', 'N30', '30 days', "
                    "'Due on receipt'. This is the terms alone — not the due date, not the "
                    "bank details, both of which belong in payment_description. Null if the "
                    "invoice states no terms.",
    )
    purchase_order: str | None = Field(
        None,
        description="The buyer's purchase order number the invoice is raised against, as "
                    "printed — labelled 'PO', 'PO Number', 'Purchase Order', 'Order Ref' or "
                    "similar. The number only, without the label. Null if the invoice quotes "
                    "no purchase order; do not substitute the invoice number, a contract "
                    "number, or the vendor's own order reference.",
    )

    # ── Rule inputs ──────────────────────────────────────────────────────────────────────
    # The fields above record the invoice faithfully. These four are the ones the ap_rules
    # decision tree reads (see analytics/ap_warehouse.py::_INVOICE_FIELDS), so they are
    # normalised rather than verbatim — a legal entity without its address, dates as ISO.
    # They are extracted in the same call because the rules run before the review agent.
    vendor: str | None = Field(
        None,
        description="The supplier being paid — the party that issued the invoice, whose name "
                    "heads the invoice or appears as 'From'. Never the billed party.",
    )
    bill_to_entity: str | None = Field(
        None,
        description="The billed party's LEGAL ENTITY NAME ONLY, from the 'Bill To' block: no "
                    "street, city, postcode, country or VAT number. From 'ACCENTURE (UK) "
                    "LIMITED, 30 Fenchurch St, London EC3M 3BD' return 'ACCENTURE (UK) "
                    "LIMITED'. This is matched against a company-code reference table, so "
                    "keep the name spelled exactly as printed.",
    )
    service_start: str | None = Field(
        None,
        description="First day of the service/subscription period this invoice covers, as ISO "
                    "YYYY-MM-DD. Vendors label the column differently ('Start Date', 'Sub "
                    "Start Date', 'Service Period'), so read the line-item table's own date "
                    "columns whatever they are called; if there is no such column, derive it "
                    "from the description wording (e.g. 'six-month subscription from 1 Jan "
                    "2026'). Null if the invoice gives no service period.",
    )
    service_end: str | None = Field(
        None,
        description="Last day of the service/subscription period this invoice covers, as ISO "
                    "YYYY-MM-DD, read the same way as service_start. Null if the invoice "
                    "gives no service period.",
    )
    tax_code: str | None = Field(
        None,
        description="The VENDOR's own tax/VAT registration number, digits with any country "
                    "prefix and nothing else (e.g. 'GB123456789') — strip spaces and "
                    "punctuation. Null if the invoice shows no vendor VAT registration "
                    "number. Do not return the billed party's VAT number.",
    )

    # Set by llm._call so the audit log can record what the extraction cost. Private, so it
    # never appears in the model's JSON schema or in a dumped result.
    _token_usage: dict[str, int] | None = PrivateAttr(default=None)

    @property
    def token_usage(self) -> dict[str, int] | None:
        return self._token_usage

    @field_serializer("invoice_description")
    def _serialize_invoice_description(
        self, table: InvoiceLineItemTable | None
    ) -> list[dict[str, str | None]] | None:
        return table.as_records() if table is not None else None


def field_descriptions() -> dict[str, str]:
    """``{field_name: description}`` — folded into the extraction prompt so the model sees
    the exact same wording as the schema."""
    return {
        name: field.description or ""
        for name, field in InvoiceExtraction.model_fields.items()
    }
