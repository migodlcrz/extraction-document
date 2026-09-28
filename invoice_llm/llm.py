"""The LLM call, mirroring AI_Enabled_RTI_AIR304298's ``invoice_extraction/llm.py``.

Same Azure OpenAI deployment and env vars (``AZURE_OPENAI_ENDPOINT``, ``AZURE_DEPLOYMENT``,
``OPENAI_API_VERSION``, ``AZURE_IDENTITY``), same system prompt, and the same Structured
Outputs schema (``schema.py`` is copied from the repo), so the only thing that differs between
the two experiments is the text handed to the model.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, AzureOpenAI, InternalServerError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from invoice_llm.schema import InvoiceExtraction

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Verbatim from the repo.
SYSTEM_PROMPT = """\
You extract structured data from vendor invoices. You will be given the raw text of one
invoice (it may include a cover email or a remittance slip alongside the invoice itself —
extract only from the invoice content).

Rules:
- Use only what the text actually says. Never infer, guess, or carry a value over from
  general knowledge of what invoices "usually" look like.
- If a field isn't present, or you can't tell which value it refers to, return null for it.
  A null is correct and expected for many invoices; a wrong guess is not.
- Invoice layouts vary between vendors — read for meaning, not position. The same field may
  appear under different headings on different invoices (e.g. a totals block may say
  "Total Amount Due", "Invoice Total due", or similar; treat these as the same concept).
"""


@dataclass
class LlmResult:
    extraction: InvoiceExtraction
    token_usage: dict[str, int]
    latency_s: float


_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        api_version = os.environ["OPENAI_API_VERSION"]
        if os.getenv("AZURE_IDENTITY", "false").strip().lower() == "true":
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            _client = AzureOpenAI(azure_endpoint=endpoint, api_version=api_version,
                                  azure_ad_token_provider=token_provider)
        else:
            _client = AzureOpenAI(azure_endpoint=endpoint, api_version=api_version,
                                  api_key=os.environ["AZURE_OPENAI_API_KEY"])
    return _client


def model_name() -> str:
    return os.environ["AZURE_DEPLOYMENT"]


@retry(retry=retry_if_exception_type((APIConnectionError, RateLimitError, InternalServerError)),
       stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, max=30), reraise=True)
def extract_invoice(user_content: str, system_prompt: str = SYSTEM_PROMPT) -> LlmResult:
    started = time.perf_counter()
    completion = _get_client().chat.completions.parse(
        model=model_name(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format=InvoiceExtraction,
        # No temperature: the reasoning deployment rejects non-default values.
        timeout=120.0,
    )
    latency = time.perf_counter() - started
    message = completion.choices[0].message
    if message.refusal:
        raise RuntimeError(f"Model refused the extraction: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("Model returned no parseable content.")
    usage = completion.usage
    token_usage = {
        "prompt": usage.prompt_tokens if usage else 0,
        "completion": usage.completion_tokens if usage else 0,
        "total": usage.total_tokens if usage else 0,
    }
    return LlmResult(message.parsed, token_usage, round(latency, 2))
