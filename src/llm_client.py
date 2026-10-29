"""
LLM Client Module
=================

Lightweight wrapper for OpenAI-compatible chat completion APIs.

Designed to work out-of-the-box with:
  - DeepSeek (https://api.deepseek.com/v1)
  - OpenAI (https://api.openai.com/v1)
  - Azure OpenAI (with custom base_url)
  - vLLM / Ollama (OpenAI-compatible server)

The module exposes a single factory function `make_summarizer()` which
returns a callable that fits the ContainerManager hook signature:

    def summarizer(entity: str, context: str) -> str: ...

The function never raises. On any failure (network, auth, rate limit) it
returns an empty string so the calling code can fall back to the template.

API key handling (security):
  - Reads from environment variable DEEPSEEK_API_KEY (or OPENAI_API_KEY).
  - NEVER hard-code keys. If the env var is missing, the client is
    instantiated in disabled mode and the summarizer returns "".
  - If a literal key is provided to `DeepSeekClient(api_key=...)` for
    ad-hoc testing, it is logged in-memory only (never printed or
    persisted to disk).

Usage:
    >>> from src.llm_client import make_summarizer
    >>> summarize = make_summarizer()  # auto-detects DEEPSEEK_API_KEY
    >>> out = summarize("Dr. Sarah Chen", "She published a paper on...")
    >>> print(out)
"""

from __future__ import annotations

import os
import logging
import time
from typing import Callable, Dict, List, Optional

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    requests = None  # type: ignore

logger = logging.getLogger(__name__)


# Default model: DeepSeek-V3 chat model (recommended for cost/perf).
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

# ── Local (self-hosted) LLM support ─────────────────────────────────────────
#
# The evaluation runs against a locally-served model (vLLM) instead of the
# hosted DeepSeek API.  A local server speaks the same OpenAI
# /chat/completions schema, so the existing `DeepSeekClient` works
# unchanged — it only needs a different base_url, model name, and a
# placeholder API key (vLLM ignores the bearer token unless
# --api-key was passed).
#
# Configuration (environment variables):
#
#   STENRAG_LLM_BASE_URL   base URL of the OpenAI-compatible server,
#                          e.g. "http://127.0.0.1:8013/v1".
#   STENRAG_LLM_MODEL      served model name, e.g. "llama-3.1-8b".
#   STENRAG_LLM_API_KEY    optional bearer token for the local server.
#
# When STENRAG_LLM_BASE_URL is set, `make_client()` targets the local
# server; otherwise it falls back to the hosted DeepSeek API.
LOCAL_LLM_BASE_URL_ENV = "STENRAG_LLM_BASE_URL"
LOCAL_LLM_MODEL_ENV = "STENRAG_LLM_MODEL"
LOCAL_LLM_API_KEY_ENV = "STENRAG_LLM_API_KEY"

# Placeholder bearer token for servers that do not enforce auth.
_LOCAL_PLACEHOLDER_KEY = "local"


class LLMClientError(Exception):
    """Raised for unrecoverable LLM client errors."""


class DeepSeekClient:
    """
    Minimal OpenAI-compatible chat client. Supports DeepSeek, OpenAI,
    Azure, vLLM, Ollama - any provider with the /chat/completions schema.

    Args:
        api_key: API key. If None, reads from env var.
        base_url: API base URL. Defaults to DeepSeek.
        model: Model name. Defaults to deepseek-chat.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on transient failures (5xx, timeout).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        model: str = DEFAULT_DEEPSEEK_MODEL,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        if requests is None:  # pragma: no cover
            raise LLMClientError(
                "The `requests` package is required. "
                "Install it via `pip install requests`."
            )

        # Resolve API key: explicit > env var
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get(
                "OPENAI_API_KEY"
            )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._enabled = bool(api_key)

    @property
    def enabled(self) -> bool:
        """True if a usable API key is configured."""
        return self._enabled

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 80,
    ) -> str:
        """
        Call /chat/completions and return the assistant text.

        Args:
            messages: OpenAI-format messages list.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Cap on response length.

        Returns:
            The assistant's text response.

        Raises:
            LLMClientError on non-recoverable failure.
        """
        if not self._enabled:
            raise LLMClientError(
                "Client is disabled: no API key configured. "
                "Set DEEPSEEK_API_KEY env var or pass api_key=..."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices") or []
                    if not choices:
                        raise LLMClientError(
                            f"Empty choices in response: {data!r}"
                        )
                    return choices[0]["message"]["content"].strip()

                # 4xx: client error → no retry (except 429 rate limit)
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    body = resp.text[:300]
                    raise LLMClientError(
                        f"HTTP {resp.status_code}: {body}"
                    )

                # 5xx / 429 → transient, retry
                last_error = LLMClientError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
            except requests.exceptions.Timeout as e:
                last_error = e
            except requests.exceptions.ConnectionError as e:
                last_error = e
            except LLMClientError:
                raise  # unrecoverable, bubble up
            except Exception as e:  # pragma: no cover - defensive
                last_error = e

            if attempt < self.max_retries:
                time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s, 2s, ...
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying...",
                    attempt + 1, self.max_retries + 1, last_error,
                )

        raise LLMClientError(
            f"LLM call failed after {self.max_retries + 1} attempts: {last_error}"
        )


# ---------------------------------------------------------------------------
# Container-summary prompt templates
# ---------------------------------------------------------------------------

_SUMMARIZE_SYSTEM = (
    "You generate concise one-sentence factual summaries for retrieval-augmented "
    "question answering. Output must be a single sentence, no preamble, no quotes."
)

_SUMMARIZE_USER_TMPL = (
    "Given the entity '{entity}' and the following sentences about it, write ONE "
    "factual sentence (12-25 words) summarizing the key claim or event. "
    "Start with the entity name. Avoid filler words.\n\n"
    "Sentences:\n{context}\n\n"
    "One-sentence summary:"
)


def make_summarizer(
    client: Optional[DeepSeekClient] = None,
    temperature: float = 0.3,
    max_tokens: int = 80,
) -> Callable[[str, str], str]:
    """
    Build a callable that fits the ContainerManager's `_llm_summarize`
    hook signature: `(entity, context) -> summary_text`.

    The returned callable NEVER raises. On any failure it returns an empty
    string so the caller can fall back to the template summary path.

    Args:
        client: Optional pre-configured DeepSeekClient. If None, one is
            created using env var DEEPSEEK_API_KEY (or OPENAI_API_KEY).
        temperature: LLM sampling temperature. Default 0.3 for moderate
            determinism while allowing natural language variation.
        max_tokens: Cap on output length. 80 is sufficient for a
            12-25 word summary with some headroom.

    Returns:
        A function `(entity: str, context: str) -> str`. Returns ""
        if the LLM is disabled (no key) or the call fails.
    """
    if client is None:
        client = make_client()

    def _summarize(entity: str, context: str) -> str:
        if not client.enabled:
            logger.debug(
                "LLM summarizer disabled (no key); caller will use template."
            )
            return ""
        if not context or not entity:
            return ""
        try:
            return client.chat(
                messages=[
                    {"role": "system", "content": _SUMMARIZE_SYSTEM},
                    {
                        "role": "user",
                        "content": _SUMMARIZE_USER_TMPL.format(
                            entity=entity, context=context
                        ),
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LLMClientError as e:
            logger.warning("LLM summarizer failed: %s", e)
            return ""
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("LLM summarizer unexpected error: %s", e)
            return ""

    return _summarize


# ---------------------------------------------------------------------------
# Client factory: local (self-hosted) LLM or hosted DeepSeek
# ---------------------------------------------------------------------------

def make_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> "DeepSeekClient":
    """
    Build the chat client used by the evaluation.

    Resolution order:

      1. Explicit `base_url` / `model` / `api_key` arguments.
      2. The local server described by STENRAG_LLM_BASE_URL +
         STENRAG_LLM_MODEL (+ optional STENRAG_LLM_API_KEY).
      3. The hosted DeepSeek API (DEEPSEEK_API_KEY / OPENAI_API_KEY).

    The returned client is always a `DeepSeekClient` — it already
    speaks the generic OpenAI /chat/completions schema, so a local
    vLLM server and the hosted API are interchangeable behind it.

    The client is *enabled* when a base URL is resolvable: for the
    local server the bearer token is a placeholder (vLLM ignores it
    unless --api-key was passed), while for the hosted API a real key
    is still required.
    """
    local_base = (os.environ.get(LOCAL_LLM_BASE_URL_ENV) or "").strip()
    local_model = (os.environ.get(LOCAL_LLM_MODEL_ENV) or "").strip()

    if base_url:
        # Explicit target (e.g. a local server passed on the command line).
        key = api_key or os.environ.get(LOCAL_LLM_API_KEY_ENV) or _LOCAL_PLACEHOLDER_KEY
        return DeepSeekClient(
            api_key=key,
            base_url=base_url,
            model=model or local_model or DEFAULT_DEEPSEEK_MODEL,
            timeout=timeout,
            max_retries=max_retries,
        )

    if local_base:
        key = api_key or os.environ.get(LOCAL_LLM_API_KEY_ENV) or _LOCAL_PLACEHOLDER_KEY
        return DeepSeekClient(
            api_key=key,
            base_url=local_base,
            model=model or local_model or DEFAULT_DEEPSEEK_MODEL,
            timeout=timeout,
            max_retries=max_retries,
        )

    # Hosted DeepSeek (or any OpenAI-compatible provider with a key).
    return DeepSeekClient(
        api_key=api_key,
        base_url=DEFAULT_DEEPSEEK_BASE_URL,
        model=model or DEFAULT_DEEPSEEK_MODEL,
        timeout=timeout,
        max_retries=max_retries,
    )
