"""Databricks AI Gateway client — the sole LLM route for FORGE.

All three LLM call paths (runtime agent, optimizer, judge) route through
the AI Gateway's unified OpenAI-compatible endpoint. The gateway routes
by FMAPI model name, so a single client serves all models — no
per-endpoint clients needed.

SP token refresh: Databricks SP tokens expire (~1h). The client refreshes
the token on each request to avoid expiry failures.

``build_gateway_client()`` builds the client; ``build_databricks_client()``
is kept as a backward-compatible wrapper that delegates to it.

The ``databricks`` SDK import (for token refresh when no static
``DATABRICKS_TOKEN`` is set) is lazy — inside the function — so importing
this module never requires the SDK to be installed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def _get_fresh_sp_token() -> str:
    """Get a fresh Databricks SP token.

    SP tokens expire (~1h), so we refresh on each call. Resolution order:

    1. ``DATABRICKS_TOKEN`` env var (static token; fine for testing/dev
       and for long-lived PATs).
    2. The Databricks SDK config chain (``~/.databrickscfg`` profile, env
       vars such as ``DATABRICKS_CONFIG_PROFILE``). The SDK mints a fresh
       OAuth token on demand.
    """
    token = os.environ.get("DATABRICKS_TOKEN")
    if token:
        return token
    # Lazy import so importing this module never requires the SDK. The
    # SDK is a core dependency, but keeping the import lazy also lets
    # the gateway-client unit tests run without exercising the SDK path.
    from databricks.sdk import WorkspaceClient

    ws = WorkspaceClient()
    # ``config.authenticate()`` returns a ``(header_name, header_value)``
    # tuple where the header value is a fresh bearer token. Resolving via
    # ``config.token`` first covers the case where a static PAT is
    # configured on the profile.
    token = ws.config.token
    if not token:
        _header, value = ws.config.authenticate()
        token = value
    if not token:
        raise RuntimeError(
            "Could not resolve a Databricks token. Set DATABRICKS_TOKEN or "
            "configure a profile in ~/.databrickscfg."
        )
    return token


def _gateway_base_url() -> str:
    """Get the AI Gateway unified URL.

    Format: ``https://<workspace_host>/ai-proxy-api/llm/v1`` — the
    OpenAI-compatible route of the Databricks AI Gateway. The same URL
    serves every FMAPI model; the ``model`` parameter in
    ``chat.completions.create`` selects which one the gateway routes to.

    Reads ``DATABRICKS_HOST`` from the environment.
    """
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    return f"{host}/ai-proxy-api/llm/v1"


class GatewayClient:
    """OpenAI-compatible client backed by the Databricks AI Gateway.

    Exposes ``chat.completions.create()`` with the same interface as
    ``openai.OpenAI``, but routes through the AI Gateway unified URL.
    Refreshes the SP token on each request to avoid expiry failures.

    The ``model`` parameter in ``chat.completions.create()`` should be an
    FMAPI model name (e.g. ``databricks-claude-sonnet-4-6``) — the gateway
    routes by that name, not by a per-endpoint URL.
    """

    class _Chat:
        class _Completions:
            def __init__(self, parent: GatewayClient) -> None:
                self._parent = parent

            def create(
                self,
                *,
                model: str,
                messages: list[dict[str, Any]],
                **kwargs: Any,
            ) -> Any:
                # Lazy import: ``openai`` is a project dependency, but
                # importing it here keeps the module side-effect-free and
                # lets tests stub ``openai.OpenAI`` without a real client.
                from openai import OpenAI

                # Fresh token on each call — SP tokens expire (~1h).
                token = self._parent._get_token()
                client = OpenAI(
                    api_key=token,
                    base_url=self._parent._base_url,
                )
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs,
                )

        def __init__(self, parent: GatewayClient) -> None:
            self.completions = self._Completions(parent)

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token_fn: Callable[[], str] | None = None,
    ) -> None:
        self._base_url = base_url or _gateway_base_url()
        self._token_fn = token_fn or _get_fresh_sp_token
        self.chat = self._Chat(self)

    def _get_token(self) -> str:
        # Always refresh — SP tokens expire, and a long-running app can
        # outlive a token minted at construction time.
        return self._token_fn()


def build_gateway_client(
    *,
    base_url: str | None = None,
    token_fn: Callable[[], str] | None = None,
) -> GatewayClient:
    """Build an AI Gateway client for LLM calls.

    The client is OpenAI-compatible (``chat.completions.create``) but
    routes through the AI Gateway unified URL with per-request SP token
    refresh. The ``model`` parameter should be an FMAPI model name.
    """
    return GatewayClient(base_url=base_url, token_fn=token_fn)


def build_databricks_client(
    profile: str | None = None,  # noqa: ARG001 — accepted for backward compat
    **kwargs: Any,
) -> GatewayClient:
    """Backward-compatible wrapper — delegates to the AI Gateway client.

    ``profile`` is accepted to preserve the legacy call signature
    (``build_databricks_client(profile=...)``) but the gateway client
    resolves host + token from the environment. Callers that set
    ``DATABRICKS_CONFIG_PROFILE`` in the env (the eval runner does this
    when a ``--profile`` is passed) are respected: the SDK reads that env
    var when minting a fresh token.
    """
    return build_gateway_client(**kwargs)
