"""OpenAI-compatible client pointed at Databricks Model Serving.

ANVIL keeps model choice out of code: the runtime endpoint lives in
``harness/config.yaml`` and is called through the ``openai`` SDK with
a base URL of ``{workspace_host}/serving-endpoints``. Databricks
Foundation Models and External Model endpoints both speak OpenAI, so
a single client works across providers and customers.

``build_databricks_client()`` auto-discovers host + token from the
standard Databricks SDK config chain (env vars, ``~/.databrickscfg``
profile, notebook credentials). Tests inject their own
``openai.OpenAI`` to avoid any real call.
"""

from __future__ import annotations

import os

from databricks.sdk import WorkspaceClient
from openai import OpenAI


def build_databricks_client(profile: str | None = None) -> OpenAI:
    """Return an ``openai.OpenAI`` client pointed at Databricks serving.

    Resolution order for host + token:

    1. ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` env vars (if both set).
    2. ``WorkspaceClient(profile=profile)`` — reads
       ``~/.databrickscfg``.

    The ``openai`` base URL is always
    ``{host}/serving-endpoints``; model names passed to
    ``client.chat.completions.create(model=...)`` are the Databricks
    endpoint names (e.g. ``databricks-claude-sonnet-4-6``).
    """
    env_host = os.environ.get("DATABRICKS_HOST")
    env_token = os.environ.get("DATABRICKS_TOKEN")
    if env_host and env_token:
        host, token = env_host, env_token
    else:
        ws = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
        host = ws.config.host
        token = ws.config.token
        if not token:
            headers = ws.config.authenticate()
            auth = headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                token = auth.removeprefix("Bearer ")
        if not host or not token:
            raise RuntimeError(
                "Could not resolve Databricks host/token. Set DATABRICKS_HOST + "
                "DATABRICKS_TOKEN, or configure a profile in ~/.databrickscfg."
            )

    base_url = host.rstrip("/") + "/serving-endpoints"
    return OpenAI(api_key=token, base_url=base_url)
