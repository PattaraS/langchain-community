from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse

import requests as http_requests

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from pydantic import Field


class MLflowGateway(LLM):
    """`MLflow AI Gateway` LLM (text completions).

    MLflow AI Gateway is a database-backed LLM proxy built into the MLflow
    tracking server (MLflow >= 3.0). It provides a unified API across multiple
    LLM providers with built-in secrets management, fallback/retry, traffic
    splitting, and usage tracing — all configured through the MLflow UI.

    No ``mlflow`` dependency is required — this class communicates with the
    gateway via its REST API.

    For more information, see https://mlflow.org/docs/latest/genai/governance/ai-gateway/

    .. note::
        For chat-based models prefer :class:`~langchain_community.chat_models.ChatMLflowGateway`.

    Example:

        .. code-block:: python

            from langchain_community.llms import MLflowGateway

            llm = MLflowGateway(
                target_uri="http://localhost:5000",
                endpoint="my-completions-endpoint",
                temperature=0.1,
            )
            llm.invoke("Explain MLflow AI Gateway in one sentence.")
    """

    endpoint: str
    """The MLflow Gateway endpoint name to use."""
    target_uri: str
    """The MLflow tracking server URI (e.g. ``http://localhost:5000``)."""
    temperature: float = 0.0
    """Sampling temperature."""
    n: int = 1
    """Number of completion choices to generate."""
    stop: Optional[List[str]] = None
    """Stop sequences."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    """Any extra parameters to pass through to the endpoint."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._validate_uri()

    def _validate_uri(self) -> None:
        if self.target_uri == "databricks":
            return
        allowed = ["http", "https", "databricks"]
        if urlparse(self.target_uri).scheme not in allowed:
            raise ValueError(
                f"Invalid target URI: {self.target_uri}. "
                f"The scheme must be one of {allowed}."
            )

    @property
    def _invocation_url(self) -> str:
        base = self.target_uri.rstrip("/")
        return f"{base}/gateway/{self.endpoint}/mlflow/invocations"

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "target_uri": self.target_uri,
            "endpoint": self.endpoint,
            "temperature": self.temperature,
            "n": self.n,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "extra_params": self.extra_params,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return self._default_params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        data: Dict[str, Any] = {
            "prompt": prompt,
            "temperature": self.temperature,
            "n": self.n,
            **self.extra_params,
            **kwargs,
        }
        if stop := self.stop or stop:
            data["stop"] = stop
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens

        resp = http_requests.post(
            self._invocation_url,
            json=data,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    @property
    def _llm_type(self) -> str:
        return "mlflow-gateway"


class Mlflow(MLflowGateway):
    """Deprecated. Use :class:`MLflowGateway` instead."""

    def __init__(self, **kwargs: Any):
        warnings.warn(
            "`Mlflow` has been renamed to `MLflowGateway`. "
            "Please update your code: `from langchain_community.llms "
            "import MLflowGateway`.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)
