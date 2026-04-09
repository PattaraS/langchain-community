from __future__ import annotations

import warnings
from typing import Any, Dict, Iterator, List
from urllib.parse import urlparse

import requests as http_requests

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


def _chunk(texts: List[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(texts), size):
        yield texts[i : i + size]


class MLflowGatewayEmbeddings(Embeddings, BaseModel):
    """`MLflow AI Gateway` embeddings.

    MLflow AI Gateway is a database-backed LLM proxy built into the MLflow
    tracking server (MLflow >= 3.0). It provides a unified API across multiple
    embedding providers with built-in secrets management, fallback/retry, and
    usage tracing — all configured through the MLflow UI.

    No ``mlflow`` dependency is required — this class communicates with the
    gateway via its REST API.

    For more information, see https://mlflow.org/docs/latest/genai/governance/ai-gateway/

    Example:

        .. code-block:: python

            from langchain_community.embeddings import MLflowGatewayEmbeddings

            embeddings = MLflowGatewayEmbeddings(
                target_uri="http://localhost:5000",
                endpoint="my-embeddings-endpoint",
            )
            embeddings.embed_query("What is MLflow AI Gateway?")
    """

    endpoint: str
    """The MLflow Gateway endpoint name to use."""
    target_uri: str
    """The MLflow tracking server URI (e.g. ``http://localhost:5000``)."""
    query_params: Dict[str, str] = {}
    """Extra parameters forwarded with every ``embed_query`` call."""
    documents_params: Dict[str, str] = {}
    """Extra parameters forwarded with every ``embed_documents`` call."""

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

    def embed(self, texts: List[str], params: Dict[str, str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for txt in _chunk(texts, 20):
            resp = http_requests.post(
                self._invocation_url,
                json={"input": txt, **params},
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            embeddings.extend(r["embedding"] for r in resp.json()["data"])
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts, params=self.documents_params)

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text], params=self.query_params)[0]


class MLflowGatewayCohereEmbeddings(MLflowGatewayEmbeddings):
    """``MLflow AI Gateway`` embeddings for Cohere models.

    Automatically sets the ``input_type`` parameter required by Cohere's
    embeddings API — ``search_query`` for queries and ``search_document``
    for documents.
    """

    query_params: Dict[str, str] = {"input_type": "search_query"}
    documents_params: Dict[str, str] = {"input_type": "search_document"}


# ---------------------------------------------------------------------------
# Deprecated aliases — kept for backwards compatibility
# ---------------------------------------------------------------------------


class MlflowEmbeddings(MLflowGatewayEmbeddings):
    """Deprecated. Use :class:`MLflowGatewayEmbeddings` instead."""

    def __init__(self, **kwargs: Any):
        warnings.warn(
            "`MlflowEmbeddings` has been renamed to `MLflowGatewayEmbeddings`. "
            "Please update your code: `from langchain_community.embeddings "
            "import MLflowGatewayEmbeddings`.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


class MlflowCohereEmbeddings(MLflowGatewayCohereEmbeddings):
    """Deprecated. Use :class:`MLflowGatewayCohereEmbeddings` instead."""

    def __init__(self, **kwargs: Any):
        warnings.warn(
            "`MlflowCohereEmbeddings` has been renamed to "
            "`MLflowGatewayCohereEmbeddings`. "
            "Please update your code: `from langchain_community.embeddings "
            "import MLflowGatewayCohereEmbeddings`.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)
