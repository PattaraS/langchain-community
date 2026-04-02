from __future__ import annotations

import warnings
from typing import Any, Dict, Iterator, List
from urllib.parse import urlparse

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, PrivateAttr


def _chunk(texts: List[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(texts), size):
        yield texts[i : i + size]


class MLflowGatewayEmbeddings(Embeddings, BaseModel):
    """`MLflow AI Gateway` embeddings.

    MLflow AI Gateway is a database-backed LLM proxy built into the MLflow
    tracking server (MLflow >= 3.0). It provides a unified API across multiple
    embedding providers with built-in secrets management, fallback/retry, and
    usage tracing — all configured through the MLflow UI.

    To use, you should have the ``mlflow[genai]`` python package installed.
    For more information, see https://mlflow.org/docs/latest/llms/gateway/index.html.

    Setup:

        Start an MLflow server and create an embeddings gateway endpoint in the UI::

            mlflow server --host 127.0.0.1 --port 5000

        Then open http://localhost:5000, navigate to **AI Gateway → Create Endpoint**,
        and configure an embeddings provider. Provider API keys are stored encrypted
        on the server.

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
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._validate_uri()
        try:
            from mlflow.deployments import get_deploy_client

            self._client = get_deploy_client(self.target_uri)
        except ImportError as e:
            raise ImportError(
                "Failed to create the MLflow deployments client. "
                "Please run `pip install mlflow[genai]` to install "
                "required dependencies."
            ) from e

    @property
    def _mlflow_extras(self) -> str:
        return "[genai]"

    def _validate_uri(self) -> None:
        if self.target_uri == "databricks":
            return
        allowed = ["http", "https", "databricks"]
        if urlparse(self.target_uri).scheme not in allowed:
            raise ValueError(
                f"Invalid target URI: {self.target_uri}. "
                f"The scheme must be one of {allowed}."
            )

    def embed(self, texts: List[str], params: Dict[str, str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for txt in _chunk(texts, 20):
            resp = self._client.predict(
                endpoint=self.endpoint,
                inputs={"input": txt, **params},
            )
            embeddings.extend(r["embedding"] for r in resp["data"])
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
