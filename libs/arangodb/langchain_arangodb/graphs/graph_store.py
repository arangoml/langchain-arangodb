from typing import Any, Dict, List, Protocol, runtime_checkable

from langchain_arangodb.graphs.graph_document import GraphDocument


@runtime_checkable
class GraphStore(Protocol):
    """Abstract class for graph operations."""

    @property
    def schema_json(self) -> str:
        """Return the schema of the Graph database as JSON."""
        ...

    @property
    def schema_yaml(self) -> str:
        """Return the schema of the Graph database as YAML."""
        ...

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Execute a database query."""
        ...

    def explain(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Explain the query."""
        ...

    def refresh_schema(self) -> None:
        """Refresh the graph schema information."""
        ...

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """Take GraphDocument as input as uses it to construct a graph."""
        ...
