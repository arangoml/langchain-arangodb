import pytest
from arango.database import StandardDatabase
from langchain_core.documents import Document

from langchain_arangodb.graphs.arangodb_graph import ArangoGraph
from langchain_arangodb.graphs.graph_document import GraphDocument, Node, Relationship

test_data = [
    GraphDocument(
        nodes=[Node(id="foo", type="foo"), Node(id="bar", type="bar")],
        relationships=[
            Relationship(
                source=Node(id="foo", type="foo"),
                target=Node(id="bar", type="bar"),
                type="REL",
                properties={"key": "val"},
            )
        ],
        source=Document(page_content="source document"),
    )
]

test_data_backticks = [
    GraphDocument(
        nodes=[Node(id="foo", type="foo`"), Node(id="bar", type="`bar")],
        relationships=[
            Relationship(
                source=Node(id="foo", type="f`oo"),
                target=Node(id="bar", type="ba`r"),
                type="`REL`",
            )
        ],
        source=Document(page_content="source document"),
    )
]


@pytest.mark.usefixtures("clear_arangodb_database")
def test_connect_arangodb(db: StandardDatabase) -> None:
    """Test that ArangoDB database is correctly instantiated and connected."""
    graph = ArangoGraph(db)

    output = graph.query("RETURN 1")
    expected_output = [1]
    assert output == expected_output
