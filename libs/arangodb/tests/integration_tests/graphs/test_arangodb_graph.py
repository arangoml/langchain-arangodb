"""Integration tests for ArangoDB graph (ArangoGraph)."""

import json
import time
from typing import Any, Dict, List

import pytest
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError
from langchain_core.documents import Document

from langchain_arangodb.graphs.graph_document import (
    GraphDocument,
    Node,
    Relationship,
)
from langchain_arangodb.graphs.arangodb_graph import ArangoGraph, get_arangodb_client
from tests.integration_tests.utils import ArangoCredentials # Assuming this exists


# Test data similar to Neo4j's test_data
TEST_GRAPH_DOC = GraphDocument(
    nodes=[
        Node(id="foo_node", type="FooNode", properties={"name": "foo name"}),
        Node(id="bar_node", type="BarNode", properties={"value": 10}),
    ],
    relationships=[
        Relationship(
            source=Node(id="foo_node", type="FooNode"),
            target=Node(id="bar_node", type="BarNode"),
            type="CONNECTS_TO",
            properties={"key": "val"},
        )
    ],
    source=Document(page_content="source document content", metadata={"origin": "test"}),
)

# --- Connection Tests ---

@pytest.mark.usefixtures("clear_arangodb_database")
def test_connect_arangodb(db: StandardDatabase) -> None:
    """Test that ArangoGraph is correctly instantiated with a db object."""
    # Don't generate schema on init for this basic connection test
    graph = ArangoGraph(db=db, generate_schema_on_init=False)
    output = graph.query('RETURN "test" AS output')
    expected_output = [{"output": "test"}]
    assert output == expected_output

@pytest.mark.usefixtures("clear_arangodb_database")
def test_connect_arangodb_credentials(arangodb_credentials: ArangoCredentials) -> None:
    """Test connection using credentials via from_db_credentials."""
    # Need to mock generate_schema or ensure DB is empty
    with patch.object(ArangoGraph, "generate_schema", return_value={}):
        graph = ArangoGraph.from_db_credentials(**arangodb_credentials)
        output = graph.query('RETURN "test" AS output')
        expected_output = [{"output": "test"}]
        assert output == expected_output

# --- Schema Tests ---

@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_schema_generation(db: StandardDatabase) -> None:
    """Test schema generation reflects created collections and edges."""
    # Create collections and an edge definition within a graph
    graph_name = "test_schema_graph"
    db.create_collection("TestNodes")
    db.create_collection("MoreNodes")
    db.create_collection("TestEdges", edge=True)
    db.create_graph(
        graph_name,
        edge_definitions=[
            {
                "edge_collection": "TestEdges",
                "from_vertex_collections": ["TestNodes"],
                "to_vertex_collections": ["MoreNodes"],
            }
        ]
    )
    db.collection("TestNodes").insert({"_key": "t1", "propA": "valA", "propB": 1})
    db.collection("MoreNodes").insert({"_key": "m1", "propC": True})
    db.collection("TestEdges").insert({"_from": "TestNodes/t1", "_to": "MoreNodes/m1", "edge_prop": 1.5})

    # Instantiate ArangoGraph and generate schema
    adb_graph = ArangoGraph(db=db, generate_schema_on_init=False)
    adb_graph.refresh_schema(graph_name=graph_name) # Use specific graph

    schema = adb_graph.schema
    print("Generated Schema:", json.dumps(schema, indent=2))

    # Assert basic structure
    assert "node_collections" in schema
    assert "edge_collections" in schema
    assert "edge_definitions" in schema

    # Check node collections
    assert "TestNodes" in schema["node_collections"]
    assert "MoreNodes" in schema["node_collections"]
    # Check properties (types might vary based on ArangoDB version/inference)
    assert any(p["property"] == "propA" for p in schema["node_collections"]["TestNodes"])
    assert any(p["property"] == "propB" for p in schema["node_collections"]["TestNodes"])
    assert any(p["property"] == "propC" for p in schema["node_collections"]["MoreNodes"])

    # Check edge collections
    assert "TestEdges" in schema["edge_collections"]
    assert any(p["property"] == "edge_prop" for p in schema["edge_collections"]["TestEdges"])

    # Check edge definitions
    assert len(schema["edge_definitions"]) == 1
    assert schema["edge_definitions"][0]["edge_collection"] == "TestEdges"
    assert schema["edge_definitions"][0]["from_vertex_collections"] == ["TestNodes"]
    assert schema["edge_definitions"][0]["to_vertex_collections"] == ["MoreNodes"]


# --- Query Execution Tests ---

@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_query(db: StandardDatabase) -> None:
    """Test basic AQL query execution."""
    db.create_collection("TestQueryCol")
    db.collection("TestQueryCol").insert({"_key": "doc1", "value": 42})
    graph = ArangoGraph(db=db, generate_schema_on_init=False)
    result = graph.query("FOR doc IN TestQueryCol FILTER doc._key == 'doc1' RETURN doc.value")
    assert result == [42]

@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_query_error(db: StandardDatabase) -> None:
    """Test handling of AQL query errors."""
    graph = ArangoGraph(db=db, generate_schema_on_init=False)
    with pytest.raises(AQLQueryExecuteError):
        graph.query("FOR doc IN NonExistentCollection RETURN doc") # Invalid query

# --- Add GraphDocuments Tests ---

@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_add_graph_documents_single_entity(db: StandardDatabase) -> None:
    """Test adding graph documents using the single entity collection strategy."""
    graph = ArangoGraph(db=db, generate_schema_on_init=False)
    entity_col_name = "Entities"
    entity_edge_col_name = "RELATIONS"

    graph.add_graph_documents(
        [TEST_GRAPH_DOC],
        use_one_entity_collection=True,
        entity_collection_name=entity_col_name,
        entity_edge_collection_name=entity_edge_col_name,
        include_source=False
    )

    # Verify collections created
    assert db.has_collection(entity_col_name)
    assert db.has_collection(entity_edge_col_name)
    assert db.collection(entity_edge_col_name).properties()["edge"] is True

    # Verify data insertion (simple checks)
    node_count = db.collection(entity_col_name).count()
    edge_count = db.collection(entity_edge_col_name).count()
    assert node_count == 2
    assert edge_count == 1

    # Check one node content example (hashed key might vary)
    cursor = db.aql.execute(f"FOR doc IN {entity_col_name} FILTER doc.name == 'foo name' RETURN doc", count=True)
    results = list(cursor)
    assert len(results) == 1
    assert results[0]["lc_type"] == "FooNode"

    # Check edge content example
    cursor_edge = db.aql.execute(f"FOR edge IN {entity_edge_col_name} RETURN edge", count=True)
    edge_results = list(cursor_edge)
    assert len(edge_results) == 1
    assert edge_results[0]["lc_type"] == "CONNECTS_TO"
    assert edge_results[0]["key"] == "val"
    assert "_from" in edge_results[0]
    assert "_to" in edge_results[0]

@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_add_graph_documents_typed_collections(db: StandardDatabase) -> None:
    """Test adding graph documents using typed collections."""
    graph = ArangoGraph(db=db, generate_schema_on_init=False)

    graph.add_graph_documents(
        [TEST_GRAPH_DOC],
        use_one_entity_collection=False,
        include_source=False
    )

    # Verify collections created based on types
    assert db.has_collection("FooNode")
    assert db.has_collection("BarNode")
    assert db.has_collection("CONNECTS_TO")
    assert db.collection("CONNECTS_TO").properties()["edge"] is True

    # Verify data insertion
    assert db.collection("FooNode").count() == 1
    assert db.collection("BarNode").count() == 1
    assert db.collection("CONNECTS_TO").count() == 1

    # Check content
    foo_doc = db.collection("FooNode").all()[0]
    bar_doc = db.collection("BarNode").all()[0]
    edge_doc = db.collection("CONNECTS_TO").all()[0]
    assert foo_doc["name"] == "foo name"
    assert bar_doc["value"] == 10
    assert edge_doc["key"] == "val"
    assert edge_doc["_from"] == foo_doc["_id"]
    assert edge_doc["_to"] == bar_doc["_id"]

@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_add_graph_documents_with_source(db: StandardDatabase) -> None:
    """Test adding graph documents including the source document."""
    graph = ArangoGraph(db=db, generate_schema_on_init=False)
    source_col = "SourceDocs"
    source_edge_col = "HAS_SOURCE"

    graph.add_graph_documents(
        [TEST_GRAPH_DOC],
        use_one_entity_collection=False,
        include_source=True,
        source_collection_name=source_col,
        source_edge_collection_name=source_edge_col
    )

    # Verify node/edge collections + source collections
    assert db.has_collection("FooNode")
    assert db.has_collection("BarNode")
    assert db.has_collection("CONNECTS_TO")
    assert db.has_collection(source_col)
    assert db.has_collection(source_edge_col)
    assert db.collection(source_edge_col).properties()["edge"] is True

    # Verify counts
    assert db.collection("FooNode").count() == 1
    assert db.collection("BarNode").count() == 1
    assert db.collection("CONNECTS_TO").count() == 1
    assert db.collection(source_col).count() == 1
    # Check source edges (one for each node/rel)
    assert db.collection(source_edge_col).count() == 3

    # Check source document content
    source_doc = db.collection(source_col).all()[0]
    assert source_doc["lc_content"] == "source document content"
    assert source_doc["lc_metadata"]["origin"] == "test"

# Note: Skipping Neo4j-specific tests like sanitize, baseEntityLabel, constraints, enhanced_schema, backticks handling (covered by sanitize method unit test)
# Note: Connection lifecycle tests are less relevant as ArangoGraph uses external db object. 