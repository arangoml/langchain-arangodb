"""Test ArangoDB graph functionality."""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from arango.database import StandardDatabase
from arango.exceptions import (
    AQLQueryExecuteError, # Used for query errors
    AuthError,
    GraphCreateError, # For graph creation issues
    DocumentInsertError, # For insertion issues
    CollectionCreateError, # For collection creation issues
)

from langchain_arangodb.graphs.graph_document import (
    Document, # Assuming this is compatible/shared
    GraphDocument,
    Node,
    Relationship,
)
from langchain_arangodb.graphs.arangodb_graph import ArangoGraph, get_arangodb_client


@pytest.fixture
def mock_arango_std_db() -> MagicMock:
    """Fixture for a mocked ArangoDB StandardDatabase instance."""
    mock_db = MagicMock(spec=StandardDatabase)
    mock_db.username = "test_user"
    # Mock methods used by ArangoGraph
    mock_db.begin_async_execution.return_value = mock_db # Mock async db
    mock_db.has_collection.return_value = True
    mock_db.has_graph.return_value = False
    mock_db.create_graph.return_value = MagicMock()
    mock_db.create_collection.return_value = MagicMock()
    mock_db.collection.return_value = MagicMock() # Mock collection object
    mock_db.graph.return_value = MagicMock() # Mock graph object
    mock_db.aql = MagicMock()
    mock_db.aql.execute.return_value = MagicMock() # Mock AQL cursor
    mock_db.aql.explain.return_value = {} # Mock explain result
    mock_db.collections.return_value = [] # Mock collections list for schema gen
    return mock_db

# --- Initialization and Connection Tests ---

def test_arangograph_init_and_properties(mock_arango_std_db: MagicMock) -> None:
    """Test basic initialization and property access."""
    # Mock schema generation during init
    with patch.object(ArangoGraph, "generate_schema", return_value={"mock": "schema"}) as mock_gen_schema:
        graph = ArangoGraph(db=mock_arango_std_db)
        mock_gen_schema.assert_called_once() # Check schema gen called by default

    assert graph.db == mock_arango_std_db
    assert graph.schema == {"mock": "schema"}
    assert graph.get_structured_schema == {"mock": "schema"}
    assert graph.schema_json == '{"mock": "schema"}'
    assert "mock: schema" in graph.schema_yaml

def test_arangograph_init_no_schema_gen(mock_arango_std_db: MagicMock) -> None:
    """Test initialization with schema generation disabled."""
    with patch.object(ArangoGraph, "generate_schema") as mock_gen_schema:
        graph = ArangoGraph(db=mock_arango_std_db, generate_schema_on_init=False)
        mock_gen_schema.assert_not_called()
    assert graph.schema == {}

def test_set_schema(mock_arango_std_db: MagicMock) -> None:
    """Test setting a custom schema."""
    graph = ArangoGraph(db=mock_arango_std_db, generate_schema_on_init=False)
    custom_schema = {"nodes": [], "edges": []}
    graph.set_schema(custom_schema)
    assert graph.schema == custom_schema

def test_arangograph_init_auth_error(mock_arango_std_db: MagicMock) -> None:
    """Test initialization raises error if DB connection has issues (e.g., auth)."""
    # Simulate error during schema generation triggered by init
    with patch.object(ArangoGraph, "generate_schema", side_effect=AuthError("Auth failed", http_exception=None)):
        with pytest.raises(AuthError):
            ArangoGraph(db=mock_arango_std_db)

# Mock get_arangodb_client for testing from_db_credentials
@patch("langchain_arangodb.graphs.arangodb_graph.get_arangodb_client")
def test_from_db_credentials(mock_get_client: MagicMock, mock_arango_std_db: MagicMock) -> None:
    """Test the from_db_credentials class method."""
    mock_get_client.return_value = mock_arango_std_db
    with patch.object(ArangoGraph, "generate_schema", return_value={}) as mock_gen_schema:
        graph = ArangoGraph.from_db_credentials(
            url="http://mockurl:8529",
            dbname="mockdb",
            username="mockuser",
            password="mockpass",
        )
        mock_get_client.assert_called_once_with(
            url="http://mockurl:8529",
            dbname="mockdb",
            username="mockuser",
            password="mockpass",
        )
        assert isinstance(graph, ArangoGraph)
        assert graph.db == mock_arango_std_db
        mock_gen_schema.assert_called_once() # Schema gen should be called

# --- Query Execution Tests ---

def test_query_execution(mock_arango_std_db: MagicMock) -> None:
    """Test successful query execution."""
    graph = ArangoGraph(db=mock_arango_std_db, generate_schema_on_init=False)
    mock_cursor = MagicMock()
    expected_result = [{"_key": "123", "value": "abc"}]
    mock_cursor.__iter__.return_value = iter(expected_result)
    mock_arango_std_db.aql.execute.return_value = mock_cursor

    query = "FOR doc IN nodes RETURN doc"
    params = {"limit": 10}
    result = graph.query(query, params)

    assert result == expected_result
    mock_arango_std_db.aql.execute.assert_called_once_with(query, bind_vars=params, count=True, batch_size=1000)

def test_query_execution_error(mock_arango_std_db: MagicMock) -> None:
    """Test query execution raising an error."""
    graph = ArangoGraph(db=mock_arango_std_db, generate_schema_on_init=False)
    error_message = "Syntax error near ..."
    mock_arango_std_db.aql.execute.side_effect = AQLQueryExecuteError(error_message, http_exception=None, error_code=1501)

    query = "FOR doc IN RETURN doc" # Invalid query
    with pytest.raises(AQLQueryExecuteError):
        graph.query(query)

# --- Explain Tests ---

def test_explain_execution(mock_arango_std_db: MagicMock) -> None:
    """Test successful explain execution."""
    graph = ArangoGraph(db=mock_arango_std_db, generate_schema_on_init=False)
    expected_plan = {"plan": {"nodes": []}}
    mock_arango_std_db.aql.explain.return_value = expected_plan

    query = "FOR doc IN nodes RETURN doc"
    params = {"limit": 10}
    result = graph.explain(query, params)

    assert result == expected_plan
    mock_arango_std_db.aql.explain.assert_called_once_with(query, bind_vars=params, all_plans=False, opt_rules=["-all", "+use-indexes"])

# --- Schema Refresh Tests ---

def test_refresh_schema(mock_arango_std_db: MagicMock) -> None:
    """Test the refresh_schema method."""
    new_schema = {"refreshed": "schema"}
    with patch.object(ArangoGraph, "generate_schema", return_value=new_schema) as mock_gen_schema:
        graph = ArangoGraph(db=mock_arango_std_db)
        # Call refresh
        graph.refresh_schema(sample_ratio=0.1, graph_name="mygraph", include_examples=False)
        # Assert generate_schema was called with refresh args
        mock_gen_schema.assert_called_with(0.1, "mygraph", False, 32, 256) # list_limit uses default
        assert graph.schema == new_schema

# --- Add Graph Documents Tests (Simplified Adaptation) ---
# Note: Full testing of add_graph_documents is complex due to its many options.
# This is a basic adaptation of the Neo4j test structure.

@patch("langchain_arangodb.graphs.arangodb_graph.ArangoGraph._import_data")
def test_add_graph_documents_simple(mock_import: MagicMock, mock_arango_std_db: MagicMock) -> None:
    """Test basic add_graph_documents call."""
    graph = ArangoGraph(db=mock_arango_std_db, generate_schema_on_init=False)
    
    # Create mock GraphDocument data
    node1 = Node(id="N1", type="Person", properties={"name": "Alice"})
    node2 = Node(id="N2", type="Person", properties={"name": "Bob"})
    rel1 = Relationship(source=node1, target=node2, type="FRIENDS")
    doc = Document(page_content="Alice and Bob are friends", metadata={})
    graph_doc = GraphDocument(nodes=[node1, node2], relationships=[rel1], source=doc)
    
    graph.add_graph_documents([graph_doc], include_source=False, graph_name="test_graph")
    
    # Check if _import_data was called (internal method for insertion)
    # Need to check calls for nodes and edges
    assert mock_import.call_count > 0 # Basic check, actual calls depend on strategy
    # More specific checks would involve inspecting the arguments passed to mock_import,
    # which depend heavily on the internal logic and strategy used (e.g., use_one_entity_collection).

@patch("langchain_arangodb.graphs.arangodb_graph.ArangoGraph._import_data")
def test_add_graph_documents_with_source(mock_import: MagicMock, mock_arango_std_db: MagicMock) -> None:
    """Test add_graph_documents with include_source=True."""
    graph = ArangoGraph(db=mock_arango_std_db, generate_schema_on_init=False)
    
    node1 = Node(id="N1", type="City", properties={"name": "Paris"})
    source_doc = Document(page_content="Paris is the capital of France.", metadata={"source_id": "wiki123"})
    graph_doc = GraphDocument(nodes=[node1], relationships=[], source=source_doc)
    
    with patch.object(ArangoGraph, "_process_source") as mock_process_source:
        graph.add_graph_documents(
            [graph_doc], 
            include_source=True, 
            source_collection_name="Sources",
            source_edge_collection_name="HAS_SOURCE"
        )
        mock_process_source.assert_called_once()
        # Check _import_data for nodes/edges as well
        assert mock_import.call_count > 0

def test_add_graph_documents_error_handling(mock_arango_std_db: MagicMock) -> None:
    """Test error handling during document insertion (simplified)."""
    graph = ArangoGraph(db=mock_arango_std_db, generate_schema_on_init=False)
    node1 = Node(id="N1", type="Person")
    graph_doc = GraphDocument(nodes=[node1], relationships=[], source=Document(page_content=" "))

    # Mock an error during the internal _import_data call
    with patch.object(ArangoGraph, "_import_data", side_effect=DocumentInsertError("Insert failed", http_exception=None)):
        # Depending on internal handling, it might raise or log. Assuming it raises for now.
        with pytest.raises(DocumentInsertError):
            graph.add_graph_documents([graph_doc])

# --- Helper Method Tests (Example: _sanitize_collection_name) ---

def test_sanitize_collection_name(mock_arango_std_db: MagicMock) -> None:
    """Test collection name sanitization."""
    graph = ArangoGraph(db=mock_arango_std_db, generate_schema_on_init=False)
    assert graph._sanitize_collection_name("ValidName") == "ValidName"
    assert graph._sanitize_collection_name("Invalid Name") == "Invalid_Name"
    assert graph._sanitize_collection_name("1StartsWithNumber") == "_1StartsWithNumber"
    assert graph._sanitize_collection_name("-StartsWithDash") == "_StartsWithDash"
    assert graph._sanitize_collection_name("With-Special$") == "With_Special_"

# Note: Need more tests for:
# - Detailed schema generation logic (sampling, examples, limits)
# - Different strategies in add_graph_documents (use_one_entity_collection, embeddings)
# - Error handling for graph/collection creation
# - Async operations 