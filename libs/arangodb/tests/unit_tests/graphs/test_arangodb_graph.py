from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from arango.request import Request
from arango.response import Response
from arango import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import ArangoServerError, ArangoClientError, ServerConnectionError
from langchain_arangodb.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_arangodb.graphs.arangodb_graph import ArangoGraph


@pytest.fixture
def mock_arangodb_driver() -> Generator[MagicMock, None, None]:
    with patch("arango.ArangoClient", autospec=True) as mock_client:
        mock_db = MagicMock()
        mock_client.return_value.db.return_value = mock_db
        mock_db.verify = MagicMock(return_value=True)
        mock_db.aql = MagicMock()
        mock_db.aql.execute = MagicMock(
            return_value=MagicMock(
                batch=lambda: [], count=lambda: 0
            )
        )
        mock_db._is_closed = False
        yield mock_db


# uses close method
# def test_driver_state_management(mock_arangodb_driver):
#     # Initialize ArangoGraph with the mocked database
#     graph = ArangoGraph(mock_arangodb_driver)

#     # Store original driver
#     original_driver = graph.db

#     # Test initial state
#     assert hasattr(graph, "db")

#     # First close
#     graph.close()
#     assert not hasattr(graph, "db")

#     # Verify methods raise error when driver is closed
#     with pytest.raises(
#         RuntimeError,
#         match="Cannot perform operations - ArangoDB connection has been closed",
#     ):
#         graph.query("RETURN 1")

#     with pytest.raises(
#         RuntimeError,
#         match="Cannot perform operations - ArangoDB connection has been closed",
#     ):
#         graph.refresh_schema()


# uses close method
# def test_arangograph_del_method() -> None:
#     """Test the __del__ method of ArangoGraph."""
#     with patch.object(ArangoGraph, "close") as mock_close:
#         graph = ArangoGraph(db=None)  # Assuming db can be None or a mock
#         mock_close.side_effect = Exception("Simulated exception during close")
#         mock_close.assert_not_called()
#         graph.__del__()
#         mock_close.assert_called_once()

# uses close method
# def test_close_method_removes_driver(mock_neo4j_driver: MagicMock) -> None:
#     """Test that close method removes the _driver attribute."""
#     graph = Neo4jGraph(
#         url="bolt://localhost:7687", username="neo4j", password="password"
#     )

#     # Store a reference to the original driver
#     original_driver = graph._driver
#     assert isinstance(original_driver.close, MagicMock)

#     # Call close method
#     graph.close()

#     # Verify driver.close was called
#     original_driver.close.assert_called_once()

#     # Verify _driver attribute is removed
#     assert not hasattr(graph, "_driver")

#     # Verify second close does not raise an error
#     graph.close()  # Should not raise any exception

# uses close method
# def test_multiple_close_calls_safe(mock_neo4j_driver: MagicMock) -> None:
#     """Test that multiple close calls do not raise errors."""
#     graph = Neo4jGraph(
#         url="bolt://localhost:7687", username="neo4j", password="password"
#     )

#     # Store a reference to the original driver
#     original_driver = graph._driver
#     assert isinstance(original_driver.close, MagicMock)

#     # First close
#     graph.close()
#     original_driver.close.assert_called_once()

#     # Verify _driver attribute is removed
#     assert not hasattr(graph, "_driver")

#     # Second close should not raise an error
#     graph.close()  # Should not raise any exception



def test_arangograph_init_with_empty_credentials() -> None:
    """Test initializing ArangoGraph with empty credentials."""
    with patch.object(ArangoClient, 'db', autospec=True) as mock_db_method:
        mock_db_instance = MagicMock()
        mock_db_method.return_value = mock_db_instance

        # Initialize ArangoClient and ArangoGraph with empty credentials
        client = ArangoClient()
        db = client.db("_system", username="", password="", verify=False)
        graph = ArangoGraph(db=db)

        # Assert that ArangoClient.db was called with empty username and password
        mock_db_method.assert_called_with(client, "_system", username="", password="", verify=False)

        # Assert that the graph instance was created successfully
        assert isinstance(graph, ArangoGraph)


def test_arangograph_init_with_invalid_credentials():
    """Test initializing ArangoGraph with incorrect credentials raises ArangoServerError."""
    # Create mock request and response objects
    mock_request = MagicMock(spec=Request)
    mock_response = MagicMock(spec=Response)

    # Initialize the client
    client = ArangoClient()

    # Patch the 'db' method of the ArangoClient instance
    with patch.object(client, 'db') as mock_db_method:
        # Configure the mock to raise ArangoServerError when called
        mock_db_method.side_effect = ArangoServerError(mock_response, mock_request, "bad username/password or token is expired")

        # Attempt to connect with invalid credentials and verify that the appropriate exception is raised
        with pytest.raises(ArangoServerError) as exc_info:
            db = client.db("_system", username="invalid_user", password="invalid_pass", verify=True)
            graph = ArangoGraph(db=db)

        # Assert that the exception message contains the expected text
        assert "bad username/password or token is expired" in str(exc_info.value)



def test_arangograph_init_missing_collection():
    """Test initializing ArangoGraph when a required collection is missing."""
    # Create mock response and request objects
    mock_response = MagicMock()
    mock_response.error_message = "collection not found"
    mock_response.status_text = "Not Found"
    mock_response.status_code = 404
    mock_response.error_code = 1203  # Example error code for collection not found

    mock_request = MagicMock()
    mock_request.method = "GET"
    mock_request.endpoint = "/_api/collection/missing_collection"

    # Patch the 'db' method of the ArangoClient instance
    with patch.object(ArangoClient, 'db') as mock_db_method:
        # Configure the mock to raise ArangoServerError when called
        mock_db_method.side_effect = ArangoServerError(
            resp=mock_response,
            request=mock_request,
            msg="collection not found"
        )

        # Initialize the client
        client = ArangoClient()

        # Attempt to connect and verify that the appropriate exception is raised
        with pytest.raises(ArangoServerError) as exc_info:
            db = client.db("_system", username="user", password="pass", verify=True)
            graph = ArangoGraph(db=db)

        # Assert that the exception message contains the expected text
        assert "collection not found" in str(exc_info.value)



@patch.object(ArangoGraph, "generate_schema")
def test_arangograph_init_refresh_schema_other_err(mock_generate_schema, socket_enabled):
    """Test that unexpected ArangoServerError during generate_schema in __init__ is re-raised."""
    # Create mock response and request objects
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.error_code = 1234
    mock_response.error_message = "Unexpected error"

    mock_request = MagicMock()

    # Configure the mock to raise ArangoServerError when called
    mock_generate_schema.side_effect = ArangoServerError(
        resp=mock_response,
        request=mock_request,
        msg="Unexpected error"
    )

    # Create a mock db object
    mock_db = MagicMock()

    # Attempt to initialize ArangoGraph and verify that the exception is re-raised
    with pytest.raises(ArangoServerError) as exc_info:
        ArangoGraph(db=mock_db)

    # Assert that the raised exception has the expected attributes
    assert exc_info.value.error_message == "Unexpected error"
    assert exc_info.value.error_code == 1234

    

def test_query_fallback_execution(socket_enabled):
    """Test the fallback mechanism when a collection is not found."""
    # Initialize the ArangoDB client and connect to the database
    client = ArangoClient()
    db = client.db("_system", username="root", password="test")

    # Define a query that accesses a non-existent collection
    query = "FOR doc IN unregistered_collection RETURN doc"

    # Patch the db.aql.execute method to raise ArangoServerError
    with patch.object(db.aql, "execute") as mock_execute:
        error = ArangoServerError(
            resp=MagicMock(),
            request=MagicMock(),
            msg="collection or view not found: unregistered_collection"
        )
        error.error_code = 1203  # ERROR_ARANGO_DATA_SOURCE_NOT_FOUND
        mock_execute.side_effect = error

        # Initialize the ArangoGraph
        graph = ArangoGraph(db=db)

        # Attempt to execute the query and verify that the appropriate exception is raised
        with pytest.raises(ArangoServerError) as exc_info:
            graph.query(query)

        # Assert that the raised exception has the expected error code and message
        assert exc_info.value.error_code == 1203
        assert "collection or view not found" in str(exc_info.value)

@patch.object(ArangoGraph, "generate_schema")
def test_refresh_schema_handles_arango_server_error(mock_generate_schema, socket_enabled):
    """Test that generate_schema handles ArangoServerError gracefully."""

    # Configure the mock to raise ArangoServerError when called
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.error_code = 1234
    mock_response.error_message = "Forbidden: insufficient permissions"

    mock_request = MagicMock()

    mock_generate_schema.side_effect = ArangoServerError(
        resp=mock_response,
        request=mock_request,
        msg="Forbidden: insufficient permissions"
    )

    # Initialize the client
    client = ArangoClient()
    db = client.db("_system", username="root", password="test", verify=True)

    # Attempt to initialize ArangoGraph and verify that the exception is re-raised
    with pytest.raises(ArangoServerError) as exc_info:
        ArangoGraph(db=db)

    # Assert that the raised exception has the expected attributes
    assert exc_info.value.error_message == "Forbidden: insufficient permissions"
    assert exc_info.value.error_code == 1234

@patch.object(ArangoGraph, "refresh_schema")
def test_get_schema(mock_refresh_schema, socket_enabled):
    """Test the schema property of ArangoGraph."""
    # Initialize the ArangoDB client and connect to the database
    client = ArangoClient()
    db = client.db("_system", username="root", password="test")

    # Initialize the ArangoGraph with refresh_schema patched
    graph = ArangoGraph(db=db)

    # Define the test schema
    test_schema = {
        "collection_schema": [{"collection_name": "TestCollection", "collection_type": "document"}],
        "graph_schema": [{"graph_name": "TestGraph", "edge_definitions": []}]
    }

    # Manually set the internal schema
    graph._ArangoGraph__schema = test_schema

    # Assert that the schema property returns the expected dictionary
    assert graph.schema == test_schema


# def test_add_graph_docs_inc_src_err(mock_arangodb_driver: MagicMock) -> None:
#     """Tests an error is raised when using add_graph_documents with include_source set
#     to True and a document is missing a source."""
#     graph = ArangoGraph(db=mock_arangodb_driver)
    
#     node_1 = Node(id=1)
#     node_2 = Node(id=2)
#     rel = Relationship(source=node_1, target=node_2, type="REL")

#     graph_doc = GraphDocument(
#         nodes=[node_1, node_2],
#         relationships=[rel],
#     )
    
#     with pytest.raises(TypeError) as exc_info:
#         graph.add_graph_documents(graph_documents=[graph_doc], include_source=True)

#     assert (
#         "include_source is set to True, but at least one document has no `source`."
#         in str(exc_info.value)
#     )


def test_add_graph_docs_inc_src_err(mock_arangodb_driver: MagicMock) -> None:
    """Test that an error is raised when using add_graph_documents with include_source=True and a document is missing a source."""
    graph = ArangoGraph(db=mock_arangodb_driver)

    node_1 = Node(id=1)
    node_2 = Node(id=2)
    rel = Relationship(source=node_1, target=node_2, type="REL")

    graph_doc = GraphDocument(
        nodes=[node_1, node_2],
        relationships=[rel],
    )

    with pytest.raises(ValueError) as exc_info:
        graph.add_graph_documents(
            graph_documents=[graph_doc],
            include_source=True,
            capitalization_strategy="lower"
        )

    assert "Source document is required." in str(exc_info.value)


def test_add_graph_docs_invalid_capitalization_strategy():
    """Test error when an invalid capitalization_strategy is provided."""
    # Mock the ArangoDB driver
    mock_arangodb_driver = MagicMock()

    # Initialize ArangoGraph with the mocked driver
    graph = ArangoGraph(db=mock_arangodb_driver)

    # Create nodes and a relationship
    node_1 = Node(id=1)
    node_2 = Node(id=2)
    rel = Relationship(source=node_1, target=node_2, type="REL")

    # Create a GraphDocument
    graph_doc = GraphDocument(
        nodes=[node_1, node_2],
        relationships=[rel],
        source={"page_content": "Sample content"}  # Provide a dummy source
    )

    # Expect a ValueError when an invalid capitalization_strategy is provided
    with pytest.raises(ValueError) as exc_info:
        graph.add_graph_documents(
            graph_documents=[graph_doc],
            capitalization_strategy="invalid_strategy"
        )

    assert (
        "**capitalization_strategy** must be 'lower', 'upper', or 'none'."
        in str(exc_info.value)
    )

