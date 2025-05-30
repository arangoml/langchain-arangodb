import json
import os
import pprint
from collections import defaultdict
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import yaml
from arango import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import (
    ArangoClientError,
    ArangoServerError,
    ServerConnectionError,
)
from arango.request import Request
from arango.response import Response

from langchain_arangodb.graphs.arangodb_graph import ArangoGraph, get_arangodb_client
from langchain_arangodb.graphs.graph_document import (
    Document,
    GraphDocument,
    Node,
    Relationship,
)


@pytest.fixture
def mock_arangodb_driver() -> Generator[MagicMock, None, None]:
    with patch("arango.ArangoClient", autospec=True) as mock_client:
        mock_db = MagicMock()
        mock_client.return_value.db.return_value = mock_db
        mock_db.verify = MagicMock(return_value=True)
        mock_db.aql = MagicMock()
        mock_db.aql.execute = MagicMock(
            return_value=MagicMock(batch=lambda: [], count=lambda: 0)
        )
        mock_db._is_closed = False
        yield mock_db


# --------------------------------------------------------------------------- #
# 1. Direct arguments only
# --------------------------------------------------------------------------- #
@patch("langchain_arangodb.graphs.arangodb_graph.ArangoClient")
def test_get_client_with_all_args(mock_client_cls):
    mock_db = MagicMock()
    mock_client = MagicMock()
    mock_client.db.return_value = mock_db
    mock_client_cls.return_value = mock_client

    result = get_arangodb_client(
        url="http://myhost:1234",
        dbname="testdb",
        username="admin",
        password="pass123",
    )

    mock_client_cls.assert_called_with("http://myhost:1234")
    mock_client.db.assert_called_with("testdb", "admin", "pass123", verify=True)
    assert result is mock_db


# --------------------------------------------------------------------------- #
# 2. Values pulled from environment variables
# --------------------------------------------------------------------------- #
@patch.dict(
    os.environ,
    {
        "ARANGODB_URL": "http://envhost:9999",
        "ARANGODB_DBNAME": "envdb",
        "ARANGODB_USERNAME": "envuser",
        "ARANGODB_PASSWORD": "envpass",
    },
    clear=True,
)
@patch("langchain_arangodb.graphs.arangodb_graph.ArangoClient")
def test_get_client_from_env(mock_client_cls):
    mock_db = MagicMock()
    mock_client = MagicMock()
    mock_client.db.return_value = mock_db
    mock_client_cls.return_value = mock_client

    result = get_arangodb_client()  # no args; should fall back on env

    mock_client_cls.assert_called_with("http://envhost:9999")
    mock_client.db.assert_called_with("envdb", "envuser", "envpass", verify=True)
    assert result is mock_db


# --------------------------------------------------------------------------- #
# 3. Defaults when no args and no env vars
# --------------------------------------------------------------------------- #
@patch("langchain_arangodb.graphs.arangodb_graph.ArangoClient")
def test_get_client_with_defaults(mock_client_cls):
    # Ensure env vars are absent
    for var in (
        "ARANGODB_URL",
        "ARANGODB_DBNAME",
        "ARANGODB_USERNAME",
        "ARANGODB_PASSWORD",
    ):
        os.environ.pop(var, None)

    mock_db = MagicMock()
    mock_client = MagicMock()
    mock_client.db.return_value = mock_db
    mock_client_cls.return_value = mock_client

    result = get_arangodb_client()

    mock_client_cls.assert_called_with("http://localhost:8529")
    mock_client.db.assert_called_with("_system", "root", "", verify=True)
    assert result is mock_db


# --------------------------------------------------------------------------- #
# 4. Propagate ArangoServerError on bad credentials (or any server failure)
# --------------------------------------------------------------------------- #
@patch("langchain_arangodb.graphs.arangodb_graph.ArangoClient")
def test_get_client_invalid_credentials_raises(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_request = MagicMock(spec=Request)
    mock_response = MagicMock(spec=Response)
    mock_client.db.side_effect = ArangoServerError(
        resp=mock_response,
        request=mock_request,
        msg="Authentication failed",
    )

    with pytest.raises(ArangoServerError, match="Authentication failed"):
        get_arangodb_client(
            url="http://localhost:8529",
            dbname="_system",
            username="bad_user",
            password="bad_pass",
        )


@pytest.fixture
def graph():
    return ArangoGraph(db=MagicMock())


class DummyCursor:
    def __iter__(self):
        yield {"name": "Alice", "tags": ["friend", "colleague"], "age": 30}


class TestArangoGraph:
    def setup_method(self):
        self.mock_db = MagicMock()
        self.graph = ArangoGraph(db=self.mock_db)
        self.graph._sanitize_input = MagicMock(
            return_value={"name": "Alice", "tags": "List of 2 elements", "age": 30}
        )

    def test_get_structured_schema_returns_correct_schema(
        self, mock_arangodb_driver: MagicMock
    ):
        # Create mock db to pass to ArangoGraph
        mock_db = MagicMock()

        # Initialize ArangoGraph
        graph = ArangoGraph(db=mock_db)

        # Manually set the private __schema attribute
        test_schema = {
            "collection_schema": [
                {"collection_name": "Users", "collection_type": "document"},
                {"collection_name": "Orders", "collection_type": "document"},
            ],
            "graph_schema": [{"graph_name": "UserOrderGraph", "edge_definitions": []}],
        }
        graph._ArangoGraph__schema = (
            test_schema  # Accessing name-mangled private attribute
        )

        # Access the property
        result = graph.get_structured_schema

        # Assert that the returned schema matches what we set
        assert result == test_schema

    def test_arangograph_init_with_empty_credentials(
        self, mock_arangodb_driver: MagicMock
    ) -> None:
        """Test initializing ArangoGraph with empty credentials."""
        with patch.object(ArangoClient, "db", autospec=True) as mock_db_method:
            mock_db_instance = MagicMock()
            mock_db_method.return_value = mock_db_instance

            # Initialize ArangoClient and ArangoGraph with empty credentials
            # client = ArangoClient()
            # db = client.db("_system", username="", password="", verify=False)
            graph = ArangoGraph(db=mock_arangodb_driver)

            # Assert that ArangoClient.db was called with empty username and password
            # mock_db_method.assert_called_with(client, "_system", username="", password="", verify=False)

            # Assert that the graph instance was created successfully
            assert isinstance(graph, ArangoGraph)

    def test_arangograph_init_with_invalid_credentials(self):
        """Test initializing ArangoGraph with incorrect credentials raises ArangoServerError."""
        # Create mock request and response objects
        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)

        # Initialize the client
        client = ArangoClient()

        # Patch the 'db' method of the ArangoClient instance
        with patch.object(client, "db") as mock_db_method:
            # Configure the mock to raise ArangoServerError when called
            mock_db_method.side_effect = ArangoServerError(
                mock_response, mock_request, "bad username/password or token is expired"
            )

            # Attempt to connect with invalid credentials and verify that the appropriate exception is raised
            with pytest.raises(ArangoServerError) as exc_info:
                db = client.db(
                    "_system",
                    username="invalid_user",
                    password="invalid_pass",
                    verify=True,
                )
                graph = ArangoGraph(db=db)

            # Assert that the exception message contains the expected text
            assert "bad username/password or token is expired" in str(exc_info.value)

    def test_arangograph_init_missing_collection(self):
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
        with patch.object(ArangoClient, "db") as mock_db_method:
            # Configure the mock to raise ArangoServerError when called
            mock_db_method.side_effect = ArangoServerError(
                resp=mock_response, request=mock_request, msg="collection not found"
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
    def test_arangograph_init_refresh_schema_other_err(
        self, mock_generate_schema, mock_arangodb_driver
    ):
        """Test that unexpected ArangoServerError during generate_schema in __init__ is re-raised."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.error_code = 1234
        mock_response.error_message = "Unexpected error"

        mock_request = MagicMock()

        mock_generate_schema.side_effect = ArangoServerError(
            resp=mock_response, request=mock_request, msg="Unexpected error"
        )

        with pytest.raises(ArangoServerError) as exc_info:
            ArangoGraph(db=mock_arangodb_driver)

        assert exc_info.value.error_message == "Unexpected error"
        assert exc_info.value.error_code == 1234

    def test_query_fallback_execution(self, mock_arangodb_driver: MagicMock):
        """Test the fallback mechanism when a collection is not found."""
        query = "FOR doc IN unregistered_collection RETURN doc"

        with patch.object(mock_arangodb_driver.aql, "execute") as mock_execute:
            error = ArangoServerError(
                resp=MagicMock(),
                request=MagicMock(),
                msg="collection or view not found: unregistered_collection",
            )
            error.error_code = 1203
            mock_execute.side_effect = error

            graph = ArangoGraph(db=mock_arangodb_driver)

            with pytest.raises(ArangoServerError) as exc_info:
                graph.query(query)

            assert exc_info.value.error_code == 1203
            assert "collection or view not found" in str(exc_info.value)

    @patch.object(ArangoGraph, "generate_schema")
    def test_refresh_schema_handles_arango_server_error(
        self, mock_generate_schema, mock_arangodb_driver: MagicMock
    ):
        """Test that generate_schema handles ArangoServerError gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.error_code = 1234
        mock_response.error_message = "Forbidden: insufficient permissions"

        mock_request = MagicMock()

        mock_generate_schema.side_effect = ArangoServerError(
            resp=mock_response,
            request=mock_request,
            msg="Forbidden: insufficient permissions",
        )

        with pytest.raises(ArangoServerError) as exc_info:
            ArangoGraph(db=mock_arangodb_driver)

        assert exc_info.value.error_message == "Forbidden: insufficient permissions"
        assert exc_info.value.error_code == 1234

    @patch.object(ArangoGraph, "refresh_schema")
    def test_get_schema(mock_refresh_schema, mock_arangodb_driver: MagicMock):
        """Test the schema property of ArangoGraph."""
        graph = ArangoGraph(db=mock_arangodb_driver)

        test_schema = {
            "collection_schema": [
                {"collection_name": "TestCollection", "collection_type": "document"}
            ],
            "graph_schema": [{"graph_name": "TestGraph", "edge_definitions": []}],
        }

        graph._ArangoGraph__schema = test_schema
        assert graph.schema == test_schema

    def test_add_graph_docs_inc_src_err(self, mock_arangodb_driver: MagicMock) -> None:
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
                capitalization_strategy="lower",
            )

        assert "Source document is required." in str(exc_info.value)

    def test_add_graph_docs_invalid_capitalization_strategy(
        self, mock_arangodb_driver: MagicMock
    ):
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
            source={"page_content": "Sample content"},  # Provide a dummy source
        )

        # Expect a ValueError when an invalid capitalization_strategy is provided
        with pytest.raises(ValueError) as exc_info:
            graph.add_graph_documents(
                graph_documents=[graph_doc], capitalization_strategy="invalid_strategy"
            )

        assert (
            "**capitalization_strategy** must be 'lower', 'upper', or 'none'."
            in str(exc_info.value)
        )

    def test_process_edge_as_type_full_flow(self):
        # Setup ArangoGraph and mock _sanitize_collection_name
        graph = ArangoGraph(db=MagicMock())
        graph._sanitize_collection_name = lambda x: f"sanitized_{x}"

        # Create source and target nodes
        source = Node(id="s1", type="User")
        target = Node(id="t1", type="Item")

        # Create an edge with properties
        edge = Relationship(
            source=source,
            target=target,
            type="LIKES",
            properties={"weight": 0.9, "timestamp": "2024-01-01"},
        )

        # Inputs
        edge_str = "User likes Item"
        edge_key = "e123"
        source_key = "s123"
        target_key = "t123"

        edges = defaultdict(list)
        edge_defs = defaultdict(lambda: defaultdict(set))

        # Call method
        graph._process_edge_as_type(
            edge=edge,
            edge_str=edge_str,
            edge_key=edge_key,
            source_key=source_key,
            target_key=target_key,
            edges=edges,
            _1="ignored_1",
            _2="ignored_2",
            edge_definitions_dict=edge_defs,
        )

        # Check edge_definitions_dict was updated
        assert edge_defs["sanitized_LIKES"]["from_vertex_collections"] == {
            "sanitized_User"
        }
        assert edge_defs["sanitized_LIKES"]["to_vertex_collections"] == {
            "sanitized_Item"
        }

        # Check edge document appended correctly
        assert edges["sanitized_LIKES"][0] == {
            "_key": "e123",
            "_from": "sanitized_User/s123",
            "_to": "sanitized_Item/t123",
            "text": "User likes Item",
            "weight": 0.9,
            "timestamp": "2024-01-01",
        }

    def test_add_graph_documents_full_flow(self, graph):
        # Mocks
        graph._create_collection = MagicMock()
        graph._hash = lambda x: f"hash_{x}"
        graph._process_source = MagicMock(return_value="hash_source_id")
        graph._import_data = MagicMock()
        graph.refresh_schema = MagicMock()
        graph._process_node_as_entity = MagicMock(return_value="ENTITY")
        graph._process_edge_as_entity = MagicMock()
        graph._get_node_key = MagicMock(side_effect=lambda n, *_: f"hash_{n.id}")
        graph.db.has_graph.return_value = False
        graph.db.create_graph = MagicMock()

        # Embedding mock
        embedding = MagicMock()
        embedding.embed_documents.return_value = [[[0.1, 0.2, 0.3]]]

        # Build GraphDocument
        node1 = Node(id="N1", type="Person", properties={})
        node2 = Node(id="N2", type="Company", properties={})
        edge = Relationship(source=node1, target=node2, type="WORKS_AT", properties={})
        source_doc = Document(page_content="source document text", metadata={})
        graph_doc = GraphDocument(
            nodes=[node1, node2], relationships=[edge], source=source_doc
        )

        # Call method
        graph.add_graph_documents(
            graph_documents=[graph_doc],
            include_source=True,
            graph_name="TestGraph",
            update_graph_definition_if_exists=True,
            batch_size=1,
            use_one_entity_collection=True,
            insert_async=False,
            source_collection_name="SRC",
            source_edge_collection_name="SRC_EDGE",
            entity_collection_name="ENTITY",
            entity_edge_collection_name="ENTITY_EDGE",
            embeddings=embedding,
            embed_source=True,
            embed_nodes=True,
            embed_relationships=True,
            capitalization_strategy="lower",
        )

        # Assertions
        graph._create_collection.assert_any_call("SRC")
        graph._create_collection.assert_any_call("SRC_EDGE", is_edge=True)
        graph._create_collection.assert_any_call("ENTITY")
        graph._create_collection.assert_any_call("ENTITY_EDGE", is_edge=True)

        graph._process_source.assert_called_once()
        graph._import_data.assert_called()
        graph.refresh_schema.assert_called_once()
        graph.db.create_graph.assert_called_once()
        assert graph._process_node_as_entity.call_count == 2
        graph._process_edge_as_entity.assert_called_once()

    def test_get_node_key_handles_existing_and_new_node(self):
        # Setup
        graph = ArangoGraph(db=MagicMock())
        graph._hash = MagicMock(side_effect=lambda x: f"hashed_{x}")

        # Data structures
        nodes = defaultdict(list)
        node_key_map = {"existing_id": "hashed_existing_id"}
        entity_collection_name = "MyEntities"
        process_node_fn = MagicMock()

        # Case 1: Node ID already in node_key_map
        existing_node = Node(id="existing_id")
        result1 = graph._get_node_key(
            node=existing_node,
            nodes=nodes,
            node_key_map=node_key_map,
            entity_collection_name=entity_collection_name,
            process_node_fn=process_node_fn,
        )
        assert result1 == "hashed_existing_id"
        process_node_fn.assert_not_called()  # It should skip processing

        # Case 2: Node ID not in node_key_map (should call process_node_fn)
        new_node = Node(id=999)  # intentionally non-str to test str conversion
        result2 = graph._get_node_key(
            node=new_node,
            nodes=nodes,
            node_key_map=node_key_map,
            entity_collection_name=entity_collection_name,
            process_node_fn=process_node_fn,
        )

        expected_key = "hashed_999"
        assert result2 == expected_key
        assert node_key_map["999"] == expected_key  # confirms key was added
        process_node_fn.assert_called_once_with(
            expected_key, new_node, nodes, entity_collection_name
        )

    def test_process_source_inserts_document_with_hash(self, graph):
        # Setup ArangoGraph with mocked hash method
        graph._hash = MagicMock(return_value="fake_hashed_id")

        # Prepare source document
        doc = Document(
            page_content="This is a test document.",
            metadata={"author": "tester", "type": "text"},
            id="doc123",
        )

        # Setup mocked insertion DB and collection
        mock_collection = MagicMock()
        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        # Run method
        source_id = graph._process_source(
            source=doc,
            source_collection_name="my_sources",
            source_embedding=[0.1, 0.2, 0.3],
            embedding_field="embedding",
            insertion_db=mock_db,
        )

        # Verify _hash was called with source.id
        graph._hash.assert_called_once_with("doc123")

        # Verify correct insertion
        mock_collection.insert.assert_called_once_with(
            {
                "author": "tester",
                "type": "Document",
                "_key": "fake_hashed_id",
                "text": "This is a test document.",
                "embedding": [0.1, 0.2, 0.3],
            },
            overwrite=True,
        )

        # Assert return value is correct
        assert source_id == "fake_hashed_id"

    def test_hash_with_string_input(self):
        result = self.graph._hash("hello")
        assert isinstance(result, str)
        assert result.isdigit()

    def test_hash_with_integer_input(self):
        result = self.graph._hash(12345)
        assert isinstance(result, str)
        assert result.isdigit()

    def test_hash_with_dict_input(self):
        value = {"key": "value"}
        result = self.graph._hash(value)
        assert isinstance(result, str)
        assert result.isdigit()

    def test_hash_raises_on_unstringable_input(self):
        class BadStr:
            def __str__(self):
                raise Exception("nope")

        with pytest.raises(
            ValueError, match="Value must be a string or have a string representation"
        ):
            self.graph._hash(BadStr())

    def test_hash_uses_farmhash(self):
        with patch(
            "langchain_arangodb.graphs.arangodb_graph.farmhash.Fingerprint64"
        ) as mock_farmhash:
            mock_farmhash.return_value = 9999999999999
            result = self.graph._hash("test")
            mock_farmhash.assert_called_once_with("test")
            assert result == "9999999999999"

    def test_empty_name_raises_error(self):
        with pytest.raises(ValueError, match="Collection name cannot be empty"):
            self.graph._sanitize_collection_name("")

    def test_name_with_valid_characters(self):
        name = "valid_name-123"
        assert self.graph._sanitize_collection_name(name) == name

    def test_name_with_invalid_characters(self):
        name = "invalid!@#name$%^"
        result = self.graph._sanitize_collection_name(name)
        assert result == "invalid___name___"

    def test_name_exceeding_max_length(self):
        long_name = "x" * 300
        result = self.graph._sanitize_collection_name(long_name)
        assert len(result) == 256

    def test_name_starting_with_number(self):
        name = "123abc"
        result = self.graph._sanitize_collection_name(name)
        assert result == "Collection_123abc"

    def test_name_starting_with_underscore(self):
        name = "_temp"
        result = self.graph._sanitize_collection_name(name)
        assert result == "Collection__temp"

    def test_name_starting_with_letter_is_unchanged(self):
        name = "a_collection"
        result = self.graph._sanitize_collection_name(name)
        assert result == name

    def test_sanitize_input_string_below_limit(self, graph):
        result = graph._sanitize_input({"text": "short"}, list_limit=5, string_limit=10)
        assert result == {"text": "short"}

    def test_sanitize_input_string_above_limit(self, graph):
        result = graph._sanitize_input(
            {"text": "a" * 50}, list_limit=5, string_limit=10
        )
        assert result == {"text": "String of 50 characters"}

    def test_sanitize_input_small_list(self, graph):
        result = graph._sanitize_input(
            {"data": [1, 2, 3]}, list_limit=5, string_limit=10
        )
        assert result == {"data": [1, 2, 3]}

    def test_sanitize_input_large_list(self, graph):
        result = graph._sanitize_input(
            {"data": [0] * 10}, list_limit=5, string_limit=10
        )
        assert result == {"data": "List of 10 elements of type <class 'int'>"}

    def test_sanitize_input_nested_dict(self, graph):
        data = {"level1": {"level2": {"long_string": "x" * 100}}}
        result = graph._sanitize_input(data, list_limit=5, string_limit=10)
        assert result == {
            "level1": {"level2": {"long_string": "String of 100 characters"}}
        }

    def test_sanitize_input_mixed_nested(self, graph):
        data = {
            "items": [
                {"text": "short"},
                {"text": "x" * 50},
                {"numbers": list(range(3))},
                {"numbers": list(range(20))},
            ]
        }
        result = graph._sanitize_input(data, list_limit=5, string_limit=10)
        assert result == {
            "items": [
                {"text": "short"},
                {"text": "String of 50 characters"},
                {"numbers": [0, 1, 2]},
                {"numbers": "List of 20 elements of type <class 'int'>"},
            ]
        }

    def test_sanitize_input_empty_list(self, graph):
        result = graph._sanitize_input([], list_limit=5, string_limit=10)
        assert result == []

    def test_sanitize_input_primitive_int(self, graph):
        assert graph._sanitize_input(123, list_limit=5, string_limit=10) == 123

    def test_sanitize_input_primitive_bool(self, graph):
        assert graph._sanitize_input(True, list_limit=5, string_limit=10) is True

    def test_from_db_credentials_uses_env_vars(self, monkeypatch):
        monkeypatch.setenv("ARANGODB_URL", "http://envhost:8529")
        monkeypatch.setenv("ARANGODB_DBNAME", "env_db")
        monkeypatch.setenv("ARANGODB_USERNAME", "env_user")
        monkeypatch.setenv("ARANGODB_PASSWORD", "env_pass")

        with patch.object(
            get_arangodb_client.__globals__["ArangoClient"], "db"
        ) as mock_db:
            fake_db = MagicMock()
            mock_db.return_value = fake_db

            graph = ArangoGraph.from_db_credentials()
            assert isinstance(graph, ArangoGraph)

            mock_db.assert_called_once_with(
                "env_db", "env_user", "env_pass", verify=True
            )

    def test_import_data_bulk_inserts_and_clears(self):
        self.graph._create_collection = MagicMock()

        data = {"MyColl": [{"_key": "1"}, {"_key": "2"}]}
        self.graph._import_data(self.mock_db, data, is_edge=False)

        self.graph._create_collection.assert_called_once_with("MyColl", False)
        self.mock_db.collection("MyColl").import_bulk.assert_called_once()
        assert data == {}

    def test_create_collection_if_not_exists(self):
        self.mock_db.has_collection.return_value = False
        self.graph._create_collection("CollX", is_edge=True)
        self.mock_db.create_collection.assert_called_once_with("CollX", edge=True)

    def test_create_collection_skips_if_exists(self):
        self.mock_db.has_collection.return_value = True
        self.graph._create_collection("Exists")
        self.mock_db.create_collection.assert_not_called()

    def test_process_node_as_entity_adds_to_dict(self):
        nodes = defaultdict(list)
        node = Node(id="n1", type="Person", properties={"age": 42})

        collection = self.graph._process_node_as_entity("key1", node, nodes, "ENTITY")
        assert collection == "ENTITY"
        assert nodes["ENTITY"][0]["_key"] == "key1"
        assert nodes["ENTITY"][0]["text"] == "n1"
        assert nodes["ENTITY"][0]["type"] == "Person"
        assert nodes["ENTITY"][0]["age"] == 42

    def test_process_node_as_type_sanitizes_and_adds(self):
        self.graph._sanitize_collection_name = lambda x: f"safe_{x}"
        nodes = defaultdict(list)
        node = Node(id="idA", type="Animal", properties={"species": "cat"})

        result = self.graph._process_node_as_type("abc123", node, nodes, "unused")
        assert result == "safe_Animal"
        assert nodes["safe_Animal"][0]["_key"] == "abc123"
        assert nodes["safe_Animal"][0]["text"] == "idA"
        assert nodes["safe_Animal"][0]["species"] == "cat"

    def test_process_edge_as_entity_adds_correctly(self):
        edges = defaultdict(list)
        edge = Relationship(
            source=Node(id="1", type="User"),
            target=Node(id="2", type="Item"),
            type="LIKES",
            properties={"strength": "high"},
        )

        self.graph._process_edge_as_entity(
            edge=edge,
            edge_str="1 LIKES 2",
            edge_key="edge42",
            source_key="s123",
            target_key="t456",
            edges=edges,
            entity_collection_name="NODE",
            entity_edge_collection_name="EDGE",
            _=defaultdict(lambda: defaultdict(set)),
        )

        e = edges["EDGE"][0]
        assert e["_key"] == "edge42"
        assert e["_from"] == "NODE/s123"
        assert e["_to"] == "NODE/t456"
        assert e["type"] == "LIKES"
        assert e["text"] == "1 LIKES 2"
        assert e["strength"] == "high"

    def test_generate_schema_invalid_sample_ratio(self):
        with pytest.raises(
            ValueError, match=r"\*\*sample_ratio\*\* value must be in between 0 to 1"
        ):
            self.graph.generate_schema(sample_ratio=2)

    def test_generate_schema_with_graph_name(self):
        mock_graph = MagicMock()
        mock_graph.edge_definitions.return_value = [{"edge_collection": "edges"}]
        mock_graph.vertex_collections.return_value = ["vertices"]
        self.mock_db.graph.return_value = mock_graph
        self.mock_db.collection().count.return_value = 5
        self.mock_db.aql.execute.return_value = DummyCursor()
        self.mock_db.collections.return_value = [
            {"name": "vertices", "system": False, "type": "document"},
            {"name": "edges", "system": False, "type": "edge"},
        ]

        result = self.graph.generate_schema(sample_ratio=0.2, graph_name="TestGraph")

        assert result["graph_schema"][0]["name"] == "TestGraph"
        assert any(col["name"] == "vertices" for col in result["collection_schema"])
        assert any(col["name"] == "edges" for col in result["collection_schema"])

    def test_generate_schema_no_graph_name(self):
        self.mock_db.graphs.return_value = [{"name": "G1", "edge_definitions": []}]
        self.mock_db.collections.return_value = [
            {"name": "users", "system": False, "type": "document"},
            {"name": "_system", "system": True, "type": "document"},
        ]
        self.mock_db.collection().count.return_value = 10
        self.mock_db.aql.execute.return_value = DummyCursor()

        result = self.graph.generate_schema(sample_ratio=0.5)

        assert result["graph_schema"][0]["graph_name"] == "G1"
        assert result["collection_schema"][0]["name"] == "users"
        assert "example" in result["collection_schema"][0]

    def test_generate_schema_include_examples_false(self):
        self.mock_db.graphs.return_value = []
        self.mock_db.collections.return_value = [
            {"name": "products", "system": False, "type": "document"}
        ]
        self.mock_db.collection().count.return_value = 10
        self.mock_db.aql.execute.return_value = DummyCursor()

        result = self.graph.generate_schema(include_examples=False)

        assert "example" not in result["collection_schema"][0]

    def test_add_graph_documents_update_graph_definition_if_exists(self):
        # Setup
        mock_graph = MagicMock()

        self.mock_db.has_graph.return_value = True
        self.mock_db.graph.return_value = mock_graph
        mock_graph.has_edge_definition.return_value = True

        # Minimal valid GraphDocument
        node1 = Node(id="1", type="Person")
        node2 = Node(id="2", type="Person")
        edge = Relationship(source=node1, target=node2, type="KNOWS")
        doc = GraphDocument(nodes=[node1, node2], relationships=[edge])

        # Patch internal methods to avoid unrelated side effects
        self.graph._hash = lambda x: str(x)
        self.graph._process_node_as_entity = lambda k, n, nodes, _: "ENTITY"
        self.graph._process_edge_as_entity = lambda *args, **kwargs: None
        self.graph._import_data = lambda *args, **kwargs: None
        self.graph.refresh_schema = MagicMock()
        self.graph._create_collection = MagicMock()

        # Act
        self.graph.add_graph_documents(
            graph_documents=[doc],
            graph_name="TestGraph",
            update_graph_definition_if_exists=True,
            capitalization_strategy="lower",
        )

        # Assert
        self.mock_db.has_graph.assert_called_once_with("TestGraph")
        self.mock_db.graph.assert_called_once_with("TestGraph")
        mock_graph.has_edge_definition.assert_called()
        mock_graph.replace_edge_definition.assert_called()

    def test_query_with_top_k_and_limits(self):
        # Simulated AQL results from ArangoDB
        raw_results = [
            {"name": "Alice", "tags": ["a", "b"], "age": 30},
            {"name": "Bob", "tags": ["c", "d"], "age": 25},
            {"name": "Charlie", "tags": ["e", "f"], "age": 40},
        ]
        # Mock AQL cursor
        self.mock_db.aql.execute.return_value = iter(raw_results)

        # Input AQL query and parameters
        query_str = "FOR u IN users RETURN u"
        params = {"top_k": 2, "list_limit": 2, "string_limit": 50}

        # Call the method
        result = self.graph.query(query_str, params.copy())

        # Expected sanitized results based on mock _sanitize_input
        expected = [
            {"name": "Alice", "tags": "List of 2 elements", "age": 30},
            {"name": "Alice", "tags": "List of 2 elements", "age": 30},
            {"name": "Alice", "tags": "List of 2 elements", "age": 30},
        ]

        # Assertions
        assert result == expected
        self.mock_db.aql.execute.assert_called_once_with(query_str)
        assert self.graph._sanitize_input.call_count == 3
        self.graph._sanitize_input.assert_any_call(raw_results[0], 2, 50)
        self.graph._sanitize_input.assert_any_call(raw_results[1], 2, 50)
        self.graph._sanitize_input.assert_any_call(raw_results[2], 2, 50)

    def test_schema_json(self):
        test_schema = {
            "collection_schema": [{"name": "Users", "type": "document"}],
            "graph_schema": [{"graph_name": "UserGraph", "edge_definitions": []}],
        }
        self.graph._ArangoGraph__schema = test_schema  # set private attribute
        result = self.graph.schema_json
        assert json.loads(result) == test_schema

    def test_schema_yaml(self):
        test_schema = {
            "collection_schema": [{"name": "Users", "type": "document"}],
            "graph_schema": [{"graph_name": "UserGraph", "edge_definitions": []}],
        }
        self.graph._ArangoGraph__schema = test_schema
        result = self.graph.schema_yaml
        assert yaml.safe_load(result) == test_schema

    def test_set_schema(self):
        new_schema = {
            "collection_schema": [{"name": "Products", "type": "document"}],
            "graph_schema": [{"graph_name": "ProductGraph", "edge_definitions": []}],
        }
        self.graph.set_schema(new_schema)
        assert self.graph._ArangoGraph__schema == new_schema

    def test_refresh_schema_sets_internal_schema(self):
        fake_schema = {
            "collection_schema": [{"name": "Test", "type": "document"}],
            "graph_schema": [{"graph_name": "TestGraph", "edge_definitions": []}],
        }

        # Mock generate_schema to return a controlled fake schema
        self.graph.generate_schema = MagicMock(return_value=fake_schema)

        # Call refresh_schema with custom args
        self.graph.refresh_schema(
            sample_ratio=0.5,
            graph_name="TestGraph",
            include_examples=False,
            list_limit=10,
        )

        # Assert generate_schema was called with those args
        self.graph.generate_schema.assert_called_once_with(0.5, "TestGraph", False, 10)

        # Assert internal schema was set correctly
        assert self.graph._ArangoGraph__schema == fake_schema

    def test_sanitize_input_large_list_returns_summary_string(self):
        # Arrange
        graph = ArangoGraph(db=MagicMock(), generate_schema_on_init=False)

        # A list longer than the list_limit (e.g., limit=5, list has 10 elements)
        test_input = [1] * 10
        list_limit = 5
        string_limit = 256  # doesn't matter for this test

        # Act
        result = graph._sanitize_input(test_input, list_limit, string_limit)

        # Assert
        assert result == "List of 10 elements of type <class 'int'>"

    def test_add_graph_documents_creates_edge_definition_if_missing(self):
        # Setup ArangoGraph instance with mocked db
        mock_db = MagicMock()
        graph = ArangoGraph(db=mock_db, generate_schema_on_init=False)

        # Setup mock for existing graph
        mock_graph = MagicMock()
        mock_graph.has_edge_definition.return_value = False
        mock_db.has_graph.return_value = True
        mock_db.graph.return_value = mock_graph

        # Minimal GraphDocument with one edge
        node1 = Node(id="1", type="Person")
        node2 = Node(id="2", type="Company")
        edge = Relationship(source=node1, target=node2, type="WORKS_AT")
        graph_doc = GraphDocument(nodes=[node1, node2], relationships=[edge])

        # Patch internals to avoid unrelated behavior
        graph._hash = lambda x: str(x)
        graph._process_node_as_type = lambda *args, **kwargs: "Entity"
        graph._import_data = lambda *args, **kwargs: None
        graph.refresh_schema = lambda *args, **kwargs: None
        graph._create_collection = lambda *args, **kwargs: None

        # Simulate _process_edge_as_type populating edge_definitions_dict

    def fake_process_edge_as_type(
        edge,
        edge_str,
        edge_key,
        source_key,
        target_key,
        edges,
        _1,
        _2,
        edge_definitions_dict,
    ):
        edge_type = "WORKS_AT"
        edges[edge_type].append({"_key": edge_key})
        edge_definitions_dict[edge_type]["from_vertex_collections"].add("Person")
        edge_definitions_dict[edge_type]["to_vertex_collections"].add("Company")

        graph._process_edge_as_type = fake_process_edge_as_type

        # Act
        graph.add_graph_documents(
            graph_documents=[graph_doc],
            graph_name="MyGraph",
            update_graph_definition_if_exists=True,
            use_one_entity_collection=False,
            capitalization_strategy="lower",
        )

        # Assert
        mock_db.graph.assert_called_once_with("MyGraph")
        mock_graph.has_edge_definition.assert_called_once_with("WORKS_AT")
        mock_graph.create_edge_definition.assert_called_once()

    def test_add_graph_documents_raises_if_embedding_missing(self):
        # Arrange
        graph = ArangoGraph(db=MagicMock(), generate_schema_on_init=False)

        # Minimal valid GraphDocument
        node1 = Node(id="1", type="Person")
        node2 = Node(id="2", type="Company")
        edge = Relationship(source=node1, target=node2, type="WORKS_AT")
        doc = GraphDocument(nodes=[node1, node2], relationships=[edge])

        # Act & Assert
        with pytest.raises(ValueError, match=r"\*\*embedding\*\* is required"):
            graph.add_graph_documents(
                graph_documents=[doc],
                embeddings=None,  # ← embeddings not provided
                embed_source=True,  # ← any of these True triggers the check
            )

    class DummyEmbeddings:
        def embed_documents(self, texts):
            return [[0.0] * 5 for _ in texts]

    @pytest.mark.parametrize(
        "strategy,input_id,expected_id",
        [
            ("none", "TeStId", "TeStId"),
            ("upper", "TeStId", "TESTID"),
        ],
    )
    def test_add_graph_documents_capitalization_strategy(
        self, strategy, input_id, expected_id
    ):
        graph = ArangoGraph(db=MagicMock(), generate_schema_on_init=False)

        graph._hash = lambda x: x
        graph._import_data = lambda *args, **kwargs: None
        graph.refresh_schema = lambda *args, **kwargs: None
        graph._create_collection = lambda *args, **kwargs: None

        mutated_nodes = []

        def track_process_node(key, node, nodes, coll):
            mutated_nodes.append(node.id)
            return "ENTITY"

        graph._process_node_as_entity = track_process_node
        graph._process_edge_as_entity = lambda *args, **kwargs: None

        node1 = Node(id=input_id, type="Person")
        node2 = Node(id="Dummy", type="Company")
        edge = Relationship(source=node1, target=node2, type="WORKS_AT")
        doc = GraphDocument(nodes=[node1, node2], relationships=[edge])

        graph.add_graph_documents(
            graph_documents=[doc],
            capitalization_strategy=strategy,
            use_one_entity_collection=True,
            embed_source=True,
            embeddings=self.DummyEmbeddings(),  # reference class properly
        )

        assert mutated_nodes[0] == expected_id
