"""Test ArangoDB vector store functionality."""

from typing import Any, Callable, List, Optional, Tuple, Type
from unittest.mock import MagicMock, patch

import arango
import pytest
from arango.database import StandardDatabase
from arango.exceptions import ( # Assuming these are relevant ArangoDB exceptions
    AuthError,
    DocumentInsertError,
    IndexCreateError,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings

from langchain_arangodb.vectorstores.arangodb_vector import (
    ArangoVector,
    DEFAULT_DISTANCE_STRATEGY,
    DistanceStrategy,
    SearchType,
)

# --- Test Helper Functions (if any were used in Neo4j version) ---
# Example: If check_if_not_null was used and is generic
# def check_if_not_null(value: Any, param_name: str) -> None:
#     if value is None or value == "":
#         raise ValueError(f"{param_name} must not be None or empty.")

# def test_check_if_not_null_happy_case() -> None:
#     check_if_not_null("test", "param")

# def test_check_if_not_null_with_empty_string() -> None:
#     with pytest.raises(ValueError):
#         check_if_not_null("", "param")

# def test_check_if_not_null_with_none_value() -> None:
#     with pytest.raises(ValueError):
#         check_if_not_null(None, "param")

# --- Mock Fixtures ---

# Define default embedding dimension for tests
TEST_EMBEDDING_DIM = 128


@pytest.fixture
def mock_arango_db() -> MagicMock:
    """Fixture for a mocked ArangoDB StandardDatabase instance."""
    mock_db = MagicMock(spec=StandardDatabase)
    mock_db.has_collection.return_value = True # Assume collection exists by default
    mock_collection = MagicMock()
    mock_collection.indexes.return_value = [] # Assume no index exists by default
    mock_collection.add_index.return_value = {"id": "indexes/123", "isNewlyCreated": True}
    mock_collection.insert_many.return_value = [] # Mock insert result
    mock_db.collection.return_value = mock_collection
    mock_aql = MagicMock()
    mock_aql.execute.return_value = MagicMock() # Mock AQL cursor
    mock_db.aql = mock_aql
    # Mock async db connection if needed
    mock_db.begin_async_execution.return_value = mock_db
    return mock_db


@pytest.fixture
def mock_embedding() -> FakeEmbeddings:
    """Fixture for FakeEmbeddings."""
    return FakeEmbeddings(size=TEST_EMBEDDING_DIM)


@pytest.fixture
def arango_vector_factory(mock_arango_db: MagicMock, mock_embedding: FakeEmbeddings) -> Callable[..., ArangoVector]:
    """Factory fixture to create ArangoVector instances with specified mocks."""

    def _factory(**kwargs: Any) -> ArangoVector:
        db = kwargs.pop("database", mock_arango_db)
        embedding = kwargs.pop("embedding", mock_embedding)
        embedding_dimension = kwargs.pop("embedding_dimension", TEST_EMBEDDING_DIM)

        # Ensure collection exists check is mocked based on input or default
        if "collection_exists" in kwargs:
            db.has_collection.return_value = kwargs.pop("collection_exists")
        else:
            db.has_collection.return_value = True # Default to True

        # Ensure index retrieval is mocked based on input or default
        if "index_exists" in kwargs:
            index_info = kwargs.pop("index_exists")
            if index_info:
                index_name = kwargs.get("index_name", "vector_index")
                embedding_field = kwargs.get("embedding_field", "embedding")
                distance_strategy = kwargs.get("distance_strategy", DEFAULT_DISTANCE_STRATEGY)
                db.collection.return_value.indexes.return_value = [
                    {
                        "id": "indexes/123",
                        "name": index_name,
                        "type": "vector",
                        "fields": [embedding_field],
                        "params": {
                            "metric": ArangoVector.DISTANCE_MAPPING[distance_strategy],
                            "dimension": embedding_dimension,
                        },
                    }
                ]
            else:
                db.collection.return_value.indexes.return_value = []
        else:
            db.collection.return_value.indexes.return_value = [] # Default to False

        return ArangoVector(
            embedding=embedding,
            embedding_dimension=embedding_dimension,
            database=db,
            **kwargs,
        )

    return _factory

# --- Basic Initialization Tests ---

def test_arango_vector_init(arango_vector_factory: Callable[..., ArangoVector], mock_arango_db: MagicMock) -> None:
    """Test basic initialization."""
    vector_store = arango_vector_factory()
    assert vector_store.db == mock_arango_db
    assert isinstance(vector_store.embedding, Embeddings)
    assert vector_store.embedding_dimension == TEST_EMBEDDING_DIM
    # Check if collection was created if mock indicated it didn't exist
    mock_arango_db.has_collection.return_value = False
    vector_store_create = arango_vector_factory(collection_exists=False)
    mock_arango_db.create_collection.assert_called_once_with("documents")
    # Check default index creation attempt
    mock_arango_db.collection.return_value.indexes.assert_called_once()
    # mock_arango_db.collection.return_value.add_index.assert_called_once() # ArangoVector doesn't create index on init


def test_arango_vector_invalid_distance_strategy(arango_vector_factory: Callable[..., ArangoVector]) -> None:
    """Test initialization with invalid distance strategy."""
    with pytest.raises(ValueError, match="distance_strategy must be"):
        arango_vector_factory(distance_strategy="invalid_strategy")


def test_arango_vector_invalid_search_type(arango_vector_factory: Callable[..., ArangoVector]) -> None:
    """Test initialization with invalid search type."""
    with pytest.raises(ValueError, match="search_type must be"):
        arango_vector_factory(search_type="invalid_search")

# --- Connection Error Tests ---

def test_arango_vector_auth_error(arango_vector_factory: Callable[..., ArangoVector], mock_arango_db: MagicMock) -> None:
    """Test handling of authentication errors during potential connection test."""
    # Simulate AuthError on a relevant DB operation (e.g., has_collection)
    mock_arango_db.has_collection.side_effect = AuthError("Auth failed", http_exception=None)
    with pytest.raises(AuthError):
         arango_vector_factory()

# --- Index Handling Tests ---

def test_retrieve_vector_index(arango_vector_factory: Callable[..., ArangoVector], mock_arango_db: MagicMock) -> None:
    """Test retrieving an existing vector index."""
    index_name = "my_index"
    mock_db = mock_arango_db
    mock_db.collection.return_value.indexes.return_value = [
        {"name": "primary", "type": "primary"},
        {"name": index_name, "type": "vector", "id": "indexes/456"},
    ]
    vector_store = arango_vector_factory(database=mock_db, index_name=index_name)
    index_info = vector_store.retrieve_vector_index()
    assert index_info is not None
    assert index_info["name"] == index_name
    mock_db.collection.return_value.indexes.assert_called_once()


def test_retrieve_vector_index_not_found(arango_vector_factory: Callable[..., ArangoVector], mock_arango_db: MagicMock) -> None:
    """Test retrieving a non-existent vector index."""
    mock_db = mock_arango_db
    mock_db.collection.return_value.indexes.return_value = [{"name": "primary", "type": "primary"}]
    vector_store = arango_vector_factory(database=mock_db, index_name="non_existent")
    index_info = vector_store.retrieve_vector_index()
    assert index_info is None
    mock_db.collection.return_value.indexes.assert_called_once()


def test_create_vector_index(arango_vector_factory: Callable[..., ArangoVector], mock_arango_db: MagicMock) -> None:
    """Test creating a vector index."""
    mock_db = mock_arango_db
    mock_collection = mock_db.collection.return_value
    mock_collection.indexes.return_value = [] # Ensure index doesn't exist initially
    vector_store = arango_vector_factory(database=mock_db)
    vector_store.create_vector_index()
    mock_collection.add_index.assert_called_once()
    args, kwargs = mock_collection.add_index.call_args
    index_config = args[0]
    assert index_config["name"] == vector_store.index_name
    assert index_config["type"] == "vector"
    assert index_config["fields"] == [vector_store.embedding_field]
    assert index_config["params"]["metric"] == "cosine" # Default
    assert index_config["params"]["dimension"] == TEST_EMBEDDING_DIM


def test_delete_vector_index(arango_vector_factory: Callable[..., ArangoVector], mock_arango_db: MagicMock) -> None:
    """Test deleting a vector index."""
    index_name = "to_delete"
    index_id = "indexes/789"
    mock_db = mock_arango_db
    mock_collection = mock_db.collection.return_value
    mock_collection.indexes.return_value = [
        {"name": index_name, "type": "vector", "id": index_id}
    ]
    vector_store = arango_vector_factory(database=mock_db, index_name=index_name)
    vector_store.delete_vector_index()
    mock_collection.delete_index.assert_called_once_with(index_id)


def test_delete_vector_index_not_found(arango_vector_factory: Callable[..., ArangoVector], mock_arango_db: MagicMock) -> None:
    """Test deleting a non-existent vector index."""
    mock_db = mock_arango_db
    mock_collection = mock_db.collection.return_value
    mock_collection.indexes.return_value = []
    vector_store = arango_vector_factory(database=mock_db, index_name="non_existent")
    vector_store.delete_vector_index()
    mock_collection.delete_index.assert_not_called()

# --- Add Texts/Embeddings Tests ---

def test_add_texts(arango_vector_factory: Callable[..., ArangoVector], mock_embedding: FakeEmbeddings, mock_arango_db: MagicMock) -> None:
    """Test adding texts."""
    vector_store = arango_vector_factory()
    texts = ["hello world", "hello arango"]
    ids = vector_store.add_texts(texts)
    assert len(ids) == len(texts)
    # Check if embeddings were generated
    assert mock_embedding.embed_documents.call_count == 1
    # Check if documents were inserted
    mock_arango_db.collection.return_value.insert_many.assert_called_once()
    inserted_data = mock_arango_db.collection.return_value.insert_many.call_args[0][0]
    assert len(inserted_data) == len(texts)
    assert inserted_data[0][vector_store.text_field] == texts[0]
    assert vector_store.embedding_field in inserted_data[0]
    assert len(inserted_data[0][vector_store.embedding_field]) == TEST_EMBEDDING_DIM


def test_add_embeddings(arango_vector_factory: Callable[..., ArangoVector], mock_arango_db: MagicMock) -> None:
    """Test adding pre-computed embeddings."""
    vector_store = arango_vector_factory()
    texts = ["doc1", "doc2"]
    embeddings = [[0.1] * TEST_EMBEDDING_DIM, [0.2] * TEST_EMBEDDING_DIM]
    metadatas = [{"source": "A"}, {"source": "B"}]
    ids_provided = ["id1", "id2"]

    ids = vector_store.add_embeddings(texts, embeddings, metadatas=metadatas, ids=ids_provided)
    assert ids == ids_provided
    mock_arango_db.collection.return_value.insert_many.assert_called_once()
    inserted_data = mock_arango_db.collection.return_value.insert_many.call_args[0][0]
    assert len(inserted_data) == len(texts)
    assert inserted_data[0]["_key"] == ids_provided[0]
    assert inserted_data[0][vector_store.text_field] == texts[0]
    assert inserted_data[0][vector_store.embedding_field] == embeddings[0]
    assert inserted_data[0]["source"] == metadatas[0]["source"]
    assert inserted_data[1]["source"] == metadatas[1]["source"]


def test_add_embeddings_mismatched_lengths(arango_vector_factory: Callable[..., ArangoVector]) -> None:
    """Test error handling for mismatched input lengths in add_embeddings."""
    vector_store = arango_vector_factory()
    texts = ["doc1"]
    embeddings = [[0.1] * TEST_EMBEDDING_DIM, [0.2] * TEST_EMBEDDING_DIM]
    with pytest.raises(ValueError, match="Length of ids, texts, embeddings and metadatas must be the same."):
        vector_store.add_embeddings(texts, embeddings)

# --- Similarity Search Tests ---

@pytest.fixture
def mock_aql_search_result(mock_arango_db: MagicMock, mock_embedding: FakeEmbeddings) -> None:
    """Mock the AQL query result for similarity search."""
    mock_cursor = MagicMock()
    # Simulate result structure: [ { doc: { _key: ..., text: ... }, score: ... }, ... ]
    mock_cursor.__iter__.return_value = [
        {
            "doc": {
                "_key": "key1",
                "text": "found doc 1",
                "embedding": mock_embedding.embed_query("found doc 1"),
                "metadata_field": "value1"
            },
            "score": 0.9,
        },
        {
            "doc": {
                "_key": "key2",
                "text": "found doc 2",
                "embedding": mock_embedding.embed_query("found doc 2"),
                "metadata_field": "value2"
            },
            "score": 0.8,
        },
    ]
    mock_arango_db.aql.execute.return_value = mock_cursor


def test_similarity_search(arango_vector_factory: Callable[..., ArangoVector], mock_embedding: FakeEmbeddings, mock_arango_db: MagicMock, mock_aql_search_result: None) -> None:
    """Test basic similarity search."""
    vector_store = arango_vector_factory(index_exists=True) # Assume index exists
    query = "search query"
    k = 2
    results = vector_store.similarity_search(query, k=k)

    assert len(results) == k
    assert isinstance(results[0], Document)
    assert results[0].page_content == "found doc 1"
    assert "metadata_field" in results[0].metadata
    assert results[0].metadata["metadata_field"] == "value1"
    # Check if query embedding was generated
    mock_embedding.embed_query.assert_called_once_with(query)
    # Check AQL query execution
    mock_arango_db.aql.execute.assert_called_once()
    args, kwargs = mock_arango_db.aql.execute.call_args
    query_text = args[0]
    bind_vars = kwargs.get("bind_vars", {})
    assert f"IN {vector_store.collection_name}" in query_text
    assert f"SEARCH ANALYZER(doc.{vector_store.text_field}" not in query_text # No hybrid search
    assert f"KNN(doc.{vector_store.embedding_field}" in query_text # Approx search
    assert f"LIMIT {k}" in query_text
    assert "query_embedding" in bind_vars


def test_similarity_search_by_vector_with_score(arango_vector_factory: Callable[..., ArangoVector], mock_arango_db: MagicMock, mock_aql_search_result: None) -> None:
    """Test similarity search by vector with scores."""
    vector_store = arango_vector_factory(index_exists=True)
    query_embedding = [0.5] * TEST_EMBEDDING_DIM
    k = 2
    results = vector_store.similarity_search_by_vector_with_score(query_embedding, k=k)

    assert len(results) == k
    assert isinstance(results[0], tuple)
    doc, score = results[0]
    assert isinstance(doc, Document)
    assert doc.page_content == "found doc 1"
    assert score == pytest.approx(0.9) # Check score directly from mocked result
    # Check AQL query execution
    mock_arango_db.aql.execute.assert_called_once()
    args, kwargs = mock_arango_db.aql.execute.call_args
    query_text = args[0]
    bind_vars = kwargs.get("bind_vars", {})
    assert "query_embedding" in bind_vars
    assert bind_vars["query_embedding"] == query_embedding

# --- From Texts Class Method Test ---

@patch("langchain_arangodb.vectorstores.arangodb_vector.ArangoVector.add_texts")
def test_from_texts(
    mock_add_texts: MagicMock,
    mock_arango_db: MagicMock,
    mock_embedding: FakeEmbeddings,
) -> None:
    """Test the from_texts class method."""
    texts = ["text1", "text2"]
    metadatas = [{"m": 1}, {"m": 2}]

    # Mock dimension retrieval from embedding if needed (ArangoVector requires it)
    mock_embedding.embed_query("test") # Call once to set up dimension if FakeEmbeddings needs it

    vector_store = ArangoVector.from_texts(
        texts=texts,
        embedding=mock_embedding,
        metadatas=metadatas,
        database=mock_arango_db,
        collection_name="test_coll",
        embedding_dimension=TEST_EMBEDDING_DIM # Provide dimension explicitly
    )

    assert isinstance(vector_store, ArangoVector)
    assert vector_store.collection_name == "test_coll"
    mock_add_texts.assert_called_once_with(texts, metadatas, ids=None)

# --- Deletion Tests ---

def test_delete(arango_vector_factory: Callable[..., ArangoVector], mock_arango_db: MagicMock) -> None:
    """Test deleting documents by ID."""
    vector_store = arango_vector_factory()
    ids_to_delete = ["id1", "id2"]
    mock_collection = mock_arango_db.collection.return_value
    mock_collection.delete_many.return_value = [] # Simulate successful deletion

    result = vector_store.delete(ids=ids_to_delete)

    assert result is True # Or check based on actual return value if specific
    mock_collection.delete_many.assert_called_once_with(ids_to_delete)


def test_delete_error(arango_vector_factory: Callable[..., ArangoVector], mock_arango_db: MagicMock) -> None:
    """Test error handling during deletion."""
    vector_store = arango_vector_factory()
    ids_to_delete = ["id1"]
    mock_collection = mock_arango_db.collection.return_value
    # Simulate an error during deletion
    mock_collection.delete_many.side_effect = DocumentInsertError("Deletion failed", http_exception=None)

    result = vector_store.delete(ids=ids_to_delete)
    # Depending on implementation, it might return False or raise the error
    assert result is False # Assuming it returns False on error
    # Or: with pytest.raises(DocumentInsertError): vector_store.delete(ids=ids_to_delete)

# Add more tests as needed, adapting from the Neo4j version:
# - MMR search (max_marginal_relevance_search)
# - Different distance strategies (EUCLIDEAN_DISTANCE)
# - Error handling for index creation (e.g., IndexCreateError)
# - Tests for specific AQL generation logic if complex cases exist
# - Tests for `get_by_ids`
# - Tests for async operations (`use_async_db=True`)

# Placeholder for ArangoDB specific types/utils if any
# from arangodb_search_types import SearchType # Example

# Placeholder for ArangoDBVector import
# from langchain_arangodb.vectorstores.arangodb import ArangoDBVector

# Placeholder for ArangoDB specific distance strategy if different
# from langchain_arangodb.vectorstores.utils import DistanceStrategy

# Helper function tests (Adapt or remove if not applicable)
# def test_remove_arangodb_specific_chars() -> None:
#     # Add tests for any ArangoDB specific character escaping if needed
#     pass

# def test_check_if_not_null_happy_case() -> None:
#     check_if_not_null("test", "param") # Assuming this util is generic

# def test_check_if_not_null_with_empty_string() -> None:
#     with pytest.raises(ValueError):
#         check_if_not_null("", "param")

# def test_check_if_not_null_with_none_value() -> None:
#     with pytest.raises(ValueError):
#         check_if_not_null(None, "param")

# Mock Fixtures (Need heavy adaptation for ArangoDB)
@pytest.fixture
def mock_vector_store() -> Any: # Replace Any with ArangoDBVector when defined
    # mock_arango_driver = MagicMock()
    # mock_db_instance = MagicMock()
    # Configure mock responses for ArangoDB connection verification, version checks etc.
    # mock_db_instance.version.return_value = { "version": "3.11.0", "server": "arango" } # Example
    # mock_arango_driver.db.return_value = mock_db_instance

    # with patch(
    #     "langchain_arangodb.vectorstores.arangodb.arango", # Patch correct location
    #     new=mock_arango_driver,
    # ):
    #     with patch.object(
    #         ArangoDBVector, # Use correct class name
    #         "query", # Adapt if ArangoDB uses a different method for queries
    #         return_value=[{ "version": "3.11.0", "server": "arango" }], # Adapt expected return
    #     ):
    #         vector_store = ArangoDBVector( # Use correct class name and constructor
    #             embedding=MagicMock(),
    #             url="http://localhost:8529",
    #             username="root",
    #             password="password",
    #             # Add ArangoDB specific parameters like db_name, collection_name, view_name
    #         )

    #     vector_store.collection_name = "LangchainDocs" # Adapt property names
    #     vector_store.embedding_field = "embedding" # Adapt property names
    #     vector_store.text_field = "text" # Adapt property names

    #     return vector_store
    pass # Placeholder

# More tests to be adapted from the Neo4j file...
# - Test different distance strategies if applicable
# - Test connection errors (Auth, Service Unavailable)
# - Test version checks if relevant for ArangoDB features
# - Test adding texts/embeddings
# - Test similarity search (vector, hybrid if supported)
# - Test metadata filtering
# - Test specific ArangoSearch/AQL query generation logic
# - Test error handling for index creation/access 