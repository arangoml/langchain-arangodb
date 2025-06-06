from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from langchain_arangodb.vectorstores.arangodb_vector import (
    ArangoVector,
    DistanceStrategy,
    StandardDatabase,
)


@pytest.fixture
def mock_vector_store() -> ArangoVector:
    """Create a mock ArangoVector instance for testing."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_async_db = MagicMock()

    mock_db.has_collection.return_value = True
    mock_db.collection.return_value = mock_collection
    mock_db.begin_async_execution.return_value = mock_async_db

    with patch(
        "langchain_arangodb.vectorstores.arangodb_vector.StandardDatabase",
        return_value=mock_db,
    ):
        vector_store = ArangoVector(
            embedding=MagicMock(),
            embedding_dimension=64,
            database=mock_db,
        )

        return vector_store


@pytest.fixture
def arango_vector_factory() -> Any:
    """Factory fixture to create ArangoVector instances
    with different configurations."""

    def _create_vector_store(
        method: Optional[str] = None,
        texts: Optional[list[str]] = None,
        text_embeddings: Optional[list[tuple[str, list[float]]]] = None,
        collection_exists: bool = True,
        vector_index_exists: bool = True,
        **kwargs: Any,
    ) -> Any:
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_async_db = MagicMock()

        # Configure has_collection
        mock_db.has_collection.return_value = collection_exists
        mock_db.collection.return_value = mock_collection
        mock_db.begin_async_execution.return_value = mock_async_db

        # Configure vector index
        if vector_index_exists:
            mock_collection.indexes.return_value = [
                {
                    "name": kwargs.get("index_name", "vector_index"),
                    "type": "vector",
                    "fields": [kwargs.get("embedding_field", "embedding")],
                    "id": "12345",
                }
            ]
        else:
            mock_collection.indexes.return_value = []

        # Create embedding instance
        embedding = kwargs.pop("embedding", MagicMock())
        if embedding is not None:
            embedding.embed_documents.return_value = [
                [0.1] * kwargs.get("embedding_dimension", 64)
            ] * (len(texts) if texts else 1)
            embedding.embed_query.return_value = [0.1] * kwargs.get(
                "embedding_dimension", 64
            )

        # Create vector store based on method
        common_kwargs = {
            "embedding": embedding,
            "database": mock_db,
            **kwargs,
        }

        if method == "from_texts" and texts:
            common_kwargs["embedding_dimension"] = kwargs.get("embedding_dimension", 64)
            vector_store = ArangoVector.from_texts(
                texts=texts,
                **common_kwargs,
            )
        elif method == "from_embeddings" and text_embeddings:
            texts = [t[0] for t in text_embeddings]
            embeddings = [t[1] for t in text_embeddings]

            with patch.object(
                ArangoVector, "add_embeddings", return_value=["id1", "id2"]
            ):
                vector_store = ArangoVector(
                    **common_kwargs,
                    embedding_dimension=len(embeddings[0]) if embeddings else 64,
                )
        else:
            vector_store = ArangoVector(
                **common_kwargs,
                embedding_dimension=kwargs.get("embedding_dimension", 64),
            )

        return vector_store

    return _create_vector_store


def test_init_with_invalid_search_type() -> None:
    """Test that initializing with an invalid search type raises ValueError."""
    mock_db = MagicMock()

    with pytest.raises(ValueError) as exc_info:
        ArangoVector(
            embedding=MagicMock(),
            embedding_dimension=64,
            database=mock_db,
            search_type="invalid_search_type",  # type: ignore
        )

    assert "search_type must be 'vector'" in str(exc_info.value)


def test_init_with_invalid_distance_strategy() -> None:
    """Test that initializing with an invalid distance strategy raises ValueError."""
    mock_db = MagicMock()

    with pytest.raises(ValueError) as exc_info:
        ArangoVector(
            embedding=MagicMock(),
            embedding_dimension=64,
            database=mock_db,
            distance_strategy="INVALID_STRATEGY",  # type: ignore
        )

    assert "distance_strategy must be 'COSINE' or 'EUCLIDEAN_DISTANCE'" in str(
        exc_info.value
    )


def test_collection_creation_if_not_exists(arango_vector_factory: Any) -> None:
    """Test that collection is created if it doesn't exist."""
    # Configure collection doesn't exist
    vector_store = arango_vector_factory(collection_exists=False)

    # Verify collection was created
    vector_store.db.create_collection.assert_called_once_with(
        vector_store.collection_name
    )


def test_collection_not_created_if_exists(arango_vector_factory: Any) -> None:
    """Test that collection is not created if it already exists."""
    # Configure collection exists
    vector_store = arango_vector_factory(collection_exists=True)

    # Verify collection was not created
    vector_store.db.create_collection.assert_not_called()


def test_retrieve_vector_index_exists(arango_vector_factory: Any) -> None:
    """Test retrieving vector index when it exists."""
    vector_store = arango_vector_factory(vector_index_exists=True)

    index = vector_store.retrieve_vector_index()

    assert index is not None
    assert index["name"] == "vector_index"
    assert index["type"] == "vector"


def test_retrieve_vector_index_not_exists(arango_vector_factory: Any) -> None:
    """Test retrieving vector index when it doesn't exist."""
    vector_store = arango_vector_factory(vector_index_exists=False)

    index = vector_store.retrieve_vector_index()

    assert index is None


def test_create_vector_index(arango_vector_factory: Any) -> None:
    """Test creating vector index."""
    vector_store = arango_vector_factory()

    vector_store.create_vector_index()

    # Verify index creation was called with correct parameters
    vector_store.collection.add_index.assert_called_once()

    call_args = vector_store.collection.add_index.call_args[0][0]
    assert call_args["name"] == "vector_index"
    assert call_args["type"] == "vector"
    assert call_args["fields"] == ["embedding"]
    assert call_args["params"]["metric"] == "cosine"
    assert call_args["params"]["dimension"] == 64


def test_delete_vector_index_exists(arango_vector_factory: Any) -> None:
    """Test deleting vector index when it exists."""
    vector_store = arango_vector_factory(vector_index_exists=True)

    with patch.object(
        vector_store,
        "retrieve_vector_index",
        return_value={"id": "12345", "name": "vector_index"},
    ):
        vector_store.delete_vector_index()

    # Verify delete_index was called with correct ID
    vector_store.collection.delete_index.assert_called_once_with("12345")


def test_delete_vector_index_not_exists(arango_vector_factory: Any) -> None:
    """Test deleting vector index when it doesn't exist."""
    vector_store = arango_vector_factory(vector_index_exists=False)

    with patch.object(vector_store, "retrieve_vector_index", return_value=None):
        vector_store.delete_vector_index()

    # Verify delete_index was not called
    vector_store.collection.delete_index.assert_not_called()


def test_delete_vector_index_with_real_index_data(arango_vector_factory: Any) -> None:
    """Test deleting vector index with real index data structure."""
    vector_store = arango_vector_factory(vector_index_exists=True)

    # Create a realistic index object with all expected fields
    mock_index = {
        "id": "vector_index_12345",
        "name": "vector_index",
        "type": "vector",
        "fields": ["embedding"],
        "selectivity": 1,
        "sparse": False,
        "unique": False,
        "deduplicate": False,
    }

    # Mock retrieve_vector_index to return our realistic index
    with patch.object(vector_store, "retrieve_vector_index", return_value=mock_index):
        # Call the method under test
        vector_store.delete_vector_index()

    # Verify delete_index was called with the exact ID from our mock index
    vector_store.collection.delete_index.assert_called_once_with("vector_index_12345")

    # Test the case where the index doesn't have an id field
    bad_index = {"name": "vector_index", "type": "vector"}
    with patch.object(vector_store, "retrieve_vector_index", return_value=bad_index):
        with pytest.raises(KeyError):
            vector_store.delete_vector_index()


def test_add_embeddings_with_mismatched_lengths(arango_vector_factory: Any) -> None:
    """Test adding embeddings with mismatched lengths raises ValueError."""
    vector_store = arango_vector_factory()

    ids = ["id1"]
    texts = ["text1", "text2"]
    embeddings = [[0.1] * 64, [0.2] * 64, [0.3] * 64]
    metadatas = [
        {"key": "value1"},
        {"key": "value2"},
        {"key": "value3"},
        {"key": "value4"},
    ]

    with pytest.raises(ValueError) as exc_info:
        vector_store.add_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    assert "Length of ids, texts, embeddings and metadatas must be the same" in str(
        exc_info.value
    )


def test_add_embeddings(arango_vector_factory: Any) -> None:
    """Test adding embeddings to the vector store."""
    vector_store = arango_vector_factory()

    texts = ["text1", "text2"]
    embeddings = [[0.1] * 64, [0.2] * 64]
    metadatas = [{"key": "value1"}, {"key": "value2"}]

    with patch(
        "langchain_arangodb.vectorstores.arangodb_vector.farmhash.Fingerprint64"
    ) as mock_hash:
        mock_hash.side_effect = ["id1", "id2"]

        ids = vector_store.add_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # Verify import_bulk was called
    vector_store.collection.import_bulk.assert_called()

    # Check the data structure
    call_args = vector_store.collection.import_bulk.call_args_list[0][0][0]
    assert len(call_args) == 2
    assert call_args[0]["_key"] == "id1"
    assert call_args[0]["text"] == "text1"
    assert call_args[0]["embedding"] == embeddings[0]
    assert call_args[0]["key"] == "value1"

    assert call_args[1]["_key"] == "id2"
    assert call_args[1]["text"] == "text2"
    assert call_args[1]["embedding"] == embeddings[1]
    assert call_args[1]["key"] == "value2"

    # Verify the correct IDs were returned
    assert ids == ["id1", "id2"]


def test_add_texts(arango_vector_factory: Any) -> None:
    """Test adding texts to the vector store."""
    vector_store = arango_vector_factory()

    texts = ["text1", "text2"]
    metadatas = [{"key": "value1"}, {"key": "value2"}]

    # Mock the embedding.embed_documents method
    mock_embeddings = [[0.1] * 64, [0.2] * 64]
    vector_store.embedding.embed_documents.return_value = mock_embeddings

    # Mock the add_embeddings method
    with patch.object(
        vector_store, "add_embeddings", return_value=["id1", "id2"]
    ) as mock_add_embeddings:
        ids = vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
        )

    # Verify embed_documents was called with texts
    vector_store.embedding.embed_documents.assert_called_once_with(texts)

    # Verify add_embeddings was called with correct parameters
    mock_add_embeddings.assert_called_once_with(
        texts=texts,
        embeddings=mock_embeddings,
        metadatas=metadatas,
        ids=None,
    )

    # Verify the correct IDs were returned
    assert ids == ["id1", "id2"]


def test_similarity_search(arango_vector_factory: Any) -> None:
    """Test similarity search."""
    vector_store = arango_vector_factory()

    # Mock the embedding.embed_query method
    mock_embedding = [0.1] * 64
    vector_store.embedding.embed_query.return_value = mock_embedding

    # Mock the similarity_search_by_vector method
    expected_docs = [MagicMock(), MagicMock()]
    with patch.object(
        vector_store, "similarity_search_by_vector", return_value=expected_docs
    ) as mock_search_by_vector:
        docs = vector_store.similarity_search(
            query="test query",
            k=2,
            return_fields={"field1", "field2"},
            use_approx=True,
        )

    # Verify embed_query was called with query
    vector_store.embedding.embed_query.assert_called_once_with("test query")

    # Verify similarity_search_by_vector was called with correct parameters
    mock_search_by_vector.assert_called_once_with(
        embedding=mock_embedding,
        k=2,
        return_fields={"field1", "field2"},
        use_approx=True,
        filter_clause="",
        metadata_clause="",
    )

    # Verify the correct documents were returned
    assert docs == expected_docs


def test_similarity_search_with_score(arango_vector_factory: Any) -> None:
    """Test similarity search with score."""
    vector_store = arango_vector_factory()

    # Mock the embedding.embed_query method
    mock_embedding = [0.1] * 64
    vector_store.embedding.embed_query.return_value = mock_embedding

    # Mock the similarity_search_by_vector_with_score method
    expected_results = [(MagicMock(), 0.8), (MagicMock(), 0.6)]
    with patch.object(
        vector_store,
        "similarity_search_by_vector_with_score",
        return_value=expected_results,
    ) as mock_search_by_vector_with_score:
        results = vector_store.similarity_search_with_score(
            query="test query",
            k=2,
            return_fields={"field1", "field2"},
            use_approx=True,
        )

    # Verify embed_query was called with query
    vector_store.embedding.embed_query.assert_called_once_with("test query")

    # Verify similarity_search_by_vector_with_score was called with correct parameters
    mock_search_by_vector_with_score.assert_called_once_with(
        embedding=mock_embedding,
        k=2,
        return_fields={"field1", "field2"},
        use_approx=True,
        filter_clause="",
        metadata_clause="",
    )

    # Verify the correct results were returned
    assert results == expected_results


def test_max_marginal_relevance_search(arango_vector_factory: Any) -> None:
    """Test max marginal relevance search."""
    vector_store = arango_vector_factory()

    # Mock the embedding.embed_query method
    mock_embedding = [0.1] * 64
    vector_store.embedding.embed_query.return_value = mock_embedding

    # Create mock documents and similarity scores
    mock_docs = [MagicMock(), MagicMock(), MagicMock()]
    mock_similarities = [0.9, 0.8, 0.7]

    with (
        patch.object(
            vector_store,
            "similarity_search_by_vector_with_score",
            return_value=list(zip(mock_docs, mock_similarities)),
        ),
        patch(
            "langchain_arangodb.vectorstores.arangodb_vector.maximal_marginal_relevance",
            return_value=[0, 2],  # Indices of selected documents
        ) as mock_mmr,
    ):
        results = vector_store.max_marginal_relevance_search(
            query="test query",
            k=2,
            fetch_k=3,
            lambda_mult=0.5,
        )

    # Verify embed_query was called with query
    vector_store.embedding.embed_query.assert_called_once_with("test query")

    mmr_call_kwargs = mock_mmr.call_args[1]
    assert mmr_call_kwargs["k"] == 2
    assert mmr_call_kwargs["lambda_mult"] == 0.5

    # Verify the selected documents were returned
    assert results == [mock_docs[0], mock_docs[2]]


def test_from_texts(arango_vector_factory: Any) -> None:
    """Test creating vector store from texts."""
    texts = ["text1", "text2"]
    mock_embedding = MagicMock()
    mock_embedding.embed_documents.return_value = [[0.1] * 64, [0.2] * 64]

    # Configure mock_db for this specific test to simulate no pre-existing index
    mock_db_instance = MagicMock(spec=StandardDatabase)
    mock_collection_instance = MagicMock()
    mock_db_instance.collection.return_value = mock_collection_instance
    mock_db_instance.has_collection.return_value = (
        True  # Assume collection exists or is created by __init__
    )
    mock_collection_instance.indexes.return_value = []

    with patch.object(ArangoVector, "add_embeddings", return_value=["id1", "id2"]):
        vector_store = ArangoVector.from_texts(
            texts=texts,
            embedding=mock_embedding,
            database=mock_db_instance,  # Use the specifically configured mock_db
            collection_name="custom_collection",
        )

        # Verify the vector store was initialized correctly
        assert vector_store.collection_name == "custom_collection"
        assert vector_store.embedding == mock_embedding
        assert vector_store.embedding_dimension == 64

        # Note: create_vector_index is not automatically called in from_texts
        # so we don't verify it was called here


def test_delete(arango_vector_factory: Any) -> None:
    """Test deleting documents from the vector store."""
    vector_store = arango_vector_factory()

    # Test deleting specific IDs
    ids = ["id1", "id2"]
    vector_store.delete(ids=ids)

    # Verify collection.delete_many was called with correct IDs
    vector_store.collection.delete_many.assert_called_once()
    # ids are passed as the first positional argument to collection.delete_many
    positional_args = vector_store.collection.delete_many.call_args[0]
    assert set(positional_args[0]) == set(ids)


def test_get_by_ids(arango_vector_factory: Any) -> None:
    """Test getting documents by IDs."""
    vector_store = arango_vector_factory()

    # Test case 1: Multiple documents returned
    # Mock documents to be returned
    mock_docs = [
        {"_key": "id1", "text": "content1", "color": "red", "size": 10},
        {"_key": "id2", "text": "content2", "color": "blue", "size": 20},
    ]

    # Mock collection.get_many to return the mock documents
    vector_store.collection.get_many.return_value = mock_docs

    ids = ["id1", "id2"]
    docs = vector_store.get_by_ids(ids)

    # Verify collection.get_many was called with correct IDs
    vector_store.collection.get_many.assert_called_with(ids)

    # Verify the correct documents were returned
    assert len(docs) == 2
    assert docs[0].page_content == "content1"
    assert docs[0].id == "id1"
    assert docs[0].metadata["color"] == "red"
    assert docs[0].metadata["size"] == 10
    assert docs[1].page_content == "content2"
    assert docs[1].id == "id2"
    assert docs[1].metadata["color"] == "blue"
    assert docs[1].metadata["size"] == 20

    # Test case 2: No documents returned (empty result)
    vector_store.collection.get_many.reset_mock()
    vector_store.collection.get_many.return_value = []

    empty_docs = vector_store.get_by_ids(["non_existent_id"])

    # Verify collection.get_many was called with the non-existent ID
    vector_store.collection.get_many.assert_called_with(["non_existent_id"])

    # Verify an empty list was returned
    assert empty_docs == []

    # Test case 3: Custom text field
    vector_store = arango_vector_factory(text_field="custom_text")

    custom_field_docs = [
        {"_key": "id3", "custom_text": "custom content", "tag": "important"},
    ]

    vector_store.collection.get_many.return_value = custom_field_docs

    result_docs = vector_store.get_by_ids(["id3"])

    # Verify collection.get_many was called
    vector_store.collection.get_many.assert_called_with(["id3"])

    # Verify the document was correctly processed with the custom text field
    assert len(result_docs) == 1
    assert result_docs[0].page_content == "custom content"
    assert result_docs[0].id == "id3"
    assert result_docs[0].metadata["tag"] == "important"

    # Test case 4: Document is missing the text field
    vector_store = arango_vector_factory()

    # Document without the text field
    incomplete_docs = [
        {"_key": "id4", "other_field": "some value"},
    ]

    vector_store.collection.get_many.return_value = incomplete_docs

    # This should raise a KeyError when trying to access the missing text field
    with pytest.raises(KeyError):
        vector_store.get_by_ids(["id4"])


def test_select_relevance_score_fn_override(arango_vector_factory: Any) -> None:
    """Test that the override relevance score function is used if provided."""

    def custom_score_fn(score: float) -> float:
        return score * 10.0

    vector_store = arango_vector_factory(relevance_score_fn=custom_score_fn)
    selected_fn = vector_store._select_relevance_score_fn()
    assert selected_fn(0.5) == 5.0
    assert selected_fn == custom_score_fn


def test_select_relevance_score_fn_default_strategies(
    arango_vector_factory: Any,
) -> None:
    """Test the default relevance score function for supported strategies."""
    # Test for COSINE
    vector_store_cosine = arango_vector_factory(
        distance_strategy=DistanceStrategy.COSINE
    )
    fn_cosine = vector_store_cosine._select_relevance_score_fn()
    assert fn_cosine(0.75) == 0.75

    # Test for EUCLIDEAN_DISTANCE
    vector_store_euclidean = arango_vector_factory(
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    fn_euclidean = vector_store_euclidean._select_relevance_score_fn()
    assert fn_euclidean(1.25) == 1.25


def test_select_relevance_score_fn_invalid_strategy_raises_error(
    arango_vector_factory: Any,
) -> None:
    """Test that an invalid distance strategy raises a ValueError
    if _distance_strategy is mutated post-init."""
    vector_store = arango_vector_factory()
    vector_store._distance_strategy = "INVALID_STRATEGY"

    with pytest.raises(ValueError) as exc_info:
        vector_store._select_relevance_score_fn()

    expected_message = (
        "No supported normalization function for distance_strategy of INVALID_STRATEGY."
        "Consider providing relevance_score_fn to ArangoVector constructor."
    )
    assert str(exc_info.value) == expected_message
