import json
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_arangodb.vectorstores.arangodb_vector import (
    ArangoVector,
    DistanceStrategy,
    SearchType,
    StandardDatabase,
)


@pytest.fixture
def mock_vector_store() -> ArangoVector:
    """Create a mock ArangoVector instance for testing."""
    mock_db = MagicMock()
    mock_collection = MagicMock()

    mock_db.has_collection.return_value = True
    mock_db.collection.return_value = mock_collection
    mock_db.begin_async_execution.return_value = MagicMock()

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
        async_database: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        mock_db = MagicMock()
        mock_collection = MagicMock()

        # Configure has_collection
        mock_db.has_collection.return_value = collection_exists
        mock_db.collection.return_value = mock_collection
        mock_db.begin_async_execution.return_value = MagicMock()

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
            "async_database": async_database,
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

    expected_message = (
        "distance_strategy must be one of: 'COSINE', 'EUCLIDEAN_DISTANCE', "
        "'JACCARD', 'DOT_PRODUCT', 'MAX_INNER_PRODUCT'"
    )
    assert expected_message in str(exc_info.value)


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


def test_init_with_hybrid_search_type(arango_vector_factory: Any) -> None:
    """Test initialization with hybrid search type."""
    from langchain_arangodb.vectorstores.arangodb_vector import SearchType

    vector_store = arango_vector_factory(search_type=SearchType.HYBRID)
    assert vector_store.search_type == SearchType.HYBRID


def test_similarity_search_hybrid(arango_vector_factory: Any) -> None:
    """Test similarity search with hybrid search type."""
    from langchain_arangodb.vectorstores.arangodb_vector import SearchType

    vector_store = arango_vector_factory(search_type=SearchType.HYBRID)

    # Mock the embedding.embed_query method
    mock_embedding = [0.1] * 64
    vector_store.embedding.embed_query.return_value = mock_embedding

    # Mock the similarity_search_by_vector_and_keyword method
    expected_docs = [MagicMock(), MagicMock()]
    with patch.object(
        vector_store,
        "similarity_search_by_vector_and_keyword",
        return_value=expected_docs,
    ) as mock_hybrid_search:
        docs = vector_store.similarity_search(
            query="test query",
            k=2,
            return_fields={"field1", "field2"},
            use_approx=True,
            vector_weight=1.0,
            keyword_weight=0.5,
        )

    # Verify embed_query was called with query
    vector_store.embedding.embed_query.assert_called_once_with("test query")

    # Verify similarity_search_by_vector_and_keyword was called with correct parameters
    mock_hybrid_search.assert_called_once_with(
        query="test query",
        embedding=mock_embedding,
        k=2,
        return_fields={"field1", "field2"},
        use_approx=True,
        filter_clause="",
        vector_weight=1.0,
        keyword_weight=0.5,
        keyword_search_clause="",
        metadata_clause="",
    )

    # Verify the correct documents were returned
    assert docs == expected_docs


def test_similarity_search_with_score_hybrid(arango_vector_factory: Any) -> None:
    """Test similarity search with score using hybrid search type."""
    from langchain_arangodb.vectorstores.arangodb_vector import SearchType

    vector_store = arango_vector_factory(search_type=SearchType.HYBRID)

    # Mock the embedding.embed_query method
    mock_embedding = [0.1] * 64
    vector_store.embedding.embed_query.return_value = mock_embedding

    # Mock the similarity_search_by_vector_and_keyword_with_score method
    expected_results = [(MagicMock(), 0.8), (MagicMock(), 0.6)]
    with patch.object(
        vector_store,
        "similarity_search_by_vector_and_keyword_with_score",
        return_value=expected_results,
    ) as mock_hybrid_search_with_score:
        results = vector_store.similarity_search_with_score(
            query="test query",
            k=2,
            return_fields={"field1", "field2"},
            use_approx=True,
            vector_weight=2.0,
            keyword_weight=1.5,
            keyword_search_clause="custom clause",
        )
    query = "test query"
    v_store = vector_store
    v_store.embedding.embed_query.assert_called_once_with(query)

    # Verify similarity_search_by_vector_and
    # _keyword_with_score was called with correct parameters
    mock_hybrid_search_with_score.assert_called_once_with(
        query="test query",
        embedding=mock_embedding,
        k=2,
        return_fields={"field1", "field2"},
        use_approx=True,
        filter_clause="",
        vector_weight=2.0,
        keyword_weight=1.5,
        keyword_search_clause="custom clause",
        metadata_clause="",
    )

    # Verify the correct results were returned
    assert results == expected_results


def test_similarity_search_by_vector_and_keyword(arango_vector_factory: Any) -> None:
    """Test similarity_search_by_vector_and_keyword method."""
    vector_store = arango_vector_factory()

    mock_embedding = [0.1] * 64
    expected_docs = [MagicMock(), MagicMock()]

    with patch.object(
        vector_store,
        "similarity_search_by_vector_and_keyword_with_score",
        return_value=[(expected_docs[0], 0.8), (expected_docs[1], 0.6)],
    ) as mock_hybrid_search_with_score:
        docs = vector_store.similarity_search_by_vector_and_keyword(
            query="test query",
            embedding=mock_embedding,
            k=2,
            return_fields={"field1"},
            use_approx=False,
            filter_clause="FILTER doc.type == 'test'",
            vector_weight=1.5,
            keyword_weight=0.8,
            keyword_search_clause="custom search",
        )

    # Verify the method was called with correct parameters
    mock_hybrid_search_with_score.assert_called_once_with(
        query="test query",
        embedding=mock_embedding,
        k=2,
        return_fields={"field1"},
        use_approx=False,
        filter_clause="FILTER doc.type == 'test'",
        vector_weight=1.5,
        keyword_weight=0.8,
        keyword_search_clause="custom search",
        metadata_clause="",
    )

    # Verify only documents (not scores) were returned
    assert docs == expected_docs


def test_similarity_search_by_vector_and_keyword_with_score(
    arango_vector_factory: Any,
) -> None:
    """Test similarity_search_by_vector_and_keyword_with_score method."""
    vector_store = arango_vector_factory()

    mock_embedding = [0.1] * 64
    mock_cursor = MagicMock()
    mock_query = "test query"
    mock_bind_vars = {"test": "value"}

    # Mock _build_hybrid_search_query
    with patch.object(
        vector_store,
        "_build_hybrid_search_query",
        return_value=(mock_query, mock_bind_vars),
    ) as mock_build_query:
        # Mock database execution
        vector_store.db.aql.execute.return_value = mock_cursor

        # Mock _process_search_query
        expected_results = [(MagicMock(), 0.9), (MagicMock(), 0.7)]
        with patch.object(
            vector_store, "_process_search_query", return_value=expected_results
        ) as mock_process:
            results = vector_store.similarity_search_by_vector_and_keyword_with_score(
                query="test query",
                embedding=mock_embedding,
                k=3,
                return_fields={"field1", "field2"},
                use_approx=True,
                filter_clause="FILTER doc.active == true",
                vector_weight=2.0,
                keyword_weight=1.0,
                keyword_search_clause="SEARCH doc.content",
            )

    # Verify _build_hybrid_search_query was called with correct parameters
    mock_build_query.assert_called_once_with(
        query="test query",
        k=3,
        embedding=mock_embedding,
        return_fields={"field1", "field2"},
        use_approx=True,
        filter_clause="FILTER doc.active == true",
        vector_weight=2.0,
        keyword_weight=1.0,
        keyword_search_clause="SEARCH doc.content",
        metadata_clause="",
    )

    # Verify database query execution
    vector_store.db.aql.execute.assert_called_once_with(
        mock_query, bind_vars=mock_bind_vars, stream=True
    )

    # Verify _process_search_query was called
    mock_process.assert_called_once_with(mock_cursor)

    # Verify results
    assert results == expected_results


def test_build_hybrid_search_query(arango_vector_factory: Any) -> None:
    """Test _build_hybrid_search_query method."""
    vector_store = arango_vector_factory(
        collection_name="test_collection",
        keyword_index_name="test_view",
        keyword_analyzer="text_en",
        rrf_constant=60,
        rrf_search_limit=100,
        text_field="text",
        embedding_field="embedding",
    )

    # Mock database version for approx search
    vector_store.db.version.return_value = "3.12.5"

    with (
        patch.object(vector_store, "retrieve_vector_index", return_value=None),
        patch.object(vector_store, "create_vector_index") as mock_create_vector_index,
    ):
        query, bind_vars = vector_store._build_hybrid_search_query(
            query="test query",
            k=5,
            embedding=[0.1] * 64,
            return_fields={"field1", "field2"},
            use_approx=True,
            filter_clause="FILTER doc.active == true",
            vector_weight=1.5,
            keyword_weight=2.0,
            keyword_search_clause="",
            metadata_clause="",
        )

    # Verify vector index was created
    mock_create_vector_index.assert_called_once()

    # Verify query string contains expected components
    assert "FOR doc IN @@collection" in query
    assert "FOR doc IN @@view" in query
    assert "SEARCH ANALYZER" in query
    assert "BM25(doc)" in query
    assert "COLLECT key = result.key AGGREGATE score = SUM(result.score)" in query

    # Verify bind variables
    assert bind_vars["@collection"] == "test_collection"
    assert bind_vars["@view"] == "test_view"
    assert bind_vars["embedding"] == [0.1] * 64
    assert bind_vars["query"] == "test query"
    assert bind_vars["analyzer"] == "text_en"


def test_build_hybrid_search_query_with_custom_keyword_search(
    arango_vector_factory: Any,
) -> None:
    """Test _build_hybrid_search_query with custom keyword search clause."""
    vector_store = arango_vector_factory()

    # Mock dependencies
    with patch.object(
        vector_store, "retrieve_keyword_index", return_value={"name": "test_view"}
    ):
        with patch.object(
            vector_store, "retrieve_vector_index", return_value={"name": "test_index"}
        ):
            vector_store.db.version.return_value = "3.12.5"

            custom_search_clause = "SEARCH doc.title IN TOKENS(@query, @analyzer)"

            query, bind_vars = vector_store._build_hybrid_search_query(
                query="test query",
                k=3,
                embedding=[0.2] * 64,
                return_fields={"title"},
                use_approx=False,
                filter_clause="",
                vector_weight=1.0,
                keyword_weight=1.0,
                keyword_search_clause=custom_search_clause,
                metadata_clause="",
            )

    # Verify custom keyword search clause is used
    assert custom_search_clause in query
    # Verify default search clause is not used
    assert "doc.text IN TOKENS" not in query


def test_keyword_index_management(arango_vector_factory: Any) -> None:
    """Test keyword index creation, retrieval, and deletion."""
    vector_store = arango_vector_factory(
        keyword_index_name="test_keyword_view",
        keyword_analyzer="text_en",
        collection_name="test_collection",
        text_field="content",
    )

    # Test retrieve_keyword_index when index exists
    mock_view = {"name": "test_keyword_view", "type": "arangosearch"}

    with patch.object(vector_store, "retrieve_keyword_index", return_value=mock_view):
        result = vector_store.retrieve_keyword_index()
        assert result == mock_view

    # Test retrieve_keyword_index when index doesn't exist
    with patch.object(vector_store, "retrieve_keyword_index", return_value=None):
        result = vector_store.retrieve_keyword_index()
        assert result is None

    # Test create_keyword_index
    with patch.object(vector_store, "retrieve_keyword_index", return_value=None):
        vector_store.create_keyword_index()

    # Verify create_view was called with correct parameters
    vector_store.db.create_view.assert_called_once()
    call_args = vector_store.db.create_view.call_args
    assert call_args[0][0] == "test_keyword_view"
    assert call_args[0][1] == "search-alias"

    view_properties = call_args[0][2]
    assert view_properties["indexes"][0]["index"] == "test_keyword_view"
    assert view_properties["indexes"][0]["collection"] == "test_collection"

    # Test create_keyword_index when index already exists (idempotent)
    vector_store.db.create_view.reset_mock()
    with patch.object(vector_store, "retrieve_keyword_index", return_value=mock_view):
        vector_store.create_keyword_index()

    # Should not create view if it already exists
    vector_store.db.create_view.assert_not_called()

    # Test delete_keyword_index
    with patch.object(vector_store, "retrieve_keyword_index", return_value=mock_view):
        vector_store.delete_keyword_index()

    vector_store.db.delete_view.assert_called_once_with("test_keyword_view")

    # Test delete_keyword_index when index doesn't exist
    vector_store.db.delete_view.reset_mock()
    with patch.object(vector_store, "retrieve_keyword_index", return_value=None):
        vector_store.delete_keyword_index()

    # Should not call delete_view if view doesn't exist
    vector_store.db.delete_view.assert_not_called()


def test_from_texts_with_hybrid_search_and_invalid_insert_text() -> None:
    """Test that from_texts raises ValueError when
    hybrid search is used without insert_text."""
    mock_embedding = MagicMock()
    mock_embedding.embed_documents.return_value = [[0.1] * 64, [0.2] * 64]
    mock_db = MagicMock()

    from langchain_arangodb.vectorstores.arangodb_vector import ArangoVector, SearchType

    with pytest.raises(ValueError) as exc_info:
        ArangoVector.from_texts(
            texts=["text1", "text2"],
            embedding=mock_embedding,
            database=mock_db,
            search_type=SearchType.HYBRID,
            insert_text=False,  # This should cause the error
        )

    assert "insert_text must be True when search_type is HYBRID" in str(exc_info.value)


def test_from_texts_with_hybrid_search_valid() -> None:
    """Test that from_texts works correctly with hybrid search when insert_text=True."""
    mock_embedding = MagicMock()
    mock_embedding.embed_documents.return_value = [[0.1] * 64, [0.2] * 64]
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.has_collection.return_value = True
    mock_db.collection.return_value = mock_collection
    mock_collection.indexes.return_value = []

    from langchain_arangodb.vectorstores.arangodb_vector import ArangoVector, SearchType

    with patch.object(ArangoVector, "add_embeddings", return_value=["id1", "id2"]):
        vector_store = ArangoVector.from_texts(
            texts=["text1", "text2"],
            embedding=mock_embedding,
            database=mock_db,
            search_type=SearchType.HYBRID,
            insert_text=True,  # This should work
        )

    assert vector_store.search_type == SearchType.HYBRID


def test_from_existing_collection_with_hybrid_search_invalid() -> None:
    """Test that from_existing_collection raises
    error with hybrid search and insert_text=False."""
    mock_embedding = MagicMock()
    mock_db = MagicMock()

    from langchain_arangodb.vectorstores.arangodb_vector import ArangoVector, SearchType

    with pytest.raises(ValueError) as exc_info:
        ArangoVector.from_existing_collection(
            collection_name="test_collection",
            text_properties_to_embed=["title", "content"],
            embedding=mock_embedding,
            database=mock_db,
            search_type=SearchType.HYBRID,
            insert_text=False,  # This should cause the error
        )

    assert "insert_text must be True when search_type is HYBRID" in str(exc_info.value)


def test_build_hybrid_search_query_euclidean_distance(
    arango_vector_factory: Any,
) -> None:
    """Test _build_hybrid_search_query with Euclidean distance strategy."""
    from langchain_arangodb.vectorstores.utils import DistanceStrategy

    vector_store = arango_vector_factory(
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
    )

    # Mock dependencies
    with patch.object(
        vector_store, "retrieve_keyword_index", return_value={"name": "test_view"}
    ):
        with patch.object(
            vector_store, "retrieve_vector_index", return_value={"name": "test_index"}
        ):
            query, bind_vars = vector_store._build_hybrid_search_query(
                query="test",
                k=2,
                embedding=[0.1] * 64,
                return_fields=set(),
                use_approx=False,
                filter_clause="",
                vector_weight=1.0,
                keyword_weight=1.0,
                keyword_search_clause="",
                metadata_clause="",
            )

    # Should use L2_DISTANCE for Euclidean distance
    assert "L2_DISTANCE" in query
    assert "SORT score ASC" in query  # Euclidean uses ascending sort


def test_build_hybrid_search_query_dot_product(
    arango_vector_factory: Any,
) -> None:
    """Test _build_hybrid_search_query with DOT_PRODUCT distance strategy."""
    from langchain_arangodb.vectorstores.utils import DistanceStrategy

    vector_store = arango_vector_factory(distance_strategy=DistanceStrategy.DOT_PRODUCT)

    # Mock dependencies
    with patch.object(
        vector_store, "retrieve_keyword_index", return_value={"name": "test_view"}
    ):
        with patch.object(
            vector_store, "retrieve_vector_index", return_value={"name": "test_index"}
        ):
            query, bind_vars = vector_store._build_hybrid_search_query(
                query="test",
                k=2,
                embedding=[0.1] * 64,
                return_fields=set(),
                use_approx=False,
                filter_clause="",
                vector_weight=0.7,
                keyword_weight=0.3,
                keyword_search_clause="",
                metadata_clause="",
            )

    # Should use manual dot product calculation
    assert "SUM(" in query
    assert "doc.embedding[i] * @embedding[i]" in query
    assert "SORT score DESC" in query
    assert 'WINDOW { preceding: "unbounded", following: 0 }' in query
    assert "AGGREGATE rank = COUNT(1)" in query


def test_build_hybrid_search_query_max_inner_product(
    arango_vector_factory: Any,
) -> None:
    """Test _build_hybrid_search_query with MAX_INNER_PRODUCT distance strategy."""
    from langchain_arangodb.vectorstores.utils import DistanceStrategy

    vector_store = arango_vector_factory(
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )

    # Mock dependencies
    with patch.object(
        vector_store, "retrieve_keyword_index", return_value={"name": "test_view"}
    ):
        with patch.object(
            vector_store, "retrieve_vector_index", return_value={"name": "test_index"}
        ):
            query, bind_vars = vector_store._build_hybrid_search_query(
                query="test",
                k=3,
                embedding=[0.2] * 64,
                return_fields=set(),
                use_approx=False,
                filter_clause="",
                vector_weight=0.6,
                keyword_weight=0.4,
                keyword_search_clause="",
                metadata_clause="",
            )

    assert "SUM(" in query
    assert "doc.embedding[i] * @embedding[i]" in query
    assert "SORT score DESC" in query
    assert "LET scored = (" in query
    assert "LET maxScore = MAX(scored[*].score)" in query
    assert "FILTER item.score == maxScore" in query
    assert "item.doc._key" in query


def test_build_hybrid_search_query_jaccard(
    arango_vector_factory: Any,
) -> None:
    """Test _build_hybrid_search_query with JACCARD distance strategy."""
    from langchain_arangodb.vectorstores.utils import DistanceStrategy

    vector_store = arango_vector_factory(distance_strategy=DistanceStrategy.JACCARD)

    # Mock dependencies
    with patch.object(
        vector_store, "retrieve_keyword_index", return_value={"name": "test_view"}
    ):
        with patch.object(
            vector_store, "retrieve_vector_index", return_value={"name": "test_index"}
        ):
            query, bind_vars = vector_store._build_hybrid_search_query(
                query="test",
                k=2,
                embedding=[0.1] * 64,
                return_fields=set(),
                use_approx=False,
                filter_clause="",
                vector_weight=0.8,
                keyword_weight=0.2,
                keyword_search_clause="",
                metadata_clause="",
            )

    # Should use JACCARD built-in function
    assert "JACCARD" in query
    assert "SORT score DESC" in query
    assert 'WINDOW { preceding: "unbounded", following: 0 }' in query
    assert "AGGREGATE rank = COUNT(1)" in query


def test_build_hybrid_search_query_version_check(arango_vector_factory: Any) -> None:
    """Test that _build_hybrid_search_query checks
    ArangoDB version for approximate search."""
    vector_store = arango_vector_factory()

    # Mock dependencies
    with patch.object(
        vector_store, "retrieve_keyword_index", return_value={"name": "test_view"}
    ):
        with patch.object(vector_store, "retrieve_vector_index", return_value=None):
            # Mock old version
            vector_store.db.version.return_value = "3.12.3"

            with pytest.raises(ValueError) as exc_info:
                vector_store._build_hybrid_search_query(
                    query="test",
                    k=2,
                    embedding=[0.1] * 64,
                    return_fields=set(),
                    use_approx=True,  # This should trigger the version check
                    filter_clause="",
                    vector_weight=1.0,
                    keyword_weight=1.0,
                    keyword_search_clause="",
                    metadata_clause="",
                )

            assert (
                "Approximate Nearest Neighbor search requires ArangoDB >= 3.12.4"
                in str(exc_info.value)
            )


def test_search_type_override_in_similarity_search(arango_vector_factory: Any) -> None:
    """Test that search_type can be overridden in similarity_search method."""
    from langchain_arangodb.vectorstores.arangodb_vector import SearchType

    # Create vector store with default vector search
    vector_store = arango_vector_factory(search_type=SearchType.VECTOR)

    mock_embedding = [0.1] * 64
    vector_store.embedding.embed_query.return_value = mock_embedding

    # Test overriding to hybrid search
    expected_docs = [MagicMock()]
    with patch.object(
        vector_store,
        "similarity_search_by_vector_and_keyword",
        return_value=expected_docs,
    ) as mock_hybrid_search:
        docs = vector_store.similarity_search(
            query="test",
            k=1,
            search_type=SearchType.HYBRID,  # Override default
        )

    # Should call hybrid search method despite default being vector
    mock_hybrid_search.assert_called_once()
    assert docs == expected_docs

    # Test overriding to vector search
    with patch.object(
        vector_store, "similarity_search_by_vector", return_value=expected_docs
    ) as mock_vector_search:
        docs = vector_store.similarity_search(
            query="test",
            k=1,
            search_type=SearchType.VECTOR,  # Explicit vector search
        )

    mock_vector_search.assert_called_once()
    assert docs == expected_docs


def _make_sync_batch_cursor(items: list) -> Any:
    """Stateful cursor mock for _iter_cursor: drives empty()/pop()/has_more()."""
    remaining = list(items)

    cursor = MagicMock()
    cursor.empty.side_effect = lambda: len(remaining) == 0
    cursor.pop.side_effect = lambda: remaining.pop(0)
    cursor.has_more.return_value = False
    return cursor


def test_jaccard_distance_strategy(arango_vector_factory: Any) -> None:
    """Test JACCARD calculation with actual vectors and query generation."""
    vector_store = arango_vector_factory(distance_strategy=DistanceStrategy.JACCARD)

    query_vector = [1.0, 1.0, 0.0]

    # Test query generation
    query, bind_vars = vector_store._build_vector_search_query(
        embedding=query_vector,
        k=1,
        return_fields={"text"},
        use_approx=False,
        filter_clause="",
        metadata_clause="",
    )

    # Verify JACCARD function is used correctly
    assert "JACCARD(doc.embedding, @embedding)" in query
    assert bind_vars["embedding"] == query_vector
    assert "DESC" in query

    # Test mathematical correctness with real document data
    docs = [
        {"_key": "doc1", "text": "identical", "embedding": [1.0, 1.0, 0.0]},
        {"_key": "doc2", "text": "partial", "embedding": [1.0, 0.0, 1.0]},
        {"_key": "doc3", "text": "different", "embedding": [0.0, 0.0, 1.0]},
    ]

    mock_cursor = _make_sync_batch_cursor(
        [
            {"data": docs[0], "score": 1.0, "metadata": {}},
            {"data": docs[1], "score": 0.33, "metadata": {}},
            {"data": docs[2], "score": 0.0, "metadata": {}},
        ]
    )
    vector_store.db.aql.execute.return_value = mock_cursor

    # Query with vector [1,1,0]
    results = vector_store.similarity_search_with_score("test", k=3, use_approx=False)
    scores = [score for _, score in results]

    # Verify results
    assert scores == [1.0, 0.33, 0.0]
    assert scores[0] > scores[1] > scores[2]
    assert len(results) == 3


def test_dot_product_distance_strategy(arango_vector_factory: Any) -> None:
    """Test DOT_PRODUCT calculation with actual vectors and query generation."""
    vector_store = arango_vector_factory(distance_strategy=DistanceStrategy.DOT_PRODUCT)

    query_vector = [1.0, 2.0, 3.0]

    # Test query generation
    query, bind_vars = vector_store._build_vector_search_query(
        embedding=query_vector,
        k=1,
        return_fields={"text"},
        use_approx=False,
        filter_clause="",
        metadata_clause="",
    )

    # Verify dot product formula: sum of element-wise multiplication
    assert "SUM(" in query
    assert "doc.embedding[i] * @embedding[i]" in query
    assert "FOR i IN 0..LENGTH(doc.embedding)-1" in query
    assert bind_vars["embedding"] == query_vector
    assert "DESC" in query

    # Test mathematical correctness with real document data
    docs = [
        {"_key": "doc1", "text": "high_score", "embedding": [2.0, 3.0, 1.0]},
        {"_key": "doc2", "text": "medium_score", "embedding": [1.0, 1.0, 1.0]},
        {"_key": "doc3", "text": "low_score", "embedding": [1.0, 0.0, 0.0]},
    ]

    mock_cursor = _make_sync_batch_cursor(
        [
            {"data": docs[0], "score": 11.0, "metadata": {}},
            {"data": docs[1], "score": 6.0, "metadata": {}},
            {"data": docs[2], "score": 1.0, "metadata": {}},
        ]
    )
    vector_store.db.aql.execute.return_value = mock_cursor

    # Query with vector [1,2,3]
    results = vector_store.similarity_search_with_score("test", k=3, use_approx=False)
    scores = [score for _, score in results]

    # Verify results
    assert scores == [11.0, 6.0, 1.0]
    assert scores[0] > scores[1] > scores[2]
    assert len(results) == 3


def test_max_inner_product_strategy(arango_vector_factory: Any) -> None:
    """Test MAX_INNER_PRODUCT calculation with actual vectors and query generation."""
    vector_store = arango_vector_factory(
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )

    query_vector = [1.0, 2.0, 3.0]

    # Test query generation
    query, bind_vars = vector_store._build_vector_search_query(
        embedding=query_vector,
        k=1,
        return_fields={"text"},
        use_approx=False,
        filter_clause="",
        metadata_clause="",
    )

    # Verify inner product calculation (similar to dot product)
    assert "SUM(" in query
    assert "doc.embedding[i] * @embedding[i]" in query
    assert bind_vars["embedding"] == query_vector
    assert "DESC" in query

    # Test mathematical correctness with real document data
    docs = [
        {"_key": "doc1", "text": "high_score", "embedding": [2.0, 3.0, 1.0]},
        {"_key": "doc2", "text": "medium_score", "embedding": [1.0, 1.0, 1.0]},
        {"_key": "doc3", "text": "low_score", "embedding": [1.0, 0.0, 0.0]},
    ]

    mock_cursor = _make_sync_batch_cursor(
        [
            {"data": docs[0], "score": 11.0, "metadata": {}},
            {"data": docs[1], "score": 6.0, "metadata": {}},
            {"data": docs[2], "score": 1.0, "metadata": {}},
        ]
    )
    vector_store.db.aql.execute.return_value = mock_cursor

    # Query with vector [1, 2, 3]
    results = vector_store.similarity_search_with_score("test", k=3, use_approx=False)
    scores = [score for _, score in results]

    # Verify inner product calculations
    assert scores == [11.0, 6.0, 1.0]
    assert len(results) == 3
    assert scores[0] > scores[1] > scores[2]


def test_distance_strategy_reject_approximate_search(
    arango_vector_factory: Any,
) -> None:
    """Test new metrics raise exception for approximate search."""
    test_vector = [1.0, 2.0, 3.0]

    for strategy in [
        DistanceStrategy.JACCARD,
        DistanceStrategy.DOT_PRODUCT,
        DistanceStrategy.MAX_INNER_PRODUCT,
    ]:
        vector_store = arango_vector_factory(distance_strategy=strategy)

        expected_error = (
            f"Unsupported metric: {strategy} is not supported for approximate search"
        )
        with pytest.raises(
            ValueError,
            match=expected_error,
        ):
            vector_store._build_vector_search_query(
                embedding=test_vector,
                k=1,
                return_fields={"text"},
                use_approx=True,
                filter_clause="",
                metadata_clause="",
            )


def test_similarity_search_returns_iterator_with_stream(
    arango_vector_factory: Any,
) -> None:
    """Test that similarity_search returns iterator when stream=True."""
    from langchain_core.documents import Document

    vector_store = arango_vector_factory()

    mock_embedding = [0.1] * 64
    vector_store.embedding.embed_query.return_value = mock_embedding

    # Mock similarity_search_by_vector to return an iterator
    mock_iterator = iter([MagicMock(spec=Document), MagicMock(spec=Document)])
    with patch.object(
        vector_store, "similarity_search_by_vector", return_value=mock_iterator
    ):
        result = vector_store.similarity_search("query", k=2, stream=True)

        # Should be an iterator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")


def test_similarity_search_returns_list_without_stream(
    arango_vector_factory: Any,
) -> None:
    """Test that similarity_search returns list when stream is not set."""
    from langchain_core.documents import Document

    vector_store = arango_vector_factory()

    mock_embedding = [0.1] * 64
    vector_store.embedding.embed_query.return_value = mock_embedding

    # Mock to return a list
    mock_docs = [MagicMock(spec=Document), MagicMock(spec=Document)]
    with patch.object(
        vector_store, "similarity_search_by_vector", return_value=mock_docs
    ):
        result = vector_store.similarity_search("query", k=2)

        # Should be a list
        assert isinstance(result, list)


def test_similarity_search_passes_stream_only_when_true(
    arango_vector_factory: Any,
) -> None:
    """Test that stream parameter is only included in kwargs when True."""
    vector_store = arango_vector_factory()

    mock_embedding = [0.1] * 64
    vector_store.embedding.embed_query.return_value = mock_embedding

    # Test stream=True
    with patch.object(
        vector_store, "similarity_search_by_vector", return_value=iter([])
    ) as mock_search:
        vector_store.similarity_search("query", k=2, stream=True)

        # Should include stream=True
        call_kwargs = mock_search.call_args[1]
        assert "stream" in call_kwargs
        assert call_kwargs["stream"] is True

    # Test stream=False
    with patch.object(
        vector_store, "similarity_search_by_vector", return_value=[]
    ) as mock_search:
        vector_store.similarity_search("query", k=2, stream=False)

        # Should NOT include stream in kwargs
        call_kwargs = mock_search.call_args[1]
        assert "stream" not in call_kwargs

    # Test stream=None (default)
    with patch.object(
        vector_store, "similarity_search_by_vector", return_value=[]
    ) as mock_search:
        vector_store.similarity_search("query", k=2)

        # Should NOT include stream in kwargs
        call_kwargs = mock_search.call_args[1]
        assert "stream" not in call_kwargs


# ---------------------------------------------------------------------------
# use_async_db tests (python-arango fire-and-forget job queue)
# ---------------------------------------------------------------------------


def test_add_embeddings_use_async_db_false(arango_vector_factory: Any) -> None:
    """When use_async_db=False (default), import_bulk is called on the sync db."""
    vector_store = arango_vector_factory()

    texts = ["text1", "text2"]
    embeddings = [[0.1] * 64, [0.2] * 64]

    vector_store.add_embeddings(texts=texts, embeddings=embeddings, use_async_db=False)

    # The sync collection (via self.db) must have been used
    vector_store.db.collection.assert_called_with(vector_store.collection_name)
    vector_store.collection.import_bulk.assert_called()


def test_add_embeddings_use_async_db_true(arango_vector_factory: Any) -> None:
    """
    When use_async_db=True, import_bulk is called on the arango async-execution db.
    """
    mock_async_exec_db = MagicMock()
    mock_async_exec_collection = MagicMock()
    mock_async_exec_db.collection.return_value = mock_async_exec_collection

    vector_store = arango_vector_factory()
    # Patch the internal _arango_async_db (begin_async_execution result)
    vector_store._arango_async_db = mock_async_exec_db

    texts = ["text1", "text2"]
    embeddings = [[0.1] * 64, [0.2] * 64]

    vector_store.add_embeddings(texts=texts, embeddings=embeddings, use_async_db=True)

    # The async-execution collection must have been used, not the sync one
    mock_async_exec_db.collection.assert_called_with(vector_store.collection_name)
    mock_async_exec_collection.import_bulk.assert_called()
    # The sync collection must NOT have been called for import_bulk
    vector_store.collection.import_bulk.assert_not_called()


def test_add_embeddings_use_async_db_default_is_false(
    arango_vector_factory: Any,
) -> None:
    """use_async_db defaults to False — sync db is used when arg is omitted."""
    vector_store = arango_vector_factory()

    mock_async_exec_db = MagicMock()
    vector_store._arango_async_db = mock_async_exec_db

    vector_store.add_embeddings(texts=["t"], embeddings=[[0.1] * 64])

    # Async-execution db must NOT have been touched
    mock_async_exec_db.collection.assert_not_called()
    vector_store.collection.import_bulk.assert_called()


def test_arango_async_db_set_on_init(arango_vector_factory: Any) -> None:
    """_arango_async_db is set via begin_async_execution during __init__."""
    vector_store = arango_vector_factory()
    vector_store.db.begin_async_execution.assert_called_once_with(return_result=False)
    assert vector_store._arango_async_db is not None


def test_use_async_db_and_async_database_are_independent(
    arango_vector_factory: Any,
) -> None:
    """use_async_db (sync driver job queue) and async_database (true async) are
    independent: one can be set without the other."""
    mock_async_exec_db = MagicMock()
    mock_true_async_db = MagicMock()

    vector_store = arango_vector_factory(async_database=mock_true_async_db)
    vector_store._arango_async_db = mock_async_exec_db

    # async_database is stored for true-async methods
    assert vector_store.async_db is mock_true_async_db
    # _arango_async_db is the fire-and-forget job queue
    assert vector_store._arango_async_db is mock_async_exec_db


# ---------------------------------------------------------------------------
# Async unit tests
# ---------------------------------------------------------------------------


def _make_async_vector_store(async_db: Any = None, **kwargs: Any) -> ArangoVector:
    """Helper: build an ArangoVector with a sync mock db and optional async_db."""
    mock_db = MagicMock()
    mock_db.has_collection.return_value = True
    mock_db.collection.return_value = MagicMock()
    return ArangoVector(
        embedding=MagicMock(),
        embedding_dimension=64,
        database=mock_db,
        async_database=async_db,
        **kwargs,
    )


def _make_async_db() -> MagicMock:
    """Return a MagicMock that mimics arangoasync StandardDatabase."""
    async_db = MagicMock()
    async_collection = MagicMock()
    async_collection.import_bulk = AsyncMock(return_value={})  # type: ignore[method-assign]
    async_collection.delete_many = AsyncMock(return_value=[])  # type: ignore[method-assign]
    async_collection.get_many = AsyncMock(return_value=[])  # type: ignore[method-assign]
    async_db.collection.return_value = async_collection
    async_db.has_collection = AsyncMock(return_value=True)  # type: ignore[method-assign]
    async_db.view = AsyncMock(return_value={"name": "keyword_index"})  # type: ignore[method-assign]
    async_db.aql = MagicMock()
    async_db.aql.execute = AsyncMock()  # type: ignore[method-assign]
    return async_db


def _make_async_batch_iterator(batches: list) -> Any:
    """Create a cursor mock simulating arangoasync cursor batch API."""

    class AsyncBatchCursor:
        def __init__(self, batches: list) -> None:
            self._batches = list(batches)
            self._index = 0

        def empty(self) -> bool:
            return (
                self._index >= len(self._batches)
                or len(self._batches[self._index]) == 0
            )

        @property
        def batch(self) -> list:
            return self._batches[self._index]

        @property
        def has_more(self) -> bool:
            return self._index + 1 < len(self._batches)

        async def fetch(self) -> list:
            self._index += 1
            return self._batches[self._index]

    return AsyncBatchCursor(batches)


# --- guard: async methods raise when async_db is None ---


async def test_aadd_texts_requires_async_db() -> None:
    store = _make_async_vector_store()
    store.embedding.aembed_documents = AsyncMock(return_value=[[0.1] * 64])  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="async_database must be provided"):
        await store.aadd_texts(["hello"])
    # Verify aembed_documents was not called (validation happens first)
    store.embedding.aembed_documents.assert_not_called()


async def test_adelete_requires_async_db() -> None:
    store = _make_async_vector_store()
    with pytest.raises(ValueError, match="async_database must be provided"):
        await store.adelete(ids=["id1"])


async def test_aget_by_ids_requires_async_db() -> None:
    store = _make_async_vector_store()
    with pytest.raises(ValueError, match="async_database must be provided"):
        await store.aget_by_ids(["id1"])


async def test_asimilarity_search_with_score_requires_async_db() -> None:
    store = _make_async_vector_store()
    store.embedding.aembed_query = AsyncMock(return_value=[0.1] * 64)  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="async_database must be provided"):
        await store.asimilarity_search_with_score("query")


# --- happy-path async tests ---


async def test_aadd_texts_calls_import_bulk() -> None:
    async_db = _make_async_db()
    store = _make_async_vector_store(async_db=async_db)
    embedding_1, embedding_2 = [0.1] * 64, [0.2] * 64
    store.embedding.aembed_documents = AsyncMock(  # type: ignore[method-assign]
        return_value=[embedding_1, embedding_2]
    )

    ids = await store.aadd_texts(["text1", "text2"])

    assert len(ids) == 2
    async_db.collection.return_value.import_bulk.assert_called_once()

    # Verify the payload is a list (arangoasync requirement)
    call_arg = async_db.collection.return_value.import_bulk.call_args[0][0]
    parsed = json.loads(call_arg)
    assert isinstance(parsed, list)
    assert parsed[0]["text"] == "text1"
    assert parsed[1]["text"] == "text2"
    assert parsed[0]["embedding"] == embedding_1
    assert parsed[1]["embedding"] == embedding_2


async def test_adelete_calls_delete_many() -> None:
    async_db = _make_async_db()
    store = _make_async_vector_store(async_db=async_db)

    result = await store.adelete(ids=["id1", "id2"])

    assert result is True
    async_db.collection.return_value.delete_many.assert_called_once_with(["id1", "id2"])


async def test_adelete_returns_none_for_empty_ids() -> None:
    async_db = _make_async_db()
    store = _make_async_vector_store(async_db=async_db)

    result = await store.adelete(ids=[])

    assert result is None
    async_db.collection.return_value.delete_many.assert_not_called()


async def test_aget_by_ids_returns_documents() -> None:
    async_db = _make_async_db()
    async_db.collection.return_value.get_many = AsyncMock(
        return_value=[
            {"_key": "id1", "text": "hello", "meta": "val"},
        ]
    )
    store = _make_async_vector_store(async_db=async_db)

    docs = await store.aget_by_ids(["id1"])

    assert len(docs) == 1
    assert docs[0].page_content == "hello"
    assert docs[0].id == "id1"
    assert docs[0].metadata["meta"] == "val"


async def test_asimilarity_search_delegates_to_with_score() -> None:
    async_db = _make_async_db()
    store = _make_async_vector_store(async_db=async_db)

    expected = [MagicMock(), MagicMock()]
    with patch.object(
        store,
        "asimilarity_search_with_score",
        new=AsyncMock(return_value=[(d, 0.9) for d in expected]),
    ) as mock_ws:
        docs = await store.asimilarity_search("query", k=2)

    mock_ws.assert_called_once_with("query", k=2)
    assert docs == expected


async def test_asimilarity_search_with_score_vector() -> None:
    async_db = _make_async_db()
    store = _make_async_vector_store(async_db=async_db)
    store.embedding.aembed_query = AsyncMock(return_value=[0.1] * 64)  # type: ignore[method-assign]

    # Use async batch iterator to mock the cursor
    mock_cursor = _make_async_batch_iterator([[]])
    async_db.aql.execute = AsyncMock(return_value=mock_cursor)  # type: ignore[method-assign]

    with patch.object(
        store,
        "_async_build_vector_search_query",
        new=AsyncMock(return_value=("AQL", {"bind": "vars"})),
    ):
        results = await store.asimilarity_search_with_score(
            "query", k=2, search_type=SearchType.VECTOR
        )

    async_db.aql.execute.assert_called_once_with("AQL", bind_vars={"bind": "vars"})
    assert results == []


async def test_asimilarity_search_with_score_hybrid() -> None:
    async_db = _make_async_db()
    store = _make_async_vector_store(async_db=async_db, search_type=SearchType.HYBRID)
    store.embedding.aembed_query = AsyncMock(return_value=[0.1] * 64)  # type: ignore[method-assign]

    # Use async batch iterator to mock the cursor
    mock_cursor = _make_async_batch_iterator([[]])
    async_db.aql.execute = AsyncMock(return_value=mock_cursor)  # type: ignore[method-assign]

    with patch.object(
        store,
        "_async_build_hybrid_search_query",
        new=AsyncMock(return_value=("AQL_HYBRID", {"bind": "vars"})),
    ):
        results = await store.asimilarity_search_with_score(
            "query", k=2, search_type=SearchType.HYBRID
        )

    async_db.aql.execute.assert_called_once_with(
        "AQL_HYBRID", bind_vars={"bind": "vars"}
    )
    assert results == []


async def test_asimilarity_search_by_vector_delegates() -> None:
    async_db = _make_async_db()
    store = _make_async_vector_store(async_db=async_db)

    expected_doc = MagicMock()
    with patch.object(
        store,
        "asimilarity_search_with_score",
        new=AsyncMock(return_value=[(expected_doc, 0.8)]),
    ) as mock_ws:
        docs = await store.asimilarity_search_by_vector([0.1] * 64, k=1)

    mock_ws.assert_called_once()
    assert docs == [expected_doc]


async def test_amax_marginal_relevance_search() -> None:
    async_db = _make_async_db()
    store = _make_async_vector_store(async_db=async_db)
    store.embedding.aembed_query = AsyncMock(return_value=[0.1] * 64)  # type: ignore[method-assign]

    mock_docs = [MagicMock() for _ in range(3)]
    for i, doc in enumerate(mock_docs):
        doc.metadata = {store.embedding_field: [float(i)] * 64}

    with (
        patch.object(
            store,
            "asimilarity_search_by_vector",
            new=AsyncMock(return_value=mock_docs),
        ),
        patch(
            "langchain_arangodb.vectorstores.arangodb_vector.maximal_marginal_relevance",
            return_value=[0, 2],
        ) as mock_mmr,
    ):
        results = await store.amax_marginal_relevance_search(
            "query", k=2, fetch_k=3, lambda_mult=0.5
        )

    mock_mmr.assert_called_once()
    assert results == [mock_docs[0], mock_docs[2]]


async def test_async_process_search_query_single_page() -> None:
    """
    Test _async_process_search_query with a cursor that has all results in one page.
    """
    store = _make_async_vector_store()

    # Mock cursor as async iterator
    mock_batch = [
        {
            "data": {"_key": "k1", "text": "hello", "extra": "x"},
            "score": 0.9,
            "metadata": {},
        },
        {
            "data": {"_key": "k2", "text": "world", "extra": "y"},
            "score": 0.7,
            "metadata": {"m": 1},
        },
    ]

    # Create async iterator mock
    mock_cursor = _make_async_batch_iterator([mock_batch])

    results = await store._async_process_search_query(mock_cursor)

    assert len(results) == 2
    doc1, score1 = results[0]
    assert doc1.page_content == "hello"
    assert doc1.id == "k1"
    assert score1 == 0.9

    doc2, score2 = results[1]
    assert doc2.page_content == "world"
    assert doc2.metadata["m"] == 1
    assert score2 == 0.7


async def test_async_process_search_query_paginated() -> None:
    """Test _async_process_search_query fetches next pages via async iteration."""
    store = _make_async_vector_store()

    mock_batches = [
        [{"data": {"_key": "k1", "text": "page1"}, "score": 0.9, "metadata": {}}],
        [{"data": {"_key": "k2", "text": "page2"}, "score": 0.8, "metadata": {}}],
    ]

    mock_cursor = _make_async_batch_iterator(mock_batches)

    results = await store._async_process_search_query(mock_cursor)

    assert len(results) == 2
    assert results[0][0].page_content == "page1"
    assert results[1][0].page_content == "page2"


def test_async_database_stored_on_init() -> None:
    """Test that async_database is stored as self.async_db."""
    mock_async_db = MagicMock()
    store = _make_async_vector_store(async_db=mock_async_db)
    assert store.async_db is mock_async_db


def test_async_db_none_by_default() -> None:
    """Test that async_db defaults to None when not provided."""
    store = _make_async_vector_store()
    assert store.async_db is None
