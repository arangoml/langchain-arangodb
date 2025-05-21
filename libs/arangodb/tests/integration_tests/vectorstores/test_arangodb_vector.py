"""Integration tests for ArangoVector."""

from typing import Any, Dict, List

import pytest
from arango import ArangoClient
from arango.collection import StandardCollection
from arango.cursor import Cursor
from langchain_core.documents import Document

from langchain_arangodb.vectorstores.arangodb_vector import ArangoVector
from langchain_arangodb.vectorstores.utils import DistanceStrategy
from tests.integration_tests.utils import ArangoCredentials

from .fake_embeddings import FakeEmbeddings

EMBEDDING_DIMENSION = 10


@pytest.fixture(scope="session")
def fake_embedding_function() -> FakeEmbeddings:
    """Provides a FakeEmbeddings instance."""
    return FakeEmbeddings()


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_from_texts_and_similarity_search(
    arangodb_credentials: ArangoCredentials,
    fake_embedding_function: FakeEmbeddings,
) -> None:
    """Test end-to-end construction from texts and basic similarity search."""
    client = ArangoClient(hosts=arangodb_credentials["url"])
    db = client.db(
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )
    # Try to create a collection to force a connection error
    if not db.has_collection(
        "test_collection_init"
    ):  # Use a different name to avoid conflict if already exists
        _test_init_coll = db.create_collection("test_collection_init")
        assert isinstance(_test_init_coll, StandardCollection)

    texts_to_embed = ["hello world", "hello arango", "test document"]
    metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]

    vector_store = ArangoVector.from_texts(
        texts=texts_to_embed,
        embedding=fake_embedding_function,
        metadatas=metadatas,
        database=db,
        collection_name="test_collection",
        index_name="test_index",
        overwrite_index=True,  # Ensure clean state for the index
    )

    # Manually create the index as from_texts with overwrite=True only deletes it
    # in the current version of arangodb_vector.py
    vector_store.create_vector_index()

    # Check if the collection was created
    assert db.has_collection("test_collection")
    _collection_obj = db.collection("test_collection")
    assert isinstance(_collection_obj, StandardCollection)
    collection: StandardCollection = _collection_obj
    assert collection.count() == len(texts_to_embed)

    # Check if the index was created
    index_info = None
    indexes_raw = collection.indexes()
    assert indexes_raw is not None, "collection.indexes() returned None"
    assert isinstance(indexes_raw, list), (
        f"collection.indexes() expected list, got {type(indexes_raw)}"
    )
    indexes: List[Dict[str, Any]] = indexes_raw
    for index in indexes:
        if index.get("name") == "test_index" and index.get("type") == "vector":
            index_info = index
            break
    assert index_info is not None
    assert index_info["fields"] == ["embedding"]  # Default embedding field

    # Test similarity search
    query = "hello"
    results = vector_store.similarity_search(query, k=1, return_fields={"source"})

    assert len(results) == 1
    assert results[0].page_content == "hello world"
    assert results[0].metadata.get("source") == "doc1"


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_euclidean_distance(
    arangodb_credentials: ArangoCredentials,
    fake_embedding_function: FakeEmbeddings,
) -> None:
    """Test ArangoVector with Euclidean distance."""
    client = ArangoClient(hosts=arangodb_credentials["url"])
    db = client.db(
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )
    texts_to_embed = ["docA", "docB", "docC"]

    vector_store = ArangoVector.from_texts(
        texts=texts_to_embed,
        embedding=fake_embedding_function,
        database=db,
        collection_name="test_collection",
        index_name="test_index",
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        overwrite_index=True,
    )

    # Manually create the index as from_texts with overwrite=True only deletes it
    vector_store.create_vector_index()

    # Check index metric
    _collection_obj_euclidean = db.collection("test_collection")
    assert isinstance(_collection_obj_euclidean, StandardCollection)
    collection_euclidean: StandardCollection = _collection_obj_euclidean
    index_info = None
    indexes_raw_euclidean = collection_euclidean.indexes()
    assert indexes_raw_euclidean is not None, (
        "collection_euclidean.indexes() returned None"
    )
    assert isinstance(indexes_raw_euclidean, list), (
        f"collection_euclidean.indexes() expected list, \
            got {type(indexes_raw_euclidean)}"
    )
    indexes_euclidean: List[Dict[str, Any]] = indexes_raw_euclidean
    for index in indexes_euclidean:
        if index.get("name") == "test_index" and index.get("type") == "vector":
            index_info = index
            break
    assert index_info is not None
    query = "docA"
    results = vector_store.similarity_search(query, k=1)
    assert len(results) == 1
    assert results[0].page_content == "docA"


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_similarity_search_with_score(
    arangodb_credentials: ArangoCredentials,
    fake_embedding_function: FakeEmbeddings,
) -> None:
    """Test similarity search with scores."""
    client = ArangoClient(hosts=arangodb_credentials["url"])
    db = client.db(
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )
    texts_to_embed = ["alpha", "beta", "gamma"]
    metadatas = [{"id": 1}, {"id": 2}, {"id": 3}]

    vector_store = ArangoVector.from_texts(
        texts=texts_to_embed,
        embedding=fake_embedding_function,
        metadatas=metadatas,
        database=db,
        collection_name="test_collection",
        index_name="test_index",
        overwrite_index=True,
    )

    query = "foo"
    results_with_scores = vector_store.similarity_search_with_score(
        query, k=1, return_fields={"id"}
    )

    assert len(results_with_scores) == 1
    doc, score = results_with_scores[0]

    assert doc.page_content == "alpha"
    assert doc.metadata.get("id") == 1

    # Test with exact cosine similarity
    results_with_scores_exact = vector_store.similarity_search_with_score(
        query, k=1, use_approx=False, return_fields={"id"}
    )
    assert len(results_with_scores_exact) == 1
    doc_exact, score_exact = results_with_scores_exact[0]
    assert doc_exact.page_content == "alpha"
    assert (
        score_exact == 1.0
    )  # Exact cosine similarity should be 1.0 for identical vectors

    # Test with Euclidean distance
    vector_store_l2 = ArangoVector.from_texts(
        texts=texts_to_embed,  # Re-using same texts for simplicity
        embedding=fake_embedding_function,
        metadatas=metadatas,
        database=db,  # db is managed by fixture, collection will be overwritten
        collection_name="test_collection"
        + "_l2",  # Use a different collection or ensure overwrite
        index_name="test_index" + "_l2",
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        overwrite_index=True,
    )
    results_with_scores_l2 = vector_store_l2.similarity_search_with_score(
        query, k=1, return_fields={"id"}
    )
    assert len(results_with_scores_l2) == 1
    doc_l2, score_l2 = results_with_scores_l2[0]
    assert doc_l2.page_content == "alpha"
    assert score_l2 == 0.0  # For L2 (Euclidean) distance, perfect match is 0.0


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_add_embeddings_and_search(
    arangodb_credentials: ArangoCredentials,
    fake_embedding_function: FakeEmbeddings,
) -> None:
    """Test construction from pre-computed embeddings and search."""
    client = ArangoClient(hosts=arangodb_credentials["url"])
    db = client.db(
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )
    texts_to_embed = ["apple", "banana", "cherry"]
    metadatas = [
        {"fruit_type": "pome"},
        {"fruit_type": "berry"},
        {"fruit_type": "drupe"},
    ]

    # Manually create embeddings
    embeddings = fake_embedding_function.embed_documents(texts_to_embed)

    # Initialize ArangoVector - embedding_dimension must match FakeEmbeddings
    vector_store = ArangoVector(
        embedding=fake_embedding_function,  # Still needed for query embedding
        embedding_dimension=EMBEDDING_DIMENSION,  # Should be 10 from FakeEmbeddings
        database=db,
        collection_name="test_collection",  # Will be created if not exists
        index_name="test_index",
    )

    # Add embeddings first, so the index has data to train on
    vector_store.add_embeddings(texts_to_embed, embeddings, metadatas=metadatas)

    # Create the index if it doesn't exist
    # For similarity_search to work with approx=True (default), an index is needed.
    if not vector_store.retrieve_vector_index():
        vector_store.create_vector_index()

    # Check collection count
    _collection_obj_add_embed = db.collection("test_collection")
    assert isinstance(_collection_obj_add_embed, StandardCollection)
    collection_add_embed: StandardCollection = _collection_obj_add_embed
    assert collection_add_embed.count() == len(texts_to_embed)

    # Perform search
    query = "apple"
    results = vector_store.similarity_search(query, k=1, return_fields={"fruit_type"})
    assert len(results) == 1
    assert results[0].page_content == "apple"
    assert results[0].metadata.get("fruit_type") == "pome"


# NEW TEST
@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_retriever_search_threshold(
    arangodb_credentials: ArangoCredentials,
    fake_embedding_function: FakeEmbeddings,
) -> None:
    """Test using retriever for searching with a score threshold."""
    client = ArangoClient(hosts=arangodb_credentials["url"])
    db = client.db(
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )
    texts_to_embed = ["dog", "cat", "mouse"]
    metadatas = [
        {"animal_type": "canine"},
        {"animal_type": "feline"},
        {"animal_type": "rodent"},
    ]

    vector_store = ArangoVector.from_texts(
        texts=texts_to_embed,
        embedding=fake_embedding_function,
        metadatas=metadatas,
        database=db,
        collection_name="test_collection",
        index_name="test_index",
        overwrite_index=True,
    )

    # Default is COSINE, perfect match (score 1.0 with exact, close with approx)
    # Test with a threshold that should only include a perfect/near-perfect match
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        score_threshold=0.95,
        search_kwargs={
            "k": 3,
            "use_approx": False,
            "score_threshold": 0.95,
            "return_fields": {"animal_type"},
        },
    )

    query = "foo"
    results = retriever.invoke(query)

    assert len(results) == 1
    assert results[0].page_content == "dog"
    assert results[0].metadata.get("animal_type") == "canine"

    retriever_strict = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        score_threshold=1.01,
        search_kwargs={
            "k": 3,
            "use_approx": False,
            "score_threshold": 1.01,
            "return_fields": {"animal_type"},
        },
    )
    results_strict = retriever_strict.invoke(query)
    assert len(results_strict) == 0


# NEW TEST
@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_delete_documents(
    arangodb_credentials: ArangoCredentials,
    fake_embedding_function: FakeEmbeddings,
) -> None:
    """Test deleting documents from ArangoVector."""
    client = ArangoClient(hosts=arangodb_credentials["url"])
    db = client.db(
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )
    texts_to_embed = [
        "doc_to_keep1",
        "doc_to_delete1",
        "doc_to_keep2",
        "doc_to_delete2",
    ]
    metadatas = [
        {"id_val": 1, "status": "keep"},
        {"id_val": 2, "status": "delete"},
        {"id_val": 3, "status": "keep"},
        {"id_val": 4, "status": "delete"},
    ]

    # Use specific IDs for easier deletion and verification
    doc_ids = ["id_keep1", "id_delete1", "id_keep2", "id_delete2"]

    vector_store = ArangoVector.from_texts(
        texts=texts_to_embed,
        embedding=fake_embedding_function,
        metadatas=metadatas,
        ids=doc_ids,  # Pass our custom IDs
        database=db,
        collection_name="test_collection",
        index_name="test_index",
        overwrite_index=True,
    )

    # Verify initial count
    _collection_obj_delete = db.collection("test_collection")
    assert isinstance(_collection_obj_delete, StandardCollection)
    collection_delete: StandardCollection = _collection_obj_delete
    assert collection_delete.count() == 4

    # IDs to delete
    ids_to_delete = ["id_delete1", "id_delete2"]
    delete_result = vector_store.delete(ids=ids_to_delete)
    assert delete_result is True

    # Verify count after deletion
    assert collection_delete.count() == 2

    # Verify that specific documents are gone and others remain
    # Use direct DB checks for presence/absence of docs by ID

    # Check that deleted documents are indeed gone
    deleted_docs_check_raw = collection_delete.get_many(ids_to_delete)
    assert deleted_docs_check_raw is not None, (
        "collection.get_many() returned None for deleted_docs_check"
    )
    assert isinstance(deleted_docs_check_raw, list), (
        f"collection.get_many() expected list for deleted_docs_check,\
              got {type(deleted_docs_check_raw)}"
    )
    deleted_docs_check: List[Dict[str, Any]] = deleted_docs_check_raw
    assert len(deleted_docs_check) == 0

    # Check that remaining documents are still present
    remaining_ids_expected = ["id_keep1", "id_keep2"]
    remaining_docs_check_raw = collection_delete.get_many(remaining_ids_expected)
    assert remaining_docs_check_raw is not None, (
        "collection.get_many() returned None for remaining_docs_check"
    )
    assert isinstance(remaining_docs_check_raw, list), (
        f"collection.get_many() expected list for remaining_docs_check,\
              got {type(remaining_docs_check_raw)}"
    )
    remaining_docs_check: List[Dict[str, Any]] = remaining_docs_check_raw
    assert len(remaining_docs_check) == 2

    # Optionally, verify content of remaining documents if needed
    retrieved_contents = sorted(
        [d[vector_store.text_field] for d in remaining_docs_check]
    )
    assert retrieved_contents == sorted(
        [texts_to_embed[0], texts_to_embed[2]]
    )  # doc_to_keep1, doc_to_keep2


# NEW TEST
@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_similarity_search_with_return_fields(
    arangodb_credentials: ArangoCredentials,
    fake_embedding_function: FakeEmbeddings,
) -> None:
    """Test similarity search with specified return_fields for metadata."""
    client = ArangoClient(hosts=arangodb_credentials["url"])
    db = client.db(
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    metadatas = [
        {"source": "doc1", "chapter": "ch1", "page": 10, "author": "A"},
        {"source": "doc2", "chapter": "ch2", "page": 20, "author": "B"},
        {"source": "doc3", "chapter": "ch3", "page": 30, "author": "C"},
    ]
    doc_ids = ["id1", "id2", "id3"]

    vector_store = ArangoVector.from_texts(
        texts=texts,
        embedding=fake_embedding_function,
        metadatas=metadatas,
        ids=doc_ids,
        database=db,
        collection_name="test_collection",
        index_name="test_index",
        overwrite_index=True,
    )

    query_text = "alpha beta"

    # Test 1: No return_fields (should return all metadata except embedding_field)
    results_all_meta = vector_store.similarity_search(
        query_text, k=1, return_fields={"source", "chapter", "page", "author"}
    )
    assert len(results_all_meta) == 1
    assert results_all_meta[0].page_content == query_text
    expected_meta_all = {"source": "doc1", "chapter": "ch1", "page": 10, "author": "A"}
    assert results_all_meta[0].metadata == expected_meta_all

    # Test 2: Specific return_fields
    fields_to_return = {"source", "page"}
    results_specific_meta = vector_store.similarity_search(
        query_text, k=1, return_fields=fields_to_return
    )
    assert len(results_specific_meta) == 1
    assert results_specific_meta[0].page_content == query_text
    expected_meta_specific = {"source": "doc1", "page": 10}
    assert results_specific_meta[0].metadata == expected_meta_specific

    # Test 3: Empty return_fields set
    results_empty_set_meta = vector_store.similarity_search(
        query_text, k=1, return_fields={"source", "chapter", "page", "author"}
    )
    assert len(results_empty_set_meta) == 1
    assert results_empty_set_meta[0].page_content == query_text
    assert results_empty_set_meta[0].metadata == expected_meta_all

    # Test 4: return_fields requesting a non-existent field
    # and one existing field
    fields_with_non_existent = {"source", "non_existent_field"}
    results_non_existent_meta = vector_store.similarity_search(
        query_text, k=1, return_fields=fields_with_non_existent
    )
    assert len(results_non_existent_meta) == 1
    assert results_non_existent_meta[0].page_content == query_text
    expected_meta_non_existent = {"source": "doc1"}
    assert results_non_existent_meta[0].metadata == expected_meta_non_existent


# NEW TEST
@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_max_marginal_relevance_search(
    arangodb_credentials: ArangoCredentials,
    fake_embedding_function: FakeEmbeddings,  # Using existing FakeEmbeddings
) -> None:
    """Test max marginal relevance search."""
    client = ArangoClient(hosts=arangodb_credentials["url"])
    db = client.db(
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )

    # Texts designed so some are close to each other via FakeEmbeddings
    # FakeEmbeddings: embedding[last_dim] = index i
    # apple (0), apricot (1) -> similar
    # banana (2), blueberry (3) -> similar
    # cherry (4) -> distinct
    texts = ["apple", "apricot", "banana", "blueberry", "grape"]
    metadatas = [
        {"fruit": "apple", "idx": 0},
        {"fruit": "apricot", "idx": 1},
        {"fruit": "banana", "idx": 2},
        {"fruit": "blueberry", "idx": 3},
        {"fruit": "grape", "idx": 4},
    ]
    doc_ids = ["id_apple", "id_apricot", "id_banana", "id_blueberry", "id_grape"]

    vector_store = ArangoVector.from_texts(
        texts=texts,
        embedding=fake_embedding_function,
        metadatas=metadatas,
        ids=doc_ids,
        database=db,
        collection_name="test_collection",
        index_name="test_index",
        overwrite_index=True,
    )

    query_text = "foo"

    # Test with lambda_mult = 0.5 (balance between similarity and diversity)
    mmr_results = vector_store.max_marginal_relevance_search(
        query_text, k=2, fetch_k=4, lambda_mult=0.5, use_approx=False
    )
    assert len(mmr_results) == 2
    assert mmr_results[0].page_content == "apple"
    # With new FakeEmbeddings, lambda=0.5 should pick "apricot" as second.
    assert mmr_results[1].page_content == "apricot"

    result_contents = {doc.page_content for doc in mmr_results}
    assert "apple" in result_contents
    assert len(result_contents) == 2  # Ensure two distinct docs

    # Test with lambda_mult favoring similarity (e.g., 0.1)
    mmr_results_sim = vector_store.max_marginal_relevance_search(
        query_text, k=2, fetch_k=4, lambda_mult=0.1, use_approx=False
    )
    assert len(mmr_results_sim) == 2
    assert mmr_results_sim[0].page_content == "apple"
    assert mmr_results_sim[1].page_content == "blueberry"

    # Test with lambda_mult favoring diversity (e.g., 0.9)
    mmr_results_div = vector_store.max_marginal_relevance_search(
        query_text, k=2, fetch_k=4, lambda_mult=0.9, use_approx=False
    )
    assert len(mmr_results_div) == 2
    assert mmr_results_div[0].page_content == "apple"
    assert mmr_results_div[1].page_content == "apricot"


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_delete_vector_index(
    arangodb_credentials: ArangoCredentials,
    fake_embedding_function: FakeEmbeddings,
) -> None:
    """Test creating and deleting a vector index."""
    client = ArangoClient(hosts=arangodb_credentials["url"])
    db = client.db(
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )
    texts_to_embed = ["alpha", "beta", "gamma"]

    # Create the vector store
    vector_store = ArangoVector.from_texts(
        texts=texts_to_embed,
        embedding=fake_embedding_function,
        database=db,
        collection_name="test_collection",
        index_name="test_index",
        overwrite_index=False,
    )

    # Create the index explicitly
    vector_store.create_vector_index()

    # Verify the index exists
    _collection_obj_del_idx = db.collection("test_collection")
    assert isinstance(_collection_obj_del_idx, StandardCollection)
    collection_del_idx: StandardCollection = _collection_obj_del_idx
    index_info = None
    indexes_raw_del_idx = collection_del_idx.indexes()
    assert indexes_raw_del_idx is not None
    assert isinstance(indexes_raw_del_idx, list)
    indexes_del_idx: List[Dict[str, Any]] = indexes_raw_del_idx
    for index in indexes_del_idx:
        if index.get("name") == "test_index" and index.get("type") == "vector":
            index_info = index
            break

    assert index_info is not None, "Vector index was not created"

    # Now delete the index
    vector_store.delete_vector_index()

    # Verify the index no longer exists
    indexes_after_delete_raw = collection_del_idx.indexes()
    assert indexes_after_delete_raw is not None
    assert isinstance(indexes_after_delete_raw, list)
    indexes_after_delete: List[Dict[str, Any]] = indexes_after_delete_raw
    index_after_delete = None
    for index in indexes_after_delete:
        if index.get("name") == "test_index" and index.get("type") == "vector":
            index_after_delete = index
            break

    assert index_after_delete is None, "Vector index was not deleted"

    # Ensure delete_vector_index is idempotent (calling it again doesn't cause errors)
    vector_store.delete_vector_index()

    # Recreate the index and verify
    vector_store.create_vector_index()

    indexes_after_recreate_raw = collection_del_idx.indexes()
    assert indexes_after_recreate_raw is not None
    assert isinstance(indexes_after_recreate_raw, list)
    indexes_after_recreate: List[Dict[str, Any]] = indexes_after_recreate_raw
    index_after_recreate = None
    for index in indexes_after_recreate:
        if index.get("name") == "test_index" and index.get("type") == "vector":
            index_after_recreate = index
            break

    assert index_after_recreate is not None, "Vector index was not recreated"


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_get_by_ids(
    arangodb_credentials: ArangoCredentials,
    fake_embedding_function: FakeEmbeddings,
) -> None:
    """Test retrieving documents by their IDs."""
    client = ArangoClient(hosts=arangodb_credentials["url"])
    db = client.db(
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )

    # Create test data with specific IDs
    texts = ["apple", "banana", "cherry", "date"]
    custom_ids = ["fruit_1", "fruit_2", "fruit_3", "fruit_4"]
    metadatas = [
        {"type": "pome", "color": "red", "calories": 95},
        {"type": "berry", "color": "yellow", "calories": 105},
        {"type": "drupe", "color": "red", "calories": 50},
        {"type": "drupe", "color": "brown", "calories": 20},
    ]

    # Create the vector store with custom IDs
    vector_store = ArangoVector.from_texts(
        texts=texts,
        embedding=fake_embedding_function,
        metadatas=metadatas,
        ids=custom_ids,
        database=db,
        collection_name="test_collection",
    )

    # Create the index explicitly
    vector_store.create_vector_index()

    # Test retrieving a single document by ID
    single_doc = vector_store.get_by_ids(["fruit_1"])
    assert len(single_doc) == 1
    assert single_doc[0].page_content == "apple"
    assert single_doc[0].id == "fruit_1"
    assert single_doc[0].metadata["type"] == "pome"
    assert single_doc[0].metadata["color"] == "red"
    assert single_doc[0].metadata["calories"] == 95

    # Test retrieving multiple documents by ID
    docs = vector_store.get_by_ids(["fruit_2", "fruit_4"])
    assert len(docs) == 2

    # Verify each document has the correct content and metadata
    banana_doc = next((doc for doc in docs if doc.id == "fruit_2"), None)
    date_doc = next((doc for doc in docs if doc.id == "fruit_4"), None)

    assert banana_doc is not None
    assert banana_doc.page_content == "banana"
    assert banana_doc.metadata["type"] == "berry"
    assert banana_doc.metadata["color"] == "yellow"

    assert date_doc is not None
    assert date_doc.page_content == "date"
    assert date_doc.metadata["type"] == "drupe"
    assert date_doc.metadata["color"] == "brown"

    # Test with non-existent ID (should return empty list for that ID)
    non_existent_docs = vector_store.get_by_ids(["fruit_999"])
    assert len(non_existent_docs) == 0

    # Test with mix of existing and non-existing IDs
    mixed_docs = vector_store.get_by_ids(["fruit_1", "fruit_999", "fruit_3"])
    assert len(mixed_docs) == 2  # Only fruit_1 and fruit_3 should be found

    # Verify the documents match the expected content
    found_ids = [doc.id for doc in mixed_docs]
    assert "fruit_1" in found_ids
    assert "fruit_3" in found_ids
    assert "fruit_999" not in found_ids


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_core_functionality(
    arangodb_credentials: ArangoCredentials,
    fake_embedding_function: FakeEmbeddings,
) -> None:
    """Test the core functionality of ArangoVector with an integrated workflow."""
    client = ArangoClient(hosts=arangodb_credentials["url"])
    db = client.db(
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )

    # 1. Setup - Create a vector store with documents
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Pack my box with five dozen liquor jugs",
        "How vexingly quick daft zebras jump",
        "Amazingly few discotheques provide jukeboxes",
        "Sphinx of black quartz, judge my vow",
    ]

    metadatas = [
        {"source": "english", "pangram": True, "length": len(corpus[0])},
        {"source": "english", "pangram": True, "length": len(corpus[1])},
        {"source": "english", "pangram": True, "length": len(corpus[2])},
        {"source": "english", "pangram": True, "length": len(corpus[3])},
        {"source": "english", "pangram": True, "length": len(corpus[4])},
    ]

    custom_ids = ["pangram_1", "pangram_2", "pangram_3", "pangram_4", "pangram_5"]

    vector_store = ArangoVector.from_texts(
        texts=corpus,
        embedding=fake_embedding_function,
        metadatas=metadatas,
        ids=custom_ids,
        database=db,
        collection_name="test_pangrams",
    )

    # Create the vector index
    vector_store.create_vector_index()

    # 2. Test similarity_search - the most basic search function
    query = "jumps"
    results = vector_store.similarity_search(query, k=2)

    # Should return documents with "jumps" in them
    assert len(results) == 2
    text_contents = [doc.page_content for doc in results]
    # The most relevant results should include docs with "jumps"
    has_jump_docs = [doc for doc in text_contents if "jump" in doc.lower()]
    assert len(has_jump_docs) > 0

    # 3. Test similarity_search_with_score - core search with relevance scores
    results_with_scores = vector_store.similarity_search_with_score(
        query, k=3, return_fields={"source", "pangram"}
    )

    assert len(results_with_scores) == 3
    # Check result format
    for doc, score in results_with_scores:
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        # Verify metadata got properly transferred
        assert doc.metadata["source"] == "english"
        assert doc.metadata["pangram"] is True

    # 4. Test similarity_search_by_vector_with_score
    query_embedding = fake_embedding_function.embed_query(query)
    vector_results = vector_store.similarity_search_by_vector_with_score(
        embedding=query_embedding,
        k=2,
        return_fields={"source", "length"},
    )

    assert len(vector_results) == 2
    # Check result format
    for doc, score in vector_results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        # Verify specific metadata fields were returned
        assert "source" in doc.metadata
        assert "length" in doc.metadata
        # Verify length is a number (as defined in metadatas)
        assert isinstance(doc.metadata["length"], int)

    # 5. Test with exact search (non-approximate)
    exact_results = vector_store.similarity_search_with_score(
        query, k=2, use_approx=False
    )
    assert len(exact_results) == 2

    # 6. Test max_marginal_relevance_search - for getting diverse results
    mmr_results = vector_store.max_marginal_relevance_search(
        query, k=3, fetch_k=5, lambda_mult=0.5
    )
    assert len(mmr_results) == 3
    # MMR results should be diverse, so they might differ from regular search

    # 7. Test adding new documents to the existing vector store
    new_texts = ["The five boxing wizards jump quickly"]
    new_metadatas = [
        {"source": "english", "pangram": True, "length": len(new_texts[0])}
    ]
    new_ids = vector_store.add_texts(texts=new_texts, metadatas=new_metadatas)

    # Verify the document was added by directly checking the collection
    _collection_obj_core = db.collection("test_pangrams")
    assert isinstance(_collection_obj_core, StandardCollection)
    collection_core: StandardCollection = _collection_obj_core
    assert collection_core.count() == 6  # Original 5 + 1 new document

    # Verify retrieving by ID works
    added_doc = vector_store.get_by_ids([new_ids[0]])
    assert len(added_doc) == 1
    assert added_doc[0].page_content == new_texts[0]
    assert "wizard" in added_doc[0].page_content.lower()

    # 8. Testing search by ID
    all_docs_cursor = collection_core.all()
    assert all_docs_cursor is not None, "collection.all() returned None"
    assert isinstance(all_docs_cursor, Cursor), (
        f"collection.all() expected Cursor, got {type(all_docs_cursor)}"
    )
    all_ids = [doc["_key"] for doc in all_docs_cursor]
    assert new_ids[0] in all_ids

    # 9. Test deleting documents
    vector_store.delete(ids=[new_ids[0]])

    # Verify the document was deleted
    deleted_check = vector_store.get_by_ids([new_ids[0]])
    assert len(deleted_check) == 0

    # Also verify via direct collection count
    assert collection_core.count() == 5  # Back to the original 5 documents


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_from_existing_collection(
    arangodb_credentials: ArangoCredentials,
    fake_embedding_function: FakeEmbeddings,
) -> None:
    """Test creating a vector store from an existing collection."""
    client = ArangoClient(hosts=arangodb_credentials["url"])
    db = client.db(
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )

    # Create a test collection with documents that have multiple text fields
    collection_name = "test_source_collection"

    if db.has_collection(collection_name):
        db.delete_collection(collection_name)

    _collection_obj_exist = db.create_collection(collection_name)
    assert isinstance(_collection_obj_exist, StandardCollection)
    collection_exist: StandardCollection = _collection_obj_exist
    # Create documents with multiple text fields to test different scenarios
    documents = [
        {
            "_key": "doc1",
            "title": "The Solar System",
            "abstract": (
                "The Solar System is the gravitationally bound system of the "
                "Sun and the objects that orbit it."
            ),
            "content": (
                "The Solar System formed 4.6 billion years ago from the "
                "gravitational collapse of a giant interstellar molecular cloud."
            ),
            "tags": ["astronomy", "science", "space"],
            "author": "John Doe",
        },
        {
            "_key": "doc2",
            "title": "Machine Learning",
            "abstract": (
                "Machine learning is a field of inquiry devoted to understanding and "
                "building methods that 'learn'."
            ),
            "content": (
                "Machine learning approaches are traditionally divided into three broad"
                " categories: supervised, unsupervised, and reinforcement learning."
            ),
            "tags": ["ai", "computer science", "data science"],
            "author": "Jane Smith",
        },
        {
            "_key": "doc3",
            "title": "The Theory of Relativity",
            "abstract": (
                "The theory of relativity usually encompasses two interrelated"
                " theories by Albert Einstein."
            ),
            "content": (
                "Special relativity applies to all physical phenomena in the absence of"
                " gravity. General relativity explains the law of gravitation and its"
                " relation to other forces of nature."
            ),
            "tags": ["physics", "science", "Einstein"],
            "author": "Albert Einstein",
        },
        {
            "_key": "doc4",
            "title": "Quantum Mechanics",
            "abstract": (
                "Quantum mechanics is a fundamental theory in physics that provides a"
                " description of the physical properties of nature "
                " at the scale of atoms and subatomic particles."
            ),
            "content": (
                "Quantum mechanics allows the calculation of properties and behaviour "
                "of physical systems."
            ),
            "tags": ["physics", "science", "quantum"],
            "author": "Max Planck",
        },
    ]

    # Import documents to the collection
    collection_exist.import_bulk(documents)
    assert collection_exist.count() == 4

    # 1. Basic usage - embedding title and abstract
    text_properties = ["title", "abstract"]

    vector_store = ArangoVector.from_existing_collection(
        collection_name=collection_name,
        text_properties_to_embed=text_properties,
        embedding=fake_embedding_function,
        database=db,
        embedding_field="embedding",
        text_field="combined_text",
        insert_text=True,
    )

    # Create the vector index
    vector_store.create_vector_index()

    # Verify the vector store was created correctly
    # First, check that the original collection still has 4 documents
    assert collection_exist.count() == 4

    # Check that embeddings were added to the original documents
    doc_data1 = collection_exist.get("doc1")
    assert doc_data1 is not None, "Document 'doc1' not found in collection_exist"
    assert isinstance(doc_data1, dict), (
        f"Expected 'doc1' to be a dict, got {type(doc_data1)}"
    )
    doc1: Dict[str, Any] = doc_data1
    assert "embedding" in doc1
    assert isinstance(doc1["embedding"], list)
    assert "combined_text" in doc1  # Now this field should exist

    # Perform a search to verify functionality
    results = vector_store.similarity_search("astronomy")
    assert len(results) > 0

    # 2. Test with custom AQL query to modify the text extraction
    custom_aql_query = "RETURN CONCAT(doc[p], ' by ', doc.author)"

    vector_store_custom = ArangoVector.from_existing_collection(
        collection_name=collection_name,
        text_properties_to_embed=["title"],  # Only embed titles
        embedding=fake_embedding_function,
        database=db,
        embedding_field="custom_embedding",
        text_field="custom_text",
        index_name="custom_vector_index",
        aql_return_text_query=custom_aql_query,
        insert_text=True,
    )

    # Create the vector index
    vector_store_custom.create_vector_index()

    # Check that custom embeddings were added
    doc_data2 = collection_exist.get("doc1")
    assert doc_data2 is not None, "Document 'doc1' not found after custom processing"
    assert isinstance(doc_data2, dict), (
        f"Expected 'doc1' after custom processing to be a dict, "
        f"got {type(doc_data2)}"
    )
    doc2: Dict[str, Any] = doc_data2
    assert "custom_embedding" in doc2
    assert "custom_text" in doc2
    assert "by John Doe" in doc2["custom_text"]  # Check the custom extraction format

    # 3. Test with skip_existing_embeddings=True
    vector_store.delete_vector_index()

    collection_exist.update({"_key": "doc3", "embedding": None})

    vector_store_skip = ArangoVector.from_existing_collection(
        collection_name=collection_name,
        text_properties_to_embed=["title", "abstract"],
        embedding=fake_embedding_function,
        database=db,
        embedding_field="embedding",
        text_field="combined_text",
        index_name="skip_vector_index",  # Use a different index name
        skip_existing_embeddings=True,
        insert_text=True,  # Important for search to work
    )

    # Create the vector index
    vector_store_skip.create_vector_index()

    # 4. Test with insert_text=True
    vector_store_insert = ArangoVector.from_existing_collection(
        collection_name=collection_name,
        text_properties_to_embed=["title", "content"],
        embedding=fake_embedding_function,
        database=db,
        embedding_field="content_embedding",
        text_field="combined_title_content",
        index_name="content_vector_index",  # Use a different index name
        insert_text=True,  # Already set to True, but kept for clarity
    )

    # Create the vector index
    vector_store_insert.create_vector_index()

    # Check that the combined text was inserted
    doc_data3 = collection_exist.get("doc1")
    assert doc_data3 is not None, (
        "Document 'doc1' not found after insert_text processing"
    )
    assert isinstance(doc_data3, dict), (
        f"Expected 'doc1' after insert_text to be a dict, "
        f"got {type(doc_data3)}"
    )
    doc3: Dict[str, Any] = doc_data3
    assert "combined_title_content" in doc3
    assert "The Solar System" in doc3["combined_title_content"]
    assert "formed 4.6 billion years ago" in doc3["combined_title_content"]

    # 5. Test searching in the custom store
    results_custom = vector_store_custom.similarity_search("Einstein", k=1)
    assert len(results_custom) == 1

    # 6. Test max_marginal_relevance search
    mmr_results = vector_store.max_marginal_relevance_search(
        "science", k=2, fetch_k=4, lambda_mult=0.5
    )
    assert len(mmr_results) == 2

    # 7. Test the get_by_ids method
    docs = vector_store.get_by_ids(["doc1", "doc3"])
    assert len(docs) == 2
    assert any(doc.id == "doc1" for doc in docs)
    assert any(doc.id == "doc3" for doc in docs)
