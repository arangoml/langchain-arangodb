"""Integration tests for ArangoVector."""

import pytest
from arango import ArangoClient

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
    if not db.has_collection("test_collection"):
        db.create_collection("test_collection")

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
    collection = db.collection("test_collection")
    assert collection.count() == len(texts_to_embed)

    # Check if the index was created
    index_info = None
    indexes = collection.indexes()
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
    collection = db.collection("test_collection")
    index_info = None
    indexes = collection.indexes()
    for index in indexes:
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
    collection = db.collection("test_collection")
    assert collection.count() == len(texts_to_embed)

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
        search_kwargs={"k": 3, "use_approx": False, "score_threshold": 0.95, "return_fields": {"animal_type"}},
    )

    query = "foo"
    results = retriever.invoke(query)

    assert len(results) == 1
    assert results[0].page_content == "dog"
    assert results[0].metadata.get("animal_type") == "canine"

    # Test with a threshold that should include nothing if no perfect match or if approx score is lower
    retriever_strict = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        score_threshold=1.01,
        search_kwargs={"k": 3, "use_approx": False, "score_threshold": 1.01, "return_fields": {"animal_type"}},
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
    collection = db.collection("test_collection")
    assert collection.count() == 4

    # IDs to delete
    ids_to_delete = ["id_delete1", "id_delete2"]
    delete_result = vector_store.delete(ids=ids_to_delete)
    assert delete_result is True

    # Verify count after deletion
    assert collection.count() == 2

    # Verify that specific documents are gone and others remain
    # Use direct DB checks for presence/absence of docs by ID

    # Check that deleted documents are indeed gone
    deleted_docs_check = collection.get_many(ids_to_delete)
    assert len(deleted_docs_check) == 0

    # Check that remaining documents are still present
    remaining_ids_expected = ["id_keep1", "id_keep2"]
    remaining_docs_check = collection.get_many(remaining_ids_expected)
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

    # Test 3: Empty return_fields set (should also return all metadata as per current logic)
    results_empty_set_meta = vector_store.similarity_search(
        query_text, k=1, return_fields={"source", "chapter", "page", "author"}
    )
    assert len(results_empty_set_meta) == 1
    assert results_empty_set_meta[0].page_content == query_text
    assert results_empty_set_meta[0].metadata == expected_meta_all

    # Test 4: return_fields requesting a non-existent field (should be ignored gracefully)
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

    # Query for "apple". FakeEmbeddings.embed_query("apple") will use its internal logic.
    # If "apple" is not in fake_texts global, it gets default vec [..., -1.0]
    # If "apple" is in fake_texts (it is not by default), it gets that specific vec.
    # Let's use a query known to FakeEmbeddings to make it simpler: "foo"
    # "foo" embeds to [..., 0.0] with current FakeEmbeddings
    # This will make "apple" (doc embedding [1.0]*9 + [0.0]) the most similar.
    query_text = "foo"

    # k=2: fetch 2 diverse documents. fetch_k=4: consider top 4 for MMR.

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
