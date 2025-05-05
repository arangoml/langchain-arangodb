"""Integration tests for ArangoDB vector store (ArangoVector)."""

import time
from typing import Any, Dict, List, Optional

import pytest
from arango.database import StandardDatabase
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Import ArangoDB specific classes
from langchain_arangodb.vectorstores.arangodb_vector import (
    ArangoVector,
    DEFAULT_DISTANCE_STRATEGY,
    DistanceStrategy,
)

# Import FakeEmbeddings from the new location
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

# Test data
TEST_COLLECTION_NAME = "langchain_list_test_integration"
TEST_INDEX_NAME = "langchain_test_vector_index"
TEST_DIMENSION = 10
texts_to_embed = ["foo", "bar", "baz", "qux is extra"]
metadatas_to_embed = [{"page": i} for i in range(len(texts_to_embed))]


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    # Add test_texts attribute for embed_query logic
    emb = FakeEmbeddings(dim=TEST_DIMENSION)
    emb.test_texts = texts_to_embed
    return emb


# --- Basic Functionality Tests ---

@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_from_texts(db: StandardDatabase, embeddings: FakeEmbeddings) -> None:
    """Test end-to-end construction and search."""
    docsearch = ArangoVector.from_texts(
        texts=texts_to_embed,
        embedding=embeddings,
        embedding_dimension=TEST_DIMENSION,
        database=db,
        collection_name=TEST_COLLECTION_NAME,
        index_name=TEST_INDEX_NAME,
        # overwrite_index=True # Ensure clean state (though clear_db helps)
    )
    # Allow time for index creation if needed
    time.sleep(1)
    output = docsearch.similarity_search("foo", k=1)
    assert len(output) == 1
    assert output[0].page_content == "foo"

@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_from_texts_with_metadatas(db: StandardDatabase, embeddings: FakeEmbeddings) -> None:
    """Test construction with metadata."""
    docsearch = ArangoVector.from_texts(
        texts=texts_to_embed,
        embedding=embeddings,
        embedding_dimension=TEST_DIMENSION,
        database=db,
        collection_name=TEST_COLLECTION_NAME,
        index_name=TEST_INDEX_NAME,
        metadatas=metadatas_to_embed
    )
    time.sleep(1)
    output = docsearch.similarity_search("bar", k=1)
    assert len(output) == 1
    assert output[0].page_content == "bar"
    assert output[0].metadata == {"page": 1}

@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_from_embeddings(db: StandardDatabase, embeddings: FakeEmbeddings) -> None:
    """Test construction from pre-computed embeddings."""
    embedded_texts = embeddings.embed_documents(texts_to_embed)
    text_embedding_pairs = list(zip(texts_to_embed, embedded_texts))
    docsearch = ArangoVector.from_embeddings(
        text_embeddings=text_embedding_pairs, # type: ignore - ArangoVector expects this specific type
        embedding=embeddings,
        embedding_dimension=TEST_DIMENSION,
        database=db,
        collection_name=TEST_COLLECTION_NAME,
        index_name=TEST_INDEX_NAME,
    )
    time.sleep(1)
    output = docsearch.similarity_search("baz", k=1)
    assert len(output) == 1
    assert output[0].page_content == "baz"

# --- Search Variations ---

@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_similarity_search_with_score(db: StandardDatabase, embeddings: FakeEmbeddings) -> None:
    """Test similarity search with scores."""
    docsearch = ArangoVector.from_texts(
        texts=texts_to_embed,
        embedding=embeddings,
        embedding_dimension=TEST_DIMENSION,
        database=db,
        collection_name=TEST_COLLECTION_NAME,
        index_name=TEST_INDEX_NAME,
        metadatas=metadatas_to_embed
    )
    time.sleep(1)
    results = docsearch.similarity_search_with_score("qux is extra", k=2)
    assert len(results) == 2
    doc1, score1 = results[0]
    doc2, score2 = results[1]
    assert doc1.page_content == "qux is extra"
    assert doc1.metadata == {"page": 3}
    assert score1 > 0.9 # Cosine score should be close to 1 for exact match
    # Score ordering depends on distance metric, assert the closest is first
    assert score1 >= score2

@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_mmr(db: StandardDatabase, embeddings: FakeEmbeddings) -> None:
    """Test max marginal relevance search."""
    # Add more diverse texts for MMR to be meaningful
    diverse_texts = texts_to_embed + ["another foo", "yet another bar", "unrelated"]
    docsearch = ArangoVector.from_texts(
        texts=diverse_texts,
        embedding=embeddings,
        embedding_dimension=TEST_DIMENSION,
        database=db,
        collection_name=TEST_COLLECTION_NAME,
        index_name=TEST_INDEX_NAME,
    )
    time.sleep(1)
    # Query for "foo", expecting "foo" first, then diverse results
    output = docsearch.max_marginal_relevance_search("foo", k=3, fetch_k=5)
    assert len(output) == 3
    assert output[0].page_content == "foo" # Closest match
    # The other results should be diverse
    page_contents = {doc.page_content for doc in output}
    assert "foo" in page_contents
    assert len(page_contents.intersection({"bar", "baz", "qux is extra", "another foo", "yet another bar", "unrelated"})) == 2


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_euclidean_distance(db: StandardDatabase, embeddings: FakeEmbeddings) -> None:
    """Test with Euclidean distance."""
    docsearch = ArangoVector.from_texts(
        texts=texts_to_embed,
        embedding=embeddings,
        embedding_dimension=TEST_DIMENSION,
        database=db,
        collection_name=TEST_COLLECTION_NAME,
        index_name=TEST_INDEX_NAME,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    time.sleep(1)
    output = docsearch.similarity_search("baz", k=1)
    assert len(output) == 1
    assert output[0].page_content == "baz"

# --- Index/Collection Handling Tests ---

@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_existing_collection_no_index(db: StandardDatabase, embeddings: FakeEmbeddings) -> None:
    """Test connecting when collection exists but index needs creation."""
    # 1. Create collection manually (or via first ArangoVector instance)
    db.create_collection(TEST_COLLECTION_NAME)
    assert db.has_collection(TEST_COLLECTION_NAME)

    # 2. Instantiate ArangoVector - it should create the index
    docsearch = ArangoVector(
        embedding=embeddings,
        embedding_dimension=TEST_DIMENSION,
        database=db,
        collection_name=TEST_COLLECTION_NAME,
        index_name=TEST_INDEX_NAME,
    )
    time.sleep(1) # Allow index creation

    # 3. Verify index exists
    index = docsearch.retrieve_vector_index()
    assert index is not None
    assert index["name"] == TEST_INDEX_NAME

    # 4. Add data and search
    docsearch.add_texts(["test data"])
    output = docsearch.similarity_search("test data", k=1)
    assert len(output) == 1
    assert output[0].page_content == "test data"

@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_existing_index(db: StandardDatabase, embeddings: FakeEmbeddings) -> None:
    """Test connecting when collection and index already exist."""
    # 1. Create instance to ensure collection and index are created
    initial_docsearch = ArangoVector.from_texts(
        texts=["initial data"],
        embedding=embeddings,
        embedding_dimension=TEST_DIMENSION,
        database=db,
        collection_name=TEST_COLLECTION_NAME,
        index_name=TEST_INDEX_NAME,
    )
    time.sleep(1)
    initial_index = initial_docsearch.retrieve_vector_index()
    assert initial_index is not None

    # 2. Create second instance connecting to existing index
    docsearch = ArangoVector(
        embedding=embeddings,
        embedding_dimension=TEST_DIMENSION,
        database=db,
        collection_name=TEST_COLLECTION_NAME,
        index_name=TEST_INDEX_NAME,
    )

    # 3. Verify it uses the existing index (e.g., dimension check pass)
    assert docsearch.retrieve_vector_index() == initial_index

    # 4. Search should find the initial data
    output = docsearch.similarity_search("initial data", k=1)
    assert len(output) == 1
    assert output[0].page_content == "initial data"

    # 5. Add more data
    docsearch.add_texts(["more data"])
    output_more = docsearch.similarity_search("more data", k=1)
    assert len(output_more) == 1
    assert output_more[0].page_content == "more data"

# --- Deletion Test ---
@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangovector_delete(db: StandardDatabase, embeddings: FakeEmbeddings) -> None:
    """Test deleting data."""
    ids_to_keep = ["id_keep1", "id_keep2"]
    ids_to_delete = ["id_del1", "id_del2"]
    all_ids = ids_to_keep + ids_to_delete
    texts_for_ids = [f"text_{i}" for i in all_ids]

    docsearch = ArangoVector.from_texts(
        texts=texts_for_ids,
        embedding=embeddings,
        embedding_dimension=TEST_DIMENSION,
        database=db,
        collection_name=TEST_COLLECTION_NAME,
        index_name=TEST_INDEX_NAME,
        ids=all_ids
    )
    time.sleep(1)

    # Verify initial state
    output_keep = docsearch.get_by_ids(ids_to_keep)
    output_del = docsearch.get_by_ids(ids_to_delete)
    assert len(output_keep) == len(ids_to_keep)
    assert len(output_del) == len(ids_to_delete)

    # Delete
    delete_result = docsearch.delete(ids=ids_to_delete)
    assert delete_result is True # Assuming method returns boolean success

    # Verify deletion
    output_keep_after = docsearch.get_by_ids(ids_to_keep)
    output_del_after = docsearch.get_by_ids(ids_to_delete)
    assert len(output_keep_after) == len(ids_to_keep)
    assert len(output_del_after) == 0

    # Verify search doesn't find deleted items
    search_res_del = docsearch.similarity_search("text_id_del1", k=4)
    assert "text_id_del1" not in [d.page_content for d in search_res_del]

# TODO: Add tests for metadata filtering once filtering logic is clear/implemented in ArangoVector.
# Need example AQL filters based on ArangoVector's expected filter format.

# TODO: Add tests for hybrid search if/when supported by ArangoVector. 