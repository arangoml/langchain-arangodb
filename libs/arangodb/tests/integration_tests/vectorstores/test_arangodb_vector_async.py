"""Async integration tests for ArangoVector using python-arango-async."""

import pytest
from arango import ArangoClient
from arangoasync import ArangoClient as AsyncArangoClient
from arangoasync.auth import Auth

from langchain_arangodb.vectorstores.arangodb_vector import (
    ArangoVector,
    SearchType,
)
from tests.integration_tests.utils import ArangoCredentials

from .fake_embeddings import FakeEmbeddings


@pytest.fixture(scope="module")
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
async def async_vector_store(
    arangodb_credentials: ArangoCredentials,
    fake_embeddings: FakeEmbeddings,
    clear_arangodb_database: None,
) -> ArangoVector:
    """Fixture providing an ArangoVector with both sync and async db clients."""
    async_client = AsyncArangoClient(hosts=arangodb_credentials["url"])
    async_db = await async_client.db(
        "_system",
        auth=Auth(
            username=arangodb_credentials["username"],
            password=arangodb_credentials["password"],
        ),
    )

    texts = ["hello world", "hello arango", "test document"]
    metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
    ids = ["id1", "id2", "id3"]

    store = ArangoVector(
        embedding=fake_embeddings,
        embedding_dimension=10,
        collection_name="test_async_collection",
        async_database=async_db,
    )
    await store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
    await store.acreate_vector_index()
    return store


async def test_aadd_texts(
    arangodb_credentials: ArangoCredentials,
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Test aadd_texts inserts documents via python-arango-async."""
    async_client = AsyncArangoClient(hosts=arangodb_credentials["url"])
    async_db = await async_client.db(
        "_system",
        auth=Auth(
            username=arangodb_credentials["username"],
            password=arangodb_credentials["password"],
        ),
    )

    store = ArangoVector(
        embedding=fake_embeddings,
        embedding_dimension=10,
        collection_name="test_async_add",
        async_database=async_db,
    )

    ids = await store.aadd_texts(
        ["async text 1", "async text 2"],
        metadatas=[{"tag": "a"}, {"tag": "b"}],
    )

    assert len(ids) == 2
    assert await async_db.collection("test_async_add").count() == 2


async def test_adelete(async_vector_store: ArangoVector) -> None:
    """Test adelete removes documents via python-arango-async."""
    result = await async_vector_store.adelete(ids=["id1", "id2"])

    assert result is True
    assert async_vector_store.async_db is not None
    collection = async_vector_store.async_db.collection(
        async_vector_store.collection_name
    )
    assert await collection.count() == 1


@pytest.mark.usefixtures("clear_arangodb_database")
async def test_aget_by_ids(async_vector_store: ArangoVector) -> None:
    """Test aget_by_ids retrieves documents via python-arango-async."""
    docs = await async_vector_store.aget_by_ids(["id1", "id3"])

    assert len(docs) == 2
    contents = {doc.page_content for doc in docs}
    assert "hello world" in contents
    assert "test document" in contents


@pytest.mark.usefixtures("clear_arangodb_database")
async def test_asimilarity_search(async_vector_store: ArangoVector) -> None:
    """Test asimilarity_search returns documents via python-arango-async."""
    docs = await async_vector_store.asimilarity_search("hello", k=1, use_approx=False)

    assert len(docs) == 1
    assert docs[0].page_content == "hello world"


@pytest.mark.usefixtures("clear_arangodb_database")
async def test_asimilarity_search_with_score(async_vector_store: ArangoVector) -> None:
    """Test asimilarity_search_with_score returns (doc, score) pairs."""
    results = await async_vector_store.asimilarity_search_with_score(
        "hello", k=2, use_approx=False
    )

    assert len(results) == 2
    for doc, score in results:
        assert isinstance(score, float)
        assert doc.page_content in {"hello world", "hello arango", "test document"}


@pytest.mark.usefixtures("clear_arangodb_database")
async def test_asimilarity_search_by_vector(async_vector_store: ArangoVector) -> None:
    """Test asimilarity_search_by_vector returns documents."""
    query_embedding = fake_embeddings_for_query()
    docs = await async_vector_store.asimilarity_search_by_vector(
        query_embedding, k=1, use_approx=False
    )

    assert len(docs) == 1


def fake_embeddings_for_query() -> list:
    """Return a query embedding matching FakeEmbeddings.embed_query('foo')."""
    return [1.0] * 9 + [-1.0]


@pytest.mark.usefixtures("clear_arangodb_database")
async def test_amax_marginal_relevance_search(
    async_vector_store: ArangoVector,
) -> None:
    """Test amax_marginal_relevance_search returns diverse documents."""
    results = await async_vector_store.amax_marginal_relevance_search(
        "hello", k=2, fetch_k=3, lambda_mult=0.5, use_approx=False
    )

    assert len(results) == 2
    # Results should be distinct
    assert results[0].page_content != results[1].page_content


@pytest.mark.usefixtures("clear_arangodb_database")
async def test_async_hybrid_search(
    arangodb_credentials: ArangoCredentials,
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Test async hybrid search (vector + keyword) via python-arango-async."""
    sync_client = ArangoClient(hosts=arangodb_credentials["url"])
    sync_db = sync_client.db(
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )
    async_client = AsyncArangoClient(hosts=arangodb_credentials["url"])
    async_db = await async_client.db(
        "_system",
        auth=Auth(
            username=arangodb_credentials["username"],
            password=arangodb_credentials["password"],
        ),
    )

    texts = [
        "machine learning algorithms",
        "deep learning neural networks",
        "data science analytics",
    ]

    store = ArangoVector.from_texts(
        texts=texts,
        embedding=fake_embeddings,
        database=sync_db,
        collection_name="test_async_hybrid",
        search_type=SearchType.HYBRID,
        overwrite_index=True,
        insert_text=True,
        async_database=async_db,
    )
    store.create_vector_index()
    store.create_keyword_index()

    results = await store.asimilarity_search_with_score(
        "machine learning",
        k=2,
        use_approx=False,
        search_type=SearchType.HYBRID,
    )

    assert len(results) >= 1
    for doc, score in results:
        assert isinstance(score, float)
        assert doc.page_content in texts
