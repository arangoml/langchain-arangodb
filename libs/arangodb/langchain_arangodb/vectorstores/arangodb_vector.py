from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Type, Union

import farmhash
import numpy as np
from arango.aql import Cursor
from arango.database import StandardDatabase
from arango.exceptions import ArangoServerError, ViewGetError
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from packaging import version

from langchain_arangodb.vectorstores.utils import DistanceStrategy

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE
DISTANCE_MAPPING = {
    DistanceStrategy.EUCLIDEAN_DISTANCE: "l2",
    DistanceStrategy.COSINE: "cosine",
}


class SearchType(str, Enum):
    """Enumerator of the search types."""

    VECTOR = "vector"
    HYBRID = "hybrid"


DEFAULT_SEARCH_TYPE = SearchType.VECTOR

# Constants for RRF
DEFAULT_RRF_CONSTANT = 60  # Standard constant for RRF
DEFAULT_SEARCH_LIMIT = 100  # Default limit for initial search results

# Full-text search analyzer options
DEFAULT_ANALYZER = "text_en"  # Default analyzer for full-text search
SUPPORTED_ANALYZERS = [
    "text_en",
    "text_de",
    "text_es",
    "text_fi",
    "text_fr",
    "text_it",
    "text_nl",
    "text_no",
    "text_pt",
    "text_ru",
    "text_sv",
    "text_zh",
]


class ArangoVector(VectorStore):
    """ArangoDB vector store implementation for LangChain.

    This class provides a vector store implementation using ArangoDB as the backend.
    It supports both vector similarity search and hybrid search (vector + keyword)
    capabilities.

    :param embedding: The embedding function to use for converting text to vectors.
        Must implement the `langchain.embeddings.base.Embeddings` interface.
    :type embedding: langchain.embeddings.base.Embeddings
    :param embedding_dimension: The dimensionality of the embedding vectors.
        Must match the output dimension of the embedding function.
    :type embedding_dimension: int
    :param database: The ArangoDB database instance to use for storage and retrieval.
    :type database: arango.database.StandardDatabase
    :param collection_name: The name of the ArangoDB collection to store
        documents. Defaults to "documents".
    :type collection_name: str
    :param search_type: The type of search to perform. Can be either SearchType.VECTOR
        for pure vector similarity search or SearchType.HYBRID for combining vector and
        keyword search. Defaults to SearchType.VECTOR.
    :type search_type: SearchType
    :param embedding_field: The field name in the document to store the embedding vector
        Defaults to "embedding".
    :type embedding_field: str
    :param text_field: The field name in the document to store the text content.
        Defaults to "text".
    :type text_field: str
    :param vector_index_name: The name of the vector index to create in ArangoDB.
        This index enables efficient vector similarity search.
        Defaults to "vector_index".
    :type vector_index_name: str
    :param distance_strategy: The distance metric to use for vector similarity.
        Can be either DistanceStrategy.COSINE or DistanceStrategy.EUCLIDEAN_DISTANCE.
        Defaults to DistanceStrategy.COSINE.
    :type distance_strategy: DistanceStrategy
    :param num_centroids: The number of centroids to use for the vector index.
        Higher values can improve search accuracy but increase memory usage.
        Defaults to 1.
    :type num_centroids: int
    :param relevance_score_fn: Optional function to normalize the relevance score.
        If not provided, uses the default normalization for the distance strategy.
    :type relevance_score_fn: Optional[Callable[[float], float]]
    :param keyword_index_name: The name of the ArangoDB View created to enable
        Full-Text-Search capabilities. Only used if search_type is set
        to SearchType.HYBRID. Defaults to "keyword_index".
    :type keyword_index_name: str
    :param keyword_analyzer: The text analyzer to use for keyword search.
        Must be one of the supported analyzers in ArangoDB.
        Defaults to "text_en".
    :type keyword_analyzer: str
    :param rrf_constant: The constant used in Reciprocal Rank Fusion (RRF) for hybrid
        search. Higher values give more weight to lower-ranked results.
        Defaults to 60.
    :type rrf_constant: int
    :param rrf_search_limit: The maximum number of results to consider in RRF scoring.
        Defaults to 100.
    :type rrf_search_limit: int
    """

    def __init__(
        self,
        embedding: Embeddings,
        embedding_dimension: int,
        database: StandardDatabase,
        collection_name: str = "documents",
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        embedding_field: str = "embedding",
        text_field: str = "text",
        vector_index_name: str = "vector_index",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        num_centroids: int = 1,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        keyword_index_name: str = "keyword_index",
        keyword_analyzer: str = DEFAULT_ANALYZER,
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
        rrf_search_limit: int = DEFAULT_SEARCH_LIMIT,
    ):
        if search_type not in [SearchType.VECTOR, SearchType.HYBRID]:
            raise ValueError("search_type must be 'vector' or 'hybrid'")

        if distance_strategy not in [
            DistanceStrategy.COSINE,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        ]:
            m = "distance_strategy must be 'COSINE' or 'EUCLIDEAN_DISTANCE'"
            raise ValueError(m)

        self.embedding = embedding
        self.embedding_dimension = int(embedding_dimension)
        self.db = database
        self.async_db = self.db.begin_async_execution(return_result=False)
        self.search_type = search_type
        self.collection_name = collection_name
        self.embedding_field = embedding_field
        self.text_field = text_field
        self.vector_index_name = vector_index_name
        self._distance_strategy = distance_strategy
        self.num_centroids = num_centroids
        self.override_relevance_score_fn = relevance_score_fn

        # Hybrid search parameters
        self.keyword_index_name = keyword_index_name
        self.keyword_analyzer = keyword_analyzer
        self.rrf_constant = rrf_constant
        self.rrf_search_limit = rrf_search_limit

        if not self.db.has_collection(collection_name):
            self.db.create_collection(collection_name)

        self.collection = self.db.collection(self.collection_name)

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def retrieve_vector_index(self) -> Union[dict[str, Any], None]:
        """Retrieve the vector index from the collection."""
        indexes = self.collection.indexes()  # type: ignore
        for index in indexes:  # type: ignore
            if index["name"] == self.vector_index_name:
                return index

        return None

    def create_vector_index(self) -> None:
        """Create the vector index on the collection."""
        self.collection.add_index(  # type: ignore
            {
                "name": self.vector_index_name,
                "type": "vector",
                "fields": [self.embedding_field],
                "params": {
                    "metric": DISTANCE_MAPPING[self._distance_strategy],
                    "dimension": self.embedding_dimension,
                    "nLists": self.num_centroids,
                },
            }
        )

    def delete_vector_index(self) -> None:
        """Delete the vector index from the collection."""
        index = self.retrieve_vector_index()

        if index is not None:
            self.collection.delete_index(index["id"])

    def retrieve_keyword_index(self) -> Optional[dict[str, Any]]:
        """Retrieve the keyword index from the collection."""
        try:
            return self.db.view(self.keyword_index_name)  # type: ignore
        except ViewGetError:
            return None

    def create_keyword_index(self) -> None:
        """Create the keyword index on the collection."""
        if self.retrieve_keyword_index():
            return

        collection = self.db.collection(self.collection_name)
        collection.add_index(
            {
                "type": "inverted",
                "name": self.keyword_index_name,
                "fields": [
                    {"name": self.text_field, "analyzer": self.keyword_analyzer}
                ],
            }
        )
        view_properties = {
            "indexes": [
                {"collection": self.collection_name, "index": self.keyword_index_name}
            ]
        }
        self.db.create_view(self.keyword_index_name, "search-alias", view_properties)

    def delete_keyword_index(self) -> None:
        """Delete the keyword index from the collection."""
        view = self.retrieve_keyword_index()
        if view:
            self.db.delete_view(self.keyword_index_name)
            self.db.collection(self.collection_name).delete_index(
                self.keyword_index_name, ignore_missing=True
            )

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 500,
        use_async_db: bool = False,
        insert_text: bool = True,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore."""
        texts = list(texts)

        if ids is None:
            ids = [str(farmhash.Fingerprint64(text.encode("utf-8"))) for text in texts]  # type: ignore

        if not metadatas:
            metadatas = [{} for _ in texts]

        if len(ids) != len(texts) != len(embeddings) != len(metadatas):
            m = "Length of ids, texts, embeddings and metadatas must be the same."
            raise ValueError(m)

        db = self.async_db if use_async_db else self.db
        collection = db.collection(self.collection_name)

        data = []
        for _key, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            doc: dict[str, Any] = {self.text_field: text} if insert_text else {}

            doc.update(
                {
                    **metadata,
                    "_key": _key,
                    self.embedding_field: embedding,
                }
            )

            data.append(doc)

            if len(data) == batch_size:
                collection.import_bulk(data, on_duplicate="update", **kwargs)
                data = []

        collection.import_bulk(data, on_duplicate="update", **kwargs)

        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store.

        This method embeds the provided texts using the embedding function and stores
        them in ArangoDB along with their embeddings and metadata.

        :param texts: An iterable of text strings to add to the vector store.
        :type texts: Iterable[str]
        :param metadatas: Optional list of metadata dictionaries to associate with each
            text. Each dictionary can contain arbitrary key-value pairs that
            will be stored alongside the text and embedding.
        :type metadatas: Optional[List[dict]]
        :param ids: Optional list of unique identifiers for each text. If not provided,
            IDs will be generated using a hash of the text content.
        :type ids: Optional[List[str]]
        :param kwargs: Additional keyword arguments passed to add_embeddings.
        :type kwargs: Any
        :return: List of document IDs that were added to the vector store.
        :rtype: List[str]

        .. code-block:: python

            # Add simple texts
            texts = ["hello world", "hello arango", "test document"]
            ids = vector_store.add_texts(texts)
            print(f"Added {len(ids)} documents")

            # Add texts with metadata
            texts = ["Machine learning tutorial", "Python programming guide"]
            metadatas = [
                {"category": "AI", "difficulty": "beginner"},
                {"category": "Programming", "difficulty": "intermediate"}
            ]
            ids = vector_store.add_texts(texts, metadatas=metadatas)

            # Add texts with custom IDs
            texts = ["Document 1", "Document 2"]
            custom_ids = ["doc_001", "doc_002"]
            ids = vector_store.add_texts(texts, ids=custom_ids)
        """
        embeddings = self.embedding.embed_documents(list(texts))

        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        embedding: Optional[List[float]] = None,
        filter_clause: str = "",
        search_type: Optional[SearchType] = None,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        keyword_search_clause: str = "",
        metadata_clause: str = "",
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents using vector similarity or hybrid search.

        This method performs a similarity search using either pure vector similarity
        or a hybrid approach combining vector and keyword search. The search type
        can be overridden for individual queries.

        :param query: The text query to search for.
        :type query: str
        :param k: The number of most similar documents to return. Defaults to 4.
        :type k: int
        :param return_fields: Set of additional document fields to return in results.
            The _key and text fields are always returned.
        :type return_fields: set[str]
        :param use_approx: Whether to use approximate nearest neighbor search.
            Enables faster but potentially less accurate results.
            Defaults to True.
        :type use_approx: bool
        :param embedding: Optional pre-computed embedding for the query.
            If not provided, the query will be embedded using the embedding
            function.
        :type embedding: Optional[List[float]]
        :param filter_clause: Optional AQL filter clause to apply to the search.
            Can be used to filter results based on document properties.
        :type filter_clause: str
        :param search_type: Override the default search type for this query.
            Can be either SearchType.VECTOR or SearchType.HYBRID.
        :type search_type: Optional[SearchType]
        :param vector_weight: Weight to apply to vector similarity
            scores in hybrid search. Only used when search_type is SearchType.HYBRID.
            Defaults to 1.0.
        :type vector_weight: float
        :param keyword_weight: Weight to apply to keyword search scores in hybrid
            search. Only used when search_type is SearchType.HYBRID. Defaults to 1.0.
        :type keyword_weight: float
        :param keyword_search_clause: Optional AQL filter clause to apply
          Full Text Search. If empty, a default search clause will be used.
        :type keyword_search_clause: str
        :param metadata_clause: Optional AQL clause to return additional metadata once
            the top k results are retrieved. If specified, the metadata will be
            added to the Document.metadata field.
        :type metadata_clause: str
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any
        :return: List of Document objects most similar to the query.
        :rtype: List[Document]

        .. code-block:: python

            # Simple vector search
            results = vector_store.similarity_search("hello", k=1)
            print(results[0].page_content)

            # Search with metadata filtering
            results = vector_store.similarity_search(
                "machine learning",
                k=2,
                filter_clause="doc.category == 'AI'",
                return_fields={"category", "difficulty"}
            )

            # Hybrid search with custom weights
            results = vector_store.similarity_search(
                "neural networks",
                k=3,
                search_type=SearchType.HYBRID,
                vector_weight=0.8,
                keyword_weight=0.2
            )
        """
        search_type = search_type or self.search_type
        embedding = embedding or self.embedding.embed_query(query)

        if search_type == SearchType.VECTOR:
            return self.similarity_search_by_vector(
                embedding=embedding,
                k=k,
                return_fields=return_fields,
                use_approx=use_approx,
                filter_clause=filter_clause,
                metadata_clause=metadata_clause,
            )

        else:
            return self.similarity_search_by_vector_and_keyword(
                query=query,
                embedding=embedding,
                k=k,
                return_fields=return_fields,
                use_approx=use_approx,
                filter_clause=filter_clause,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                keyword_search_clause=keyword_search_clause,
                metadata_clause=metadata_clause,
            )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        embedding: Optional[List[float]] = None,
        filter_clause: str = "",
        search_type: Optional[SearchType] = None,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        keyword_search_clause: str = "",
        metadata_clause: str = "",
    ) -> List[tuple[Document, float]]:
        """Search for similar documents and return their similarity scores.

        Similar to similarity_search but returns a tuple of (Document, score) for each
        result. The score represents the similarity between the query and the document.

        :param query: The text query to search for.
        :type query: str
        :param k: The number of most similar documents to return. Defaults to 4.
        :type k: int
        :param return_fields: Set of additional document fields to return in results.
            The _key and text fields are always returned.
        :type return_fields: set[str]
        :param use_approx: Whether to use approximate nearest neighbor search.
            Enables faster but potentially less accurate results. Defaults to True.
        :type use_approx: bool
        :param embedding: Optional pre-computed embedding for the query.
            If not provided, the query will be embedded using the embedding function.
        :type embedding: Optional[List[float]]
        :param filter_clause: Optional AQL filter clause to apply to the search.
            Can be used to filter results based on document properties.
        :type filter_clause: str
        :param search_type: Override the default search type for this query.
            Can be either SearchType.VECTOR or SearchType.HYBRID.
        :type search_type: Optional[SearchType]
        :param vector_weight: Weight to apply to vector similarity scores in
            hybrid search. Only used when search_type is SearchType.HYBRID.
            Defaults to 1.0.
        :type vector_weight: float
        :param keyword_weight: Weight to apply to keyword search scores
            in hybrid search. Only used when search_type is SearchType.HYBRID.
            Defaults to 1.0.
        :type keyword_weight: float
        :param keyword_search_clause: Optional AQL filter clause to apply
            Full Text Search. If empty, a default search clause will be used.
        :type keyword_search_clause: str
        :param metadata_clause: Optional AQL clause to return additional metadata once
            the top k results are retrieved.
        :type metadata_clause: str
        :return: List of tuples containing (Document, score) pairs, sorted by score.
        :rtype: List[tuple[Document, float]]
        """
        search_type = search_type or self.search_type
        embedding = embedding or self.embedding.embed_query(query)

        if search_type == SearchType.VECTOR:
            return self.similarity_search_by_vector_with_score(
                embedding=embedding,
                k=k,
                return_fields=return_fields,
                use_approx=use_approx,
                filter_clause=filter_clause,
                metadata_clause=metadata_clause,
            )

        else:
            return self.similarity_search_by_vector_and_keyword_with_score(
                query=query,
                embedding=embedding,
                k=k,
                return_fields=return_fields,
                use_approx=use_approx,
                filter_clause=filter_clause,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                keyword_search_clause=keyword_search_clause,
                metadata_clause=metadata_clause,
            )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        filter_clause: str = "",
        metadata_clause: str = "",
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        :param embedding: Embedding to look up documents similar to.
        :type embedding: List[float]
        :param k: Number of Documents to return. Defaults to 4.
        :type k: int
        :param return_fields: Fields to return in the result. For example,
            {"foo", "bar"} will return the "foo" and "bar" fields of the document,
            in addition to the _key & text field. Defaults to an empty set.
        :type return_fields: set[str]
        :param use_approx: Whether to use approximate vector search via ANN.
            Defaults to True. If False, exact vector search will be used.
        :type use_approx: bool
        :param filter_clause: Filter clause to apply to the query.
        :type filter_clause: str
        :param metadata_clause: Optional AQL clause to return additional metadata once
            the top k results are retrieved. If specified, the metadata will be
            added to the Document.metadata field.
        :type metadata_clause: str
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any
        :return: List of Documents most similar to the query vector.
        :rtype: List[Document]
        """
        results = self.similarity_search_by_vector_with_score(
            embedding=embedding,
            k=k,
            return_fields=return_fields,
            use_approx=use_approx,
            filter_clause=filter_clause,
            metadata_clause=metadata_clause,
        )

        return [doc for doc, _ in results]

    def similarity_search_by_vector_and_keyword(
        self,
        query: str,
        embedding: List[float],
        k: int = 4,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        filter_clause: str = "",
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        keyword_search_clause: str = "",
        metadata_clause: str = "",
    ) -> List[Document]:
        results = self.similarity_search_by_vector_and_keyword_with_score(
            query=query,
            embedding=embedding,
            k=k,
            return_fields=return_fields,
            use_approx=use_approx,
            filter_clause=filter_clause,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            keyword_search_clause=keyword_search_clause,
            metadata_clause=metadata_clause,
        )

        return [doc for doc, _ in results]

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        filter_clause: str = "",
        metadata_clause: str = "",
    ) -> List[tuple[Document, float]]:
        """Return docs most similar to embedding vector with scores.

        :param embedding: Embedding to look up documents similar to.
        :type embedding: List[float]
        :param k: Number of Documents to return. Defaults to 4.
        :type k: int
        :param return_fields: Fields to return in the result. For example,
            {"foo", "bar"} will return the "foo" and "bar" fields of the document,
            in addition to the _key & text field. Defaults to an empty set.
        :type return_fields: set[str]
        :param use_approx: Whether to use approximate vector search via ANN.
            Defaults to True. If False, exact vector search will be used.
        :type use_approx: bool
        :param filter_clause: Filter clause to apply to the query.
        :type filter_clause: str
        :param metadata_clause: Optional AQL clause to return additional metadata once
            the top k results are retrieved. If specified, the metadata will be
            added to the Document.metadata field.
        :type metadata_clause: str
        :return: List of tuples containing (Document, score)
            pairs most similar to the query vector.
        :rtype: List[tuple[Document, float]]
        """
        aql_query, bind_vars = self._build_vector_search_query(
            embedding=embedding,
            k=k,
            return_fields=return_fields,
            use_approx=use_approx,
            filter_clause=filter_clause,
            metadata_clause=metadata_clause,
        )

        cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars, stream=True)

        results = self._process_search_query(cursor)  # type: ignore

        return results

    def similarity_search_by_vector_and_keyword_with_score(
        self,
        query: str,
        embedding: List[float],
        k: int = 4,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        filter_clause: str = "",
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        keyword_search_clause: str = "",
        metadata_clause: str = "",
    ) -> List[tuple[Document, float]]:
        """Run hybrid similarity search combining vector and keyword search with scores.

        :param query: Query text to search for.
        :type query: str
        :param embedding: Embedding vector for the query.
        :type embedding: List[float]
        :param k: Number of results to return. Defaults to 4.
        :type k: int
        :param return_fields: Fields to return in the result. For example,
            {"foo", "bar"} will return the "foo" and "bar" fields of the document,
            in addition to the _key & text field. Defaults to an empty set.
        :type return_fields: set[str]
        :param use_approx: Whether to use approximate vector search via ANN.
            Defaults to True. If False, exact vector search will be used.
        :type use_approx: bool
        :param filter_clause: Filter clause to apply to the query.
        :type filter_clause: str
        :param vector_weight: Weight to apply to vector similarity scores
            in hybrid search. Only used when search_type is SearchType.HYBRID.
            Defaults to 1.0.
        :type vector_weight: float
        :param keyword_weight: Weight to apply to keyword search scores in
            hybrid search. Only used when search_type is SearchType.HYBRID.
            Defaults to 1.0.
        :type keyword_weight: float
        :param keyword_search_clause: Optional AQL filter clause to apply
            Full Text Search. If empty, a default search clause will be used.
        :type keyword_search_clause: str
        :param metadata_clause: Optional AQL clause to return additional metadata once
            the top k results are retrieved. If specified, the metadata will be
            added to the Document.metadata field.
        :type metadata_clause: str
        :return: List of tuples containing (Document, score)
            pairs most similar to thequery.
        :rtype: List[tuple[Document, float]]
        """

        aql_query, bind_vars = self._build_hybrid_search_query(
            query=query,
            k=k,
            embedding=embedding,
            return_fields=return_fields,
            use_approx=use_approx,
            filter_clause=filter_clause,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            keyword_search_clause=keyword_search_clause,
            metadata_clause=metadata_clause,
        )

        cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars, stream=True)

        results = self._process_search_query(cursor)  # type: ignore

        return results

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        :param ids: List of ids to delete.
        :type ids: Optional[List[str]]
        :param kwargs: Other keyword arguments that can be used to delete vectors.
        :type kwargs: Any
        :return: True if deletion is successful, None if no ids are provided,
            or raises an exception if an error occurs.
        :rtype: Optional[bool]
        """
        if not ids:
            return None

        for result in self.collection.delete_many(ids, **kwargs):  # type: ignore
            if isinstance(result, ArangoServerError):
                raise result

        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their IDs.

        :param ids: List of ids to get.
        :type ids: Sequence[str]
        :return: List of Documents with the given ids.
        :rtype: list[Document]
        """
        docs = []
        doc: dict[str, Any]

        for doc in self.collection.get_many(ids):  # type: ignore
            _key = doc.pop("_key")
            page_content = doc.pop(self.text_field)

            docs.append(Document(page_content=page_content, id=_key, metadata=doc))

        return docs

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        embedding: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for documents using Maximal Marginal Relevance (MMR).

        MMR optimizes for both similarity to the query and diversity among the results.
        It helps avoid returning redundant or very similar documents by balancing
        relevance and diversity in the selection process.

        :param query: The text query to search for.
        :type query: str
        :param k: The number of documents to return. Defaults to 4.
        :type k: int
        :param fetch_k: The number of documents to fetch for MMR selection.
            Should be larger than k to allow for diversity selection.
            Defaults to 20.
        :type fetch_k: int
        :param lambda_mult: Controls the diversity vs relevance tradeoff.
            Values between 0 and 1, where 0 = maximum diversity, 1 = maximum relevance.
            Defaults to 0.5.
        :type lambda_mult: float
        :param return_fields: Set of additional document fields to return in results.
            The _key and text fields are always returned.
        :type return_fields: set[str]
        :param use_approx: Whether to use approximate nearest neighbor search.
            Enables faster but potentially less accurate results.
            Defaults to True.
        :type use_approx: bool
        :param embedding: Optional pre-computed embedding for the query.
            If not provided, the query will be embedded using the embedding
            function.
        :type embedding: Optional[List[float]]
        :param kwargs: Additional keyword arguments passed to the search methods.
        :type kwargs: Any
        :return: List of Document objects selected by MMR algorithm.
        :rtype: List[Document]

        .. code-block:: python

            # Search with balanced diversity and relevance
            results = vector_store.max_marginal_relevance_search(
                "machine learning",
                k=3,
                fetch_k=10
            )

            # Emphasize diversity over relevance
            diverse_results = vector_store.max_marginal_relevance_search(
                "neural networks",
                k=5,
                fetch_k=20,
                lambda_mult=0.1  # More diverse results
            )

            # Emphasize relevance over diversity
            relevant_results = vector_store.max_marginal_relevance_search(
                "deep learning",
                k=3,
                fetch_k=15,
                lambda_mult=0.9  # More relevant results
            )
        """
        return_fields.add(self.embedding_field)

        # Embed the query
        query_embedding = embedding or self.embedding.embed_query(query)

        # Fetch the initial documents
        docs = self.similarity_search_by_vector(
            embedding=query_embedding,
            k=fetch_k,
            return_fields=return_fields,
            use_approx=use_approx,
            **kwargs,
        )

        # Get the embeddings for the fetched documents
        embeddings = [doc.metadata[self.embedding_field] for doc in docs]

        # Select documents using maximal marginal relevance
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding), embeddings, lambda_mult=lambda_mult, k=k
        )

        selected_docs = [docs[i] for i in selected_indices]

        return selected_docs

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._distance_strategy in [
            DistanceStrategy.COSINE,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        ]:
            return lambda x: x
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to ArangoVector constructor."
            )

    @classmethod
    def from_texts(
        cls: Type[ArangoVector],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        database: Optional[StandardDatabase] = None,
        collection_name: str = "documents",
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        embedding_field: str = "embedding",
        text_field: str = "text",
        index_name: str = "vector_index",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        num_centroids: int = 1,
        ids: Optional[List[str]] = None,
        overwrite_index: bool = False,
        insert_text: bool = True,
        keyword_index_name: str = "keyword_index",
        keyword_analyzer: str = DEFAULT_ANALYZER,
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
        rrf_search_limit: int = DEFAULT_SEARCH_LIMIT,
        **kwargs: Any,
    ) -> ArangoVector:
        """Create an ArangoVector instance from a list of texts.

        This is a convenience method that creates a new ArangoVector instance,
        embeds the provided texts, and stores them in ArangoDB.

        :param texts: List of text strings to add to the vector store.
        :type texts: List[str]
        :param embedding: The embedding function to use for converting text to vectors.
        :type embedding: langchain.embeddings.base.Embeddings
        :param metadatas: Optional list of metadata dictionaries to associate with each
            text.
        :type metadatas: Optional[List[dict]]
        :param database: The ArangoDB database instance to use.
        :type database: Optional[arango.database.StandardDatabase]
        :param collection_name: The name of the ArangoDB collection to use.
            Defaults to "documents".
        :type collection_name: str
        :param search_type: The type of search to perform.
            Can be either SearchType.VECTOR or SearchType.HYBRID.
            Defaults to SearchType.VECTOR.
        :type search_type: SearchType
        :param embedding_field: The field name to store embeddings. Defaults to
            "embedding".
        :type embedding_field: str
        :param text_field: The field name to store text content. Defaults to "text".
        :type text_field: str
        :param index_name: The name of the vector index. Defaults to "vector_index".
        :type index_name: str
        :param distance_strategy: The distance metric to use. Can be
            DistanceStrategy.COSINE or DistanceStrategy.EUCLIDEAN_DISTANCE.
            Defaults to DistanceStrategy.COSINE.
        :type distance_strategy: DistanceStrategy
        :param num_centroids: Number of centroids for vector index. Defaults to 1.
        :type num_centroids: int
        :param ids: Optional list of unique identifiers for each text.
        :type ids: Optional[List[str]]
        :param overwrite_index: Whether to delete and recreate existing indexes.
            Defaults to False.
        :type overwrite_index: bool
        :param insert_text: Whether to store the text content in the database.
            Required for hybrid search. Defaults to True.
        :type insert_text: bool
        :param keyword_index_name: Name of the keyword search index. Defaults to
            "keyword_index".
        :type keyword_index_name: str
        :param keyword_analyzer: Text analyzer for keyword search.
            Defaults to "text_en".
        :type keyword_analyzer: str
        :param rrf_constant: Constant for RRF scoring in hybrid search. Defaults to 60.
        :type rrf_constant: int
        :param rrf_search_limit: Maximum results for RRF scoring. Defaults to 100.
        :type rrf_search_limit: int
        :param kwargs: Additional keyword arguments passed to the constructor.
        :type kwargs: Any
        :return: A new ArangoVector instance with the texts embedded and stored.
        :rtype: ArangoVector

        .. code-block:: python

            from arango import ArangoClient
            from langchain_arangodb.vectorstores import ArangoVector
            from langchain_community.embeddings import OpenAIEmbeddings

            # Connect to ArangoDB
            client = ArangoClient("http://localhost:8529")
            db = client.db("test", username="root", password="openSesame")

            # Create vector store from texts
            texts = ["hello world", "hello arango", "test document"]
            metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]

            vector_store = ArangoVector.from_texts(
                texts=texts,
                embedding=OpenAIEmbeddings(),
                metadatas=metadatas,
                database=db,
                collection_name="test_collection"
            )

            # Create hybrid search store
            hybrid_store = ArangoVector.from_texts(
                texts=["Machine learning algorithms", "Deep neural networks"],
                embedding=OpenAIEmbeddings(),
                database=db,
                search_type=SearchType.HYBRID,
                collection_name="hybrid_docs",
                overwrite_index=True  # Clean start
            )
        """
        if not database:
            raise ValueError("Database must be provided.")

        if not insert_text and search_type == SearchType.HYBRID:
            raise ValueError("insert_text must be True when search_type is HYBRID")

        embeddings = embedding.embed_documents(list(texts))

        embedding_dimension = len(embeddings[0])

        store = cls(
            embedding,
            embedding_dimension=embedding_dimension,
            database=database,
            collection_name=collection_name,
            search_type=search_type,
            embedding_field=embedding_field,
            text_field=text_field,
            vector_index_name=index_name,
            distance_strategy=distance_strategy,
            num_centroids=num_centroids,
            keyword_index_name=keyword_index_name,
            keyword_analyzer=keyword_analyzer,
            rrf_constant=rrf_constant,
            rrf_search_limit=rrf_search_limit,
            **kwargs,
        )

        if overwrite_index:
            store.delete_vector_index()

            if search_type == SearchType.HYBRID:
                store.delete_keyword_index()

        store.add_embeddings(
            texts, embeddings, metadatas=metadatas, ids=ids, insert_text=insert_text
        )

        return store

    @classmethod
    def from_existing_collection(
        cls: Type[ArangoVector],
        collection_name: str,
        text_properties_to_embed: List[str],
        embedding: Embeddings,
        database: StandardDatabase,
        embedding_field: str = "embedding",
        text_field: str = "text",
        batch_size: int = 1000,
        aql_return_text_query: str = "",
        insert_text: bool = False,
        skip_existing_embeddings: bool = False,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        keyword_index_name: str = "keyword_index",
        keyword_analyzer: str = DEFAULT_ANALYZER,
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
        rrf_search_limit: int = DEFAULT_SEARCH_LIMIT,
        **kwargs: Any,
    ) -> ArangoVector:
        """Create an ArangoVector instance from an existing ArangoDB collection.

        This method reads documents from an existing collection, extracts specified
        text properties, embeds them, and creates a new vector store.

        :param collection_name: Name of the existing ArangoDB collection.
        :type collection_name: str
        :param text_properties_to_embed: List of document properties containing text to
            embed. These properties will be concatenated to create
            the text for embedding.
        :type text_properties_to_embed: List[str]
        :param embedding: The embedding function to use for converting text to vectors.
        :type embedding: Embeddings
        :param database: The ArangoDB database instance to use.
        :type database: StandardDatabase
        :param embedding_field: The field name to store embeddings.
            Defaults to "embedding".
        :type embedding_field: str
        :param text_field: The field name to store text content. Defaults to "text".
        :type text_field: str
        :param batch_size: Number of documents to process in each batch.
            Defaults to 1000.
        :type batch_size: int
        :param aql_return_text_query: Custom AQL query to extract text from properties.
            Defaults to "RETURN doc[p]".
        :type aql_return_text_query: str
        :param insert_text: Whether to store the concatenated text in the database.
            Required for hybrid search. Defaults to False.
        :type insert_text: bool
        :param skip_existing_embeddings: Whether to skip documents that already have
            embeddings. Defaults to False.
        :type skip_existing_embeddings: bool
        :param search_type: The type of search to perform.
            Can be either SearchType.VECTOR or SearchType.HYBRID.
            Defaults to SearchType.VECTOR.
        :type search_type: SearchType
        :param keyword_index_name: Name of the keyword search index.
            Defaults to "keyword_index".
        :type keyword_index_name: str
        :param keyword_analyzer: Text analyzer for keyword search.
            Defaults to "text_en".
        :type keyword_analyzer: str
        :param rrf_constant: Constant for RRF scoring in hybrid search. Defaults to 60.
        :type rrf_constant: int
        :param rrf_search_limit: Maximum results for RRF scoring. Defaults to 100.
        :type rrf_search_limit: int
        :param kwargs: Additional keyword arguments passed to the constructor.
        :type kwargs: Any
        :return: A new ArangoVector instance with embeddings created from the
            collection.
        :rtype: ArangoVector
        """
        if not text_properties_to_embed:
            m = "Parameter `text_properties_to_embed` must not be an empty list"
            raise ValueError(m)

        if text_field in text_properties_to_embed:
            m = "Parameter `text_field` must not be in `text_properties_to_embed`"
            raise ValueError(m)

        if not insert_text and search_type == SearchType.HYBRID:
            raise ValueError("insert_text must be True when search_type is HYBRID")

        if not aql_return_text_query:
            aql_return_text_query = "RETURN doc[p]"

        filter_clause = ""
        if skip_existing_embeddings:
            filter_clause = f"FILTER doc.{embedding_field} == null"

        aql_query = f"""
            FOR doc IN @@collection
                {filter_clause}

                LET texts = (
                    FOR p IN @properties
                        FILTER doc[p] != null
                        {aql_return_text_query}
                )

                RETURN {{
                    key: doc._key,
                    text: CONCAT_SEPARATOR(" ", texts),
                }}
        """

        bind_vars = {
            "@collection": collection_name,
            "properties": text_properties_to_embed,
        }

        cursor: Cursor = database.aql.execute(
            aql_query,
            bind_vars=bind_vars,  # type: ignore
            batch_size=batch_size,
            stream=True,
        )

        store: ArangoVector | None = None

        while not cursor.empty():
            batch = cursor.batch()
            batch_list = list(batch)  # type: ignore

            texts = [doc["text"] for doc in batch_list]
            ids = [doc["key"] for doc in batch_list]

            store = cls.from_texts(
                texts=texts,
                embedding=embedding,
                database=database,
                collection_name=collection_name,
                embedding_field=embedding_field,
                text_field=text_field,
                ids=ids,
                insert_text=insert_text,
                search_type=search_type,
                keyword_index_name=keyword_index_name,
                keyword_analyzer=keyword_analyzer,
                rrf_constant=rrf_constant,
                rrf_search_limit=rrf_search_limit,
                **kwargs,
            )

            batch.clear()  # type: ignore

            if cursor.has_more():
                cursor.fetch()

        if store is None:
            raise ValueError(f"No documents found in collection in {collection_name}")

        return store

    def _process_search_query(self, cursor: Cursor) -> List[tuple[Document, float]]:
        data: dict[str, Any]
        score: float
        results = []

        while not cursor.empty():
            for result in cursor:
                data, score, metadata = (
                    result["data"],
                    result["score"],
                    result["metadata"],
                )

                if not data:
                    continue

                _key = data.pop("_key")
                page_content = data.pop(self.text_field)
                doc = Document(
                    page_content=page_content, id=_key, metadata={**data, **metadata}
                )

                results.append((doc, score))

            if cursor.has_more():
                cursor.fetch()

        return results

    def _build_vector_search_query(
        self,
        embedding: List[float],
        k: int,
        return_fields: set[str],
        use_approx: bool,
        filter_clause: str,
        metadata_clause: str,
    ) -> Tuple[str, dict[str, Any]]:
        if self._distance_strategy == DistanceStrategy.COSINE:
            score_func = "APPROX_NEAR_COSINE" if use_approx else "COSINE_SIMILARITY"
            sort_order = "DESC"
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            score_func = "APPROX_NEAR_L2" if use_approx else "L2_DISTANCE"
            sort_order = "ASC"
        else:
            raise ValueError(f"Unsupported metric: {self._distance_strategy}")

        if use_approx:
            if version.parse(self.db.version()) < version.parse("3.12.4"):  # type: ignore
                m = "Approximate Nearest Neighbor search requires ArangoDB >= 3.12.4."
                raise ValueError(m)

            if not self.retrieve_vector_index():
                self.create_vector_index()

        return_fields.update({"_key", self.text_field})
        return_fields_list = list(return_fields)

        aql_query = f"""
            FOR doc IN @@collection
                {filter_clause if not use_approx else ""}
                LET score = {score_func}(doc.{self.embedding_field}, @embedding)
                SORT score {sort_order}
                LIMIT {k}
                {filter_clause if use_approx else ""}
                LET data = KEEP(doc, {return_fields_list})
                LET metadata = {f"({metadata_clause})" if metadata_clause else "{}"}
                RETURN {{data, score, metadata}}
        """

        bind_vars = {
            "@collection": self.collection_name,
            "embedding": embedding,
        }

        return aql_query, bind_vars

    def _build_hybrid_search_query(
        self,
        query: str,
        k: int,
        embedding: List[float],
        return_fields: set[str],
        use_approx: bool,
        filter_clause: str,
        vector_weight: float,
        keyword_weight: float,
        keyword_search_clause: str,
        metadata_clause: str,
    ) -> Tuple[str, dict[str, Any]]:
        """Build the hybrid search query using RRF."""

        if not self.retrieve_keyword_index():
            self.create_keyword_index()

        if self._distance_strategy == DistanceStrategy.COSINE:
            score_func = "APPROX_NEAR_COSINE" if use_approx else "COSINE_SIMILARITY"
            sort_order = "DESC"
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            score_func = "APPROX_NEAR_L2" if use_approx else "L2_DISTANCE"
            sort_order = "ASC"
        else:
            raise ValueError(f"Unsupported metric: {self._distance_strategy}")

        if use_approx:
            if version.parse(self.db.version()) < version.parse("3.12.4"):  # type: ignore
                m = "Approximate Nearest Neighbor search requires ArangoDB >= 3.12.4."
                raise ValueError(m)

            if not self.retrieve_vector_index():
                self.create_vector_index()

        return_fields.update({"_key", self.text_field})
        return_fields_list = list(return_fields)

        if not keyword_search_clause:
            keyword_search_clause = f"""
                SEARCH ANALYZER(
                    doc.{self.text_field} IN TOKENS(@query, @analyzer),
                    @analyzer
                )
            """

        aql_query = f"""
            LET vector_results = (
                FOR doc IN @@collection
                    {filter_clause if not use_approx else ""}
                    LET score = {score_func}(doc.{self.embedding_field}, @embedding)
                    SORT score {sort_order}
                    LIMIT {k}
                    {filter_clause if use_approx else ""}
                    WINDOW {{ preceding: "unbounded", following: 0 }}
                    AGGREGATE rank = COUNT(1)
                    LET rrf_score = {vector_weight} / ({self.rrf_constant} + rank)
                    RETURN {{ key: doc._key, score: rrf_score }}
            )

            LET keyword_results = (
                FOR doc IN @@view
                    {keyword_search_clause}
                    {filter_clause}
                    LET score = BM25(doc)
                    SORT score DESC
                    LIMIT {k}
                    WINDOW {{ preceding: "unbounded", following: 0 }}
                    AGGREGATE rank = COUNT(1)
                    LET rrf_score = {keyword_weight} / ({self.rrf_constant} + rank)
                    RETURN {{ key: doc._key, score: rrf_score }}
            )

            FOR result IN APPEND(vector_results, keyword_results)
                COLLECT key = result.key AGGREGATE score = SUM(result.score)
                SORT score DESC
                LIMIT {self.rrf_search_limit}
                LET data = FIRST(
                    FOR doc IN @@collection
                        FILTER doc._key == key
                        LIMIT 1
                        RETURN KEEP(doc, {return_fields_list})
                )
                LET metadata = {f"({metadata_clause})" if metadata_clause else "{}"}
                RETURN {{ data, score, metadata }}
        """

        bind_vars = {
            "@collection": self.collection_name,
            "@view": self.keyword_index_name,
            "embedding": embedding,
            "query": query,
            "analyzer": self.keyword_analyzer,
        }

        return aql_query, bind_vars
