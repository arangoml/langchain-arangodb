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
    """ArangoDB vector index.

        To use this, you should have the `python-arango` python package installed.

        Args:
            embedding: Any embedding function implementing
                `langchain.embeddings.base.Embeddings` interface.
            embedding_dimension: The dimension of the to-be-inserted embedding vectors.
            database: The python-arango database instance.
            collection_name: The name of the collection to use. (default: "documents")
            search_type: The type of search to be performed, currently only 'vector'
                is supported.
            embedding_field: The field name storing the embedding vector.
                (default: "embedding")
            text_field: The field name storing the text. (default: "text")
            index_name: The name of the vector index to use. (default: "vector_index")
            distance_strategy: The distance strategy to use. (default: "COSINE")
            num_centroids: The number of centroids for the vector index. (default: 1)
            relevance_score_fn: A function to normalize the relevance score.
                If not provided, the default normalization function for
                the distance strategy will be used.
            keyword_index_name: The name of the keyword index.
            full_text_search_options: Full text search options.
            rrf_constant: The RRF k value.
            search_limit: The search limit.

        Example:
            .. code-block:: python

                from arango import ArangoClient
                from langchain_community.embeddings.openai import OpenAIEmbeddings
                from langchain_community.vectorstores.arangodb_vector import (
                    ArangoVector
                )

                db = ArangoClient("http://localhost:8529").db(
                    "test",
                    username="root",
                    password="openSesame"
    )

                embedding = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    dimensions=dimension
                )

                vector_store = ArangoVector.from_texts(
                    texts=["hello world", "hello langchain", "hello arangodb"],
                    embedding=embedding,
                    database=db,
                    collection_name="Documents"
                )

                print(vector_store.similarity_search("arangodb", k=1))
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

        view_properties = {
            "links": {
                self.collection_name: {
                    "analyzers": [self.keyword_analyzer],
                    "fields": {self.text_field: {"analyzers": [self.keyword_analyzer]}},
                }
            }
        }

        self.db.create_view(self.keyword_index_name, "arangosearch", view_properties)

    def delete_keyword_index(self) -> None:
        """Delete the keyword index from the collection."""
        view = self.retrieve_keyword_index()
        if view:
            self.db.delete_view(self.keyword_index_name)

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
        """Add texts to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
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
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with ArangoDB.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            return_fields: Fields to return in the result. For example,
                {"foo", "bar"} will return the "foo" and "bar" fields of the document,
                in addition to the _key & text field. Defaults to an empty set.
            use_approx: Whether to use approximate vector search via ANN.
                Defaults to True. If False, exact vector search will be used.
            embedding: Optional embedding to use for the query. If not provided,
                the query will be embedded using the embedding function provided
                in the constructor.

        Returns:
            List of Documents most similar to the query.
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
    ) -> List[tuple[Document, float]]:
        """Run similarity search with ArangoDB.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            return_fields: Fields to return in the result. For example,
                {"foo", "bar"} will return the "foo" and "bar" fields of the document,
                in addition to the _key & text field. Defaults to an empty set.
            use_approx: Whether to use approximate vector search via ANN.
                Defaults to True. If False, exact vector search will be used.
            embedding: Optional embedding to use for the query. If not provided,
                the query will be embedded using the embedding function provided
                in the constructor.
            filter_clause: Filter clause to apply to the query.

        Returns:
            List of Documents most similar to the query.
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
            )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        filter_clause: str = "",
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            return_fields: Fields to return in the result. For example,
                {"foo", "bar"} will return the "foo" and "bar" fields of the document,
                in addition to the _key & text field. Defaults to an empty set.
            use_approx: Whether to use approximate vector search via ANN.
                Defaults to True. If False, exact vector search will be used.

        Returns:
            List of Documents most similar to the query vector.
        """
        results = self.similarity_search_by_vector_with_score(
            embedding=embedding,
            k=k,
            return_fields=return_fields,
            use_approx=use_approx,
            filter_clause=filter_clause,
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
    ) -> List[Document]:
        """Run similarity search with ArangoDB."""

        results = self.similarity_search_by_vector_and_keyword_with_score(
            query=query,
            embedding=embedding,
            k=k,
            return_fields=return_fields,
            use_approx=use_approx,
            filter_clause=filter_clause,
        )

        return [doc for doc, _ in results]

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        filter_clause: str = "",
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            return_fields: Fields to return in the result. For example,
                {"foo", "bar"} will return the "foo" and "bar" fields of the document,
                in addition to the _key & text field. Defaults to an empty set.
            use_approx: Whether to use approximate vector search via ANN.
                Defaults to True. If False, exact vector search will be used.

        Returns:
            List of Documents most similar to the query vector.
        """
        aql_query, bind_vars = self._build_vector_search_query(
            embedding=embedding,
            k=k,
            return_fields=return_fields,
            use_approx=use_approx,
            filter_clause=filter_clause,
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
    ) -> List[tuple[Document, float]]:
        """Run similarity search with ArangoDB.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            return_fields: Fields to return in the result. For example,
                {"foo", "bar"} will return the "foo" and "bar" fields of the document,
                in addition to the _key & text field. Defaults to an empty set.
            use_approx: Whether to use approximate vector search via ANN.
                Defaults to True. If False, exact vector search will be used.
            filter_clause: Filter clause to apply to the query.

        Returns:
            List of Documents most similar to the query.
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
        )

        cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars, stream=True)

        results = self._process_search_query(cursor)  # type: ignore

        return results

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that can be used to delete vectors.

        Returns:
            Optional[bool]: True if deletion is successful,
                None if no ids are provided, or raises an exception if an error occurs.
        """
        if not ids:
            return None

        for result in self.collection.delete_many(ids, **kwargs):  # type: ignore
            if isinstance(result, ArangoServerError):
                raise result

        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their IDs.

        Args:
            ids: List of ids to get.

        Returns:
            List of Documents with the given ids.
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
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            return_fields: Fields to return in the result. For example,
                {"foo", "bar"} will return the "foo" and "bar" fields of the document,
                in addition to the _key & text field. Defaults to an empty set.
            use_approx: Whether to use approximate vector search via ANN.
                Defaults to True. If False, exact vector search will be used.
            embedding: Optional embedding to use for the query. If not provided,
                the query will be embedded using the embedding function provided
                in the constructor.

        Returns:
            List of Documents selected by maximal marginal relevance.
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
        """
        Return ArangoDBVector initialized from texts, embeddings and a database.
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
        """
        Return ArangoDBVector initialized from existing collection.

        Args:
            collection_name: Name of the collection to use.
            text_properties_to_embed: List of properties to embed.
            embedding: Embedding function to use.
            database: Database to use.
            embedding_field: Field name to store the embedding.
            text_field: Field name to store the text.
            batch_size: Read batch size.
            aql_return_text_query: Custom AQL query to return the content of
                the text properties.
            insert_text: Whether to insert the new text (i.e concatenated text
                properties) into the collection.
            skip_existing_embeddings: Whether to skip documents with existing
                embeddings.
            search_type: The type of search to be performed.
            keyword_index_name: The name of the keyword index.
            full_text_search_options: Full text search options.
            **kwargs: Additional keyword arguments passed to the ArangoVector
                constructor.

        Returns:
            ArangoDBVector initialized from existing collection.
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
                data, score = result["data"], result["score"]
                _key = data.pop("_key")
                page_content = data.pop(self.text_field)
                doc = Document(page_content=page_content, id=_key, metadata=data)

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
                RETURN {{data, score}}
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
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
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

        aql_query = f"""
            LET vector_results = (
                FOR doc IN @@collection
                    {filter_clause if not use_approx else ""}
                    LET score = {score_func}(doc.{self.embedding_field}, @embedding)
                    SORT score {sort_order}
                    LIMIT {k}
                    {filter_clause if use_approx else ""}
                    RETURN {{ doc, score }}
            )

            LET keyword_results = (
                FOR doc IN @@view
                    SEARCH ANALYZER(
                        doc.{self.text_field} IN TOKENS(@query, @analyzer),
                        @analyzer
                    )
                    {filter_clause}
                    LET score = BM25(doc)
                    SORT score DESC
                    LIMIT {k}
                    RETURN {{ doc, score }}
            )

            LET rrf_vector = (
                FOR i IN RANGE(0, LENGTH(vector_results) - 1)
                    LET doc = vector_results[i].doc
                    FILTER doc != null
                    RETURN {{
                        doc,
                        score: {vector_weight} / (@rrf_constant + i + 1)
                    }}
            )

            LET rrf_keyword = (
                FOR i IN RANGE(0, LENGTH(keyword_results) - 1)
                    LET doc = keyword_results[i].doc
                    FILTER doc != null
                    RETURN {{
                        doc,
                        score: {keyword_weight} / (@rrf_constant + i + 1)
                    }}
            )

            FOR result IN APPEND(rrf_vector, rrf_keyword)
                COLLECT doc_key = result.doc._key INTO group
                LET rrf_score = SUM(group[*].result.score)
                LET doc = group[0].result.doc
                SORT rrf_score DESC
                LIMIT @rrf_search_limit
                RETURN {{
                    data: KEEP(doc, {return_fields_list}),
                    score: rrf_score
                }}
        """

        bind_vars = {
            "@collection": self.collection_name,
            "@view": self.keyword_index_name,
            "embedding": embedding,
            "query": query,
            "analyzer": self.keyword_analyzer,
            "rrf_constant": self.rrf_constant,
            "rrf_search_limit": self.rrf_search_limit,
        }

        return aql_query, bind_vars
