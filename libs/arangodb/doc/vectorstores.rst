Vector Stores
============

LangChain ArangoDB provides powerful vector store implementations that allow you to store, index, and retrieve embeddings using ArangoDB's native vector search capabilities.

Overview
--------

The ``ArangoVector`` class is the main vector store implementation that integrates with LangChain's embedding interfaces and provides:

- Efficient vector similarity search with cosine and Euclidean distance metrics
- Approximate and exact nearest neighbor search
- Maximal marginal relevance (MMR) search for diverse results
- Batch operations for adding and managing documents
- Configurable vector indexing with customizable parameters
- Integration with ArangoDB's distributed architecture

Quick Start
-----------

.. code-block:: python

    from arango import ArangoClient
    from langchain_openai import OpenAIEmbeddings
    from langchain_arangodb.vectorstores import ArangoVector

    # Connect to ArangoDB
    client = ArangoClient("http://localhost:8529")
    db = client.db("langchain", username="root", password="openSesame")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create vector store from texts
    texts = [
        "ArangoDB is a multi-model database",
        "LangChain provides tools for AI applications",
        "Vector search enables semantic similarity"
    ]
    
    vectorstore = ArangoVector.from_texts(
        texts=texts,
        embedding=embeddings,
        database=db,
        collection_name="my_documents"
    )

    # Search for similar documents
    results = vectorstore.similarity_search("database technology", k=2)
    for doc in results:
        print(doc.page_content)

Configuration
-------------

Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~

.. py:class:: ArangoVector(embedding, embedding_dimension, database, **kwargs)

   :param embedding: Any embedding function implementing LangChain's Embeddings interface
   :param embedding_dimension: The dimension of the embedding vectors (must match your embedding model)
   :param database: ArangoDB database instance from python-arango
   :param collection_name: Name of the collection to store documents (default: "documents")
   :param search_type: Type of search - currently only "vector" is supported
   :param embedding_field: Field name for storing embedding vectors (default: "embedding")
   :param text_field: Field name for storing text content (default: "text")
   :param index_name: Name of the vector index (default: "vector_index")
   :param distance_strategy: Distance metric for similarity calculation (default: "COSINE")
   :param num_centroids: Number of centroids for vector index clustering (default: 1)
   :param relevance_score_fn: Custom function to normalize relevance scores (optional)

Distance Strategies
~~~~~~~~~~~~~~~~~~

The vector store supports multiple distance metrics:

- **COSINE**: Cosine similarity (default) - good for normalized vectors
- **EUCLIDEAN_DISTANCE**: L2 distance - good for absolute distance measurements
- **MAX_INNER_PRODUCT**: Maximum inner product similarity
- **DOT_PRODUCT**: Dot product similarity
- **JACCARD**: Jaccard similarity coefficient

*Note: Currently only COSINE and EUCLIDEAN_DISTANCE are fully supported in the vector search implementation.*

.. code-block:: python

    from langchain_arangodb.vectorstores.utils import DistanceStrategy

    # Using Euclidean distance
    vectorstore = ArangoVector(
        embedding=embeddings,
        embedding_dimension=1536,
        database=db,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
    )

Search Methods
--------------

Basic Similarity Search
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Simple similarity search
    results = vectorstore.similarity_search("your query", k=5)

    # Search with additional fields returned
    results = vectorstore.similarity_search(
        "your query", 
        k=5,
        return_fields={"metadata_field", "custom_field"}
    )

    # Search with custom embedding
    custom_embedding = embeddings.embed_query("your query")
    results = vectorstore.similarity_search_by_vector(custom_embedding, k=5)

Search with Scores
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get similarity scores with results
    docs_and_scores = vectorstore.similarity_search_with_score("your query", k=5)
    
    for doc, score in docs_and_scores:
        print(f"Score: {score:.3f} - Content: {doc.page_content[:100]}...")

Maximal Marginal Relevance (MMR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MMR search helps ensure diverse results by balancing relevance and diversity:

.. code-block:: python

    # MMR search for diverse results
    diverse_results = vectorstore.max_marginal_relevance_search(
        query="your query",
        k=5,                    # Number of final results
        fetch_k=20,            # Number of initial candidates to fetch
        lambda_mult=0.5        # Balance between relevance (0) and diversity (1)
    )

Approximate vs Exact Search
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Approximate search (faster, requires ArangoDB >= 3.12.4)
    results = vectorstore.similarity_search("query", use_approx=True)

    # Exact search (slower but precise)
    results = vectorstore.similarity_search("query", use_approx=False)

Hybrid Search
~~~~~~~~~~~~~

Hybrid search combines vector similarity with traditional keyword search using Reciprocal Rank Fusion (RRF), providing more comprehensive and accurate results:

.. code-block:: python

    from langchain_arangodb.vectorstores import ArangoVector, SearchType

    # Create vector store with hybrid search enabled
    vectorstore = ArangoVector.from_texts(
        texts=["Machine learning algorithms", "AI-powered applications"],
        embedding=embeddings,
        database=db,
        search_type=SearchType.HYBRID,
        insert_text=True,  # Required for hybrid search
    )

    # Create both vector and keyword indexes
    vectorstore.create_vector_index()
    vectorstore.create_keyword_index()

    # Perform hybrid search
    results = vectorstore.similarity_search_with_score(
        query="AI technology",
        k=3,
        search_type=SearchType.HYBRID,
        vector_weight=1.0,      # Weight for vector similarity
        keyword_weight=1.0,     # Weight for keyword matching
    )

**Hybrid Search Parameters:**

- ``search_type=SearchType.HYBRID``: Enables hybrid search mode
- ``vector_weight``: Weight for vector similarity scores (default: 1.0)
- ``keyword_weight``: Weight for keyword search scores (default: 1.0)
- ``rrf_search_limit``: Number of top results to consider for RRF fusion (default: 50)
- ``keyword_search_clause``: Custom AQL search clause for keyword matching (optional)

**Custom Keyword Search:**

.. code-block:: python

    # Custom keyword search with metadata filtering
    custom_keyword_clause = f"""
        SEARCH ANALYZER(
            doc.{vectorstore.text_field} IN TOKENS(@query, @analyzer),
            @analyzer
        ) AND doc.category == "technology"
    """

    results = vectorstore.similarity_search_with_score(
        query="machine learning",
        k=5,
        search_type=SearchType.HYBRID,
        keyword_search_clause=custom_keyword_clause,
    )

**Keyword Index Management:**

.. code-block:: python

    # Create keyword index for hybrid search
    vectorstore.create_keyword_index()

    # Check if keyword index exists
    keyword_index = vectorstore.retrieve_keyword_index()
    if keyword_index:
        print(f"Keyword index: {keyword_index['name']}")

    # Delete keyword index
    vectorstore.delete_keyword_index()

**Weight Balancing Examples:**

.. code-block:: python

    # Favor vector similarity (semantic search)
    semantic_results = vectorstore.similarity_search_with_score(
        query="artificial intelligence",
        k=3,
        search_type=SearchType.HYBRID,
        vector_weight=10.0,
        keyword_weight=1.0,
    )

    # Favor keyword matching (traditional search)
    keyword_results = vectorstore.similarity_search_with_score(
        query="machine learning algorithms",
        k=3,
        search_type=SearchType.HYBRID,
        vector_weight=1.0,
        keyword_weight=10.0,
    )

    # Balanced hybrid approach
    balanced_results = vectorstore.similarity_search_with_score(
        query="AI applications",
        k=3,
        search_type=SearchType.HYBRID,
        vector_weight=1.0,
        keyword_weight=1.0,
    )

Document Management
------------------

Adding Documents
~~~~~~~~~~~~~~~

.. code-block:: python

    # Add texts with metadata
    texts = ["Document 1", "Document 2"]
    metadatas = [{"category": "tech"}, {"category": "science"}]
    ids = vectorstore.add_texts(texts, metadatas=metadatas)

    # Add pre-computed embeddings
    embeddings_list = [embedding.embed_query(text) for text in texts]
    ids = vectorstore.add_embeddings(
        texts=texts,
        embeddings=embeddings_list,
        metadatas=metadatas,
        ids=["custom_id_1", "custom_id_2"],  # Optional custom IDs
        batch_size=1000,                     # Custom batch size
        use_async_db=True                    # Use async operations
    )

    # IDs are automatically generated using farmhash if not provided
    # This ensures consistent, deterministic IDs based on content
    auto_ids = vectorstore.add_texts(["New document"])
    print(f"Auto-generated ID: {auto_ids[0]}")

Retrieving Documents
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get documents by IDs
    documents = vectorstore.get_by_ids(["doc_id_1", "doc_id_2"])

Deleting Documents
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Delete specific documents
    vectorstore.delete(ids=["doc_id_1", "doc_id_2"])

    # Delete all documents in collection
    vectorstore.delete()

Advanced Configuration
---------------------

Vector Index Management
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Check if vector index exists
    index_info = vectorstore.retrieve_vector_index()
    if index_info:
        print(f"Index exists: {index_info['name']}")

    # Create vector index manually
    vectorstore.create_vector_index()

    # Delete vector index
    vectorstore.delete_vector_index()

Batch Operations
~~~~~~~~~~~~~~~

.. code-block:: python

    # Large batch insertion with custom batch size
    large_texts = ["text"] * 10000
    vectorstore.add_texts(
        texts=large_texts,
        batch_size=1000,      # Process in batches of 1000
        use_async_db=True     # Use async operations for better performance
    )

Custom Collection Setup
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize with specific collection settings
    vectorstore = ArangoVector(
        embedding=embeddings,
        embedding_dimension=1536,
        database=db,
        collection_name="custom_vectors",
        index_name="my_vector_index",
        num_centroids=10,     # More centroids for larger datasets
        search_type=SearchType.VECTOR,  # or SearchType.HYBRID
    )

    # Create from texts with index management
    vectorstore = ArangoVector.from_texts(
        texts=["Document content"],
        embedding=embeddings,
        database=db,
        collection_name="my_collection",
        overwrite_index=True,    # Recreate existing indexes
        embedding_field="custom_embedding",
        text_field="custom_text",
        ids=["custom_id_1"],     # Custom document IDs
    )

Custom Relevance Scoring
~~~~~~~~~~~~~~~~~~~~~~~

You can provide custom relevance score normalization functions:

.. code-block:: python

    def custom_relevance_function(score: float) -> float:
        """Custom normalization that inverts and scales scores."""
        return 1.0 / (1.0 + score)

    vectorstore = ArangoVector(
        embedding=embeddings,
        embedding_dimension=1536,
        database=db,
        relevance_score_fn=custom_relevance_function
    )

    # Get relevance scores with custom normalization
    docs_with_scores = vectorstore.similarity_search_with_score("query", k=3)
    for doc, score in docs_with_scores:
        print(f"Custom score: {score}")

Performance Tips
---------------

1. **Choose the right distance strategy**: Use COSINE for normalized embeddings, EUCLIDEAN for raw distances
2. **Use approximate search**: Enable ``use_approx=True`` for large datasets (requires ArangoDB >= 3.12.4)
3. **Optimize batch size**: Use larger batch sizes (500-1000) for bulk operations
4. **Configure centroids**: Increase ``num_centroids`` for larger collections (rule of thumb: sqrt(num_documents))
5. **Use async operations**: Enable ``use_async_db=True`` for non-blocking operations
6. **Hybrid search optimization**: For hybrid search, ensure both vector and keyword indexes are created
7. **Custom field names**: Use descriptive field names for embedding and text fields to avoid conflicts
8. **Memory management**: Use ``import_bulk`` operations with appropriate batch sizes for large datasets

Example: Complete Workflow
-------------------------

.. code-block:: python

    from arango import ArangoClient
    from langchain_openai import OpenAIEmbeddings
    from langchain_arangodb.vectorstores import ArangoVector
    from langchain_arangodb.vectorstores.utils import DistanceStrategy

    # Setup
    client = ArangoClient("http://localhost:8529")
    db = client.db("vectorstore_demo", username="root", password="openSesame")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create vector store with hybrid search support
    vectorstore = ArangoVector(
        embedding=embeddings,
        embedding_dimension=1536,
        database=db,
        collection_name="knowledge_base",
        search_type=SearchType.HYBRID,
        distance_strategy=DistanceStrategy.COSINE,
        num_centroids=5,
        insert_text=True  # Required for hybrid search
    )

    # Add documents with metadata
    documents = [
        "Python is a programming language",
        "Machine learning uses algorithms",
        "Databases store structured data",
        "APIs enable system integration"
    ]
    
    metadatas = [
        {"topic": "programming", "difficulty": "beginner"},
        {"topic": "ai", "difficulty": "intermediate"},
        {"topic": "database", "difficulty": "beginner"},
        {"topic": "integration", "difficulty": "intermediate"}
    ]

    # Add to vector store
    doc_ids = vectorstore.add_texts(documents, metadatas=metadatas)
    print(f"Added {len(doc_ids)} documents")

    # Create indexes for hybrid search
    vectorstore.create_vector_index()
    vectorstore.create_keyword_index()

    # Perform searches
    print("\n--- Vector Search ---")
    vector_results = vectorstore.similarity_search(
        "programming languages", 
        k=2, 
        search_type=SearchType.VECTOR
    )
    for doc in vector_results:
        print(f"- {doc.page_content}")

    print("\n--- Hybrid Search ---")
    hybrid_results = vectorstore.similarity_search_with_score(
        "data storage algorithms", 
        k=2,
        search_type=SearchType.HYBRID,
        vector_weight=1.0,
        keyword_weight=1.0
    )
    for doc, score in hybrid_results:
        print(f"Score: {score:.3f} - {doc.page_content}")

    print("\n--- MMR Search for Diversity ---")
    diverse_results = vectorstore.max_marginal_relevance_search(
        "technology concepts", 
        k=3, 
        lambda_mult=0.7
    )
    for doc in diverse_results:
        print(f"- {doc.page_content}")

API Reference
-------------

.. automodule:: langchain_arangodb.vectorstores.arangodb_vector
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: langchain_arangodb.vectorstores.utils
   :members:
   :undoc-members:
   :show-inheritance:

Future Enhancements
-------------------

Additional Distance Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Support for additional distance strategies is planned:

- **MAX_INNER_PRODUCT**: Maximum inner product similarity
- **DOT_PRODUCT**: Dot product similarity  
- **JACCARD**: Jaccard similarity coefficient

Graph-Enhanced Search
~~~~~~~~~~~~~~~~~~~~

Integration with ArangoDB's graph capabilities for enhanced semantic search:

.. code-block:: python

    # Future graph-enhanced search API (planned)
    # results = vectorstore.graph_enhanced_search(
    #     query="your query",
    #     k=5,
    #     graph_traversal_depth=2,
    #     include_connected_nodes=True
    # )

Multi-Modal Search
~~~~~~~~~~~~~~~~~

Support for multi-modal embeddings and cross-modal search capabilities:

.. code-block:: python

    # Future multi-modal search API (planned)
    # results = vectorstore.multi_modal_search(
    #     query="text query",
    #     image_query=image_embedding,
    #     modality_weights={"text": 0.7, "image": 0.3}
    # ) 