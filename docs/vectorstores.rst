Vector Stores
============

LangChain ArangoDB provides vector store implementations that allow you to store and retrieve embeddings using ArangoDB.

ArangoDBVectorStore
------------------

The main vector store implementation that uses ArangoDB for storing and retrieving vector embeddings.

.. code-block:: python

    from langchain_arangodb.vectorstores import ArangoDBVectorStore
    from langchain.embeddings import OpenAIEmbeddings

    # Initialize the vector store
    vectorstore = ArangoDBVectorStore(
        embedding=OpenAIEmbeddings(),
        arango_url="http://localhost:8529",
        username="root",
        password="",
        database="langchain",
        collection_name="vectors"
    )

    # Add texts to the vector store
    texts = ["Hello world", "How are you"]
    vectorstore.add_texts(texts)

    # Search for similar texts
    results = vectorstore.similarity_search("Hello", k=2)

Features
--------

- Efficient vector similarity search
- Support for metadata filtering
- Batch operations for adding texts
- Configurable collection settings
- Integration with LangChain's embedding interfaces

Configuration Options
--------------------

The vector store can be configured with various options:

- ``embedding``: The embedding model to use
- ``arango_url``: URL of the ArangoDB instance
- ``username``: ArangoDB username
- ``password``: ArangoDB password
- ``database``: Database name
- ``collection_name``: Collection name for storing vectors
- ``index_name``: Name of the vector index (default: "vector_index")
- ``index_type``: Type of vector index to use
- ``index_fields``: Fields to include in the index 