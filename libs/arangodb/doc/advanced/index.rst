.. _advanced:

Advanced Topics
===============

This section covers advanced usage patterns and configuration options for LangChain ArangoDB.

Performance Optimization
------------------------

**Batch Processing**

For large document imports, use batch processing to improve performance:

.. code-block:: python

    from langchain_arangodb.vectorstores import ArangoVector
    
    # Large batch processing
    vector_store = ArangoVector.from_texts(
        texts=large_text_list,
        embedding=embeddings,
        database=db,
        batch_size=1000  # Process 1000 documents at a time
    )

**Index Management**

Optimize search performance with proper index configuration:

.. code-block:: python

    # Vector index configuration
    vector_store.create_vector_index()
    
    # For hybrid search, also create keyword index
    vector_store.create_keyword_index()
    
    # Check index status
    vector_index_info = vector_store.retrieve_vector_index()
    keyword_index_info = vector_store.retrieve_keyword_index()

**Connection Pooling**

For production deployments, use connection pooling:

.. code-block:: python

    from arango import ArangoClient
    
    # Configure connection pool
    client = ArangoClient(
        hosts="http://localhost:8529",
        host_resolver="roundrobin",  # Load balancing
        http_timeout=60,
        max_retries=3
    )

Security Best Practices
-----------------------

**Database Permissions**

Always use minimal required permissions:

.. code-block:: python

    # Create dedicated database user for LangChain
    system_db = client.db("_system", username="root", password="admin_pass")
    
    # Create user with limited permissions
    system_db.create_user(
        username="langchain_user",
        password="secure_password",
        active=True
    )
    
    # Grant only necessary permissions
    system_db.update_permission(
        username="langchain_user",
        database="langchain_db",
        permission="rw"  # read-write on specific database only
    )

**Query Validation**

For graph QA chains, enable query validation:

.. code-block:: python

    from langchain_arangodb.chains import ArangoGraphQAChain
    
    qa_chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=False,  # Disable by default
        return_direct=False,  # Review queries before execution
        validate_aql=True  # Enable AQL validation
    )

Error Handling and Monitoring
-----------------------------

**Connection Error Handling**

Implement robust error handling for database connections:

.. code-block:: python

    from arango.exceptions import ArangoError
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        db = client.db("langchain", username="user", password="pass")
        vector_store = ArangoVector.from_texts(texts, embeddings, database=db)
    except ArangoError as e:
        logger.error(f"ArangoDB connection failed: {e}")
        # Implement fallback strategy
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

**Query Performance Monitoring**

Monitor query performance and optimize as needed:

.. code-block:: python

    import time
    
    # Time vector searches
    start_time = time.time()
    results = vector_store.similarity_search("query", k=10)
    search_time = time.time() - start_time
    
    logger.info(f"Vector search completed in {search_time:.2f} seconds")
    
    # Profile AQL queries
    explain_result = graph.explain("FOR doc IN collection RETURN doc")
    logger.info(f"Query execution plan: {explain_result}")

Custom Implementations
----------------------

**Custom Distance Functions**

Implement custom similarity metrics:

.. code-block:: python

    from langchain_arangodb.vectorstores.utils import DistanceStrategy
    
    # Use different distance strategies
    vector_store_cosine = ArangoVector.from_texts(
        texts=texts,
        embedding=embeddings,
        database=db,
        distance_strategy=DistanceStrategy.COSINE
    )
    
    vector_store_euclidean = ArangoVector.from_texts(
        texts=texts,
        embedding=embeddings,
        database=db,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
    )

**Custom Graph Schemas**

Define custom graph schemas for specialized use cases:

.. code-block:: python

    # Define custom schema
    custom_schema = {
        "collection_schema": [
            {
                "collection": "Person",
                "properties": ["name", "age", "occupation"],
                "examples": {"name": "John Doe", "age": 30}
            }
        ],
        "graph_schema": [
            {
                "edge_collection": "WorksAt",
                "from_collections": ["Person"],
                "to_collections": ["Company"]
            }
        ]
    }
    
    # Apply custom schema
    graph.set_schema(custom_schema)

Deployment Considerations
------------------------

**Docker Deployment**

Example Docker Compose setup:

.. code-block:: yaml

    version: '3.8'
    services:
      arangodb:
        image: arangodb/arangodb:latest
        environment:
          - ARANGO_ROOT_PASSWORD=secure_password
        ports:
          - "8529:8529"
        volumes:
          - arango_data:/var/lib/arangodb3
          - arango_apps:/var/lib/arangodb3-apps
        
      langchain_app:
        build: .
        depends_on:
          - arangodb
        environment:
          - ARANGO_URL=http://arangodb:8529
          - ARANGO_USERNAME=root
          - ARANGO_PASSWORD=secure_password
    
    volumes:
      arango_data:
      arango_apps:

**Environment Variables**

Configure using environment variables for production:

.. code-block:: python

    import os
    from langchain_arangodb.graphs import ArangoGraph
    
    # Use environment variables
    graph = ArangoGraph.from_db_credentials(
        url=os.getenv("ARANGO_URL", "http://localhost:8529"),
        dbname=os.getenv("ARANGO_DBNAME", "langchain"),
        username=os.getenv("ARANGO_USERNAME", "root"),
        password=os.getenv("ARANGO_PASSWORD")
    )

Troubleshooting
---------------

**Common Issues and Solutions**

1. **Connection Timeout**: Increase HTTP timeout in client configuration
2. **Memory Usage**: Use batch processing for large datasets
3. **Index Creation Fails**: Ensure ArangoDB version >= 3.12.4 for vector indexes
4. **Query Performance**: Add appropriate indexes and use query profiling

**Debug Mode**

Enable verbose logging for debugging:

.. code-block:: python

    import logging
    
    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create chains with verbose output
    qa_chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,  # Show intermediate steps
        return_aql_query=True,  # Return generated queries
        return_aql_result=True  # Return raw results
    )