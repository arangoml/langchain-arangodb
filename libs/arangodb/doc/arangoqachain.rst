ArangoGraphQAChain
========================

This guide demonstrates how to use the ArangoGraphQAChain for question-answering against an ArangoDB graph database.

Basic Setup
-----------

First, let's set up the necessary imports and create a basic instance:

.. code-block:: python

    from langchain_arangodb import ArangoGraphQAChain, ArangoGraph, ArangoChatMessageHistory
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from arango import ArangoClient

    # Initialize ArangoDB connection
    client = ArangoClient()
    db = client.db("your_database", username="user", password="pass")
    
    # Create graph instance
    graph = ArangoGraph(db)
    
    # Initialize LLM
    llm = ChatOpenAI(temperature=0)
    
    # Create the chain
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True  # Be cautious with this setting
    )

Individual Method Usage
-----------------------

1. Basic Query Execution
~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to use the chain is with a direct query:

.. code-block:: python

    response = chain.invoke("Who starred in Pulp Fiction?")
    print(response["result"])

2. Using Custom Input/Output Keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can customize the input and output keys:

.. code-block:: python

    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        input_key="question",
        output_key="answer"
    )
    
    response = chain.invoke("Who directed Inception?")
    print(response["answer"])

3. Limiting Results
~~~~~~~~~~~~~~~~~~~

Control the number of results returned:

.. code-block:: python

    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        top_k=5,  # Return only top 5 results
        output_list_limit=16,  # Limit list length in response
        output_string_limit=128  # Limit string length in response
    )

4. Query Explanation Mode
~~~~~~~~~~~~~~~~~~~~~~~~~

Get query explanation without execution:

.. code-block:: python

    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        execute_aql_query=False  # Only explain, don't execute
    )
    
    explanation = chain.invoke("Find all movies released after 2020")
    print(explanation["aql_result"])  # Contains query plan

5. Read-Only Mode
~~~~~~~~~~~~~~~~~

Enforce read-only operations:

.. code-block:: python

    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        force_read_only_query=True  # Prevents write operations
    )

6. Custom AQL Examples
~~~~~~~~~~~~~~~~~~~~~~

Provide example AQL queries for better generation:

.. code-block:: python

    example_queries = """
    FOR m IN Movies
        FILTER m.year > 2020
        RETURN m.title
    
    FOR a IN Actors
        FILTER a.awards > 0
        RETURN a.name
    """
    
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        aql_examples=example_queries
    )

7. Detailed Output
~~~~~~~~~~~~~~~~~~

Get more detailed output including AQL query and results:

.. code-block:: python

    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        return_aql_query=True,
        return_aql_result=True
    )
    
    response = chain.invoke("Who acted in The Matrix?")
    print("Query:", response["aql_query"])
    print("Raw Results:", response["aql_result"])
    print("Final Answer:", response["result"])

8. Query Cache
~~~~~~~~~~~~~~

Enable query caching to reuse past queries, reducing response time and LLM cost:

.. code-block:: python

    # Initialize Embedding Model (required for query cache)
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        use_query_cache=True, # Enables query caching (default: False)
        embedding=embedding, # Required if use_query_cache is True
        query_cache_collection_name="Queries", # Optional (default: "Queries")
        query_cache_similarity_threshold=0.80, # Only fetch cached queries with similarity >= 0.80 (default: 0.80)
    )
    
    query1 = "Who directed The Matrix?"
    response1 = chain.invoke({"query": query1, "use_query_cache": False}) # Disable query cache to force fresh LLM generation
    print(response1["result"])

    # Cache the query and its result if you are satisfied
    chain.cache_query() # Caches the most recent query by default

    # Alternatively, you can cache a query-AQL pair manually
    chain.cache_query(
        text="Who directed The Matrix?", 
        aql="FOR m IN Movies FILTER m.title == 'The Matrix' RETURN m.director"
    )

    # Similar query: uses exact match or vector similarity to fetch a cached AQL query and its result
    query2 = "Who is the director of The Matrix?"
    response2 = chain.invoke({
        "query": query2, 
        "query_cache_similarity_threshold": 0.90}) # Adjust threshold if needed
    print(response2["result"])

    # Clear all cached queries
    chain.clear_query_cache()

    # Or, clear a specific cached query
    chain.clear_query_cache(text="Who directed The Matrix?")

9. Chat History
~~~~~~~~~~~~~~~

Enable context-aware responses by including chat history:

.. code-block:: python

    # Initialize chat message history (required for chat history)
    history = ArangoChatMessageHistory(
        session_id="user_123",
        db=db,
        collection_name="chat_sessions"
    )

    chain = ArangoGraphQAChain.from_llm(
        llm=llm, 
        graph=graph, 
        allow_dangerous_requests=True, 
        include_history=True, # Enables chat history (default: False)
        chat_history_store=history, # Instance of ArangoChatMessageHistory. Required if include_history is True
        max_history_messages=10  # Optional: maximum number of messages to include (default: 10)
    )

    query = "What movies were released in 1999?"
    response = chain.invoke({"query": query, "include_history": False}) # Disable chat history (on function call only)
    print(response["result"])

    query = "Among all those movies, which one is directed by Lana Wachowski?"
    response = chain.invoke({"query": query}) # include_history already set to True in the chain, enabling the LLM to understand what "those movies" refer to
    print(response["result"])


Complete Workflow Example
-------------------------

Here's a complete workflow showing how to use multiple features together:

.. code-block:: python

    from langchain_arangodb import ArangoGraphQAChain, ArangoGraph, ArangoChatMessageHistory
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from arango import ArangoClient

    # 1. Setup Database Connection
    client = ArangoClient()
    db = client.db("movies_db", username="user", password="pass")
    
    # 2. Initialize Graph
    graph = ArangoGraph(db)
    
    # 3. Create Collections and Sample Data
    if not db.has_collection("Movies"):
        movies = db.create_collection("Movies")
        movies.insert({"_key": "matrix", "title": "The Matrix", "year": 1999})
    
    if not db.has_collection("Actors"):
        actors = db.create_collection("Actors")
        actors.insert({"_key": "keanu", "name": "Keanu Reeves"})
    
    if not db.has_collection("ActedIn"):
        acted_in = db.create_collection("ActedIn", edge=True)
        acted_in.insert({
            "_from": "Actors/keanu",
            "_to": "Movies/matrix"
        })
    
    # 4. Refresh Schema
    graph.refresh_schema()
    
    # 5. Initialize Chain with Advanced Features
    llm = ChatOpenAI(temperature=0)

    # 6. Initialize Embedding Model (required for query cache)
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    # 7. Initialize chat message history (required for chat history)
    history = ArangoChatMessageHistory(
        session_id="user_123",
        db=db,
        collection_name="chat_sessions"
    )

    # 8. Initialize Chain with Advanced Features
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        top_k=5,
        force_read_only_query=True,
        return_aql_query=True,
        return_aql_result=True,
        output_list_limit=20,
        output_string_limit=200,
        use_query_cache=True,
        embedding=embedding,
        query_cache_collection_name="Queries",
        query_cache_similarity_threshold=0.80,
        include_history=True,
        chat_history_store=history,
        max_history_messages=10
    )
    
    # 9. Run Multiple Queries
    queries = [
        "Who acted in The Matrix?",
        "Who starred in The Matrix?",
        "What is the last name of this actor?"
        "What movies were released in 1999?",
        "List all actors in the database"
    ]
    
    for query in queries:
        print(f"\nProcessing query: {query}")
        response = chain.invoke(query)
        
        print("AQL Query:", response["aql_query"])
        print("Raw Results:", response["aql_result"])
        print("Final Answer:", response["result"])
        chain.cache_query()
        print("-" * 50)

Security Considerations
-----------------------

1. Always use appropriate database credentials with minimal required permissions
2. Be cautious with ``allow_dangerous_requests=True``
3. Use ``force_read_only_query=True`` when only read operations are needed
4. Monitor and log query execution in production environments
5. Regularly review and update AQL examples to prevent injection risks

Error Handling
--------------      

The chain includes built-in error handling:

.. code-block:: python

    try:
        response = chain.invoke("Find all movies")
    except ValueError as e:
        if "Maximum amount of AQL Query Generation attempts" in str(e):
            print("Failed to generate valid AQL after multiple attempts")
        elif "Write operations are not allowed" in str(e):
            print("Attempted write operation in read-only mode")
        else:
            print(f"Other error: {e}")

The chain will automatically attempt to fix invalid AQL queries up to 
``max_aql_generation_attempts`` times (default: 3) before raising an error.