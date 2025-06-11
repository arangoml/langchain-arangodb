ArangoGraphQAChain
========================

This guide demonstrates how to use the ArangoGraphQAChain for question-answering against an ArangoDB graph database.

Basic Setup
-----------

First, let's set up the necessary imports and create a basic instance:

.. code-block:: python

    from langchain_arangodb.chains.graph_qa.arangodb import ArangoGraphQAChain
    from langchain_arangodb.graphs.arangodb_graph import ArangoGraph
    from langchain.chat_models import ChatOpenAI
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

    response = chain.invoke({"query": "Who starred in Pulp Fiction?"})
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
    
    response = chain.invoke({"question": "Who directed Inception?"})
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
    
    explanation = chain.invoke({"query": "Find all movies released after 2020"})
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
    
    response = chain.invoke({"query": "Who acted in The Matrix?"})
    print("Query:", response["aql_query"])
    print("Raw Results:", response["aql_result"])
    print("Final Answer:", response["result"])

Complete Workflow Example
-------------------------

Here's a complete workflow showing how to use multiple features together:

.. code-block:: python

    from langchain_arangodb.chains.graph_qa.arangodb import ArangoGraphQAChain
    from langchain_arangodb.graphs.arangodb_graph import ArangoGraph
    from langchain.chat_models import ChatOpenAI
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
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        top_k=5,
        force_read_only_query=True,
        return_aql_query=True,
        return_aql_result=True,
        output_list_limit=20,
        output_string_limit=200
    )
    
    # 6. Run Multiple Queries
    queries = [
        "Who acted in The Matrix?",
        "What movies were released in 1999?",
        "List all actors in the database"
    ]
    
    for query in queries:
        print(f"\nProcessing query: {query}")
        response = chain.invoke({"query": query})
        
        print("AQL Query:", response["aql_query"])
        print("Raw Results:", response["aql_result"])
        print("Final Answer:", response["result"])
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
        response = chain.invoke({"query": "Find all movies"})
    except ValueError as e:
        if "Maximum amount of AQL Query Generation attempts" in str(e):
            print("Failed to generate valid AQL after multiple attempts")
        elif "Write operations are not allowed" in str(e):
            print("Attempted write operation in read-only mode")
        else:
            print(f"Other error: {e}")

The chain will automatically attempt to fix invalid AQL queries up to 
``max_aql_generation_attempts`` times (default: 3) before raising an error.