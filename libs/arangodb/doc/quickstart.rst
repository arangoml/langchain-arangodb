Quickstart
==========

Get started with LangChain ArangoDB in 5 simple steps:

1. Set up ArangoDB
2. Set environment variables  
3. Instantiate a Vector Store
4. Instantiate an ArangoDB Graph
5. Instantiate an ArangoDB Graph QA Chain

1. Set up ArangoDB
------------------

**Option A: Local Instance via Docker**

Appears on ``localhost:8529`` with the user ``root`` & password ``openSesame``.

More info: `arangodb.com/download-major <https://arangodb.com/download-major/>`_.

.. code-block:: bash

    docker run -e ARANGO_ROOT_PASSWORD=openSesame -p 8529:8529 arangodb/arangodb

**Option B: ArangoDB Cloud Trial**

`ArangoGraph <https://dashboard.arangodb.cloud/home>`_ is ArangoDB's Cloud offering to use ArangoDB as a managed service.

A 14-day trial is available upon sign up.

**Option C: Temporary Cloud Instance via Python**

A temporary cloud database can be provisioned using the `adb-cloud-connector <https://github.com/arangodb/adb-cloud-connector?tab=readme-ov-file#arangodb-cloud-connector>`_ Python package.

.. code-block:: bash

    pip install adb-cloud-connector

.. code-block:: python

    from adb_cloud_connector import get_temp_credentials

    credentials = get_temp_credentials()
    print(credentials)

2. Set environment variables
----------------------------

Set up your LLM Environment Variables:

.. code-block:: bash

    export OPENAI_API_KEY=sk-proj-....

Or via python:

.. code-block:: python

    import os
    os.environ["OPENAI_API_KEY"] = "sk-proj-...."

3. Instantiate a Vector Store
-----------------------------

Create an ArangoDB vector store for semantic search and embeddings:

.. code-block:: python

    from arango import ArangoClient
    from langchain_openai import OpenAIEmbeddings
    from langchain_arangodb.vectorstores import ArangoVector

    # Connect to ArangoDB
    client = ArangoClient("http://localhost:8529")
    db = client.db("langchain_demo", username="root", password="openSesame")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create vector store
    texts = [
        "ArangoDB is a multi-model database supporting documents, graphs, and search",
        "LangChain enables building applications with large language models",
        "Vector databases enable semantic search and RAG applications"
    ]

    vectorstore = ArangoVector.from_texts(
        texts=texts,
        embedding=embeddings,
        database=db,
        collection_name="documents"
    )

    # Test similarity search
    results = vectorstore.similarity_search("What is ArangoDB?", k=2)
    for doc in results:
        print(doc.page_content)

**Advanced: Hybrid Search**

.. code-block:: python

    from langchain_arangodb.vectorstores import SearchType

    # Create vector store with hybrid search
    hybrid_vectorstore = ArangoVector.from_texts(
        texts=texts,
        embedding=embeddings,
        database=db,
        collection_name="hybrid_docs",
        search_type=SearchType.HYBRID,
        insert_text=True  # Required for hybrid search
    )

    # Create indexes
    hybrid_vectorstore.create_vector_index()
    hybrid_vectorstore.create_keyword_index()

    # Perform hybrid search
    hybrid_results = hybrid_vectorstore.similarity_search_with_score(
        "multi-model database technology",
        k=2,
        search_type=SearchType.HYBRID,
        vector_weight=1.0,
        keyword_weight=1.0
    )

4. Instantiate an ArangoDB Graph
---------------------------------

Create and work with knowledge graphs using ArangoDB:

.. code-block:: python

    from langchain_arangodb.graphs import ArangoGraph

    # Initialize the graph
    graph = ArangoGraph(
        database=db,
        vertex_collections=["Person", "Company", "Technology"],
        edge_collections=["WorksAt", "Uses", "DeveloperOf"]
    )

    # Add nodes and relationships
    graph.add_graph_documents([
        {
            "nodes": [
                {"id": "person1", "type": "Person", "properties": {"name": "Alice", "role": "Developer"}},
                {"id": "company1", "type": "Company", "properties": {"name": "TechCorp", "industry": "Software"}},
                {"id": "tech1", "type": "Technology", "properties": {"name": "ArangoDB", "category": "Database"}}
            ],
            "relationships": [
                {"source": "person1", "target": "company1", "type": "WorksAt", "properties": {"since": "2023"}},
                {"source": "company1", "target": "tech1", "type": "Uses", "properties": {"purpose": "Data storage"}}
            ]
        }
    ])

    # Query the graph
    query_result = graph.query("FOR v, e, p IN 1..2 OUTBOUND 'Person/person1' GRAPH 'knowledge_graph' RETURN {vertex: v, edge: e}")
    print(query_result)

**Schema Management**

.. code-block:: python

    # Get current schema
    schema = graph.get_schema
    print("Graph Schema:", schema)

    # Refresh schema after changes
    graph.refresh_schema()

5. Instantiate an ArangoDB Graph QA Chain
------------------------------------------

Create a question-answering system that leverages your graph data:

.. code-block:: python

    from langchain_openai import ChatOpenAI
    from langchain_arangodb.chains import ArangoGraphQAChain

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Create the QA chain
    qa_chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True
    )

    # Ask questions about your graph
    response = qa_chain.run("Who works at TechCorp and what technologies do they use?")
    print(response)

    # Ask about relationships
    response = qa_chain.run("What is the relationship between Alice and ArangoDB?")
    print(response)

**Advanced: Custom Prompts**

.. code-block:: python

    from langchain_arangodb.chains.graph_qa import CYPHER_GENERATION_PROMPT

    # Customize the prompt for better AQL generation
    custom_prompt = CYPHER_GENERATION_PROMPT.partial(
        schema=graph.get_schema,
        examples="Example: To find all people working at companies that use ArangoDB:\n"
                "FOR person IN Person\n"
                "  FOR company IN Company\n"
                "    FILTER person._id IN (FOR v IN 1..1 OUTBOUND company._id WorksAt RETURN v._id)\n"
                "    FILTER 'ArangoDB' IN company.technologies\n"
                "    RETURN person"
    )

    qa_chain_custom = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=custom_prompt,
        verbose=True
    )

**Chat Message History Integration**

.. code-block:: python

    from langchain_arangodb.chat_message_histories import ArangoChatMessageHistory
    from langchain.memory import ConversationBufferMemory

    # Set up chat history storage
    chat_history = ArangoChatMessageHistory(
        arango_url="http://localhost:8529",
        username="root",
        password="openSesame",
        database="langchain_demo",
        collection_name="chat_sessions",
        session_id="user_123"
    )

    # Create memory with persistent storage
    memory = ConversationBufferMemory(
        chat_memory=chat_history,
        return_messages=True
    )

    # Use with the QA chain for conversation history
    qa_chain_with_memory = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        memory=memory,
        verbose=True
    )

    # Now your conversations are persisted
    response1 = qa_chain_with_memory.run("Tell me about the people in our database")
    response2 = qa_chain_with_memory.run("What companies do they work for?")

Complete Example: RAG with Graph and Vector Search
--------------------------------------------------

Combine all components for a powerful RAG application:

.. code-block:: python

    # Complete setup
    from arango import ArangoClient
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_arangodb.vectorstores import ArangoVector, SearchType
    from langchain_arangodb.graphs import ArangoGraph
    from langchain_arangodb.chains import ArangoGraphQAChain
    from langchain_arangodb.chat_message_histories import ArangoChatMessageHistory

    # Database connection
    client = ArangoClient("http://localhost:8529")
    db = client.db("rag_demo", username="root", password="openSesame")

    # Embeddings and LLM
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Vector store for documents
    vectorstore = ArangoVector.from_texts(
        texts=[
            "ArangoDB combines document, graph, and search in one database",
            "LangChain provides tools for building LLM applications",
            "RAG systems improve LLM responses with external knowledge"
        ],
        embedding=embeddings,
        database=db,
        collection_name="rag_documents",
        search_type=SearchType.HYBRID,
        insert_text=True
    )

    # Graph for structured knowledge
    graph = ArangoGraph(
        database=db,
        vertex_collections=["Entity", "Concept"],
        edge_collections=["RelatedTo", "PartOf"]
    )

    # QA chain with graph reasoning
    qa_chain = ArangoGraphQAChain.from_llm(llm=llm, graph=graph)

    # Chat history for context
    chat_history = ArangoChatMessageHistory(
        arango_url="http://localhost:8529",
        username="root",
        password="openSesame",
        database="rag_demo",
        collection_name="conversations",
        session_id="session_1"
    )

    print("🚀 RAG system ready! You can now:")
    print("- Search documents with hybrid vector/keyword search")
    print("- Query structured knowledge with graph traversal")
    print("- Maintain conversation context with persistent chat history")

Next Steps
----------

- Explore the :doc:`vectorstores` guide for advanced search capabilities
- Learn about graph operations in the graphs documentation
- Check out chat message histories for conversation management
- See the API reference for complete method documentation