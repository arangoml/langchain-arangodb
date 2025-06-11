LangChain ArangoDB
==================

LangChain ArangoDB is a Python package that provides ArangoDB integrations for LangChain, enabling vector storage, graph operations, and chat message history management.

.. raw:: html

   <div style="display: flex; align-items: center; gap: 10px;">
      <a href="https://www.langchain.com/">

         <img src="https://python.langchain.com/img/brand/wordmark.png" alt="LangChain" style="height: 60px;">
      </a>
      <a href="https://www.arangodb.com/">
         <img src="https://arangodb.com/wp-content/uploads/2016/05/ArangoDB_logo_avocado_@1.png" alt="ArangoDB" style="height: 60px;">
      </a>
   </div>

.. raw:: html

   <br>

Key Features
------------

LangChain ArangoDB provides comprehensive integrations for building AI applications:

**Vector Operations**
  - High-performance vector similarity search
  - Support for multiple distance metrics (cosine, Euclidean)
  - Approximate and exact nearest neighbor search
  - Maximal marginal relevance (MMR) search for diverse results

**Graph Operations**
  - Knowledge graph construction and querying
  - Graph-based question answering chains
  - Integration with LangChain's graph interfaces

**Chat Memory**
  - Persistent chat message history storage
  - Session-based conversation management
  - Efficient message retrieval and filtering

**Query Construction**
  - AQL (ArangoDB Query Language) integration
  - Structured query generation from natural language

Requirements
------------
- Python 3.9+
- LangChain
- ArangoDB
- python-arango

Installation
------------

Latest Release

.. code-block:: bash

   pip install langchain-arangodb

Current Development State

.. code-block:: bash

   pip install git+https://github.com/arangodb/langchain-arangodb

Quick Start
-----------

.. code-block:: python

    from arango import ArangoClient
    from langchain_openai import OpenAIEmbeddings
    from langchain_arangodb.vectorstores import ArangoVector

    # Connect to ArangoDB
    client = ArangoClient("http://localhost:8529")
    db = client.db("langchain", username="root", password="openSesame")

    # Create vector store
    vectorstore = ArangoVector.from_texts(
        texts=["Hello world", "LangChain with ArangoDB"],
        embedding=OpenAIEmbeddings(),
        database=db
    )

    # Search
    results = vectorstore.similarity_search("greeting", k=1)

Documentation Contents

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   quickstart
   vectorstores
   chat_message_histories

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api_reference