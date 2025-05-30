Chains
======

LangChain ArangoDB provides chain implementations that integrate with ArangoDB for various operations.

ArangoDBChain
------------

The main chain implementation that uses ArangoDB for storing and retrieving chain data.

.. code-block:: python

    from langchain_arangodb.chains import ArangoDBChain
    from langchain.llms import OpenAI

    # Initialize the chain
    chain = ArangoDBChain(
        llm=OpenAI(),
        arango_url="http://localhost:8529",
        username="root",
        password="",
        database="langchain",
        collection_name="chain_data"
    )

    # Run the chain
    result = chain.run("What is the capital of France?")

Features
--------

- Chain execution with ArangoDB storage
- Integration with LangChain's chain interfaces
- Support for various chain types
- Persistent storage of chain data
- Configurable chain parameters

Configuration Options
--------------------

The chain implementation can be configured with various options:

- ``llm``: The language model to use
- ``arango_url``: URL of the ArangoDB instance
- ``username``: ArangoDB username
- ``password``: ArangoDB password
- ``database``: Database name
- ``collection_name``: Collection name for storing chain data
- ``chain_type``: Type of chain to use
- ``chain_kwargs``: Additional chain parameters 