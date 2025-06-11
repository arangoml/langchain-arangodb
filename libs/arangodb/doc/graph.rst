.. _arangograph_graph_store:

===========
ArangoGraph
===========

The ``ArangoGraph`` class is a comprehensive wrapper for ArangoDB, designed to facilitate graph operations within the LangChain ecosystem. It implements the ``GraphStore`` interface, providing robust functionalities for schema generation, AQL querying, and constructing complex graphs from ``GraphDocument`` objects.

.. warning::
    **Security Note**: This class interacts directly with your database. Ensure that the database connection credentials are narrowly-scoped with the minimum necessary permissions. Failure to do so can result in data corruption, data loss, or exposure of sensitive information if the calling code attempts unintended mutations or reads. See `LangChain Security Docs <https://python.langchain.com/docs/security>`_ for more information.

Overview
--------

``ArangoGraph`` simplifies the integration of ArangoDB as a knowledge graph backend for LLM applications.

Core Features:
~~~~~~~~~~~~~~

* **Automatic Schema Generation**: Introspects the database to generate a detailed schema, which is crucial for providing context to LLMs. The schema can be customized based on a specific graph or sampled from collections.
* **Graph Construction**: Ingests lists of ``GraphDocument`` objects, efficiently creating nodes and relationships in ArangoDB.
* **Flexible Data Modeling**: Supports two primary strategies for storing graph data:
    1.  **Unified Entity Collections**: All nodes and relationships are stored in single, designated collections (e.g., "ENTITY", "LINKS_TO").
    2.  **Type-Based Collections**: Nodes and relationships are stored in separate collections based on their assigned `type` (e.g., "Person", "Company", "WORKS_FOR").
* **Embedding Integration**: Seamlessly generates and stores vector embeddings for nodes, relationships, and source documents using any LangChain-compatible embedding provider.
* **AQL Querying**: Provides direct methods to execute and explain AQL queries, with built-in sanitization to manage large data fields for LLM processing.
* **Convenience Initializers**: Allows for easy instantiation from environment variables or direct credentials.

Initialization
--------------

The primary way to initialize ``ArangoGraph`` is by providing a `python-arango` database instance.

.. code-block:: python

    from arango import ArangoClient
    from langchain_arangodb import ArangoGraph

    # 1. Connect to ArangoDB
    client = ArangoClient(hosts="http://localhost:8529")
    db = client.db("your_db_name", username="root", password="your_password")

    # 2. Initialize ArangoGraph
    # Schema will be generated automatically on initialization
    graph = ArangoGraph(db=db)

    # You can now access the schema
    print(graph.schema_yaml)


Convenience Constructor
~~~~~~~~~~~~~~~~~~~~~~~

For ease of use, you can initialize directly from credentials or environment variables using the ``from_db_credentials`` class method.

**Environment Variables:**

* ``ARANGODB_URL`` (default: "http://localhost:8529")
* ``ARANGODB_DBNAME`` (default: "_system")
* ``ARANGODB_USERNAME`` (default: "root")
* ``ARANGODB_PASSWORD`` (default: "")

.. code-block:: python

    # This will automatically use credentials from environment variables
    graph_from_env = ArangoGraph.from_db_credentials()

    # Or pass them directly
    graph_from_args = ArangoGraph.from_db_credentials(
        url="http://localhost:8529",
        dbname="my_app_db",
        username="my_user",
        password="my_password"
    )

Configuration
-------------

The behavior of ``ArangoGraph`` can be configured during initialization:

.. py:class:: ArangoGraph(db, generate_schema_on_init=True, schema_sample_ratio=0, schema_graph_name=None, schema_include_examples=True, schema_list_limit=32, schema_string_limit=256)

   :param db: An instance of `arango.database.StandardDatabase`.
   :type db: arango.database.StandardDatabase
   :param generate_schema_on_init: If ``True``, automatically generates the graph schema upon initialization.
   :type generate_schema_on_init: bool
   :param schema_sample_ratio: The ratio (0 to 1) of documents to sample from each collection for schema generation. A value of `0` samples one document.
   :type schema_sample_ratio: float
   :param schema_graph_name: If specified, the schema generation will be limited to the collections within this named graph.
   :type schema_graph_name: str, optional
   :param schema_include_examples: If ``True``, includes example values from sampled documents in the schema.
   :type schema_include_examples: bool
   :param schema_list_limit: The maximum length for lists to be included as examples in the schema.
   :type schema_list_limit: int
   :param schema_string_limit: The maximum length for strings to be included as examples in the schema.
   :type schema_string_limit: int

Schema Management
-----------------

The graph schema provides a structured view of your data, which is essential for LLMs to generate accurate AQL queries.

### Accessing the Schema

Once initialized or refreshed, the schema is cached and can be accessed in various formats.

.. code-block:: python

    # Get schema as a Python dictionary
    structured_schema = graph.schema

    # Get schema as a JSON string
    json_schema = graph.schema_json

    # Get schema as a YAML string (often best for LLM prompts)
    yaml_schema = graph.schema_yaml
    print(yaml_schema)


### Refreshing the Schema

If your graph's structure changes, you can refresh the schema at any time.

.. code-block:: python

    # Refresh schema using default settings
    graph.refresh_schema()

    # Refresh schema for a specific graph with more samples
    graph.refresh_schema(graph_name="my_specific_graph", sample_ratio=0.1)


Adding Graph Documents
----------------------

The ``add_graph_documents`` method is the primary way to populate your graph. It takes a list of ``GraphDocument`` objects and intelligently creates nodes and relationships.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from langchain_core.documents import Document
    from langchain_arangodb.graphs.graph_document import Node, Relationship, GraphDocument
    from langchain_openai import OpenAIEmbeddings

    # 1. Define nodes and relationships
    node1 = Node(id="Alice", type="Person", properties={"age": 30})
    node2 = Node(id="Bob", type="Person", properties={"age": 32})
    relationship = Relationship(source=node1, target=node2, type="KNOWS", properties={"since": 2021})

    # 2. Define the source document
    source_doc = Document(page_content="Alice and Bob have been friends since 2021.")

    # 3. Create a GraphDocument
    graph_doc = GraphDocument(nodes=[node1, node2], relationships=[relationship], source=source_doc)

    # 4. Add to the graph
    graph.add_graph_documents([graph_doc])


Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

The method offers extensive options for controlling how data is stored.

.. py:method:: add_graph_documents(graph_documents, include_source=False, graph_name=None, use_one_entity_collection=True, embeddings=None, embed_nodes=False, ...)

   :param graph_documents: A list of ``GraphDocument`` objects to add.
   :type graph_documents: List[GraphDocument]
   :param include_source: If ``True``, stores the source document and links it to the extracted nodes.
   :type include_source: bool
   :param graph_name: The name of an ArangoDB graph to create or update with the new edge definitions.
   :type graph_name: str, optional
   :param update_graph_definition_if_exists: If ``True``, adds new edge definitions to an existing graph. Recommended when `use_one_entity_collection` is ``False``.
   :type update_graph_definition_if_exists: bool
   :param use_one_entity_collection: If ``True``, all nodes are stored in a single "ENTITY" collection. If ``False``, nodes are stored in collections named after their `type`.
   :type use_one_entity_collection: bool
   :param embeddings: An embedding model to generate vectors for nodes, relationships, or sources.
   :type embeddings: Embeddings, optional
   :param embed_nodes: If ``True``, generates and stores embeddings for nodes.
   :type embed_nodes: bool
   :param capitalization_strategy: Applies capitalization ("lower", "upper", "none") to node IDs to aid in entity resolution.
   :type capitalization_strategy: str
   :param ...: Other parameters include `batch_size`, `insert_async`, and custom collection names.

Example: Using Type-Based Collections and Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    graph.add_graph_documents(
        [graph_doc],
        graph_name="people_graph",
        use_one_entity_collection=False,  # Creates 'Person' node collection and 'KNOWS' edge collection
        update_graph_definition_if_exists=True,
        include_source=True,
        embeddings=OpenAIEmbeddings(),
        embed_nodes=True  # Embeds 'Alice' and 'Bob' nodes
    )


Querying the Graph
------------------

You can execute AQL queries directly through the ``query`` method or get their execution plan using ``explain``.

.. code-block:: python

    # Execute a query
    aql_query = "FOR p IN Person FILTER p.age > 30 RETURN p"
    results = graph.query(aql_query)
    print(results)

    # Get the query plan without executing it
    plan = graph.explain(aql_query)
    print(plan)


The ``query`` method automatically sanitizes results by truncating long strings and lists, making the output suitable for LLM processing.

.. code-block:: python

    # Example of sanitization
    long_text_query = "FOR doc IN my_docs LIMIT 1 RETURN doc"
    results = graph.query(
        long_text_query,
        params={"top_k": 1, "string_limit": 64} # Custom limits
    )
    # The 'text' field in the result will be truncated if it exceeds 64 chars.


API Reference
-------------

.. automodule:: langchain_arangodb.graphs.arangodb_graph
   :members:
   :undoc-members:
   :show-inheritance:



