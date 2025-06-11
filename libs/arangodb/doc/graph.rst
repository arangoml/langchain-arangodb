ArangoGraph
===========

The ``ArangoGraph`` class provides an interface to interact with ArangoDB for graph operations in LangChain.

Installation
------------

.. code-block:: bash

   pip install langchain-arangodb

Basic Usage
-----------

.. code-block:: python

   from langchain_arangodb.graphs.arangodb_graph import ArangoGraph, get_arangodb_client

   # Connect to ArangoDB
   db = get_arangodb_client(
       url="http://localhost:8529",
       dbname="_system",
       username="root",
       password="password"
   )

   # Initialize ArangoGraph
   graph = ArangoGraph(db)


Factory Methods
---------------

get_arangodb_client
~~~~~~~~~~~~~~~~~~~~

Creates a connection to ArangoDB.

.. code-block:: python

   from langchain_arangodb.graphs.arangodb_graph import get_arangodb_client

   # Using direct credentials
   db = get_arangodb_client(
       url="http://localhost:8529",
       dbname="_system", 
       username="root",
       password="password"
   )

   # Using environment variables
   # ARANGODB_URL
   # ARANGODB_DBNAME
   # ARANGODB_USERNAME
   # ARANGODB_PASSWORD
   db = get_arangodb_client()

from_db_credentials
~~~~~~~~~~~~~~~~~~

Alternative constructor that creates an ArangoGraph instance directly from credentials.

.. code-block:: python

   graph = ArangoGraph.from_db_credentials(
       url="http://localhost:8529",
       dbname="_system",
       username="root", 
       password="password"
   )

Core Methods
------------

add_graph_documents
~~~~~~~~~~~~~~~~~~~

Adds graph documents to the database.

.. code-block:: python

   from langchain_core.documents import Document
   from langchain_arangodb.graphs.graph_document import GraphDocument, Node, Relationship

   # Create nodes and relationships
   nodes = [
       Node(id="1", type="Person", properties={"name": "Alice"}),
       Node(id="2", type="Company", properties={"name": "Acme"})
   ]
   
   relationship = Relationship(
       source=nodes[0],
       target=nodes[1], 
       type="WORKS_AT",
       properties={"since": 2020}
   )

   # Create graph document
   doc = GraphDocument(
       nodes=nodes,
       relationships=[relationship],
       source=Document(page_content="Employee record")
   )

   # Add to database
   graph.add_graph_documents(
       graph_documents=[doc],
       include_source=True,
       graph_name="EmployeeGraph",
       update_graph_definition_if_exists=True,
       capitalization_strategy="lower"
   )
Example: Using LLMGraphTransformer 

.. code-block:: python

   from langchain.experimental import LLMGraphTransformer
   from langchain_core.chat_models import ChatOpenAI
   from langchain_openai import OpenAIEmbeddings

   # Text to transform into a graph
   text = "Bob knows Alice, John knows Bob."

   # Initialize transformer with ChatOpenAI
   transformer = LLMGraphTransformer(
       llm=ChatOpenAI(temperature=0)
   )

   # Create graph document from text
   graph_doc = transformer.create_graph_doc(text)

   # Add to ArangoDB with embeddings
   graph.add_graph_documents(
       [graph_doc],
       graph_name="people_graph",
       use_one_entity_collection=False,  # Creates 'Person' node collection and 'KNOWS' edge collection
       update_graph_definition_if_exists=True,
       include_source=True,
       embeddings=OpenAIEmbeddings(),
       embed_nodes=True  # Embeds 'Alice' and 'Bob' nodes
   )

query
~~~~~

Executes AQL queries against the database.

.. code-block:: python

   # Simple query
   result = graph.query("FOR doc IN users RETURN doc")

   # Query with parameters
   result = graph.query(
       "FOR u IN users FILTER u.age > @min_age RETURN u",
       params={"min_age": 21}
   )



explain
~~~~~~~

Gets the query execution plan.

.. code-block:: python

   plan = graph.explain(
       "FOR doc IN users RETURN doc"
   )

Schema Management
-----------------

refresh_schema
~~~~~~~~~~~~~~

Updates the internal schema representation.

.. code-block:: python

   graph.refresh_schema(
       sample_ratio=0.1,  # Sample 10% of documents
       graph_name="MyGraph",
       include_examples=True
   )

generate_schema
~~~~~~~~~~~~~~~

Generates a schema representation of the database.

.. code-block:: python

   schema = graph.generate_schema(
       sample_ratio=0.1,
       graph_name="MyGraph",
       include_examples=True,
       list_limit=32
   )

set_schema
~~~~~~~~~~

Sets a custom schema.

.. code-block:: python

   custom_schema = {
       "collections": {
           "users": {"fields": ["name", "age"]},
           "products": {"fields": ["name", "price"]}
       }
   }
   
   graph.set_schema(custom_schema)

Schema Properties
-----------------

schema
~~~~~~

Gets the current schema as a dictionary.

.. code-block:: python

   current_schema = graph.schema

schema_json
~~~~~~~~~~~~

Gets the schema as a JSON string.

.. code-block:: python

   schema_json = graph.schema_json

schema_yaml
~~~~~~~~~~~

Gets the schema as a YAML string.

.. code-block:: python

   schema_yaml = graph.schema_yaml

get_structured_schema
~~~~~~~~~~~~~~~~~~~~~

Gets the schema in a structured format.

.. code-block:: python

   structured_schema = graph.get_structured_schema

Internal Utility Methods
-----------------------

These methods are used internally but may be useful for advanced use cases:

_sanitize_collection_name
~~~~~~~~~~~~~~~~~~~~~~~~

Sanitizes collection names to be valid in ArangoDB.

.. code-block:: python

   safe_name = graph._sanitize_collection_name("My Collection!")
   # Returns: "My_Collection_"

_sanitize_input
~~~~~~~~~~~~~~~~

Sanitizes input data by truncating long strings and lists.

.. code-block:: python

   sanitized = graph._sanitize_input(
       {"list": [1,2,3,4,5,6]}, 
       list_limit=5,
       string_limit=100
   )

_hash
~~~~~

Generates a hash string for a value.

.. code-block:: python

   hash_str = graph._hash("some value")

_process_source
~~~~~~~~~~~~~~~~

Processes a source document for storage.

.. code-block:: python

   from langchain_core.documents import Document
   
   source = Document(
       page_content="test content",
       metadata={"author": "Alice"}
   )
   
   source_id = graph._process_source(
       source=source,
       source_collection_name="sources",
       source_embedding=[0.1, 0.2, 0.3],
       embedding_field="embedding",
       insertion_db=db
   )

_import_data
~~~~~~~~~~~~~

Bulk imports data into collections.

.. code-block:: python

   data = {
       "users": [
           {"_key": "1", "name": "Alice"},
           {"_key": "2", "name": "Bob"}
       ]
   }
   
   graph._import_data(db, data, is_edge=False)


Example Workflow
----------------    

Here's a complete example demonstrating a typical workflow using ArangoGraph to create a knowledge graph from documents:

.. code-block:: python

   from langchain_core.documents import Document
   from langchain_core.embeddings import Embeddings
   from langchain_arangodb.graphs.arangodb_graph import ArangoGraph, get_arangodb_client
   from langchain_arangodb.graphs.graph_document import GraphDocument, Node, Relationship

   # 1. Setup embeddings (example using OpenAI - you can use any embeddings model)
   from langchain_openai import OpenAIEmbeddings
   embeddings = OpenAIEmbeddings()
   # 2. Connect to ArangoDB and initialize graph
   db = get_arangodb_client(
       url="http://localhost:8529",
       dbname="knowledge_base",
       username="root",
       password="password"
   )
   graph = ArangoGraph(db)

   # 3. Create sample documents with relationships
   documents = [
       Document(
           page_content="Alice is a software engineer at Acme Corp.",
           metadata={"source": "employee_records", "date": "2024-01-01"}
       ),
       Document(
           page_content="Bob is a project manager working with Alice on Project X.",
           metadata={"source": "project_docs", "date": "2024-01-02"}
       )
   ]

   # 4. Create nodes and relationships for each document
   graph_documents = []
   for doc in documents:
       # Extract entities and relationships (simplified example)
       if "Alice" in doc.page_content:
           alice_node = Node(id="alice", type="Person", properties={"name": "Alice", "role": "Software Engineer"})
           company_node = Node(id="acme", type="Company", properties={"name": "Acme Corp"})
           works_at_rel = Relationship(
               source=alice_node,
               target=company_node,
               type="WORKS_AT"
           )
           graph_doc = GraphDocument(
               nodes=[alice_node, company_node],
               relationships=[works_at_rel],
               source=doc
           )
           graph_documents.append(graph_doc)
       
       if "Bob" in doc.page_content:
           bob_node = Node(id="bob", type="Person", properties={"name": "Bob", "role": "Project Manager"})
           project_node = Node(id="project_x", type="Project", properties={"name": "Project X"})
           manages_rel = Relationship(
               source=bob_node,
               target=project_node,
               type="MANAGES"
           )
           works_with_rel = Relationship(
               source=bob_node,
               target=alice_node,
               type="WORKS_WITH"
           )
           graph_doc = GraphDocument(
               nodes=[bob_node, project_node],
               relationships=[manages_rel, works_with_rel],
               source=doc
           )
           graph_documents.append(graph_doc)

   # 5. Add documents to the graph with embeddings
   graph.add_graph_documents(
       graph_documents=graph_documents,
       include_source=True,  # Store original documents
       graph_name="CompanyGraph",
       update_graph_definition_if_exists=True,
       embed_source=True,  # Generate embeddings for documents
       embed_nodes=True,  # Generate embeddings for nodes
       embed_relationships=True,  # Generate embeddings for relationships
       embeddings=embeddings,
       batch_size=100,
       capitalization_strategy="lower"
   )

   # 6. Query the graph
   # Find all people who work at Acme Corp
   employees = graph.query("""
       FOR v, e IN 1..1 OUTBOUND 
           (FOR c IN ENTITY FILTER c.type == 'Company' AND c.name == 'Acme Corp' RETURN c)._id
           ENTITY_EDGE
       RETURN {
           name: v.name,
           role: v.role,
           company: 'Acme Corp'
       }
   """)

   # Find all projects and their managers
   projects = graph.query("""
       FOR v, e IN 1..1 INBOUND 
           (FOR p IN ENTITY FILTER p.type == 'Project' RETURN p)._id
           ENTITY_EDGE
       FILTER e.type == 'MANAGES'
       RETURN {
           project: v.name,
           manager: e._from
       }
   """)

   # 7. Generate and inspect schema
   schema = graph.generate_schema(
       sample_ratio=1.0,  # Use all documents for schema
       graph_name="CompanyGraph",
       include_examples=True
   )

   print("Schema:", schema)

   # 8. Error handling for queries
   try:
       # Complex query with potential for errors
       result = graph.query("""
           FOR v, e, p IN 1..3 OUTBOUND 
               (FOR p IN ENTITY FILTER p.name == 'Alice' RETURN p)._id
               ENTITY_EDGE
           RETURN p
       """)
   except ArangoServerError as e:
       print(f"Query error: {e}")

This workflow demonstrates:

1. Setting up the environment with embeddings
2. Connecting to ArangoDB
3. Creating documents with structured relationships
4. Adding documents to the graph with embeddings
5. Querying the graph using AQL
6. Schema management
7. Error handling

The example creates a simple company knowledge graph with:

- People (employees)
- Companies
- Projects
- Various relationships (WORKS_AT, MANAGES, WORKS_WITH)
- Document sources with embeddings

Key Features Used:

- Document embedding
- Node and relationship embedding
- Source document storage
- Graph schema management
- AQL queries
- Error handling
- Batch processing


Best Practices
--------------

1. Always use appropriate capitalization strategy for consistency
2. Use batch operations for large data imports
3. Consider using embeddings for semantic search capabilities
4. Implement proper error handling for database operations
5. Use schema management for better data organization

Error Handling
--------------

.. code-block:: python

   from arango.exceptions import ArangoServerError

   try:
       result = graph.query("FOR doc IN nonexistent RETURN doc")
   except ArangoServerError as e:
       print(f"Database error: {e}")


--------------




