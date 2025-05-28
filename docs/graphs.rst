Graphs
======

LangChain ArangoDB provides graph implementations that allow you to work with graph data in ArangoDB.

ArangoDBGraph
------------

The main graph implementation that uses ArangoDB for storing and querying graph data.

.. code-block:: python

    from langchain_arangodb.graphs import ArangoDBGraph

    # Initialize the graph
    graph = ArangoDBGraph(
        arango_url="http://localhost:8529",
        username="root",
        password="",
        database="langchain",
        graph_name="knowledge_graph"
    )

    # Add nodes and edges
    graph.add_node("person", {"name": "John", "age": 30})
    graph.add_node("person", {"name": "Alice", "age": 25})
    graph.add_edge("knows", "person/John", "person/Alice")

    # Query the graph
    results = graph.query("FOR v IN person RETURN v")

Features
--------

- Graph data modeling
- Node and edge management
- AQL query support
- Graph traversal capabilities
- Integration with LangChain's graph interfaces

Configuration Options
--------------------

The graph implementation can be configured with various options:

- ``arango_url``: URL of the ArangoDB instance
- ``username``: ArangoDB username
- ``password``: ArangoDB password
- ``database``: Database name
- ``graph_name``: Name of the graph
- ``edge_definitions``: Edge collection definitions
- ``orphan_collections``: Collections that can contain orphan vertices 