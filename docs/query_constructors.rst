Query Constructors
================

LangChain ArangoDB provides query constructor implementations that help build AQL queries for ArangoDB.

ArangoDBQueryConstructor
-----------------------

The main query constructor implementation that helps build AQL queries.

.. code-block:: python

    from langchain_arangodb.query_constructors import ArangoDBQueryConstructor

    # Initialize the query constructor
    constructor = ArangoDBQueryConstructor(
        collection_name="documents",
        filter_fields=["category", "tags"],
        sort_fields=["created_at", "updated_at"]
    )

    # Build a query
    query = constructor.construct_query(
        filter_criteria={
            "category": "news",
            "tags": ["important", "urgent"]
        },
        sort_by="created_at",
        sort_order="DESC",
        limit=10
    )

Features
--------

- AQL query construction
- Support for filtering
- Support for sorting
- Support for pagination
- Support for aggregation
- Integration with LangChain's query interfaces

Configuration Options
--------------------

The query constructor can be configured with various options:

- ``collection_name``: Name of the collection to query
- ``filter_fields``: Fields that can be used for filtering
- ``sort_fields``: Fields that can be used for sorting
- ``default_limit``: Default number of results to return
- ``default_sort_field``: Default field to sort by
- ``default_sort_order``: Default sort order (ASC/DESC) 