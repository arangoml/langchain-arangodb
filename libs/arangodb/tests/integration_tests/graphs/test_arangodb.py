import pytest
import os
import urllib.parse
from arango.database import StandardDatabase
from langchain_core.documents import Document
import pytest
from arango import ArangoClient
from arango.exceptions import ArangoServerError, ServerConnectionError, ArangoClientError


from langchain_arangodb.graphs.arangodb_graph import ArangoGraph
from langchain_arangodb.graphs.graph_document import GraphDocument, Node, Relationship

test_data = [
    GraphDocument(
        nodes=[Node(id="foo", type="foo"), Node(id="bar", type="bar")],
        relationships=[
            Relationship(
                source=Node(id="foo", type="foo"),
                target=Node(id="bar", type="bar"),
                type="REL",
                properties={"key": "val"},
            )
        ],
        source=Document(page_content="source document"),
    )
]

test_data_backticks = [
    GraphDocument(
        nodes=[Node(id="foo", type="foo`"), Node(id="bar", type="`bar")],
        relationships=[
            Relationship(
                source=Node(id="foo", type="f`oo"),
                target=Node(id="bar", type="ba`r"),
                type="`REL`",
            )
        ],
        source=Document(page_content="source document"),
    )
]
url = os.environ.get("ARANGO_URL", "http://localhost:8529")
username = os.environ.get("ARANGO_USERNAME", "root")
password = os.environ.get("ARANGO_PASSWORD", "test")

os.environ["ARANGO_URL"] = url
os.environ["ARANGO_USERNAME"] = username
os.environ["ARANGO_PASSWORD"] = password


@pytest.mark.usefixtures("clear_arangodb_database")
def test_connect_arangodb(db: StandardDatabase) -> None:
    """Test that ArangoDB database is correctly instantiated and connected."""
    graph = ArangoGraph(db)

    output = graph.query("RETURN 1")
    expected_output = [1]
    assert output == expected_output


@pytest.mark.usefixtures("clear_arangodb_database")
def test_connect_arangodb_env(db: StandardDatabase) -> None:
    """Test that Neo4j database environment variables."""
    assert os.environ.get("ARANGO_URL") is not None
    assert os.environ.get("ARANGO_USERNAME") is not None
    assert os.environ.get("ARANGO_PASSWORD") is not None
    graph = ArangoGraph(db)

    output = graph.query('RETURN 1')
    expected_output = [1]
    assert output == expected_output


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_schema_structure(db: StandardDatabase) -> None:
    """Test that nodes and relationships with properties are correctly inserted and queried in ArangoDB."""
    graph = ArangoGraph(db)

    # Create nodes and relationships using the ArangoGraph API
    doc = GraphDocument(
        nodes=[
            Node(id="label_a", type="LabelA", properties={"property_a": "a"}),
            Node(id="label_b", type="LabelB"),
            Node(id="label_c", type="LabelC"),
        ],
        relationships=[
            Relationship(
                source=Node(id="label_a", type="LabelA"),
                target=Node(id="label_b", type="LabelB"),
                type="REL_TYPE"
            ),
            Relationship(
                source=Node(id="label_a", type="LabelA"),
                target=Node(id="label_c", type="LabelC"),
                type="REL_TYPE",
                properties={"rel_prop": "abc"}
            ),
        ],
        source=Document(page_content="sample document"),
    )

    # Use 'lower' to avoid capitalization_strategy bug
    graph.add_graph_documents(
        [doc],
        capitalization_strategy="lower"
    )

    node_query = """
    FOR doc IN @@collection
      FILTER doc.type == @label
      RETURN {
        type: doc.type,
        properties: KEEP(doc, ["property_a"])
      }
    """

    rel_query = """
    FOR edge IN @@collection
      RETURN {
        text: edge.text,
         }
    """

    node_output = graph.query(
        node_query,
        params={"bind_vars": {"@collection": "ENTITY", "label": "LabelA"}}
    )

    relationship_output = graph.query(
        rel_query,
        params={"bind_vars": {"@collection": "LINKS_TO"}}
    )

    expected_node_properties = [
        {"type": "LabelA", "properties": {"property_a": "a"}}
    ]

    expected_relationships = [
    {
        "text": "label_a REL_TYPE label_b"
    },
    {
        "text": "label_a REL_TYPE label_c"
    }
    ]

    assert node_output == expected_node_properties
    assert  relationship_output == expected_relationships

   



@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_query_timeout(db: StandardDatabase):
    
    long_running_query = "FOR i IN 1..10000000 FILTER i == 0 RETURN i"

    # Set a short maxRuntime to trigger a timeout
    try:
        cursor = db.aql.execute(
            long_running_query,
            max_runtime=0.1  # maxRuntime in seconds
        )
        # Force evaluation of the cursor
        list(cursor)
        assert False, "Query did not timeout as expected"
    except ArangoServerError as e:
        # Check if the error code corresponds to a query timeout
        assert e.error_code == 1500
        assert "query killed" in str(e)


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_sanitize_values(db: StandardDatabase) -> None:
    """Test that large lists are appropriately handled in the results."""
    # Insert a document with a large list
    collection_name = "test_collection"
    if not db.has_collection(collection_name):
        db.create_collection(collection_name)
    collection = db.collection(collection_name)
    large_list = list(range(130))
    collection.insert({"_key": "test_doc", "large_list": large_list})

    # Query the document
    query = f"""
        FOR doc IN {collection_name}
            RETURN doc.large_list
    """
    cursor = db.aql.execute(query)
    result = list(cursor)

    # Assert that the large list is present and has the expected length
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert len(result[0]) == 130





@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_add_data(db: StandardDatabase) -> None:
    """Test that ArangoDB correctly imports graph documents."""
    graph = ArangoGraph(db)

    # Define test data
    test_data = GraphDocument(
        nodes=[
            Node(id="foo", type="foo", properties={}),
            Node(id="bar", type="bar", properties={}),
        ],
        relationships=[],
        source=Document(page_content="test document"),
    )

    # Add graph documents
    graph.add_graph_documents([test_data],capitalization_strategy="lower")

    # Query to count nodes by type
    query = """
        FOR doc IN @@collection
            COLLECT label = doc.type WITH COUNT INTO count
            filter label == @type
            RETURN { label, count }
    """

    # Execute the query for each collection
    foo_result = graph.query(query, params={"bind_vars": {"@collection": "ENTITY", "type": "foo"}})
    bar_result = graph.query(query, params={"bind_vars": {"@collection": "ENTITY", "type": "bar"}})

    # Combine results
    output = foo_result + bar_result

    # Expected output
    expected_output = [{"label": "foo", "count": 1}, {"label": "bar", "count": 1}]

    # Assert the output matches expected
    assert sorted(output, key=lambda x: x["label"]) == sorted(expected_output, key=lambda x: x["label"])



@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_backticks(db: StandardDatabase) -> None:
    """Test that backticks in identifiers are correctly handled."""
    graph = ArangoGraph(db)

    # Define test data with identifiers containing backticks
    test_data_backticks = GraphDocument(
        nodes=[
            Node(id="foo`", type="foo"),
            Node(id="bar`", type="bar"),
        ],
        relationships=[
            Relationship(
                source=Node(id="foo`", type="foo"),
                target=Node(id="bar`", type="bar"),
                type="REL"
            ),
        ],
        source=Document(page_content="sample document"),
    )

    # Add graph documents
    graph.add_graph_documents([test_data_backticks], capitalization_strategy="lower")

    # Query nodes
    node_query = """
        
        FOR doc IN @@collection
            FILTER doc.type == @type
            RETURN { labels: doc.type }

    """
    foo_nodes = graph.query(node_query, params={"bind_vars": {"@collection": "ENTITY", "type": "foo"}})
    bar_nodes = graph.query(node_query, params={"bind_vars": {"@collection": "ENTITY", "type": "bar"}})

    # Query relationships
    rel_query = """
        FOR edge IN @@edge
            RETURN { type: edge.type }
    """
    rels = graph.query(rel_query, params={"bind_vars": {"@edge": "LINKS_TO"}})

    # Expected results
    expected_nodes = [{"labels": "foo"}, {"labels": "bar"}]
    expected_rels = [{"type": "REL"}]

    # Combine node results
    nodes = foo_nodes + bar_nodes

    # Assertions
    assert sorted(nodes, key=lambda x: x["labels"]) == sorted(expected_nodes, key=lambda x: x["labels"])
    assert rels == expected_rels

@pytest.mark.usefixtures("clear_arangodb_database")
def test_invalid_url() -> None:
    """Test initializing with an invalid URL raises ArangoClientError."""
    # Original URL
    original_url = "http://localhost:8529"
    parsed_url = urllib.parse.urlparse(original_url)
    # Increment the port number by 1 and wrap around if necessary
    original_port = parsed_url.port or 8529
    new_port = (original_port + 1) % 65535 or 1
    # Reconstruct the netloc (hostname:port)
    new_netloc = f"{parsed_url.hostname}:{new_port}"
    # Rebuild the URL with the new netloc
    new_url = parsed_url._replace(netloc=new_netloc).geturl()

    client = ArangoClient(hosts=new_url)

    with pytest.raises(ArangoClientError) as exc_info:
        # Attempt to connect with invalid URL
        client.db("_system", username="root", password="passwd", verify=True)

    assert "bad connection" in str(exc_info.value)


@pytest.mark.usefixtures("clear_arangodb_database")
def test_invalid_credentials() -> None:
    """Test initializing with invalid credentials raises ArangoServerError."""
    client = ArangoClient(hosts="http://localhost:8529")

    with pytest.raises(ArangoServerError) as exc_info:
        # Attempt to connect with invalid username and password
        client.db("_system", username="invalid_user", password="invalid_pass", verify=True)

    assert "bad username/password" in str(exc_info.value)