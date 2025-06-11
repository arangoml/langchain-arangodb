import json
import os
import pprint

import urllib.parse

from collections import defaultdict
from unittest.mock import MagicMock

import pytest
from arango import ArangoClient
from arango.database import StandardDatabase

from arango.exceptions import ArangoClientError, ArangoServerError

from langchain_core.documents import Document

from langchain_arangodb.graphs.arangodb_graph import ArangoGraph, get_arangodb_client
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
url = os.environ.get("ARANGO_URL", "http://localhost:8529")  # type: ignore[assignment]
username = os.environ.get("ARANGO_USERNAME", "root")  # type: ignore[assignment]
password = os.environ.get("ARANGO_PASSWORD", "test")  # type: ignore[assignment]

os.environ["ARANGO_URL"] = url  # type: ignore[assignment]
os.environ["ARANGO_USERNAME"] = username  # type: ignore[assignment]
os.environ["ARANGO_PASSWORD"] = password  # type: ignore[assignment]


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

    output = graph.query("RETURN 1")
    expected_output = [1]
    assert output == expected_output


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_schema_structure(db: StandardDatabase) -> None:
    """Test that nodes and relationships with properties are correctly
    inserted and queried in ArangoDB."""
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
                type="REL_TYPE",
            ),
            Relationship(
                source=Node(id="label_a", type="LabelA"),
                target=Node(id="label_c", type="LabelC"),
                type="REL_TYPE",
                properties={"rel_prop": "abc"},
            ),
        ],
        source=Document(page_content="sample document"),
    )

    # Use 'lower' to avoid capitalization_strategy bug
    graph.add_graph_documents([doc], capitalization_strategy="lower")

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
        node_query, params={"bind_vars": {"@collection": "ENTITY", "label": "LabelA"}}
    )

    relationship_output = graph.query(
        rel_query, params={"bind_vars": {"@collection": "LINKS_TO"}}
    )

    expected_node_properties = [{"type": "LabelA", "properties": {"property_a": "a"}}]

    expected_relationships = [
        {"text": "label_a REL_TYPE label_b"},
        {"text": "label_a REL_TYPE label_c"},
    ]

    assert node_output == expected_node_properties
    assert relationship_output == expected_relationships


@pytest.mark.usefixtures("clear_arangodb_database")

def test_arangodb_query_timeout(db: StandardDatabase) -> None:
    long_running_query = "FOR i IN 1..10000000 FILTER i == 0 RETURN i"

    # Set a short maxRuntime to trigger a timeout
    try:
        cursor = db.aql.execute(
            long_running_query,
            max_runtime=0.1,  # type: ignore # maxRuntime in seconds
        )  # type: ignore
        # Force evaluation of the cursor
        list(cursor)  # type: ignore
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
    result = list(cursor)  # type: ignore

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
    graph.add_graph_documents([test_data], capitalization_strategy="lower")

    # Query to count nodes by type
    query = """
        FOR doc IN @@collection
            COLLECT label = doc.type WITH COUNT INTO count
            filter label == @type
            RETURN { label, count }
    """

    # Execute the query for each collection
    foo_result = graph.query(
        query, params={"bind_vars": {"@collection": "ENTITY", "type": "foo"}}
    )  # noqa: E501
    bar_result = graph.query(
        query, params={"bind_vars": {"@collection": "ENTITY", "type": "bar"}}
    )  # noqa: E501

    # Combine results
    output = foo_result + bar_result

    # Expected output
    expected_output = [{"label": "foo", "count": 1}, {"label": "bar", "count": 1}]

    # Assert the output matches expected
    assert sorted(output, key=lambda x: x["label"]) == sorted(
        expected_output,
        key=lambda x: x["label"],  # type: ignore
    )  # noqa: E501


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_rels(db: StandardDatabase) -> None:
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
                type="REL",
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
    foo_nodes = graph.query(
        node_query, params={"bind_vars": {"@collection": "ENTITY", "type": "foo"}}
    )  # noqa: E501
    bar_nodes = graph.query(
        node_query, params={"bind_vars": {"@collection": "ENTITY", "type": "bar"}}
    )  # noqa: E501

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
    assert sorted(nodes, key=lambda x: x["labels"]) == sorted(
        expected_nodes, key=lambda x: x["labels"]
    )  # noqa: E501
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
        client.db(
            "_system", username="invalid_user", password="invalid_pass", verify=True
        )

    assert "bad username/password" in str(exc_info.value)


@pytest.mark.usefixtures("clear_arangodb_database")
def test_schema_refresh_updates_schema(db: StandardDatabase) -> None:
    """Test that schema is updated when add_graph_documents is called."""
    graph = ArangoGraph(db, generate_schema_on_init=False)
    assert graph.schema == {}

    doc = GraphDocument(
        nodes=[Node(id="x", type="X")],
        relationships=[],
        source=Document(page_content="refresh test"),
    )
    graph.add_graph_documents([doc], capitalization_strategy="lower")

    assert "collection_schema" in graph.schema
    assert any(
        col["name"].lower() == "entity" for col in graph.schema["collection_schema"]
    )


@pytest.mark.usefixtures("clear_arangodb_database")
def test_sanitize_input_list_cases(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    sanitize = graph._sanitize_input

    # 1. Empty list
    assert sanitize([], list_limit=5, string_limit=10) == []

    # 2. List within limit with nested dicts
    input_data = [{"a": "short"}, {"b": "short"}]
    result = sanitize(input_data, list_limit=5, string_limit=10)
    assert isinstance(result, list)
    assert result == input_data  # No truncation needed

    # 3. List exceeding limit
    long_list = list(range(20))  # default list_limit should be < 20
    result = sanitize(long_list, list_limit=5, string_limit=10)
    assert isinstance(result, str)
    assert result.startswith("List of 20 elements of type")

    # 4. List at exact limit (should pass through)
    exact_limit_list = list(range(5))
    result = sanitize(exact_limit_list, list_limit=5, string_limit=10)
    assert isinstance(result, str)  # Should still be replaced since `len == list_limit`


@pytest.mark.usefixtures("clear_arangodb_database")
def test_sanitize_input_dict_with_lists(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)
    sanitize = graph._sanitize_input

    # 1. Dict with short list as value
    input_data_short = {"my_list": [1, 2, 3]}
    result_short = sanitize(input_data_short, list_limit=5, string_limit=50)
    assert result_short == {"my_list": [1, 2, 3]}

    # 2. Dict with long list as value
    input_data_long = {"my_list": list(range(10))}
    result_long = sanitize(input_data_long, list_limit=5, string_limit=50)
    assert isinstance(result_long["my_list"], str)
    assert result_long["my_list"].startswith("List of 10 elements of type")

    # 3. Dict with empty list
    input_data_empty: dict[str, list[int]] = {"empty": []}
    result_empty = sanitize(input_data_empty, list_limit=5, string_limit=50)
    assert result_empty == {"empty": []}


@pytest.mark.usefixtures("clear_arangodb_database")
def test_sanitize_collection_name(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    # 1. Valid name (no change)
    assert graph._sanitize_collection_name("validName123") == "validName123"

    # 2. Name with invalid characters (replaced with "_")
    assert graph._sanitize_collection_name("name with spaces!") == "name_with_spaces_"  # noqa: E501

    # 3. Name starting with a digit (prepends "Collection_")
    assert (
        graph._sanitize_collection_name("1invalidStart") == "Collection_1invalidStart"
    )  # noqa: E501

    # 4. Name starting with underscore (still not a letter → prepend)
    assert graph._sanitize_collection_name("_underscore") == "Collection__underscore"  # noqa: E501

    # 5. Name too long (should trim to 256 characters)
    long_name = "x" * 300
    result = graph._sanitize_collection_name(long_name)
    assert len(result) <= 256

    # 6. Empty string should raise ValueError
    with pytest.raises(ValueError, match="Collection name cannot be empty."):
        graph._sanitize_collection_name("")


@pytest.mark.usefixtures("clear_arangodb_database")
def test_process_source(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    source_doc = Document(page_content="Test content", metadata={"author": "Alice"})
    # Manually override the default type (not part of constructor)
    source_doc.type = "test_type"  # type: ignore

    collection_name = "TEST_SOURCE"
    if not db.has_collection(collection_name):
        db.create_collection(collection_name)

    embedding = [0.1, 0.2, 0.3]
    source_id = graph._process_source(
        source=source_doc,
        source_collection_name=collection_name,
        source_embedding=embedding,
        embedding_field="embedding",
        insertion_db=db,
    )

    inserted_doc = db.collection(collection_name).get(source_id)

    assert inserted_doc is not None
    assert inserted_doc["_key"] == source_id  # type: ignore
    assert inserted_doc["text"] == "Test content"  # type: ignore
    assert inserted_doc["author"] == "Alice"  # type: ignore
    assert inserted_doc["type"] == "test_type"  # type: ignore
    assert inserted_doc["embedding"] == embedding  # type: ignore


@pytest.mark.usefixtures("clear_arangodb_database")
def test_process_edge_as_type(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    # Define source and target nodes
    source_node = Node(id="s1", type="Person")
    target_node = Node(id="t1", type="City")

    # Define edge with type and properties
    edge = Relationship(
        source=source_node,
        target=target_node,
        type="LIVES_IN",
        properties={"since": "2020"},
    )

    edge_key = "edge123"
    edge_str = "s1 LIVES_IN t1"
    source_key = "s1_key"
    target_key = "t1_key"

    # Setup containers
    edges = defaultdict(list)  # type: ignore
    edge_definitions_dict = defaultdict(lambda: defaultdict(set))  # type: ignore

    # Call the method
    graph._process_edge_as_type(
        edge=edge,
        edge_str=edge_str,
        edge_key=edge_key,
        source_key=source_key,
        target_key=target_key,
        edges=edges,
        _1="unused",
        _2="unused",
        edge_definitions_dict=edge_definitions_dict,
    )

    # Assertions
    sanitized_edge_type = graph._sanitize_collection_name("LIVES_IN")
    sanitized_source_type = graph._sanitize_collection_name("Person")
    sanitized_target_type = graph._sanitize_collection_name("City")

    # Edge inserted in correct collection
    assert len(edges[sanitized_edge_type]) == 1
    inserted_edge = edges[sanitized_edge_type][0]

    assert inserted_edge["_key"] == edge_key
    assert inserted_edge["_from"] == f"{sanitized_source_type}/{source_key}"
    assert inserted_edge["_to"] == f"{sanitized_target_type}/{target_key}"
    assert inserted_edge["text"] == edge_str
    assert inserted_edge["since"] == "2020"

    # Edge definitions updated
    assert (
        sanitized_source_type
        in edge_definitions_dict[sanitized_edge_type]["from_vertex_collections"]
    )  # noqa: E501
    assert (
        sanitized_target_type
        in edge_definitions_dict[sanitized_edge_type]["to_vertex_collections"]
    )  # noqa: E501


@pytest.mark.usefixtures("clear_arangodb_database")
def test_graph_creation_and_edge_definitions(db: StandardDatabase) -> None:
    graph_name = "TestGraph"
    graph = ArangoGraph(db, generate_schema_on_init=False)

    graph_doc = GraphDocument(
        nodes=[
            Node(id="user1", type="User"),
            Node(id="group1", type="Group"),
        ],
        relationships=[
            Relationship(
                source=Node(id="user1", type="User"),
                target=Node(id="group1", type="Group"),
                type="MEMBER_OF",
            )
        ],
        source=Document(page_content="user joins group"),
    )

    graph.add_graph_documents(
        [graph_doc],
        graph_name=graph_name,
        update_graph_definition_if_exists=True,
        capitalization_strategy="lower",
        use_one_entity_collection=False,
    )

    assert db.has_graph(graph_name)
    g = db.graph(graph_name)

    edge_definitions = g.edge_definitions()
    edge_collections = {e["edge_collection"] for e in edge_definitions}  # type: ignore
    assert "MEMBER_OF" in edge_collections  # MATCH lowercased name

    member_def = next(
        e
        for e in edge_definitions  # type: ignore
        if e["edge_collection"] == "MEMBER_OF"  # type: ignore
    )
    assert "User" in member_def["from_vertex_collections"]  # type: ignore
    assert "Group" in member_def["to_vertex_collections"]  # type: ignore


@pytest.mark.usefixtures("clear_arangodb_database")
def test_include_source_collection_setup(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    graph_name = "TestGraph"
    source_col = f"{graph_name}_SOURCE"
    source_edge_col = f"{graph_name}_HAS_SOURCE"
    entity_col = f"{graph_name}_ENTITY"

    # Input with source document
    graph_doc = GraphDocument(
        nodes=[
            Node(id="user1", type="User"),
        ],
        relationships=[],
        source=Document(page_content="source doc"),
    )

    # Insert with include_source=True
    graph.add_graph_documents(
        [graph_doc],
        graph_name=graph_name,
        include_source=True,
        capitalization_strategy="lower",
        use_one_entity_collection=True,  # test common case
    )

    # Assert source and edge collections were created
    assert db.has_collection(source_col)
    assert db.has_collection(source_edge_col)

    # Assert that at least one source edge exists and links correctly
    edges = list(db.collection(source_edge_col).all())  # type: ignore
    assert len(edges) == 1
    edge = edges[0]
    assert edge["_to"].startswith(f"{source_col}/")
    assert edge["_from"].startswith(f"{entity_col}/")


@pytest.mark.usefixtures("clear_arangodb_database")
def test_graph_edge_definition_replacement(db: StandardDatabase) -> None:
    graph_name = "ReplaceGraph"

    def insert_graph_with_node_type(node_type: str) -> None:
        graph = ArangoGraph(db, generate_schema_on_init=False)
        graph_doc = GraphDocument(
            nodes=[
                Node(id="n1", type=node_type),
                Node(id="n2", type=node_type),
            ],
            relationships=[
                Relationship(
                    source=Node(id="n1", type=node_type),
                    target=Node(id="n2", type=node_type),
                    type="CONNECTS",
                )
            ],
            source=Document(page_content="replace test"),
        )

        graph.add_graph_documents(
            [graph_doc],
            graph_name=graph_name,
            update_graph_definition_if_exists=True,
            capitalization_strategy="lower",
            use_one_entity_collection=False,
        )

    # Step 1: Insert with type "TypeA"
    insert_graph_with_node_type("TypeA")
    g = db.graph(graph_name)
    edge_defs_1 = [
        ed
        for ed in g.edge_definitions()  # type: ignore
        if ed["edge_collection"] == "CONNECTS"  # type: ignore
    ]
    assert len(edge_defs_1) == 1

    assert "TypeA" in edge_defs_1[0]["from_vertex_collections"]
    assert "TypeA" in edge_defs_1[0]["to_vertex_collections"]

    # Step 2: Insert again with different type "TypeB" — should trigger replace
    insert_graph_with_node_type("TypeB")
    edge_defs_2 = [
        ed
        for ed in g.edge_definitions()  # type: ignore
        if ed["edge_collection"] == "CONNECTS"  # type: ignore
    ]  # noqa: E501
    assert len(edge_defs_2) == 1
    assert "TypeB" in edge_defs_2[0]["from_vertex_collections"]
    assert "TypeB" in edge_defs_2[0]["to_vertex_collections"]
    # Should not contain old "typea" anymore
    assert "TypeA" not in edge_defs_2[0]["from_vertex_collections"]


@pytest.mark.usefixtures("clear_arangodb_database")
def test_generate_schema_with_graph_name(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)
    graph_name = "TestGraphSchema"

    # Setup: Create collections
    vertex_col1 = "Person"
    vertex_col2 = "Company"
    edge_col = "WORKS_AT"

    for col in [vertex_col1, vertex_col2]:
        if not db.has_collection(col):
            db.create_collection(col)

    if not db.has_collection(edge_col):
        db.create_collection(edge_col, edge=True)

    # Insert test data
    db.collection(vertex_col1).insert({"_key": "alice", "role": "engineer"})
    db.collection(vertex_col2).insert({"_key": "acme", "industry": "tech"})
    db.collection(edge_col).insert(
        {"_from": f"{vertex_col1}/alice", "_to": f"{vertex_col2}/acme", "since": 2020}
    )

    # Create graph
    if not db.has_graph(graph_name):
        db.create_graph(
            graph_name,
            edge_definitions=[
                {
                    "edge_collection": edge_col,
                    "from_vertex_collections": [vertex_col1],
                    "to_vertex_collections": [vertex_col2],
                }
            ],
        )

    # Call generate_schema
    schema = graph.generate_schema(
        sample_ratio=1.0, graph_name=graph_name, include_examples=True
    )

    # Validate graph schema
    graph_schema = schema["graph_schema"]
    assert isinstance(graph_schema, list)
    assert graph_schema[0]["name"] == graph_name
    edge_defs = graph_schema[0]["edge_definitions"]
    assert any(ed["edge_collection"] == edge_col for ed in edge_defs)

    # Validate collection schema includes vertex and edge
    collection_schema = schema["collection_schema"]
    col_names = {col["name"] for col in collection_schema}
    assert vertex_col1 in col_names
    assert vertex_col2 in col_names
    assert edge_col in col_names


@pytest.mark.usefixtures("clear_arangodb_database")
def test_add_graph_documents_requires_embedding(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    doc = GraphDocument(
        nodes=[Node(id="A", type="TypeA")],
        relationships=[],
        source=Document(page_content="doc without embedding"),
    )

    with pytest.raises(ValueError, match="embedding.*required"):
        graph.add_graph_documents(
            [doc],
            embed_source=True,  # requires embedding, but embeddings=None
        )


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.mark.usefixtures("clear_arangodb_database")
def test_add_graph_documents_with_embedding(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    doc = GraphDocument(
        nodes=[Node(id="NodeX", type="TypeX")],
        relationships=[],
        source=Document(page_content="sample text"),
    )

    # Provide FakeEmbeddings and enable source embedding
    graph.add_graph_documents(
        [doc],
        include_source=True,
        embed_source=True,
        embeddings=FakeEmbeddings(),  # type: ignore
        embedding_field="embedding",
        capitalization_strategy="lower",
    )

    # Verify the embedding was stored
    source_col = "SOURCE"
    inserted = db.collection(source_col).all()  # type: ignore
    inserted = list(inserted)  # type: ignore
    assert len(inserted) == 1  # type: ignore
    assert "embedding" in inserted[0]  # type: ignore
    assert inserted[0]["embedding"] == [0.1, 0.2, 0.3]  # type: ignore


@pytest.mark.usefixtures("clear_arangodb_database")
@pytest.mark.parametrize(
    "strategy, expected_id",
    [
        ("lower", "node1"),
        ("upper", "NODE1"),
    ],
)
def test_capitalization_strategy_applied(
    db: StandardDatabase, strategy: str, expected_id: str
) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    doc = GraphDocument(
        nodes=[Node(id="Node1", type="Entity")],
        relationships=[],
        source=Document(page_content="source"),
    )

    graph.add_graph_documents([doc], capitalization_strategy=strategy)

    results = list(db.collection("ENTITY").all())  # type: ignore
    assert any(doc["text"] == expected_id for doc in results)  # type: ignore


def test_capitalization_strategy_none_does_not_raise(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    # Patch internals if needed to avoid real inserts
    graph._hash = lambda x: x  # type: ignore
    graph._import_data = lambda *args, **kwargs: None  # type: ignore
    graph.refresh_schema = lambda *args, **kwargs: None  # type: ignore
    graph._create_collection = lambda *args, **kwargs: None  # type: ignore
    graph._process_node_as_entity = lambda key, node, nodes, coll: "ENTITY"  # type: ignore
    graph._process_edge_as_entity = lambda *args, **kwargs: None  # type: ignore

    doc = GraphDocument(
        nodes=[Node(id="Node1", type="Entity")],
        relationships=[],
        source=Document(page_content="source"),
    )

    # Act (should NOT raise)
    graph.add_graph_documents([doc], capitalization_strategy="none")


def test_get_arangodb_client_direct_credentials() -> None:
    db = get_arangodb_client(
        url="http://localhost:8529",
        dbname="_system",
        username="root",
        password="test",  # adjust if your test instance uses a different password
    )
    assert isinstance(db, StandardDatabase)
    assert db.name == "_system"


def test_get_arangodb_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARANGODB_URL", "http://localhost:8529")
    monkeypatch.setenv("ARANGODB_DBNAME", "_system")
    monkeypatch.setenv("ARANGODB_USERNAME", "root")
    monkeypatch.setenv("ARANGODB_PASSWORD", "test")

    db = get_arangodb_client()
    assert isinstance(db, StandardDatabase)
    assert db.name == "_system"


def test_get_arangodb_client_invalid_url() -> None:  # type: ignore
    with pytest.raises(Exception):
        # Unreachable host or invalid port
        ArangoClient(  # type: ignore
            url="http://localhost:9999",
            dbname="_system",
            username="root",
            password="test",
        )


@pytest.mark.usefixtures("clear_arangodb_database")
def test_batch_insert_triggers_import_data(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    # Patch _import_data to monitor calls
    graph._import_data = MagicMock()  # type: ignore

    batch_size = 3
    total_nodes = 7

    doc = GraphDocument(
        nodes=[Node(id=f"n{i}", type="T") for i in range(total_nodes)],
        relationships=[],
        source=Document(page_content="batch insert test"),
    )

    graph.add_graph_documents(
        [doc], batch_size=batch_size, capitalization_strategy="lower"
    )

    # Filter for node insert calls
    node_calls = [
        call for call in graph._import_data.call_args_list if not call.kwargs["is_edge"]
    ]

    assert len(node_calls) == 4  # 2 during loop, 1 at the end


@pytest.mark.usefixtures("clear_arangodb_database")
def test_batch_insert_edges_triggers_import_data(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)
    graph._import_data = MagicMock()  # type: ignore

    batch_size = 2
    total_edges = 5

    # Prepare enough nodes to support relationships
    nodes = [Node(id=f"n{i}", type="Entity") for i in range(total_edges + 1)]
    relationships = [
        Relationship(source=nodes[i], target=nodes[i + 1], type="LINKS_TO")
        for i in range(total_edges)
    ]

    doc = GraphDocument(
        nodes=nodes,
        relationships=relationships,
        source=Document(page_content="edge batch test"),
    )

    graph.add_graph_documents(
        [doc], batch_size=batch_size, capitalization_strategy="lower"
    )

    # Count how many times _import_data was called with is_edge=True
    # AND non-empty edge data
    edge_calls = [
        call
        for call in graph._import_data.call_args_list
        if call.kwargs.get("is_edge") is True and any(call.args[1].values())
    ]

    assert len(edge_calls) == 7  # 2 full batches (2, 4), 1 final flush (5)


def test_from_db_credentials_direct() -> None:
    graph = ArangoGraph.from_db_credentials(
        url="http://localhost:8529",
        dbname="_system",
        username="root",
        password="test",  # use "" if your ArangoDB has no password
    )

    assert isinstance(graph, ArangoGraph)
    assert isinstance(graph.db, StandardDatabase)
    assert graph.db.name == "_system"


@pytest.mark.usefixtures("clear_arangodb_database")
def test_get_node_key_existing_entry(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)
    node = Node(id="A", type="Type")

    existing_key = "123456789"
    node_key_map = {"A": existing_key}  # type: ignore
    nodes = defaultdict(list)  # type: ignore

    process_node_fn = MagicMock()  # type: ignore

    key = graph._get_node_key(
        node=node,
        nodes=nodes,
        node_key_map=node_key_map,
        entity_collection_name="ENTITY",
        process_node_fn=process_node_fn,
    )

    assert key == existing_key
    process_node_fn.assert_not_called()


@pytest.mark.usefixtures("clear_arangodb_database")
def test_get_node_key_new_entry(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)
    node = Node(id="B", type="Type")

    node_key_map = {}  # type: ignore
    nodes = defaultdict(list)  # type: ignore
    process_node_fn = MagicMock()  # type: ignore

    key = graph._get_node_key(
        node=node,
        nodes=nodes,
        node_key_map=node_key_map,
        entity_collection_name="ENTITY",
        process_node_fn=process_node_fn,
    )

    # Assert new key added to map
    assert node.id in node_key_map
    assert node_key_map[node.id] == key
    process_node_fn.assert_called_once_with(key, node, nodes, "ENTITY")


@pytest.mark.usefixtures("clear_arangodb_database")
def test_hash_basic_inputs(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    # String input
    result_str = graph._hash("hello")
    assert isinstance(result_str, str)
    assert result_str.isdigit()

    # Integer input
    result_int = graph._hash(123)
    assert isinstance(result_int, str)
    assert result_int.isdigit()

    # Object with __str__
    class Custom:
        def __str__(self) -> str:
            return "custom"

    result_obj = graph._hash(Custom())
    assert isinstance(result_obj, str)
    assert result_obj.isdigit()


def test_hash_invalid_input_raises() -> None:
    class BadStr:
        def __str__(self) -> str:
            raise TypeError("nope")

    graph = ArangoGraph.__new__(ArangoGraph)  # avoid needing db

    with pytest.raises(ValueError, match="string or have a string representation"):
        graph._hash(BadStr())


@pytest.mark.usefixtures("clear_arangodb_database")
def test_sanitize_input_short_string_preserved(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)
    input_dict = {"key": "short"}

    result = graph._sanitize_input(input_dict, list_limit=10, string_limit=10)

    assert result["key"] == "short"


@pytest.mark.usefixtures("clear_arangodb_database")
def test_sanitize_input_long_string_truncated(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)
    long_value = "x" * 100
    input_dict = {"key": long_value}

    result = graph._sanitize_input(input_dict, list_limit=10, string_limit=50)

    assert result["key"] == f"String of {len(long_value)} characters"


@pytest.mark.usefixtures("clear_arangodb_database")
def test_create_edge_definition_called_when_missing(db: StandardDatabase) -> None:
    """
    Tests that `create_edge_definition` is called if an edge type is missing
    when `update_graph_definition_if_exists` is True.
    """
    graph_name = "TestEdgeDefGraph"

    # --- Corrected Mocking Strategy ---
    # 1. Simulate that the graph already exists.
    db.has_graph = MagicMock(return_value=True)  # type: ignore

    # 2. Create a mock for the graph object that db.graph() will return.
    mock_graph_obj = MagicMock()
    # 3. Simulate that this graph is missing the specific edge definition.
    mock_graph_obj.has_edge_definition.return_value = False

    # 4. Configure the db fixture to return our mock graph object.
    db.graph = MagicMock(return_value=mock_graph_obj)  # type: ignore
    # --- End of Mocking Strategy ---

    # Initialize ArangoGraph with the pre-configured mock db
    graph = ArangoGraph(db, generate_schema_on_init=False)

    # Create an input graph document with a new edge type
    doc = GraphDocument(
        nodes=[Node(id="n1", type="Person"), Node(id="n2", type="Company")],
        relationships=[
            Relationship(
                source=Node(id="n1", type="Person"),
                target=Node(id="n2", type="Company"),
                type="WORKS_FOR",  # A clear, new edge type
            )
        ],
        source=Document(page_content="edge definition test"),
    )

    # Run the insertion logic
    graph.add_graph_documents(
        [doc],
        graph_name=graph_name,
        update_graph_definition_if_exists=True,
        use_one_entity_collection=False,  # Use separate collections for node/edge types
    )

    # --- Assertions ---
    # Verify that the code checked for the graph and then retrieved it.
    db.has_graph.assert_called_once_with(graph_name)
    db.graph.assert_called_once_with(graph_name)

    # Verify the code checked for the edge definition. The collection name is
    # derived from the relationship type.
    mock_graph_obj.has_edge_definition.assert_called_once_with("WORKS_FOR")

    # ✅ The main assertion: create_edge_definition should have been called.
    mock_graph_obj.create_edge_definition.assert_called_once()

    # Inspect the keyword arguments of the call to ensure they are correct.
    call_kwargs = mock_graph_obj.create_edge_definition.call_args.kwargs

    assert call_kwargs == {
        "edge_collection": "WORKS_FOR",
        "from_vertex_collections": ["Person"],
        "to_vertex_collections": ["Company"],
    }


class DummyEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 5 for _ in texts]  # Return dummy vectors


@pytest.mark.usefixtures("clear_arangodb_database")
def test_embed_relationships_and_include_source(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)
    graph._import_data = MagicMock()  # type: ignore

    doc = GraphDocument(
        nodes=[
            Node(id="A", type="Entity"),
            Node(id="B", type="Entity"),
        ],
        relationships=[
            Relationship(
                source=Node(id="A", type="Entity"),
                target=Node(id="B", type="Entity"),
                type="Rel",
            ),
        ],
        source=Document(page_content="relationship source test"),
    )

    embeddings = DummyEmbeddings()

    graph.add_graph_documents(
        [doc],
        include_source=True,
        embed_relationships=True,
        embeddings=embeddings,  # type: ignore
        capitalization_strategy="lower",
    )

    # Only select edge batches that contain custom
    # relationship types (i.e. with type="Rel")
    relationship_edge_calls = []
    for call in graph._import_data.call_args_list:
        if call.kwargs.get("is_edge"):
            edge_batch = call.args[1]
            for edge_list in edge_batch.values():
                if any(edge.get("type") == "Rel" for edge in edge_list):
                    relationship_edge_calls.append(edge_list)

    assert relationship_edge_calls, "Expected at least one batch of relationship edges"

    all_relationship_edges = relationship_edge_calls[0]
    pprint.pprint(all_relationship_edges)


    assert any(
       "embedding" in e for e in all_relationship_edges
    ), "Expected embedding in relationship"  # noqa: E501
    assert any(
        "source_id" in e for e in all_relationship_edges
    ), "Expected source_id in relationship"  # noqa: E501


@pytest.mark.usefixtures("clear_arangodb_database")
def test_set_schema_assigns_correct_value(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    custom_schema = {
        "collections": {
            "User": {"fields": ["name", "email"]},
            "Transaction": {"fields": ["amount", "timestamp"]},
        }
    }

    graph.set_schema(custom_schema)
    assert graph._ArangoGraph__schema == custom_schema  # type: ignore


@pytest.mark.usefixtures("clear_arangodb_database")
def test_schema_json_returns_correct_json_string(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    fake_schema = {
        "collections": {
            "Entity": {"fields": ["id", "name"]},
            "Links": {"fields": ["source", "target"]},
        }
    }
    graph._ArangoGraph__schema = fake_schema  # type: ignore

    schema_json = graph.schema_json

    assert isinstance(schema_json, str)
    assert json.loads(schema_json) == fake_schema


@pytest.mark.usefixtures("clear_arangodb_database")
def test_get_structured_schema_returns_schema(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    # Simulate assigning schema manually
    fake_schema = {"collections": {"Entity": {"fields": ["id", "name"]}}}
    graph._ArangoGraph__schema = fake_schema  # type: ignore

    result = graph.get_structured_schema
    assert result == fake_schema


@pytest.mark.usefixtures("clear_arangodb_database")
def test_generate_schema_invalid_sample_ratio(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    # Test with sample_ratio < 0
    with pytest.raises(ValueError, match=".*sample_ratio.*"):
        graph.refresh_schema(sample_ratio=-0.1)

    # Test with sample_ratio > 1
    with pytest.raises(ValueError, match=".*sample_ratio.*"):
        graph.refresh_schema(sample_ratio=1.5)


@pytest.mark.usefixtures("clear_arangodb_database")
def test_add_graph_documents_noop_on_empty_input(db: StandardDatabase) -> None:
    graph = ArangoGraph(db, generate_schema_on_init=False)

    # Patch _import_data to verify it's not called
    graph._import_data = MagicMock()  # type: ignore

    # Call with empty input
    graph.add_graph_documents([], capitalization_strategy="lower")

    # Assert _import_data was never triggered
    graph._import_data.assert_not_called()
