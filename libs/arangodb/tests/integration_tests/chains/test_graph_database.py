"""Test Graph Database Chain."""

import pytest
from arango.database import StandardDatabase

from langchain_arangodb.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_arangodb.graphs.arangodb_graph import ArangoGraph
from tests.llms.fake_llm import FakeLLM


@pytest.mark.usefixtures("clear_arangodb_database")
def test_aql_generating_run(db: StandardDatabase) -> None:
    """Test that AQL statement is correctly generated and executed."""
    graph = ArangoGraph(db)

    assert graph.schema == {
        "collection_schema": [],
        "graph_schema": [],
    }

    # Create two nodes and a relationship
    graph.db.create_collection("Actor")
    graph.db.create_collection("Movie")
    graph.db.create_collection("ActedIn", edge=True)

    graph.db.collection("Actor").insert({"_key": "BruceWillis", "name": "Bruce Willis"})
    graph.db.collection("Movie").insert(
        {"_key": "PulpFiction", "title": "Pulp Fiction"}
    )
    graph.db.collection("ActedIn").insert(
        {"_from": "Actor/BruceWillis", "_to": "Movie/PulpFiction"}
    )

    # Refresh schema information
    graph.refresh_schema()

    assert len(graph.schema["collection_schema"]) == 3
    assert len(graph.schema["graph_schema"]) == 0

    query = """```
        FOR m IN Movie
            FILTER m.title == 'Pulp Fiction'
            FOR actor IN 1..1 INBOUND m ActedIn
                RETURN actor.name
    ```"""

    llm = FakeLLM(
        queries={"query": query, "response": "Bruce Willis"}, sequential_responses=True
    )

    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        max_aql_generation_attempts=1,
    )

    output = chain.invoke("Who starred in Pulp Fiction?")  # type: ignore
    assert output["result"] == "Bruce Willis"
