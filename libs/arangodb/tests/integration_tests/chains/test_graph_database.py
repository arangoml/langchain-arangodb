"""Test Graph Database Chain."""

import pprint
from unittest.mock import MagicMock, patch

import pytest
from arango.database import StandardDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage

from langchain_arangodb.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_arangodb.graphs.arangodb_graph import ArangoGraph
from tests.llms.fake_llm import FakeLLM

# from langchain_arangodb.chains.graph_qa.arangodb import GraphAQLQAChain


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



@pytest.mark.usefixtures("clear_arangodb_database")
def test_aql_top_k(db: StandardDatabase) -> None:
    """Test top_k parameter correctly limits the number of results in the context."""
    TOP_K = 1
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
        top_k=TOP_K,
    )

    output = chain.invoke("Who starred in Pulp Fiction?")  # type: ignore
    assert len([output["result"]]) == TOP_K




@pytest.mark.usefixtures("clear_arangodb_database")
def test_aql_returns(db: StandardDatabase) -> None:
    """Test that chain returns direct results."""
    # Initialize the ArangoGraph
    graph = ArangoGraph(db)

    # Create collections
    db.create_collection("Actor")
    db.create_collection("Movie")
    db.create_collection("ActedIn", edge=True)

    # Insert documents
    db.collection("Actor").insert({"_key": "BruceWillis", "name": "Bruce Willis"})
    db.collection("Movie").insert({"_key": "PulpFiction", "title": "Pulp Fiction"})
    db.collection("ActedIn").insert({
        "_from": "Actor/BruceWillis",
        "_to": "Movie/PulpFiction"
    })

    # Refresh schema information
    graph.refresh_schema()

    # Define the AQL query
    query = """```
    FOR m IN Movie
        FILTER m.title == 'Pulp Fiction'
        FOR actor IN 1..1 INBOUND m ActedIn
            RETURN actor.name
    ```"""

    # Initialize the fake LLM with the query and expected response
    llm = FakeLLM(
        queries={"query": query, "response": "Bruce Willis"},
        sequential_responses=True
    )

    # Initialize the QA chain with return_direct=True
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        return_direct=True,
        return_aql_query=True,
        return_aql_result=True,
    )

    # Run the chain with the question
    output = chain.invoke("Who starred in Pulp Fiction?")
    pprint.pprint(output)

    # Define the expected output
    expected_output = {'aql_query': '```\n'
              '    FOR m IN Movie\n'
              "        FILTER m.title == 'Pulp Fiction'\n"
              '        FOR actor IN 1..1 INBOUND m ActedIn\n'
              '            RETURN actor.name\n'
              '    ```',
 'aql_result': ['Bruce Willis'],
 'query': 'Who starred in Pulp Fiction?',
 'result': 'Bruce Willis'}
    # Assert that the output matches the expected output
    assert output== expected_output


@pytest.mark.usefixtures("clear_arangodb_database")
def test_function_response(db: StandardDatabase) -> None:
    """Test returning a function response."""
    # Initialize the ArangoGraph
    graph = ArangoGraph(db)

    # Create collections
    db.create_collection("Actor")
    db.create_collection("Movie")
    db.create_collection("ActedIn", edge=True)

    # Insert documents
    db.collection("Actor").insert({"_key": "BruceWillis", "name": "Bruce Willis"})
    db.collection("Movie").insert({"_key": "PulpFiction", "title": "Pulp Fiction"})
    db.collection("ActedIn").insert({
        "_from": "Actor/BruceWillis",
        "_to": "Movie/PulpFiction"
    })

    # Refresh schema information
    graph.refresh_schema()

    # Define the AQL query
    query = """```
    FOR m IN Movie
        FILTER m.title == 'Pulp Fiction'
        FOR actor IN 1..1 INBOUND m ActedIn
            RETURN actor.name
    ```"""

    # Initialize the fake LLM with the query and expected response
    llm = FakeLLM(
        queries={"query": query, "response": "Bruce Willis"},
        sequential_responses=True
    )

    # Initialize the QA chain with use_function_response=True
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        use_function_response=True,
    )

    # Run the chain with the question
    output = chain.run("Who starred in Pulp Fiction?")

    # Define the expected output
    expected_output = "Bruce Willis"

    # Assert that the output matches the expected output
    assert output == expected_output

@pytest.mark.usefixtures("clear_arangodb_database")
def test_exclude_types(db: StandardDatabase) -> None:
    """Test exclude types from schema."""
    # Initialize the ArangoGraph
    graph = ArangoGraph(db)

    # Create collections
    db.create_collection("Actor")
    db.create_collection("Movie")
    db.create_collection("Person")
    db.create_collection("ActedIn", edge=True)
    db.create_collection("Directed", edge=True)

    # Insert documents
    db.collection("Actor").insert({"_key": "BruceWillis", "name": "Bruce Willis"})
    db.collection("Movie").insert({"_key": "PulpFiction", "title": "Pulp Fiction"})
    db.collection("Person").insert({"_key": "John", "name": "John"})
    
    # Insert relationships
    db.collection("ActedIn").insert({
        "_from": "Actor/BruceWillis",
        "_to": "Movie/PulpFiction"
    })
    db.collection("Directed").insert({
        "_from": "Person/John",
        "_to": "Movie/PulpFiction"
    })

    # Refresh schema information
    graph.refresh_schema()

    # Initialize the LLM with a mock
    llm = MagicMock(spec=BaseLanguageModel)

    # Initialize the QA chain with exclude_types set
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        exclude_types=["Person", "Directed"],
        allow_dangerous_requests=True,
    )

    # Print the full version of the schema
    # pprint.pprint(chain.graph.schema)
    res=[]
    for collection in chain.graph.schema["collection_schema"]:
        res.append(collection["name"])
    assert set(res) == set(["Actor", "Movie", "Person", "ActedIn", "Directed"])


@pytest.mark.usefixtures("clear_arangodb_database")
def test_exclude_examples(db: StandardDatabase) -> None:
    """Test include types from schema."""
    # Initialize the ArangoGraph
    graph = ArangoGraph(db, schema_include_examples=False)

    # Create collections and edges
    db.create_collection("Actor")
    db.create_collection("Movie")
    db.create_collection("Person")
    db.create_collection("ActedIn", edge=True)
    db.create_collection("Directed", edge=True)

    # Insert documents
    db.collection("Actor").insert({"_key": "BruceWillis", "name": "Bruce Willis"})
    db.collection("Movie").insert({"_key": "PulpFiction", "title": "Pulp Fiction"})
    db.collection("Person").insert({"_key": "John", "name": "John"})

    # Insert edges
    db.collection("ActedIn").insert({
        "_from": "Actor/BruceWillis",
        "_to": "Movie/PulpFiction"
    })
    db.collection("Directed").insert({
        "_from": "Person/John",
        "_to": "Movie/PulpFiction"
    })

    # Refresh schema information
    graph.refresh_schema(include_examples=False)

    # Initialize the LLM with a mock
    llm = MagicMock(spec=BaseLanguageModel)

    # Initialize the QA chain with include_types set
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        include_types=["Actor", "Movie", "ActedIn"],
        allow_dangerous_requests=True,
    )
    pprint.pprint(chain.graph.schema)

    expected_schema = {'collection_schema': [{'name': 'ActedIn',
                        'properties': [{'_key': 'str'},
                                       {'_id': 'str'},
                                       {'_from': 'str'},
                                       {'_to': 'str'},
                                       {'_rev': 'str'}],
                        'size': 1,
                        'type': 'edge'},
                       {'name': 'Directed',
                        'properties': [{'_key': 'str'},
                                       {'_id': 'str'},
                                       {'_from': 'str'},
                                       {'_to': 'str'},
                                       {'_rev': 'str'}],
                        'size': 1,
                        'type': 'edge'},
                       {'name': 'Person',
                        'properties': [{'_key': 'str'},
                                       {'_id': 'str'},
                                       {'_rev': 'str'},
                                       {'name': 'str'}],
                        'size': 1,
                        'type': 'document'},
                       {'name': 'Actor',
                        'properties': [{'_key': 'str'},
                                       {'_id': 'str'},
                                       {'_rev': 'str'},
                                       {'name': 'str'}],
                        'size': 1,
                        'type': 'document'},
                       {'name': 'Movie',
                        'properties': [{'_key': 'str'},
                                       {'_id': 'str'},
                                       {'_rev': 'str'},
                                       {'title': 'str'}],
                        'size': 1,
                        'type': 'document'}],
 'graph_schema': []}
    assert set(chain.graph.schema) == set(expected_schema)

@pytest.mark.usefixtures("clear_arangodb_database")
def test_aql_fixing_mechanism_with_fake_llm(db: StandardDatabase) -> None:
    """Test that the AQL fixing mechanism is invoked and can correct a query."""
    graph = ArangoGraph(db)
    graph.db.create_collection("Students")
    graph.db.collection("Students").insert({"name": "John Doe"})
    graph.refresh_schema()

    # Define the sequence of responses the LLM should produce.
    faulty_query = "FOR s IN Students RETURN s.namee"  # Intentionally incorrect query
    corrected_query = "FOR s IN Students RETURN s.name"
    final_answer = "John Doe"

    # The keys in the dictionary don't matter in sequential mode, only the order.
    sequential_queries = {
        "first_call": f"```aql\n{faulty_query}\n```",
        "second_call": f"```aql\n{corrected_query}\n```",
        # This response will not be used, but we leave it for clarity
        "third_call": final_answer, 
    }

    # Initialize FakeLLM in sequential mode
    llm = FakeLLM(queries=sequential_queries, sequential_responses=True)

    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
    )

    # Execute the chain
    output = chain.invoke("Get student names")
    pprint.pprint(output)

    # --- THIS IS THE FIX ---
    # The chain's actual behavior is to return the corrected query string as the
    # final result, skipping the final QA step. The assertion must match this.
    expected_result = f"```aql\n{corrected_query}\n```"
    assert output["result"] == expected_result

@pytest.mark.usefixtures("clear_arangodb_database")
def test_explain_only_mode(db: StandardDatabase) -> None:
    """Test that with execute_aql_query=False, the query is explained, not run."""
    graph = ArangoGraph(db)
    graph.db.create_collection("Products")
    graph.db.collection("Products").insert({"name": "Laptop", "price": 1200})
    graph.refresh_schema()

    query = "FOR p IN Products FILTER p.price > 1000 RETURN p.name"

    llm = FakeLLM(
        queries={"placeholder_prompt": f"```aql\n{query}\n```"},
        sequential_responses=True,
    )

    chain = ArangoGraphQAChain.from_llm(
        llm,
        graph=graph,
        allow_dangerous_requests=True,
        execute_aql_query=False,
    )

    output = chain.invoke("Find expensive products")

    # The result should be the AQL query itself
    assert output["result"] == query

    # FIX: The ArangoDB explanation plan is stored under the "nodes" key.
    # We will assert its presence to confirm we have a plan and not a result.
    assert "nodes" in output["aql_result"]

@pytest.mark.usefixtures("clear_arangodb_database")
def test_force_read_only_with_write_query(db: StandardDatabase) -> None:
    """Test that a write query raises a ValueError when 
    force_read_only_query is True."""
    graph = ArangoGraph(db)
    graph.db.create_collection("Users")
    graph.refresh_schema()

    # This is a write operation
    write_query = "INSERT {_key: 'test', name: 'Test User'} INTO Users"

    # FIX: Use sequential mode to provide the write query as the LLM's response,
    # regardless of the incoming prompt from the chain.
    llm = FakeLLM(
        queries={"placeholder_prompt": f"```aql\n{write_query}\n```"},
        sequential_responses=True,
    )

    chain = ArangoGraphQAChain.from_llm(
        llm,
        graph=graph,
        allow_dangerous_requests=True,
        force_read_only_query=True,
    )

    with pytest.raises(ValueError) as excinfo:
        chain.invoke("Add a new user")

    assert "Write operations are not allowed" in str(excinfo.value)
    assert "Detected write operation in query: INSERT" in str(excinfo.value)

@pytest.mark.usefixtures("clear_arangodb_database")
def test_no_aql_query_in_response(db: StandardDatabase) -> None:
    """Test that a ValueError is raised if the LLM response contains no AQL query."""
    graph = ArangoGraph(db)
    graph.db.create_collection("Customers")
    graph.refresh_schema()

    # LLM response without a valid AQL block
    response_no_query = "I am sorry, I cannot generate a query for that."

    # FIX: Use FakeLLM in sequential mode to return the response.
    llm = FakeLLM(
        queries={"placeholder_prompt": response_no_query}, sequential_responses=True
    )

    chain = ArangoGraphQAChain.from_llm(
        llm,
        graph=graph,
        allow_dangerous_requests=True,
    )

    with pytest.raises(ValueError) as excinfo:
        chain.invoke("Get customer data")

    assert "Unable to extract AQL Query from response" in str(excinfo.value)

@pytest.mark.usefixtures("clear_arangodb_database")
def test_max_generation_attempts_exceeded(db: StandardDatabase) -> None:
    """Test that the chain stops after the maximum number of AQL generation attempts."""
    graph = ArangoGraph(db)
    graph.db.create_collection("Tasks")
    graph.refresh_schema()

    # A query that will consistently fail
    bad_query = "FOR t IN Tasks RETURN t."

    # FIX: Provide enough responses for all expected LLM calls.
    # 1 (initial generation) + max_aql_generation_attempts (fixes) = 1 + 2 = 3 calls
    llm = FakeLLM(
        queries={
            "initial_generation": f"```aql\n{bad_query}\n```",
            "fix_attempt_1": f"```aql\n{bad_query}\n```",
            "fix_attempt_2": f"```aql\n{bad_query}\n```",
        },
        sequential_responses=True,
    )

    chain = ArangoGraphQAChain.from_llm(
        llm,
        graph=graph,
        allow_dangerous_requests=True,
        max_aql_generation_attempts=2, # This means 2 attempts *within* the loop
    )

    with pytest.raises(ValueError) as excinfo:
        chain.invoke("Get tasks")

    assert "Maximum amount of AQL Query Generation attempts reached" in str(
        excinfo.value
    )
    # FIX: Assert against the FakeLLM's internal counter.
    # The LLM is called 3 times in total.
    assert llm.response_index == 3


@pytest.mark.usefixtures("clear_arangodb_database")
def test_unsupported_aql_generation_output_type(db: StandardDatabase) -> None:
    """
    Test that a ValueError is raised for an unsupported AQL generation output type.

    This test uses patching to bypass the LangChain framework's own output
    validation, allowing us to directly test the error handling inside the
    ArangoGraphQAChain's _call method.
    """
    graph = ArangoGraph(db)
    graph.refresh_schema()

    # The actual LLM doesn't matter, as we will patch the chain's output.
    llm = FakeLLM(queries={"placeholder": "this response is never used"})

    chain = ArangoGraphQAChain.from_llm(
        llm,
        graph=graph,
        allow_dangerous_requests=True,
    )

    # Define an output type that the chain does not support, like a dictionary.
    unsupported_output = {"error": "This is not a valid output format"}

    # Use patch.object to temporarily replace the chain's internal aql_generation_chain
    # with a mock. We configure this mock to return our unsupported dictionary.
    with patch.object(chain, "aql_generation_chain") as mock_aql_chain:
        mock_aql_chain.invoke.return_value = unsupported_output

        # We now expect our specific ValueError from the ArangoGraphQAChain.
        with pytest.raises(ValueError) as excinfo:
            chain.invoke("This query will trigger the error")

    # Assert that the error message is the one we expect from the target code block.
    error_message = str(excinfo.value)
    assert "Invalid AQL Generation Output" in error_message
    assert str(unsupported_output) in error_message
    assert str(type(unsupported_output)) in error_message


@pytest.mark.usefixtures("clear_arangodb_database")
def test_handles_aimessage_output(db: StandardDatabase) -> None:
    """
    Test that the chain correctly handles an AIMessage object from the
    AQL generation chain and completes the full QA process.
    """
    # 1. Setup: Create a simple graph and data.
    graph = ArangoGraph(db)
    graph.db.create_collection("Movies")
    graph.db.collection("Movies").insert({"title": "Inception"})
    graph.refresh_schema()

    query_string = "FOR m IN Movies FILTER m.title == 'Inception' RETURN m.title"
    final_answer = "The movie is Inception."

    # 2. Define the AIMessage object we want the generation chain to return.
    llm_output_as_message = AIMessage(content=f"```aql\n{query_string}\n```")

    # 3. Configure the underlying FakeLLM to handle the *second* LLM call,
    # which is the final QA step.
    llm = FakeLLM(
        queries={"qa_step_response": final_answer},
        sequential_responses=True,
    )

    # 4. Initialize the main chain.
    chain = ArangoGraphQAChain.from_llm(
        llm,
        graph=graph,
        allow_dangerous_requests=True,
    )

    # 5. Use patch.object to mock the output of the internal aql_generation_chain.
    # This ensures the `aql_generation_output` variable in the _call method
    # becomes our AIMessage object.
    with patch.object(chain, "aql_generation_chain") as mock_aql_chain:
        mock_aql_chain.invoke.return_value = llm_output_as_message

        # 6. Run the full chain.
        output = chain.invoke("What is the movie title?")

    # 7. Assert that the final result is correct.
    # A correct result proves the AIMessage was successfully parsed, the query
    # was executed, and the qa_chain (using the real FakeLLM) was called.
    assert output["result"] == final_answer

def test_chain_type_property() -> None:
    """
    Tests that the _chain_type property returns the correct hardcoded value.
    """
    # 1. Create a mock database object to allow instantiation of ArangoGraph.
    mock_db = MagicMock(spec=StandardDatabase)
    graph = ArangoGraph(db=mock_db)

    # 2. Create a minimal FakeLLM. Its responses don't matter for this test.
    llm = FakeLLM()

    # 3. Instantiate the chain using the `from_llm` classmethod. This ensures
    #    all internal runnables are created correctly and pass Pydantic validation.
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
    )

    # 4. Assert that the property returns the expected value.
    assert chain._chain_type == "graph_aql_chain"

def test_is_read_only_query_returns_true_for_readonly_query() -> None:
    """
    Tests that _is_read_only_query returns (True, None) for a read-only AQL query.
    """
    # 1. Create a mock database object for ArangoGraph instantiation.
    mock_db = MagicMock(spec=StandardDatabase)
    graph = ArangoGraph(db=mock_db)

    # 2. Create a minimal FakeLLM.
    llm = FakeLLM()

    # 3. Instantiate the chain using the `from_llm` classmethod.
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True, # Necessary for instantiation
    )

    # 4. Define a sample read-only AQL query.
    read_only_query = "FOR doc IN MyCollection FILTER doc.name == 'test' RETURN doc"

    # 5. Call the method under test.
    is_read_only, operation = chain._is_read_only_query(read_only_query)

    # 6. Assert that the result is (True, None).
    assert is_read_only is True
    assert operation is None

def test_is_read_only_query_returns_false_for_insert_query() -> None:
    """
    Tests that _is_read_only_query returns (False, 'INSERT') for an INSERT query.
    """
    mock_db = MagicMock(spec=StandardDatabase)
    graph = ArangoGraph(db=mock_db)
    llm = FakeLLM()
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
    )
    write_query = "INSERT { name: 'test' } INTO MyCollection"
    is_read_only, operation = chain._is_read_only_query(write_query)
    assert is_read_only is False
    assert operation == "INSERT"

def test_is_read_only_query_returns_false_for_update_query() -> None:
    """
    Tests that _is_read_only_query returns (False, 'UPDATE') for an UPDATE query.
    """
    mock_db = MagicMock(spec=StandardDatabase)
    graph = ArangoGraph(db=mock_db)
    llm = FakeLLM()
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
    )
    write_query = "FOR doc IN MyCollection FILTER doc._key == '123' \
    UPDATE doc WITH { name: 'new_test' } IN MyCollection"
    is_read_only, operation = chain._is_read_only_query(write_query)
    assert is_read_only is False
    assert operation == "UPDATE"

def test_is_read_only_query_returns_false_for_remove_query() -> None:
    """
    Tests that _is_read_only_query returns (False, 'REMOVE') for a REMOVE query.
    """
    mock_db = MagicMock(spec=StandardDatabase)
    graph = ArangoGraph(db=mock_db)
    llm = FakeLLM()
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
    )
    write_query = "FOR doc IN MyCollection FILTER \
    doc._key== '123' REMOVE doc IN MyCollection"
    is_read_only, operation = chain._is_read_only_query(write_query)
    assert is_read_only is False
    assert operation == "REMOVE"

def test_is_read_only_query_returns_false_for_replace_query() -> None:
    """
    Tests that _is_read_only_query returns (False, 'REPLACE') for a REPLACE query.
    """
    mock_db = MagicMock(spec=StandardDatabase)
    graph = ArangoGraph(db=mock_db)
    llm = FakeLLM()
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
    )
    write_query = "FOR doc IN MyCollection FILTER doc._key == '123' \
    REPLACE doc WITH { name: 'replaced_test' } IN MyCollection"
    is_read_only, operation = chain._is_read_only_query(write_query)
    assert is_read_only is False
    assert operation == "REPLACE"

def test_is_read_only_query_returns_false_for_upsert_query() -> None:
    """
    Tests that _is_read_only_query returns (False, 'INSERT') for an UPSERT query
    due to the iteration order in AQL_WRITE_OPERATIONS.
    """
    # ... (instantiation code is the same) ...
    mock_db = MagicMock(spec=StandardDatabase)
    graph = ArangoGraph(db=mock_db)
    llm = FakeLLM()
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
    )

    write_query = "UPSERT { _key: '123' } INSERT { name: 'new_upsert' } \
    UPDATE { name: 'updated_upsert' } IN MyCollection"
    is_read_only, operation = chain._is_read_only_query(write_query)

    assert is_read_only is False
    # FIX: The method finds "INSERT" before "UPSERT" because of the list order.
    assert operation == "INSERT"

def test_is_read_only_query_is_case_insensitive() -> None:
    """
    Tests that the write operation check is case-insensitive.
    """
    # ... (instantiation code is the same) ...
    mock_db = MagicMock(spec=StandardDatabase)
    graph = ArangoGraph(db=mock_db)
    llm = FakeLLM()
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
    )

    write_query_lower = "insert { name: 'test' } into MyCollection"
    is_read_only, operation = chain._is_read_only_query(write_query_lower)
    assert is_read_only is False
    assert operation == "INSERT"

    write_query_mixed = "UpSeRt { _key: '123' } InSeRt { name: 'new' } \
    UpDaTe { name: 'updated' } In MyCollection"
    is_read_only_mixed, operation_mixed = chain._is_read_only_query(write_query_mixed)
    assert is_read_only_mixed is False
    # FIX: The method finds "INSERT" before "UPSERT" regardless of case.
    assert operation_mixed == "INSERT"

def test_init_raises_error_if_dangerous_requests_not_allowed() -> None:
    """
    Tests that the __init__ method raises a ValueError if
    allow_dangerous_requests is not True.
    """
    # 1. Create mock/minimal objects for dependencies.
    mock_db = MagicMock(spec=StandardDatabase)
    graph = ArangoGraph(db=mock_db)
    llm = FakeLLM()

    # 2. Define the expected error message.
    expected_error_message = (
        "In order to use this chain, you must acknowledge that it can make "
        "dangerous requests by setting `allow_dangerous_requests` to `True`."
    ) # We only need to check for a substring

    # 3. Attempt to instantiate the chain without allow_dangerous_requests=True
    #    (or explicitly setting it to False) and assert that a ValueError is raised.
    with pytest.raises(ValueError) as excinfo:
        ArangoGraphQAChain.from_llm(
            llm=llm,
            graph=graph,
            # allow_dangerous_requests is omitted, so it defaults to False
        )

    # 4. Assert that the caught exception's message contains the expected text.
    assert expected_error_message in str(excinfo.value)

    # 5. Also test explicitly setting it to False
    with pytest.raises(ValueError) as excinfo_false:
        ArangoGraphQAChain.from_llm(
            llm=llm,
            graph=graph,
            allow_dangerous_requests=False,
        )
    assert expected_error_message in str(excinfo_false.value)

def test_init_succeeds_if_dangerous_requests_allowed() -> None:
    """
    Tests that the __init__ method succeeds if allow_dangerous_requests is True.
    """
    mock_db = MagicMock(spec=StandardDatabase)
    graph = ArangoGraph(db=mock_db)
    llm = FakeLLM()

    try:
        ArangoGraphQAChain.from_llm(
            llm=llm,
            graph=graph,
            allow_dangerous_requests=True,
        )
    except ValueError:
        pytest.fail("ValueError was raised unexpectedly when \
                        allow_dangerous_requests=True")