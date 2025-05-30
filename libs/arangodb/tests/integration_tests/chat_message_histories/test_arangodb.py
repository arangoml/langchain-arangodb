import os

import pytest
from arango import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import ArangoError
from langchain_core.messages import AIMessage, HumanMessage

from langchain_arangodb.chat_message_histories.arangodb import ArangoChatMessageHistory
from langchain_arangodb.graphs.arangodb_graph import ArangoGraph
from tests.integration_tests.utils import ArangoCredentials


@pytest.mark.usefixtures("clear_arangodb_database")
def test_add_messages(db: StandardDatabase) -> None:
    """Basic testing: adding messages to the ArangoDBChatMessageHistory."""
    message_store = ArangoChatMessageHistory("123", db=db)
    message_store.clear()
    assert len(message_store.messages) == 0
    message_store.add_user_message("Hello! Language Chain!")
    message_store.add_ai_message("Hi Guys!")

    # create another message store to check if the messages are stored correctly
    message_store_another = ArangoChatMessageHistory("456", db=db)
    message_store_another.clear()
    assert len(message_store_another.messages) == 0
    message_store_another.add_user_message("Hello! Bot!")
    message_store_another.add_ai_message("Hi there!")
    message_store_another.add_user_message("How's this pr going?")

    # Now check if the messages are stored in the database correctly
    assert len(message_store.messages) == 2
    assert isinstance(message_store.messages[0], HumanMessage)
    assert isinstance(message_store.messages[1], AIMessage)
    assert message_store.messages[0].content == "Hello! Language Chain!"
    assert message_store.messages[1].content == "Hi Guys!"

    assert len(message_store_another.messages) == 3
    assert isinstance(message_store_another.messages[0], HumanMessage)
    assert isinstance(message_store_another.messages[1], AIMessage)
    assert isinstance(message_store_another.messages[2], HumanMessage)
    assert message_store_another.messages[0].content == "Hello! Bot!"
    assert message_store_another.messages[1].content == "Hi there!"
    assert message_store_another.messages[2].content == "How's this pr going?"

    # Now clear the first history
    message_store.clear()
    assert len(message_store.messages) == 0
    assert len(message_store_another.messages) == 3
    message_store_another.clear()
    assert len(message_store.messages) == 0
    assert len(message_store_another.messages) == 0


@pytest.mark.usefixtures("clear_arangodb_database")
def test_add_messages_graph_object(arangodb_credentials: ArangoCredentials) -> None:
    """Basic testing: Passing driver through graph object."""
    graph = ArangoGraph.from_db_credentials(
        url=arangodb_credentials["url"],
        username=arangodb_credentials["username"],
        password=arangodb_credentials["password"],
    )

    # rewrite env for testing
    old_username = os.environ.get("ARANGO_USERNAME", "root")
    os.environ["ARANGO_USERNAME"] = "foo"

    message_store = ArangoChatMessageHistory("23334", db=graph.db)
    message_store.clear()
    assert len(message_store.messages) == 0
    message_store.add_user_message("Hello! Language Chain!")
    message_store.add_ai_message("Hi Guys!")
    # Now check if the messages are stored in the database correctly
    assert len(message_store.messages) == 2

    # Restore original environment
    os.environ["ARANGO_USERNAME"] = old_username


def test_invalid_credentials(arangodb_credentials: ArangoCredentials) -> None:
    """Test initializing with invalid credentials raises an authentication error."""
    with pytest.raises(ArangoError) as exc_info:
        client = ArangoClient(arangodb_credentials["url"])
        db = client.db(username="invalid_username", password="invalid_password")
        # Try to perform a database operation to trigger an authentication error
        db.collections()

    # Check for any authentication-related error message
    error_msg = str(exc_info.value)
    # Just check for "error" which should be in any auth error
    assert "not authorized" in error_msg


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_message_history_clear_messages(
    db: StandardDatabase,
) -> None:
    """Test adding multiple messages at once to ArangoChatMessageHistory."""
    # Specify a custom collection name that includes the session_id
    collection_name = "chat_history_123"
    message_history = ArangoChatMessageHistory(
        session_id="123", db=db, collection_name=collection_name
    )
    message_history.add_messages(
        [
            HumanMessage(content="You are a helpful assistant."),
            AIMessage(content="Hello"),
        ]
    )
    assert len(message_history.messages) == 2
    assert isinstance(message_history.messages[0], HumanMessage)
    assert isinstance(message_history.messages[1], AIMessage)
    assert message_history.messages[0].content == "You are a helpful assistant."
    assert message_history.messages[1].content == "Hello"

    message_history.clear()
    assert len(message_history.messages) == 0

    # Verify all messages are removed but collection still exists
    assert db.has_collection(message_history._collection_name)
    assert message_history._collection_name == collection_name


@pytest.mark.usefixtures("clear_arangodb_database")
def test_arangodb_message_history_clear_session_collection(
    db: StandardDatabase,
) -> None:
    """Test clearing messages and removing the collection for a session."""
    # Create a test collection specific to the session
    session_id = "456"
    collection_name = f"chat_history_{session_id}"

    if not db.has_collection(collection_name):
        db.create_collection(collection_name)

    message_history = ArangoChatMessageHistory(
        session_id=session_id, db=db, collection_name=collection_name
    )

    message_history.add_messages(
        [
            HumanMessage(content="You are a helpful assistant."),
            AIMessage(content="Hello"),
        ]
    )
    assert len(message_history.messages) == 2

    # Clear messages
    message_history.clear()
    assert len(message_history.messages) == 0

    # The collection should still exist after clearing messages
    assert db.has_collection(collection_name)

    # Delete the collection (equivalent to delete_session_node in Neo4j)
    db.delete_collection(collection_name)
    assert not db.has_collection(collection_name)
