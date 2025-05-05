"""Integration tests for ArangoDB Chat Message History."""

from typing import List

import pytest
from arango.database import StandardDatabase
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_arangodb.chat_message_histories.arangodb import ArangoChatMessageHistory

TEST_COLLECTION_NAME = "test_chat_history"

@pytest.fixture
def history_store(db: StandardDatabase) -> ArangoChatMessageHistory:
    """Yields a ArangoChatMessageHistory instance for a specific session_id."""
    # Use a fixed session_id for predictability in tests
    session_id = "test_session_integration"
    store = ArangoChatMessageHistory(
        session_id=session_id,
        db=db,
        collection_name=TEST_COLLECTION_NAME
    )
    # Ensure clean state for this specific session before test
    store.clear()
    # Verify clear worked (or collection is empty for this session)
    assert len(store.messages) == 0
    yield store
    # Optional: Clean up after test if clear_arangodb_database isn't used or sufficient
    # store.clear()

@pytest.mark.usefixtures("clear_arangodb_database")
def test_add_and_retrieve_messages(history_store: ArangoChatMessageHistory) -> None:
    """Test adding messages and retrieving them."""
    history_store.add_message(HumanMessage(content="Hello there!"))
    history_store.add_message(AIMessage(content="General Kenobi!"))

    messages = history_store.messages
    assert len(messages) == 2
    # ArangoDB implementation retrieves messages sorted by insertion time (implicitly)
    # Check order and content
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hello there!"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "General Kenobi!"

@pytest.mark.usefixtures("clear_arangodb_database")
def test_add_messages_multiple_sessions(db: StandardDatabase) -> None:
    """Test isolation between different session IDs."""
    session_id_1 = "session_abc"
    session_id_2 = "session_xyz"

    history1 = ArangoChatMessageHistory(
        session_id=session_id_1,
        db=db,
        collection_name=TEST_COLLECTION_NAME
    )
    history2 = ArangoChatMessageHistory(
        session_id=session_id_2,
        db=db,
        collection_name=TEST_COLLECTION_NAME
    )

    # Clear both to be sure
    history1.clear()
    history2.clear()
    assert len(history1.messages) == 0
    assert len(history2.messages) == 0

    # Add messages to session 1
    history1.add_message(HumanMessage(content="Message 1 for session 1"))
    history1.add_message(AIMessage(content="Message 2 for session 1"))

    # Add messages to session 2
    history2.add_message(HumanMessage(content="Message 1 for session 2"))

    # Verify retrieval
    messages1 = history1.messages
    messages2 = history2.messages

    assert len(messages1) == 2
    assert messages1[0].content == "Message 1 for session 1"
    assert messages1[1].content == "Message 2 for session 1"

    assert len(messages2) == 1
    assert messages2[0].content == "Message 1 for session 2"

@pytest.mark.usefixtures("clear_arangodb_database")
def test_clear_messages(history_store: ArangoChatMessageHistory) -> None:
    """Test clearing messages for a session."""
    # Add messages first
    history_store.add_message(HumanMessage(content="To be cleared"))
    history_store.add_message(AIMessage(content="Also cleared"))

    assert len(history_store.messages) == 2

    # Clear the history
    history_store.clear()

    # Verify messages are gone
    assert len(history_store.messages) == 0

@pytest.mark.usefixtures("clear_arangodb_database")
def test_clear_messages_isolation(db: StandardDatabase) -> None:
    """Test that clearing one session doesn't affect another."""
    session_id_1 = "session_clear_1"
    session_id_2 = "session_clear_2"

    history1 = ArangoChatMessageHistory(
        session_id=session_id_1,
        db=db,
        collection_name=TEST_COLLECTION_NAME
    )
    history2 = ArangoChatMessageHistory(
        session_id=session_id_2,
        db=db,
        collection_name=TEST_COLLECTION_NAME
    )

    # Add messages to both
    history1.add_message(HumanMessage(content="Session 1 msg 1"))
    history1.add_message(AIMessage(content="Session 1 msg 2"))
    history2.add_message(HumanMessage(content="Session 2 msg 1"))

    assert len(history1.messages) == 2
    assert len(history2.messages) == 1

    # Clear session 1
    history1.clear()

    # Verify session 1 is empty, session 2 is not
    assert len(history1.messages) == 0
    assert len(history2.messages) == 1
    assert history2.messages[0].content == "Session 2 msg 1"

    # Clear session 2 as well for cleanup
    history2.clear()
    assert len(history2.messages) == 0

# Note: Connection error tests (invalid URL/creds) are harder to integration test reliably
# as they depend on the fixture providing a valid connection.
# Note: Neo4j specific test `clear(delete_session_node=True)` is not applicable.
# Note: Sorting by time in retrieval relies on implicit insertion order or a missing `time` field. 