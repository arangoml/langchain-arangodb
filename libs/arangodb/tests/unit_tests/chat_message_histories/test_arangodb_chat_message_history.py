from unittest.mock import MagicMock

import pytest
from arango.database import StandardDatabase

from langchain_arangodb.chat_message_histories.arangodb import ArangoChatMessageHistory


def test_init_without_session_id() -> None:
    """Test initializing without session_id raises ValueError."""
    mock_db = MagicMock(spec=StandardDatabase)
    with pytest.raises(ValueError) as exc_info:
        ArangoChatMessageHistory(None, db=mock_db)  # type: ignore[arg-type]
    assert "Please ensure that the session_id parameter is provided" in str(
        exc_info.value
    )


def test_messages_setter() -> None:
    """Test that assigning to messages raises NotImplementedError."""
    mock_db = MagicMock(spec=StandardDatabase)
    mock_collection = MagicMock()
    mock_db.collection.return_value = mock_collection
    mock_db.has_collection.return_value = True

    message_store = ArangoChatMessageHistory(
        session_id="test_session",
        db=mock_db,
    )

    with pytest.raises(NotImplementedError) as exc_info:
        message_store.messages = []
    assert "Direct assignment to 'messages' is not allowed." in str(exc_info.value)


def test_collection_creation() -> None:
    """Test that collection is created if it doesn't exist."""
    mock_db = MagicMock(spec=StandardDatabase)
    mock_collection = MagicMock()
    mock_db.collection.return_value = mock_collection

    # First test when collection doesn't exist
    mock_db.has_collection.return_value = False

    ArangoChatMessageHistory(
        session_id="test_session",
        db=mock_db,
        collection_name="TestCollection",
    )

    # Verify collection creation was called
    mock_db.create_collection.assert_called_once_with("TestCollection")
    mock_db.collection.assert_called_once_with("TestCollection")

    # Now test when collection exists
    mock_db.reset_mock()
    mock_db.has_collection.return_value = True

    ArangoChatMessageHistory(
        session_id="test_session",
        db=mock_db,
        collection_name="TestCollection",
    )

    # Verify collection creation was not called
    mock_db.create_collection.assert_not_called()
    mock_db.collection.assert_called_once_with("TestCollection")


def test_index_creation() -> None:
    """Test that index on session_id is created if it doesn't exist."""
    mock_db = MagicMock(spec=StandardDatabase)
    mock_collection = MagicMock()
    mock_db.collection.return_value = mock_collection
    mock_db.has_collection.return_value = True

    # First test when index doesn't exist
    mock_collection.indexes.return_value = []

    ArangoChatMessageHistory(
        session_id="test_session",
        db=mock_db,
    )

    # Verify index creation was called
    mock_collection.add_persistent_index.assert_called_once_with(
        ["session_id"], unique=False
    )

    # Now test when index exists
    mock_db.reset_mock()
    mock_collection.reset_mock()
    mock_collection.indexes.return_value = [{"fields": ["session_id"]}]

    ArangoChatMessageHistory(
        session_id="test_session",
        db=mock_db,
    )

    # Verify index creation was not called
    mock_collection.add_persistent_index.assert_not_called()


def test_add_message() -> None:
    """Test adding a message to the collection."""
    mock_db = MagicMock(spec=StandardDatabase)
    mock_collection = MagicMock()
    mock_db.collection.return_value = mock_collection
    mock_db.has_collection.return_value = True
    mock_collection.indexes.return_value = [{"fields": ["session_id"]}]

    message_store = ArangoChatMessageHistory(
        session_id="test_session",
        db=mock_db,
    )

    # Create a mock message
    mock_message = MagicMock()
    mock_message.type = "human"
    mock_message.content = "Hello, world!"

    # Add the message
    message_store.add_message(mock_message)

    # Verify the message was added to the collection
    mock_db.collection.assert_called_with("ChatHistory")
    mock_collection.insert.assert_called_once_with(
        {
            "role": "human",
            "content": "Hello, world!",
            "session_id": "test_session",
        }
    )


def test_clear() -> None:
    """Test clearing messages from the collection."""
    mock_db = MagicMock(spec=StandardDatabase)
    mock_collection = MagicMock()
    mock_aql = MagicMock()
    mock_db.collection.return_value = mock_collection
    mock_db.aql = mock_aql
    mock_db.has_collection.return_value = True
    mock_collection.indexes.return_value = [{"fields": ["session_id"]}]

    message_store = ArangoChatMessageHistory(
        session_id="test_session",
        db=mock_db,
    )

    # Clear the messages
    message_store.clear()

    # Verify the AQL query was executed
    mock_aql.execute.assert_called_once()
    # Check that the bind variables are correct
    call_args = mock_aql.execute.call_args[1]
    assert call_args["bind_vars"]["@col"] == "ChatHistory"
    assert call_args["bind_vars"]["session_id"] == "test_session"


def test_messages_property() -> None:
    """Test retrieving messages from the collection."""
    mock_db = MagicMock(spec=StandardDatabase)
    mock_collection = MagicMock()
    mock_aql = MagicMock()
    mock_cursor = MagicMock()
    mock_db.collection.return_value = mock_collection
    mock_db.aql = mock_aql
    mock_db.has_collection.return_value = True
    mock_collection.indexes.return_value = [{"fields": ["session_id"]}]
    mock_aql.execute.return_value = mock_cursor

    # Mock cursor to return two messages
    mock_cursor.__iter__.return_value = [
        {"role": "human", "content": "Hello"},
        {"role": "ai", "content": "Hi there"},
    ]

    message_store = ArangoChatMessageHistory(
        session_id="test_session",
        db=mock_db,
    )

    # Get the messages
    messages = message_store.messages

    # Verify the AQL query was executed
    mock_aql.execute.assert_called_once()
    # Check that the bind variables are correct
    call_args = mock_aql.execute.call_args[1]
    assert call_args["bind_vars"]["@col"] == "ChatHistory"
    assert call_args["bind_vars"]["session_id"] == "test_session"

    # Check that we got the right number of messages
    assert len(messages) == 2
    assert messages[0].type == "human"
    assert messages[0].content == "Hello"
    assert messages[1].type == "ai"
    assert messages[1].content == "Hi there"
