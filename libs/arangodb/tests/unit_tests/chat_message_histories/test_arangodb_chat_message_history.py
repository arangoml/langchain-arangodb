"""Unit tests for ArangoDB Chat Message History."""

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from arango.database import StandardDatabase
from arango.exceptions import CollectionCreateError, IndexCreateError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, messages_to_dict

from langchain_arangodb.chat_message_histories.arangodb import ArangoChatMessageHistory


@pytest.fixture
def mock_arango_db_chat() -> MagicMock:
    """Fixture for a mocked ArangoDB StandardDatabase specifically for chat history."""
    mock_db = MagicMock(spec=StandardDatabase)
    mock_db.has_collection.return_value = True
    mock_collection = MagicMock()
    # Mock index list to simulate index check
    mock_collection.indexes.return_value = [{"fields": ["session_id"]}] # Assume index exists
    mock_collection.add_persistent_index.return_value = {}
    mock_collection.insert.return_value = {}
    mock_db.collection.return_value = mock_collection
    mock_db.create_collection.return_value = mock_collection

    mock_aql = MagicMock()
    mock_aql.execute.return_value = MagicMock() # Mock AQL cursor
    mock_db.aql = mock_aql
    return mock_db

# --- Initialization Tests ---

def test_init_requires_session_id(mock_arango_db_chat: MagicMock) -> None:
    """Test initializing without session_id raises ValueError."""
    with pytest.raises(ValueError, match="session_id parameter is provided"):
        ArangoChatMessageHistory(session_id=None, db=mock_arango_db_chat) # type: ignore
    with pytest.raises(ValueError, match="session_id parameter is provided"):
        ArangoChatMessageHistory(session_id="", db=mock_arango_db_chat)

def test_init_collection_creation(mock_arango_db_chat: MagicMock) -> None:
    """Test that the collection is created if it doesn't exist."""
    mock_arango_db_chat.has_collection.return_value = False
    history = ArangoChatMessageHistory(session_id="test1", db=mock_arango_db_chat, collection_name="NewChat")
    mock_arango_db_chat.create_collection.assert_called_once_with("NewChat")
    assert history._collection_name == "NewChat"

def test_init_index_creation(mock_arango_db_chat: MagicMock) -> None:
    """Test that the index is created if it doesn't exist."""
    mock_collection = mock_arango_db_chat.collection.return_value
    mock_collection.indexes.return_value = [] # Simulate index NOT existing
    history = ArangoChatMessageHistory(session_id="test2", db=mock_arango_db_chat)
    mock_collection.add_persistent_index.assert_called_once_with(["session_id"], unique=True)

def test_init_collection_creation_error(mock_arango_db_chat: MagicMock) -> None:
    """Test error handling during collection creation."""
    mock_arango_db_chat.has_collection.return_value = False
    mock_arango_db_chat.create_collection.side_effect = CollectionCreateError("Failed to create", http_exception=None)
    with pytest.raises(CollectionCreateError):
        ArangoChatMessageHistory(session_id="test3", db=mock_arango_db_chat)

def test_init_index_creation_error(mock_arango_db_chat: MagicMock) -> None:
    """Test error handling during index creation."""
    mock_collection = mock_arango_db_chat.collection.return_value
    mock_collection.indexes.return_value = []
    mock_collection.add_persistent_index.side_effect = IndexCreateError("Failed to create index", http_exception=None)
    with pytest.raises(IndexCreateError):
        ArangoChatMessageHistory(session_id="test4", db=mock_arango_db_chat)

# --- Message Property Tests ---

def test_messages_setter_error(mock_arango_db_chat: MagicMock) -> None:
    """Test that assigning to messages raises NotImplementedError."""
    history = ArangoChatMessageHistory(session_id="test_session", db=mock_arango_db_chat)
    with pytest.raises(NotImplementedError, match="Direct assignment to 'messages' is not allowed"):
        history.messages = [] # type: ignore

def test_messages_retrieval(mock_arango_db_chat: MagicMock) -> None:
    """Test retrieving messages."""
    history = ArangoChatMessageHistory(session_id="sid1", db=mock_arango_db_chat, collection_name="History")

    # Mock AQL result
    mock_cursor = MagicMock()
    mock_db_results = [
        {"role": "human", "content": "Hello"},
        {"role": "ai", "content": "Hi there"},
    ]
    mock_cursor.__iter__.return_value = iter(mock_db_results)
    mock_arango_db_chat.aql.execute.return_value = mock_cursor

    messages = history.messages

    # Check AQL execution
    mock_arango_db_chat.aql.execute.assert_called_once()
    args, kwargs = mock_arango_db_chat.aql.execute.call_args
    expected_query = """
            FOR doc IN @@col
                FILTER doc.session_id == @session_id
                SORT doc.time DESC
                RETURN UNSET(doc, ["session_id", "_id", "_rev"])
        """
    # Use strip/replace to make query comparison less fragile
    assert args[0].strip().replace("\n", "") == expected_query.strip().replace("\n", "")
    assert kwargs["bind_vars"] == {"@col": "History", "session_id": "sid1"}

    # Check message conversion
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hello"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Hi there"

# --- Add Message Tests ---

def test_add_message(mock_arango_db_chat: MagicMock) -> None:
    """Test adding a single message."""
    history = ArangoChatMessageHistory(session_id="sid2", db=mock_arango_db_chat, collection_name="TestChat")
    mock_collection = mock_arango_db_chat.collection.return_value

    human_message = HumanMessage(content="What is the weather?")
    history.add_message(human_message)

    mock_collection.insert.assert_called_once_with({
        "role": "human",
        "content": "What is the weather?",
        "session_id": "sid2",
    })

    mock_collection.insert.reset_mock()

    ai_message = AIMessage(content="It is sunny.")
    history.add_message(ai_message)

    mock_collection.insert.assert_called_once_with({
        "role": "ai",
        "content": "It is sunny.",
        "session_id": "sid2",
    })

# --- Clear Tests ---

def test_clear(mock_arango_db_chat: MagicMock) -> None:
    """Test clearing the history for a session."""
    history = ArangoChatMessageHistory(session_id="sid3", db=mock_arango_db_chat, collection_name="ClearMe")

    history.clear()

    # Check AQL execution
    mock_arango_db_chat.aql.execute.assert_called_once()
    args, kwargs = mock_arango_db_chat.aql.execute.call_args
    expected_query = """
            FOR doc IN @@col
                FILTER doc.session_id == @session_id
                REMOVE doc IN @@col
        """
    assert args[0].strip().replace("\n", "") == expected_query.strip().replace("\n", "")
    assert kwargs["bind_vars"] == {"@col": "ClearMe", "session_id": "sid3"}

# Note: No driver closing test needed as ArangoChatMessageHistory uses an external db object.
# Note: Windowing functionality mentioned in __init__ (TODO) is not tested as it's not implemented. 