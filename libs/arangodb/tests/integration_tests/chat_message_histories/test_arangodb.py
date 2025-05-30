import pytest
from arango.database import StandardDatabase
from langchain_core.messages import AIMessage, HumanMessage

from langchain_arangodb.chat_message_histories.arangodb import ArangoChatMessageHistory


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
