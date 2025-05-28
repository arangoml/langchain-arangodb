Chat Message Histories
====================

LangChain ArangoDB provides chat message history implementations that allow you to store and retrieve chat messages using ArangoDB.

ArangoDBChatMessageHistory
-------------------------

The main chat message history implementation that uses ArangoDB for storing and retrieving chat messages.

.. code-block:: python

    from langchain_arangodb.chat_message_histories import ArangoDBChatMessageHistory
    from langchain.schema import HumanMessage, AIMessage

    # Initialize the chat message history
    history = ArangoDBChatMessageHistory(
        arango_url="http://localhost:8529",
        username="root",
        password="",
        database="langchain",
        collection_name="chat_history",
        session_id="user123"
    )

    # Add messages
    history.add_user_message("Hello!")
    history.add_ai_message("Hi there!")

    # Get all messages
    messages = history.messages

Features
--------

- Persistent storage of chat messages
- Session-based message organization
- Support for different message types
- Efficient message retrieval
- Integration with LangChain's chat interfaces

Configuration Options
--------------------

The chat message history can be configured with various options:

- ``arango_url``: URL of the ArangoDB instance
- ``username``: ArangoDB username
- ``password``: ArangoDB password
- ``database``: Database name
- ``collection_name``: Collection name for storing messages
- ``session_id``: Unique identifier for the chat session
- ``ttl``: Time-to-live for messages (optional) 