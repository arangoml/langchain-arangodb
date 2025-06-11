Chat Message Histories
=====================

LangChain ArangoDB provides persistent chat message history storage using ArangoDB's document database capabilities. The ``ArangoChatMessageHistory`` class enables you to store, retrieve, and manage conversation history across sessions.

Overview
--------

The ``ArangoChatMessageHistory`` class integrates with LangChain's chat memory system to provide:

- **Persistent Storage**: Chat messages are stored permanently in ArangoDB
- **Session Management**: Organize conversations by session ID
- **Automatic Indexing**: Efficient retrieval with automatic session-based indexing
- **Message Ordering**: Messages are retrieved in chronological order
- **Memory Integration**: Works seamlessly with LangChain's memory components

Quick Start
-----------

.. code-block:: python

    from arango import ArangoClient
    from langchain_arangodb.chat_message_histories import ArangoChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage

    # Connect to ArangoDB
    client = ArangoClient("http://localhost:8529")
    db = client.db("langchain_demo", username="root", password="openSesame")

    # Initialize chat history for a specific session
    chat_history = ArangoChatMessageHistory(
        session_id="user_123",
        db=db,
        collection_name="chat_sessions"
    )

    # Add messages to the conversation
    chat_history.add_message(HumanMessage(content="Hello, how are you?"))
    chat_history.add_message(AIMessage(content="I'm doing well, thank you! How can I help you today?"))

    # Retrieve all messages in the session
    messages = chat_history.messages
    for message in messages:
        print(f"{message.type}: {message.content}")

Configuration
-------------

Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~

.. py:class:: ArangoChatMessageHistory(session_id, db, collection_name="ChatHistory", window=3)

   :param session_id: Unique identifier for the chat session (string or int)
   :param db: ArangoDB database instance from python-arango
   :param collection_name: Name of the collection to store messages (default: "ChatHistory")
   :param window: Message window size for future windowing feature (default: 3)

The class automatically:

- Creates the collection if it doesn't exist
- Creates a persistent index on ``session_id`` for efficient queries
- Handles message serialization and deserialization

Core Methods
------------

Adding Messages
~~~~~~~~~~~~~~

.. code-block:: python

    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    # Add different types of messages
    chat_history.add_message(HumanMessage(content="What is machine learning?"))
    chat_history.add_message(AIMessage(content="Machine learning is a subset of AI..."))
    chat_history.add_message(SystemMessage(content="System: Conversation started"))

    # Messages are automatically timestamped and stored with session context

Retrieving Messages
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get all messages for the current session
    all_messages = chat_history.messages

    # Messages are returned in chronological order (most recent first in database, 
    # but converted to proper order for LangChain)
    for i, message in enumerate(all_messages):
        print(f"Message {i+1}: [{message.type}] {message.content}")

Clearing History
~~~~~~~~~~~~~~~

.. code-block:: python

    # Clear all messages for the current session
    chat_history.clear()

    # Verify the session is cleared
    print(f"Messages after clear: {len(chat_history.messages)}")

Integration with LangChain Memory
---------------------------------

Conversation Buffer Memory
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from langchain.memory import ConversationBufferMemory
    from langchain_openai import ChatOpenAI
    from langchain_arangodb.chat_message_histories import ArangoChatMessageHistory

    # Create chat history
    chat_history = ArangoChatMessageHistory(
        session_id="conversation_1",
        db=db,
        collection_name="conversations"
    )

    # Create memory with persistent storage
    memory = ConversationBufferMemory(
        chat_memory=chat_history,
        return_messages=True,
        memory_key="chat_history"
    )

    # Use with any LangChain chain
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # The memory will automatically persist conversations
    conversation_input = {"input": "Tell me about Python programming"}

Conversation Summary Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from langchain.memory import ConversationSummaryMemory

    # Create summary memory with persistent storage
    summary_memory = ConversationSummaryMemory(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        chat_memory=chat_history,
        return_messages=True
    )

    # Conversation summaries are also persisted

Integration with Chains
-----------------------

QA Chain with Memory
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from langchain_arangodb.chains import ArangoGraphQAChain
    from langchain_arangodb.graphs import ArangoGraph

    # Set up graph and chat history
    graph = ArangoGraph(database=db)
    chat_history = ArangoChatMessageHistory(
        session_id="qa_session_1",
        db=db,
        collection_name="qa_conversations"
    )

    # Create memory
    memory = ConversationBufferMemory(
        chat_memory=chat_history,
        return_messages=True
    )

    # Create QA chain with persistent memory
    qa_chain = ArangoGraphQAChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        graph=graph,
        memory=memory,
        verbose=True
    )

    # Conversations are automatically persisted
    response1 = qa_chain.run("What entities are in our knowledge graph?")
    response2 = qa_chain.run("Tell me more about the first one you mentioned")

Conversation Chain
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from langchain.chains import ConversationChain

    # Create a simple conversation chain with persistent memory
    conversation = ConversationChain(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        memory=ConversationBufferMemory(
            chat_memory=ArangoChatMessageHistory(
                session_id="simple_chat",
                db=db
            ),
            return_messages=True
        ),
        verbose=True
    )

    # Each interaction is persisted
    response1 = conversation.predict(input="Hi, I'm interested in learning about databases")
    response2 = conversation.predict(input="What makes ArangoDB special?")
    response3 = conversation.predict(input="Can you elaborate on the multi-model aspect?")

Advanced Usage
--------------

Multiple Sessions
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Manage different conversation sessions
    user_sessions = {}

    def get_chat_history(user_id: str) -> ArangoChatMessageHistory:
        if user_id not in user_sessions:
            user_sessions[user_id] = ArangoChatMessageHistory(
                session_id=f"user_{user_id}",
                db=db,
                collection_name="user_conversations"
            )
        return user_sessions[user_id]

    # Use for different users
    alice_history = get_chat_history("alice")
    bob_history = get_chat_history("bob")

    # Each user maintains separate conversation history
    alice_history.add_message(HumanMessage(content="Hello from Alice"))
    bob_history.add_message(HumanMessage(content="Hello from Bob"))

Custom Collection Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Use different collections for different purposes
    support_history = ArangoChatMessageHistory(
        session_id="support_ticket_123",
        db=db,
        collection_name="customer_support"
    )

    training_history = ArangoChatMessageHistory(
        session_id="training_session_1",
        db=db,
        collection_name="ai_training_conversations"
    )

    # Each collection can have different retention policies or indexes

Session Analytics
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Query conversation statistics directly from ArangoDB
    def get_session_stats(db, collection_name: str, session_id: str) -> dict:
        query = """
            FOR doc IN @@collection
                FILTER doc.session_id == @session_id
                COLLECT WITH COUNT INTO length
                RETURN {
                    message_count: length,
                    session_id: @session_id
                }
        """
        
        bind_vars = {
            "@collection": collection_name,
            "session_id": session_id
        }
        
        result = list(db.aql.execute(query, bind_vars=bind_vars))
        return result[0] if result else {"message_count": 0, "session_id": session_id}

    # Get conversation statistics
    stats = get_session_stats(db, "chat_sessions", "user_123")
    print(f"Session user_123 has {stats['message_count']} messages")

Data Structure
--------------

Storage Format
~~~~~~~~~~~~~

Messages are stored in ArangoDB with the following structure:

.. code-block:: json

    {
        "_key": "auto_generated_key",
        "_id": "collection_name/auto_generated_key",
        "_rev": "revision_id",
        "session_id": "user_123",
        "role": "human",
        "content": "Hello, how are you?",
        "time": "2024-01-01T12:00:00Z"
    }

**Field Descriptions:**

- ``session_id``: Groups messages by conversation session
- ``role``: Message type (human, ai, system, etc.)
- ``content``: The actual message content
- ``time``: Timestamp for message ordering (automatically added by ArangoDB)

Indexing Strategy
~~~~~~~~~~~~~~~

The class automatically creates a persistent index on ``session_id`` to ensure efficient retrieval:

.. code-block:: aql

    // Automatic index creation
    CREATE INDEX session_idx ON ChatHistory (session_id) OPTIONS {type: "persistent", unique: false}

This index enables fast filtering of messages by session while maintaining good performance even with large message volumes.

Best Practices
--------------

Session ID Management
~~~~~~~~~~~~~~~~~~~

1. **Use descriptive session IDs**: Include user context or conversation type
2. **Avoid special characters**: Stick to alphanumeric characters and underscores
3. **Include timestamps for analytics**: Consider formats like ``user_123_2024_01_01``

.. code-block:: python

    # Good session ID patterns
    session_id = f"user_{user_id}_{datetime.now().strftime('%Y_%m_%d')}"
    session_id = f"support_ticket_{ticket_id}"
    session_id = f"training_{model_version}_{session_counter}"

Memory Management
~~~~~~~~~~~~~~~

1. **Choose appropriate memory types** based on conversation length
2. **Implement session cleanup** for privacy or storage management
3. **Monitor collection size** and implement archiving if needed

.. code-block:: python

    # Cleanup old sessions
    def cleanup_old_sessions(db, collection_name: str, days_old: int = 30):
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        query = """
            FOR doc IN @@collection
                FILTER doc.time < @cutoff_date
                REMOVE doc IN @@collection
        """
        
        bind_vars = {
            "@collection": collection_name,
            "cutoff_date": cutoff_date.isoformat()
        }
        
        db.aql.execute(query, bind_vars=bind_vars)

Error Handling
~~~~~~~~~~~~~

.. code-block:: python

    from arango.exceptions import ArangoError

    try:
        chat_history = ArangoChatMessageHistory(
            session_id="test_session",
            db=db,
            collection_name="chat_test"
        )
        
        chat_history.add_message(HumanMessage(content="Test message"))
        messages = chat_history.messages
        
    except ValueError as e:
        print(f"Invalid session ID: {e}")
    except ArangoError as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

Performance Considerations
-------------------------

1. **Session ID indexing**: Automatic indexing ensures O(log n) lookup performance
2. **Message ordering**: Uses ArangoDB's built-in sorting capabilities
3. **Batch operations**: Consider bulk operations for high-volume scenarios
4. **Collection sizing**: Monitor and archive old conversations as needed

Example: Complete Chat Application
---------------------------------

.. code-block:: python

    from arango import ArangoClient
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    from langchain_arangodb.chat_message_histories import ArangoChatMessageHistory

    class ChatApplication:
        def __init__(self, db_url: str, username: str, password: str):
            # Initialize ArangoDB connection
            self.client = ArangoClient(db_url)
            self.db = self.client.db("chat_app", username=username, password=password)
            
            # Initialize LLM
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            
            # Session storage
            self.sessions = {}
        
        def get_conversation(self, session_id: str) -> ConversationChain:
            """Get or create a conversation for a session."""
            if session_id not in self.sessions:
                # Create persistent chat history
                chat_history = ArangoChatMessageHistory(
                    session_id=session_id,
                    db=self.db,
                    collection_name="app_conversations"
                )
                
                # Create memory with chat history
                memory = ConversationBufferMemory(
                    chat_memory=chat_history,
                    return_messages=True
                )
                
                # Create conversation chain
                conversation = ConversationChain(
                    llm=self.llm,
                    memory=memory,
                    verbose=True
                )
                
                self.sessions[session_id] = conversation
            
            return self.sessions[session_id]
        
        def chat(self, session_id: str, message: str) -> str:
            """Send a message and get a response."""
            conversation = self.get_conversation(session_id)
            return conversation.predict(input=message)
        
        def get_history(self, session_id: str) -> list:
            """Get conversation history for a session."""
            chat_history = ArangoChatMessageHistory(
                session_id=session_id,
                db=self.db,
                collection_name="app_conversations"
            )
            return chat_history.messages
        
        def clear_session(self, session_id: str):
            """Clear a conversation session."""
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            chat_history = ArangoChatMessageHistory(
                session_id=session_id,
                db=self.db,
                collection_name="app_conversations"
            )
            chat_history.clear()

    # Usage example
    app = ChatApplication("http://localhost:8529", "root", "openSesame")

    # Start conversations with different users
    response1 = app.chat("user_alice", "Hello, I need help with Python programming")
    response2 = app.chat("user_bob", "What's the weather like?")
    response3 = app.chat("user_alice", "Can you explain list comprehensions?")

    # Get conversation history
    alice_history = app.get_history("user_alice")
    print(f"Alice has {len(alice_history)} messages in her conversation")

    # Clear a session when done
    app.clear_session("user_bob")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~

**ValueError: Please ensure that the session_id parameter is provided**
   - Ensure session_id is not None, empty string, or 0
   - Use descriptive, non-empty session identifiers

**Database connection errors**
   - Verify ArangoDB is running and accessible
   - Check connection credentials and database permissions
   - Ensure the database exists or the user has create permissions

**Index creation failures**
   - Verify the user has index creation permissions
   - Check if the collection already has conflicting indexes
   - Ensure adequate disk space for index creation

**Message retrieval issues**
   - Verify session_id matches exactly (case-sensitive)
   - Check if messages exist in the collection using ArangoDB web interface
   - Ensure proper message format in the database
