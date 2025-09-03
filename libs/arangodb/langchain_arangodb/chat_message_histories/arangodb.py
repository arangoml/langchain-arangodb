import time
from typing import Any, List, Optional, Union

from arango.database import StandardDatabase
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict


class ArangoChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in an ArangoDB database.

    This class provides persistent storage for chat message histories using ArangoDB
    as the backend. It supports session-based message storage with automatic
    collection creation and indexing.

    :param session_id: Unique identifier for the chat session.
    :type session_id: Union[str, int]
    :param db: ArangoDB database instance for storing chat messages.
    :type db: arango.database.StandardDatabase
    :param collection_name: Name of the ArangoDB collection to store messages.
        Defaults to "ChatHistory".
    :type collection_name: str
    :param window: Maximum number of messages to keep in memory (currently unused).
        Defaults to 3.
    :type window: int
    :param args: Additional positional arguments passed to BaseChatMessageHistory.
    :type args: Any
    :param kwargs: Additional keyword arguments passed to BaseChatMessageHistory.
    :type kwargs: Any

    .. code-block:: python

        from arango import ArangoClient
        from langchain_arangodb.chat_message_histories import ArangoChatMessageHistory

        # Connect to ArangoDB
        client = ArangoClient("http://localhost:8529")
        db = client.db("test", username="root", password="openSesame")

        # Create chat message history
        history = ArangoChatMessageHistory(
            session_id="user_123",
            db=db,
            collection_name="chat_sessions"
        )

        # Add messages
        history.add_user_message("Hello! How are you?")
        history.add_ai_message("I'm doing well, thank you!")

        # Add QA message
        history.add_qa_message(
            user_input="Who is the first character?",
            aql_query="FOR doc IN Characters LIMIT 1 RETURN doc",
            result="The first character is Arya Stark."
        )

        # Retrieve messages
        messages = history.messages
        print(f"Found {len(messages)} messages")

        # Retrieve messages by role
        human_messages = history.get_messages(role="human")
        ai_messages = history.get_messages(role="ai")
        qa_messages = history.get_messages(role="qa")

        # Clear session
        history.clear()
    """

    def __init__(
        self,
        session_id: Union[str, int],
        db: StandardDatabase,
        collection_name: str = "ChatHistory",
        window: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Make sure session id is not null
        if not session_id:
            raise ValueError("Please ensure that the session_id parameter is provided")

        self._session_id = str(session_id)
        self._db = db
        self._collection_name = collection_name
        self._window = window  # TODO: Use this

        if not self._db.has_collection(collection_name):
            self._db.create_collection(collection_name)

        self._collection = self._db.collection(self._collection_name)

        has_index = False
        for index in self._collection.indexes():  # type: ignore
            if "session_id" in index["fields"]:
                has_index = True
                break

        if not has_index:
            self._collection.add_persistent_index(["session_id"], unique=False)

        super().__init__(*args, **kwargs)

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from ArangoDB.

        Retrieves all messages for the current session from the ArangoDB collection,
        sorted by timestamp in descending order (most recent first).

        :return: List of chat messages for the current session,
            sorted in reverse chronological order (most recent first).
        :rtype: List[BaseMessage]

        .. code-block:: python

            # Get all messages for the session
            messages = history.messages
            for msg in messages:
                print(f"{msg.type}: {msg.content}")

            # Check if session has any messages
            if history.messages:
                print(f"Session has {len(history.messages)} messages")
            else:
                print("No messages in this session")
        """
        query = """
            FOR doc IN @@col
                FILTER doc.session_id == @session_id
                SORT doc.time DESC
                RETURN UNSET(doc, ["session_id", "_id", "_rev"])
        """

        bind_vars = {"@col": self._collection_name, "session_id": self._session_id}

        cursor = self._db.aql.execute(query, bind_vars=bind_vars)  # type: ignore

        messages = [
            {"data": {"content": res["content"]}, "type": res["role"]}
            for res in cursor  # type: ignore
        ]

        return messages_from_dict(messages)

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        raise NotImplementedError(
            "Direct assignment to 'messages' is not allowed."
            " Use the 'add_messages' instead."
        )

    def get_messages(
        self,
        role: Optional[str] = None,
        n_messages: int = 10,
        excluded_fields: list[str] = ["_id", "_key", "_rev", "session_id", "time"],
    ) -> list:
        """Retrieve messages from ArangoDB, optionally filtered by role.

        :param role: Optional filter to retrieve messages of a specific role.
        :type role: Optional[str]
        :param n_messages: Number of messages to retrieve.
        :type n_messages: int
        :param excluded_fields: Fields to exclude from the returned messages.
        :type excluded_fields: list[str]

        .. code-block:: python

            # Get all types of messages, default is 10 messages
            messages = history.get_messages()

            # Get the first 20 human messages
            messages = history.get_messages(role="human", n_messages=20)

            # Get the first 20 AI messages
            messages = history.get_messages(role="ai", n_messages=20)

        """
        query = f"""
            FOR doc IN @@col
                FILTER doc.session_id == @session_id
                {"AND doc.role == @role" if role else ""}
                SORT doc.time DESC
                LIMIT @n
                RETURN UNSET(doc, @excluded_fields)
        """
        bind_vars = {
            "@col": self._collection_name,
            "session_id": self._session_id,
            "n": n_messages,
            "excluded_fields": excluded_fields,
        }
        if role is not None:
            bind_vars["role"] = role
        cursor = self._db.aql.execute(query, bind_vars=bind_vars)  # type: ignore

        # return in chronological order
        return [d for d in cursor][::-1]  # type: ignore

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in ArangoDB.

        Stores a single chat message in the ArangoDB collection associated with
        the current session. The message is stored with its type, content, and
        session identifier.

        :param message: The chat message to add to the history.
        :type message: BaseMessage

        .. code-block:: python

            from langchain_core.messages import HumanMessage, AIMessage

            # Add user message
            user_msg = HumanMessage(content="What is the weather like?")
            history.add_message(user_msg)

            # Add AI response
            ai_msg = AIMessage(content="I don't have access to current weather data.")
            history.add_message(ai_msg)

            # Or use convenience methods
            history.add_user_message("Hello!")
            history.add_ai_message("Hi there!")
        """

        self._db.collection(self._collection_name).insert(
            {
                "role": message.type,
                "content": message.content,
                "session_id": self._session_id,
                "time": time.time(),
            },
        )

    def add_qa_message(self, user_input: str, aql_query: str, result: str) -> None:
        """Add a QA message to the chat history.

        :param user_input: The user's input.
        :type user_input: str
        :param aql_query: The AQL query to execute.
        :type aql_query: str
        :param result: The result of the AQL query.
        :type result: str

        .. code-block:: python

            history.add_qa_message(
                user_input="Who is the first character?",
                aql_query="FOR doc IN Characters LIMIT 1 RETURN doc",
                result="The first character is Arya Stark."
            )
        """
        self._db.collection(self._collection_name).insert(
            {
                "role": "qa",
                "session_id": self._session_id,
                "time": time.time(),
                "user_input": user_input,
                "aql_query": aql_query,
                "result": result,
            }
        )

    def clear(self) -> None:
        """Clear session memory from ArangoDB.

        Removes all messages associated with the current session from the ArangoDB
        collection. The collection itself is preserved for future use.

        .. code-block:: python

            # Add some messages
            history.add_user_message("Hello")
            history.add_ai_message("Hi!")
            print(f"Messages before clear: {len(history.messages)}")  # Output: 2

            # Clear all messages for this session
            history.clear()
            print(f"Messages after clear: {len(history.messages)}")   # Output: 0

            # Collection still exists for future messages
            history.add_user_message("Starting fresh conversation")
            print(f"New message count: {len(history.messages)}")      # Output: 1
        """
        query = """
            FOR doc IN @@col
                FILTER doc.session_id == @session_id
                REMOVE doc IN @@col
        """

        bind_vars = {"@col": self._collection_name, "session_id": self._session_id}

        self._db.aql.execute(query, bind_vars=bind_vars)  # type: ignore
