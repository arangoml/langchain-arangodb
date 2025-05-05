from typing import Any, List, Union

from arango.database import StandardDatabase
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict


class ArangoChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in an ArangoDB database."""

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
        """Retrieve the messages from ArangoDB"""
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

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in ArangoDB"""

        self._db.collection(self._collection_name).insert(
            {
                "role": message.type,
                "content": message.content,
                "session_id": self._session_id,
            },
        )

    def clear(self) -> None:
        """Clear session memory from ArangoDB"""
        query = """
            FOR doc IN @@col
                FILTER doc.session_id == @session_id
                REMOVE doc IN @@col
        """

        bind_vars = {"@col": self._collection_name, "session_id": self._session_id}

        self._db.aql.execute(query, bind_vars=bind_vars)  # type: ignore
