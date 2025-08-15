"""Question answering over a graph."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from arango import AQLQueryExecuteError, AQLQueryExplainError
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from pydantic import Field

from langchain_arangodb.chains.graph_qa.prompts import (
    AQL_FIX_PROMPT,
    AQL_GENERATION_PROMPT,
    AQL_QA_PROMPT,
)
from langchain_arangodb.chat_message_histories.arangodb import ArangoChatMessageHistory
from langchain_arangodb.graphs.arangodb_graph import ArangoGraph

AQL_WRITE_OPERATIONS: List[str] = [
    "INSERT",
    "UPDATE",
    "REPLACE",
    "REMOVE",
    "UPSERT",
]


class ArangoGraphQAChain(Chain):
    """Chain for question-answering against a graph by generating AQL statements.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    graph: ArangoGraph = Field(exclude=True)
    embedding: Optional[Embeddings] = Field(default=None, exclude=True)
    query_cache_collection_name: str = Field(default="Queries")
    aql_generation_chain: Runnable[Dict[str, Any], Any]
    aql_fix_chain: Runnable[Dict[str, Any], Any]
    qa_chain: Runnable[Dict[str, Any], Any]
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    use_query_cache: bool = Field(default=False)
    query_cache_similarity_threshold: float = Field(default=0.80)
    include_history: bool = Field(default=False)
    max_history_messages: int = Field(default=10)
    chat_history_store: Optional[ArangoChatMessageHistory] = Field(default=None)

    top_k: int = 10
    """Number of results to return from the query"""
    aql_examples: str = ""
    """Specifies the set of AQL Query Examples that promote few-shot-learning"""
    return_aql_query: bool = False
    """ Specify whether to return the AQL Query in the output dictionary"""
    return_aql_result: bool = False
    """Specify whether to return the AQL JSON Result in the output dictionary"""
    max_aql_generation_attempts: int = 3
    """Specify the maximum amount of AQL Generation attempts that should be made"""
    execute_aql_query: bool = True
    """If False, the AQL Query is only explained & returned, not executed"""
    allow_dangerous_requests: bool = False
    """Forced user opt-in to acknowledge that the chain can make dangerous requests."""
    output_list_limit: int = 32
    """Maximum list length to include in the response prompt. Truncated if longer."""
    output_string_limit: int = 256
    """Maximum string length to include in the response prompt. Truncated if longer."""
    force_read_only_query: bool = False
    """If True, the query is checked for write operations and raises an
    error if a write operation is detected."""

    """
    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chain."""
        super().__init__(**kwargs)
        if self.allow_dangerous_requests is not True:
            raise ValueError(
                "In order to use this chain, you must acknowledge that it can make "
                "dangerous requests by setting `allow_dangerous_requests` to `True`."
                "You must narrowly scope the permissions of the database connection "
                "to only include necessary permissions. Failure to do so may result "
                "in data corruption or loss or reading sensitive data if such data is "
                "present in the database."
                "Only use this chain if you understand the risks and have taken the "
                "necessary precautions. "
                "See https://python.langchain.com/docs/security for more information."
            )
        self._last_user_input: Optional[str] = None
        self._last_aql_query: Optional[str] = None

    @property
    def input_keys(self) -> List[str]:
        """Get the input keys for the chain."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Get the output keys for the chain."""
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
        """Get the chain type."""
        return "graph_aql_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        qa_prompt: Optional[BasePromptTemplate] = None,
        aql_generation_prompt: Optional[BasePromptTemplate] = None,
        aql_fix_prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> ArangoGraphQAChain:
        """Initialize from LLM.

        :param llm: The language model to use.
        :type llm: BaseLanguageModel
        :param embedding: The embedding model to use.
        :type embedding: Embeddings
        :param query_cache_collection_name: The name of the collection
            to use for the query cache.
        :type query_cache_collection_name: str
        :param include_history: Whether to include the chat history in the prompt.
        :type include_history: bool
        :param max_history_messages: The maximum number of messages to
            include in the chat history.
        :type max_history_messages: int
        :param chat_history_store: The chat history store to use.
        :type chat_history_store: ArangoChatMessageHistory
        :param qa_prompt: The prompt to use for the QA chain.
        :type qa_prompt: BasePromptTemplate
        :param aql_generation_prompt: The prompt to use for the AQL generation chain.
        :type aql_generation_prompt: BasePromptTemplate
        :param aql_fix_prompt: The prompt to use for the AQL fix chain.
        :type aql_fix_prompt: BasePromptTemplate
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any
        :return: The initialized ArangoGraphQAChain.
        :rtype: ArangoGraphQAChain
        :raises ValueError: If the LLM is not provided.
        """
        if qa_prompt is None:
            qa_prompt = AQL_QA_PROMPT
        if aql_generation_prompt is None:
            aql_generation_prompt = AQL_GENERATION_PROMPT
        if aql_fix_prompt is None:
            aql_fix_prompt = AQL_FIX_PROMPT

        qa_chain = qa_prompt | llm
        aql_generation_chain = aql_generation_prompt | llm
        aql_fix_chain = aql_fix_prompt | llm

        return cls(
            qa_chain=qa_chain,
            aql_generation_chain=aql_generation_chain,
            aql_fix_chain=aql_fix_chain,
            **kwargs,
        )

    def _check_and_insert_query(self, text: str, aql: str) -> str:
        """
        Check if a query is already in the cache and insert it if it's not.

        :param text: The text of the query to check.
        :type text: str
        :param aql: The AQL query to check.
        :type aql: str
        :return: A message indicating the result of the operation.
        """
        text = text.strip().lower()
        text_hash = self.graph._hash(text)
        collection = self.graph.db.collection(self.query_cache_collection_name)

        if collection.has(text_hash):
            return f"This query is already in the cache: {text}"

        if self.embedding is None:
            raise ValueError("Cannot cache queries without an embedding model.")

        query_embedding = self.embedding.embed_query(text)
        collection.insert(
            {
                "_key": text_hash,
                "text": text,
                "embedding": query_embedding,
                "aql": aql,
            }
        )

        return f"Cached: {text}"

    def cache_query(self, text: Optional[str] = None, aql: Optional[str] = None) -> str:
        """
        Cache a query generated by the LLM only if it's not already stored.

        :param text: The text of the query to cache.
        :param aql: The AQL query to cache.
        :return: A message indicating the result of the operation.
        """
        if self.embedding is None:
            raise ValueError("Cannot cache queries without an embedding model.")

        if not self.graph.db.has_collection(self.query_cache_collection_name):
            m = f"Collection {self.query_cache_collection_name} does not exist"  # noqa: E501
            raise ValueError(m)

        if not text and aql:
            raise ValueError("Text is required to cache a query")

        if text and not aql:
            raise ValueError("AQL is required to cache a query")

        if not text and not aql:
            if self._last_user_input is None or self._last_aql_query is None:
                m = "No previous query to cache. Please provide **text** and **aql**."
                raise ValueError(m)

            # Fallback: cache the most recent query
            return self._check_and_insert_query(
                self._last_user_input,
                self._last_aql_query,
            )

        if not isinstance(text, str) or not isinstance(aql, str):
            raise ValueError("Text and AQL must be strings")

        return self._check_and_insert_query(text, aql)

    def clear_query_cache(self, text: Optional[str] = None) -> str:
        """
        Clear the query cache.

        :param text: The text of the query to delete from the cache.
        :type text: str
        :return: A message indicating the result of the operation.
        """

        if not self.graph.db.has_collection(self.query_cache_collection_name):
            m = f"Collection {self.query_cache_collection_name} does not exist"
            raise ValueError(m)

        collection = self.graph.db.collection(self.query_cache_collection_name)

        if text is None:
            collection.truncate()
            return "Cleared all queries from the cache"

        text = text.strip().lower()
        text_hash = self.graph._hash(text)

        if collection.has(text_hash):
            collection.delete(text_hash)
            return f"Removed: {text}"

        return f"Not found: {text}"

    def _get_cached_query(
        self, user_input: str, query_cache_similarity_threshold: float
    ) -> Optional[Tuple[str, str]]:
        """Get the cached query for the user input. Only used if embedding
        is provided and **use_query_cache** is True.

        :param user_input: The user input to search for in the cache.
        :type user_input: str

        :return: The cached query and score, if found.
        :rtype: Optional[Tuple[str, int]]
        """
        if self.embedding is None:
            raise ValueError("Cannot enable query cache without passing embedding")

        if self.graph.db.collection(self.query_cache_collection_name).count() == 0:
            return None

        user_input = user_input.strip().lower()

        # 1. Exact Search

        query = f"""
            FOR q IN {self.query_cache_collection_name}
                FILTER q.text == @user_input
                LIMIT 1
                RETURN q.aql
        """

        cursor = self.graph.db.aql.execute(
            query,
            bind_vars={"user_input": user_input},
        )

        result = list(cursor)  # type: ignore

        if result:
            return result[0], "1.0"

        # 2. Vector Search

        embedding = self.embedding.embed_query(user_input)
        query = """
            FOR q IN @@col
                LET score = COSINE_SIMILARITY(q.embedding, @embedding)
                SORT score DESC
                LIMIT 1
                FILTER score > @score_threshold
                RETURN {aql: q.aql, score: score}   
        """

        result = list(
            self.graph.db.aql.execute(
                query,
                bind_vars={
                    "@col": self.query_cache_collection_name,
                    "embedding": embedding,  # type: ignore
                    "score_threshold": query_cache_similarity_threshold,  # type: ignore
                },
            )
        )

        if result:
            return result[0]["aql"], str(round(result[0]["score"], 2))

        return None

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Generate an AQL statement from user input, use it retrieve a response
        from an ArangoDB Database instance, and respond to the user input
        in natural language.

        Users can modify the following ArangoGraphQAChain Class Variables:

        :param top_k: The maximum number of AQL Query Results to return
        :type top_k: int

        :param aql_examples: A set of AQL Query Examples that are passed to
            the AQL Generation Prompt Template to promote few-shot-learning.
            Defaults to an empty string.
        :type aql_examples: str

        :param return_aql_query: Whether to return the AQL Query in the
            output dictionary. Defaults to False.
        :type return_aql_query: bool

        :param return_aql_result: Whether to return the AQL Query in the
            output dictionary. Defaults to False
        :type return_aql_result: bool

        :param max_aql_generation_attempts: The maximum amount of AQL
            Generation attempts to be made prior to raising the last
            AQL Query Execution Error. Defaults to 3.
        :type max_aql_generation_attempts: int

        :param execute_aql_query: If False, the AQL Query is only
            explained & returned, not executed. Defaults to True.
        :type execute_aql_query: bool

        :param output_list_limit: The maximum list length to display
            in the output. If the list is longer, it will be truncated.
            Defaults to 32.
        :type output_list_limit: int

        :param output_string_limit: The maximum string length to display
            in the output. If the string is longer, it will be truncated.
            Defaults to 256.
        :type output_string_limit: int
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        user_input = inputs[self.input_key].strip().lower()

        # Query Cache Parameters (can be overridden by inputs at runtime)
        use_query_cache = inputs.get("use_query_cache", self.use_query_cache)
        query_cache_similarity_threshold = inputs.get(
            "query_cache_similarity_threshold", self.query_cache_similarity_threshold
        )

        # Chat History Parameters (can be overridden by inputs at runtime)
        include_history = inputs.get("include_history", self.include_history)
        max_history_messages = inputs.get(
            "max_history_messages", self.max_history_messages
        )

        if use_query_cache and self.embedding is None:
            raise ValueError("Cannot enable query cache without passing embedding")

        # ######################
        # # Get Chat History #
        # ######################

        if include_history and self.chat_history_store is None:
            raise ValueError(
                "Chat message history is required if include_history is True"
            )

        if max_history_messages <= 0:
            raise ValueError("max_history_messages must be greater than 0")

        chat_history = []
        if include_history and self.chat_history_store is not None:
            for msg in self.chat_history_store.messages[-max_history_messages:]:
                cls = HumanMessage if msg.type == "human" else AIMessage
                chat_history.append(cls(content=msg.content))

        ######################
        # Check Query Cache #
        ######################

        cached_query, score = None, None
        if use_query_cache:
            if self.embedding is None:
                m = "Embedding must be provided when using query cache"
                raise ValueError(m)

            if not self.graph.db.has_collection(self.query_cache_collection_name):
                self.graph.db.create_collection(self.query_cache_collection_name)

            cache_result = self._get_cached_query(
                user_input, query_cache_similarity_threshold
            )

            if cache_result is not None:
                cached_query, score = cache_result

        if cached_query:
            aql_generation_output = f"```aql{cached_query}```"
        else:
            aql_generation_output = self.aql_generation_chain.invoke(
                {
                    "adb_schema": self.graph.schema_yaml,
                    "aql_examples": self.aql_examples,
                    "user_input": user_input,
                    "chat_history": chat_history,
                },
                callbacks=callbacks,
            )

        aql_query = ""
        aql_error = ""
        aql_result = None
        aql_generation_attempt = 1

        aql_execution_func = (
            self.graph.query if self.execute_aql_query else self.graph.explain
        )

        while (
            aql_result is None
            and aql_generation_attempt < self.max_aql_generation_attempts + 1
        ):
            if isinstance(aql_generation_output, str):
                aql_generation_output_content = aql_generation_output
            elif isinstance(aql_generation_output, AIMessage):
                aql_generation_output_content = str(aql_generation_output.content)
            else:
                m = f"Invalid AQL Generation Output: {aql_generation_output} (type: {type(aql_generation_output)})"  # noqa: E501
                raise ValueError(m)

            #####################
            # Extract AQL Query #
            #####################

            pattern = r"```(?i:aql)?(.*?)```"
            matches: List[str] = re.findall(
                pattern, aql_generation_output_content, re.DOTALL
            )

            if not matches:
                _run_manager.on_text(
                    "Invalid Response: ", end="\n", verbose=self.verbose
                )

                _run_manager.on_text(
                    aql_generation_output_content,
                    color="red",
                    end="\n",
                    verbose=self.verbose,
                )

                m = f"Unable to extract AQL Query from response: {aql_generation_output_content}"  # noqa: E501
                raise ValueError(m)

            aql_query = matches[0].strip()

            if self.force_read_only_query:
                is_read_only, write_operation = self._is_read_only_query(aql_query)

                if not is_read_only:
                    error_msg = f"""
                        Security violation: Write operations are not allowed.
                        Detected write operation in query: {write_operation}
                    """
                    raise ValueError(error_msg)

            query_message = f"AQL Query ({aql_generation_attempt})\n"
            if cached_query:
                score_string = score if score is not None else "1.0"
                query_message = (
                    f"AQL Query (used cached query, score: {score_string})\n"  # noqa: E501
                )

            _run_manager.on_text(query_message, verbose=self.verbose)
            _run_manager.on_text(
                aql_query, color="green", end="\n", verbose=self.verbose
            )

            #############################
            # Execute/Explain AQL Query #
            #############################

            try:
                params = {
                    "top_k": self.top_k,
                    "list_limit": self.output_list_limit,
                    "string_limit": self.output_string_limit,
                }
                aql_result = aql_execution_func(aql_query, params)
            except (AQLQueryExecuteError, AQLQueryExplainError) as e:
                aql_error = str(e.error_message)

                _run_manager.on_text(
                    "AQL Query Execution Error: ", end="\n", verbose=self.verbose
                )
                _run_manager.on_text(
                    aql_error, color="yellow", end="\n\n", verbose=self.verbose
                )

                ########################
                # Retry AQL Generation #
                ########################

                aql_generation_output = self.aql_fix_chain.invoke(
                    {
                        "adb_schema": self.graph.schema_yaml,
                        "aql_query": aql_query,
                        "aql_error": aql_error,
                    },
                    callbacks=callbacks,
                )

            aql_generation_attempt += 1

        if aql_result is None:
            m = f"""
                Maximum amount of AQL Query Generation attempts reached.
                Unable to execute the AQL Query due to the following error:
                {aql_error}
            """
            raise ValueError(m)

        text = "AQL Result:" if self.execute_aql_query else "AQL Explain:"
        _run_manager.on_text(text, end="\n", verbose=self.verbose)
        _run_manager.on_text(
            str(aql_result), color="green", end="\n", verbose=self.verbose
        )

        if not self.execute_aql_query:
            result = {self.output_key: aql_query, "aql_result": aql_result}

            return result

        ########################
        # Interpret AQL Result #
        ########################

        result = self.qa_chain.invoke(  # type: ignore
            {
                "adb_schema": self.graph.schema_yaml,
                "user_input": user_input,
                "aql_query": aql_query,
                "aql_result": aql_result,
            },
            callbacks=callbacks,
        )

        # Add summary
        text = "Summary:"
        _run_manager.on_text(text, end="\n", verbose=self.verbose)
        _run_manager.on_text(
            str(result.content) if isinstance(result, AIMessage) else result,
            color="green",
            end="\n",
            verbose=self.verbose,
        )

        results: Dict[str, Any] = {self.output_key: result}

        if self.return_aql_query:
            results["aql_query"] = aql_generation_output

        if self.return_aql_result:
            results["aql_result"] = aql_result

        self._last_user_input = user_input
        self._last_aql_query = aql_query

        ########################
        # Store Chat History #
        ########################

        if self.chat_history_store:
            self.chat_history_store.add_user_message(user_input)
            self.chat_history_store.add_ai_message(aql_query)
            self.chat_history_store.add_ai_message(result)

        return results

    def _is_read_only_query(self, aql_query: str) -> Tuple[bool, Optional[str]]:
        """Check if the AQL query is read-only.

        :param aql_query: The AQL query to check.
        :type aql_query: str

        :return: True if the query is read-only, False otherwise.
        :rtype: Tuple[bool, Optional[str]]
        """
        normalized_query = aql_query.upper()

        for op in AQL_WRITE_OPERATIONS:
            if op in normalized_query:
                return False, op

        return True, None
