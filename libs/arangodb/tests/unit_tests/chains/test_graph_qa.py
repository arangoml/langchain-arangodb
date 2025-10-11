"""Unit tests for ArangoGraphQAChain."""

import math
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest
from arango import AQLQueryExecuteError
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from langchain_arangodb.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_arangodb.chat_message_histories.arangodb import ArangoChatMessageHistory
from langchain_arangodb.graphs.arangodb_graph import ArangoGraph
from tests.llms.fake_llm import FakeLLM


class FakeGraphStore(ArangoGraph):
    """A fake ArangoGraph implementation for testing purposes."""

    def __init__(self) -> None:
        self._schema_yaml = "node_props:\n Movie:\n - property: title\n   type: STRING"
        self._schema_json = (
            '{"node_props": {"Movie": [{"property": "title", "type": "STRING"}]}}'  # noqa: E501
        )
        self.queries_executed = []  # type: ignore
        self.explains_run = []  # type: ignore
        self.refreshed = False
        self.graph_documents_added = []  # type: ignore

        # Mock the database interface
        self.__db = Mock()
        self.__db.collection = Mock()
        mock_queries_collection = Mock()
        mock_queries_collection.find = Mock(return_value=[])
        mock_queries_collection.insert = Mock()
        self.__db.collection.return_value = mock_queries_collection
        self.__db.aql = Mock()
        self.__db.aql.execute = Mock(return_value=[])

    @property
    def schema_yaml(self) -> str:
        return self._schema_yaml

    @property
    def db(self) -> Mock:  # type: ignore
        return self.__db  # type: ignore

    @property
    def schema_json(self) -> str:
        return self._schema_json

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        self.queries_executed.append((query, params))
        return [{"title": "Inception", "year": 2010}]

    def explain(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        self.explains_run.append((query, params))
        return [{"plan": "This is a fake AQL query plan."}]

    def refresh_schema(self) -> None:  # type: ignore
        self.refreshed = True

    def add_graph_documents(  # type: ignore
        self,
        graph_documents,  # type: ignore
        include_source: bool = False,  # type: ignore
    ) -> None:
        self.graph_documents_added.append((graph_documents, include_source))


class TestArangoGraphQAChain:
    """Test suite for ArangoGraphQAChain."""

    @pytest.fixture
    def fake_graph_store(self) -> FakeGraphStore:
        """Create a fake GraphStore."""
        return FakeGraphStore()

    @pytest.fixture
    def fake_llm(self) -> FakeLLM:
        """Create a fake LLM."""
        return FakeLLM()

    @pytest.fixture
    def mock_embedding(self) -> Mock:
        """Create a mock embedding model."""
        mock_emb = Mock()
        mock_emb.embed_query = Mock(
            return_value=[0.1, 0.2, 0.3]
        )  # Simple mock embedding vector
        return mock_emb

    @pytest.fixture
    def mock_db_with_queries(self) -> Mock:
        """Create a mock database with a Queries collection."""
        mock_db = Mock()
        mock_collection = Mock()
        mock_collection.find = Mock(return_value=[])  # Default to empty results
        mock_db.collection = Mock(return_value=mock_collection)
        mock_db.aql = Mock()
        mock_db.aql.execute = Mock(return_value=[])  # Default to empty results
        return mock_db

    @pytest.fixture
    def mock_chains(self) -> Dict[str, Runnable]:
        """Create mock chains that correctly implement the Runnable abstract class."""

        class CompliantRunnable(Runnable):
            def invoke(self, *args, **kwargs) -> None:  # type: ignore
                pass

            def stream(self, *args, **kwargs) -> None:  # type: ignore
                yield

            def batch(self, *args, **kwargs) -> List[Any]:  # type: ignore
                return []

        qa_chain = CompliantRunnable()
        qa_chain.invoke = MagicMock(return_value="This is a test answer")  # type: ignore

        aql_generation_chain = CompliantRunnable()
        aql_generation_chain.invoke = MagicMock(  # type: ignore
            return_value="```aql\nFOR doc IN Movies RETURN doc\n```"
        )  # noqa: E501

        aql_fix_chain = CompliantRunnable()
        aql_fix_chain.invoke = MagicMock(  # type: ignore
            return_value="```aql\nFOR doc IN Movies LIMIT 10 RETURN doc\n```"
        )  # noqa: E501

        return {
            "qa_chain": qa_chain,
            "aql_generation_chain": aql_generation_chain,
            "aql_fix_chain": aql_fix_chain,
        }

    def test_initialize_chain_with_dangerous_requests_false(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test that initialization fails when allow_dangerous_requests is False."""
        with pytest.raises(ValueError, match="dangerous requests"):
            ArangoGraphQAChain(
                graph=fake_graph_store,
                aql_generation_chain=mock_chains["aql_generation_chain"],
                aql_fix_chain=mock_chains["aql_fix_chain"],
                qa_chain=mock_chains["qa_chain"],
                allow_dangerous_requests=False,
            )

    def test_initialize_chain_with_dangerous_requests_true(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test successful initialization when allow_dangerous_requests is True."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
        )
        assert isinstance(chain, ArangoGraphQAChain)
        assert chain.graph == fake_graph_store
        assert chain.allow_dangerous_requests is True

    def test_from_llm_class_method(
        self, fake_graph_store: FakeGraphStore, fake_llm: FakeLLM
    ) -> None:
        """Test the from_llm class method."""
        chain = ArangoGraphQAChain.from_llm(
            llm=fake_llm,
            graph=fake_graph_store,
            allow_dangerous_requests=True,
        )
        assert isinstance(chain, ArangoGraphQAChain)
        assert chain.graph == fake_graph_store

    def test_input_keys_property(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test the input_keys property."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
        )
        assert chain.input_keys == ["query"]

    def test_output_keys_property(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test the output_keys property."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
        )
        assert chain.output_keys == ["result"]

    def test_chain_type_property(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test the _chain_type property."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
        )
        assert chain._chain_type == "graph_aql_chain"

    def test_call_successful_execution(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test successful AQL query execution."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
        )

        result = chain._call({"query": "Find all movies"})

        assert "result" in result
        assert result["result"] == "This is a test answer"
        assert len(fake_graph_store.queries_executed) == 1

    def test_call_with_ai_message_response(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test AQL generation with AIMessage response."""
        mock_chains["aql_generation_chain"].invoke.return_value = AIMessage(  # type: ignore
            content="```aql\nFOR doc IN Movies RETURN doc\n```"
        )

        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
        )

        result = chain._call({"query": "Find all movies"})

        assert "result" in result
        assert len(fake_graph_store.queries_executed) == 1

    def test_call_with_return_aql_query_true(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test returning AQL query in output."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
            return_aql_query=True,
        )

        result = chain._call({"query": "Find all movies"})

        assert "result" in result
        assert "aql_query" in result

    def test_call_with_return_aql_result_true(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test returning AQL result in output."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
            return_aql_result=True,
        )

        result = chain._call({"query": "Find all movies"})

        assert "result" in result
        assert "aql_result" in result

    def test_call_with_execute_aql_query_false(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test when execute_aql_query is False (explain only)."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
            execute_aql_query=False,
        )

        result = chain._call({"query": "Find all movies"})

        assert "result" in result
        assert "aql_result" in result
        assert len(fake_graph_store.explains_run) == 1
        assert len(fake_graph_store.queries_executed) == 0

    def test_call_no_aql_code_blocks(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test error when no AQL code blocks are found."""
        mock_chains["aql_generation_chain"].invoke.return_value = "No AQL query here"  # type: ignore

        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
        )

        with pytest.raises(ValueError, match="Unable to extract AQL Query"):
            chain._call({"query": "Find all movies"})

    def test_call_invalid_generation_output_type(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test error with invalid AQL generation output type."""
        mock_chains["aql_generation_chain"].invoke.return_value = 12345  # type: ignore

        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
        )

        with pytest.raises(ValueError, match="Invalid AQL Generation Output"):
            chain._call({"query": "Find all movies"})

    def test_call_with_aql_execution_error_and_retry(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test AQL execution error and retry mechanism."""
        error_graph_store = FakeGraphStore()

        # Create a real exception instance without calling its complex __init__
        error_instance = AQLQueryExecuteError.__new__(AQLQueryExecuteError)
        error_instance.error_message = "Mocked AQL execution error"

        def query_side_effect(query: str, params: dict = {}) -> List[Dict[str, Any]]:
            if error_graph_store.query.call_count == 1:  # type: ignore
                raise error_instance
            else:
                return [{"title": "Inception"}]

        error_graph_store.query = Mock(side_effect=query_side_effect)  # type: ignore

        chain = ArangoGraphQAChain(
            graph=error_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
            max_aql_generation_attempts=3,
        )

        result = chain._call({"query": "Find all movies"})

        assert "result" in result
        assert mock_chains["aql_fix_chain"].invoke.call_count == 1  # type: ignore

    def test_call_max_attempts_exceeded(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test when maximum AQL generation attempts are exceeded."""
        error_graph_store = FakeGraphStore()

        # Create a real exception instance to be raised on every call
        error_instance = AQLQueryExecuteError.__new__(AQLQueryExecuteError)
        error_instance.error_message = "Persistent error"
        error_graph_store.query = Mock(side_effect=error_instance)  # type: ignore

        chain = ArangoGraphQAChain(
            graph=error_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
            max_aql_generation_attempts=2,
        )

        with pytest.raises(
            ValueError, match="Maximum amount of AQL Query Generation attempts"
        ):  # noqa: E501
            chain._call({"query": "Find all movies"})

    def test_is_read_only_query_with_read_operation(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test _is_read_only_query with a read operation."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
        )

        is_read_only, write_op = chain._is_read_only_query(
            "FOR doc IN Movies RETURN doc"
        )  # noqa: E501
        assert is_read_only is True
        assert write_op is None

    def test_is_read_only_query_with_write_operation(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test _is_read_only_query with a write operation."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
        )

        is_read_only, write_op = chain._is_read_only_query(
            "INSERT {name: 'test'} INTO Movies"
        )  # noqa: E501
        assert is_read_only is False
        assert "INSERT" in write_op

    def test_force_read_only_query_with_write_operation(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test force_read_only_query flag with write operation."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
            force_read_only_query=True,
        )

        mock_chains[  # type: ignore
            "aql_generation_chain"
        ].invoke.return_value = "```aql\nINSERT {name: 'test'} INTO Movies\n```"  # noqa: E501

        with pytest.raises(
            ValueError, match="Security violation: Write operations are not allowed"
        ):  # noqa: E501
            chain._call({"query": "Add a movie"})

    def test_custom_input_output_keys(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test custom input and output keys."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
            input_key="question",
            output_key="answer",
        )

        assert chain.input_keys == ["question"]
        assert chain.output_keys == ["answer"]

        result = chain._call({"question": "Find all movies"})
        assert "answer" in result

    def test_custom_limits_and_parameters(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test custom limits and parameters."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
            top_k=5,
            output_list_limit=16,
            output_string_limit=128,
        )

        chain._call({"query": "Find all movies"})

        executed_query = fake_graph_store.queries_executed[0]
        params = executed_query[1]
        assert params["top_k"] == 5
        assert params["list_limit"] == 16
        assert params["string_limit"] == 128

    def test_aql_examples_parameter(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test that AQL examples are passed to the generation chain."""
        example_queries = "FOR doc IN Movies RETURN doc.title"

        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
            aql_examples=example_queries,
        )

        chain._call({"query": "Find all movies"})

        call_args, _ = mock_chains["aql_generation_chain"].invoke.call_args  # type: ignore
        assert call_args[0]["aql_examples"] == example_queries

    @pytest.mark.parametrize(
        "write_op", ["INSERT", "UPDATE", "REPLACE", "REMOVE", "UPSERT"]
    )
    def test_all_write_operations_detected(
        self,
        fake_graph_store: FakeGraphStore,
        mock_chains: Dict[str, Runnable],
        write_op: str,
    ) -> None:
        """Test that all write operations are correctly detected."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
        )

        query = f"{write_op} {{name: 'test'}} INTO Movies"
        is_read_only, detected_op = chain._is_read_only_query(query)
        assert is_read_only is False
        assert write_op in detected_op

    def test_call_with_callback_manager(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """Test _call with callback manager."""
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
        )

        mock_run_manager = Mock(spec=CallbackManagerForChainRun)
        mock_run_manager.get_child.return_value = Mock()

        result = chain._call({"query": "Find all movies"}, run_manager=mock_run_manager)

        assert "result" in result
        assert mock_run_manager.get_child.called

    def test_query_cache(self, fake_graph_store: FakeGraphStore) -> None:
        """Test query cache in 4 situations:
        1. Exact search
        2. Vector search
        3. AQL generation and store new query
        4. Query cache disabled
        """

        def cosine_similarity(v1, v2):  # type: ignore
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            if norm1 == 0.0 or norm2 == 0.0:
                return 0.0
            return dot / (norm1 * norm2)

        fake_graph_store.db.collection("Queries").find = Mock(return_value=[])

        # Simulate a previously cached query
        stored_query = {
            "text": "List all movies",
            "aql": "FOR m IN Movies RETURN m",
            "embedding": [0.123, 0.456, 0.789, 0.321, 0.654],
        }

        def mock_execute(query, bind_vars):  # type: ignore
            # Handle vector search query
            if "query_embedding" in bind_vars:
                query_embedding = bind_vars["query_embedding"]
                score = cosine_similarity(query_embedding, stored_query["embedding"])
                if score > bind_vars.get("score_threshold", 0.75):
                    return [{"aql": stored_query["aql"], "score": score}]
                return []
            # Handle regular query execution
            return [{"title": "Inception"}]

        fake_graph_store.db.aql.execute = Mock(side_effect=mock_execute)

        dummy_llm = RunnableLambda(
            lambda prompt: "```FOR m IN Movies LIMIT 1 RETURN m```"
        )

        chain = ArangoGraphQAChain.from_llm(
            llm=dummy_llm,  # type: ignore
            graph=fake_graph_store,
            allow_dangerous_requests=True,
            verbose=True,
            return_aql_result=True,
        )

        chain.embedding = type(
            "FakeEmbedding",
            (),
            {
                "embed_query": staticmethod(
                    lambda text: {
                        "Find all movies": [0.123, 0.456, 0.789, 0.321, 0.654],
                        "Show me all movies": [0.120, 0.460, 0.780, 0.330, 0.650],
                    }.get(text, [0.0] * 5)
                )
            },
        )()

        # 1. Test with exact search
        result1 = chain.invoke({"query": "Find all movies", "use_query_cache": True})
        assert result1["aql_result"][0]["title"] == "Inception"

        # 2. Test with vector search
        result2 = chain.invoke({"query": "Show me all movies", "use_query_cache": True})
        assert result2["aql_result"][0]["title"] == "Inception"

        # 3. Test with aql generation and store new query
        result3 = chain.invoke(
            {"query": "What is the name of the first movie?", "use_query_cache": True}
        )
        assert result3["result"] == "```FOR m IN Movies LIMIT 1 RETURN m```"

        # 4. Test with query cache disabled
        result4 = chain.invoke({"query": "What is the name of the first movie?"})
        assert result4["result"] == "```FOR m IN Movies LIMIT 1 RETURN m```"

    def test_chat_history(
        self, fake_graph_store: FakeGraphStore, mock_chains: Dict[str, Runnable]
    ) -> None:
        """test _call with chat history"""

        chat_history_store = Mock(spec=ArangoChatMessageHistory)

        chat_history_store._collection_name = "ChatHistory"

        # Add fake message history
        chat_history_store.add_qa_message(
            user_input="What is 1+1?",
            aql_query="RETURN 1+1",
            result="2",
        )

        chat_history_store.get_messages.return_value = [
            {
                "user_input": "What is 1+1?",
                "aql_query": "RETURN 1+1",
                "result": "2",
                "role": "qa",
                "session_id": "test",
            },
            {
                "user_input": "What is 2+2?",
                "aql_query": "RETURN 2+2",
                "result": "4",
                "role": "qa",
                "session_id": "test",
            },
        ]

        # Mock LLM chains
        mock_chains[  # type: ignore
            "aql_generation_chain"
        ].invoke.return_value = "```aql\nFOR m IN Movies RETURN m\n```"  # noqa: E501
        mock_chains["qa_chain"].invoke.return_value = AIMessage(  # type: ignore
            content="Here are the movies."
        )  # noqa: E501

        # Build the chain
        chain = ArangoGraphQAChain(
            graph=fake_graph_store,
            aql_generation_chain=mock_chains["aql_generation_chain"],
            aql_fix_chain=mock_chains["aql_fix_chain"],
            qa_chain=mock_chains["qa_chain"],
            allow_dangerous_requests=True,
            include_history=True,
            chat_history_store=chat_history_store,
            max_history_messages=10,
            return_aql_result=True,
        )

        # Run the call
        result = chain.invoke({"query": "List all movies"})

        # LLM received the latest 2 docs
        llm_input = mock_chains["aql_generation_chain"].invoke.call_args[0][0]  # type: ignore
        chat_history = llm_input["chat_history"]
        assert len(chat_history) == 2

        # result has expected fields
        assert result["result"].content == "Here are the movies."
        assert result["aql_result"][0]["title"] == "Inception"

        # Error: chat history enabled but store is missing
        chain.chat_history_store = None
        with pytest.raises(
            ValueError,
            match="Chat message history is required if include_history is True",
        ):
            chain.invoke({"query": "List again"})

        # Error: invalid max_history_messages
        chain.chat_history_store = chat_history_store
        chain.max_history_messages = 0
        with pytest.raises(
            ValueError, match="max_history_messages must be greater than 0"
        ):
            chain.invoke({"query": "List again"})
