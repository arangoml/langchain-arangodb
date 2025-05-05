"""Unit tests for ArangoDB Graph QA chain."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from arango import AQLQueryExecuteError
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable

# Import ArangoDB specific classes and prompts
from langchain_arangodb.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_arangodb.chains.graph_qa.prompts import (
    AQL_FIX_PROMPT,
    AQL_GENERATION_PROMPT,
    AQL_QA_PROMPT,
)
from langchain_arangodb.graphs.arangodb_graph import ArangoGraph

# Assuming a FakeLLM exists similar to the one in Neo4j tests
class FakeLLM(BaseLanguageModel):
    """Fake LLM for testing."""
    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs: Any) -> str:
        return f"Response based on: {prompt[:100]}"

    async def _acall(self, prompt: str, stop: list[str] | None = None, **kwargs: Any) -> str:
        return f"Response based on: {prompt[:100]}"

    def _generate(
        self, prompts: list[str], stop: list[str] | None = None, **kwargs: Any
    ) -> Any:
        responses = [self._call(prompt) for prompt in prompts]
        # Simplified mock response generation
        return MagicMock(generations=[[MagicMock(text=r)] for r in responses])

    async def _agenerate(
        self, prompts: list[str], stop: list[str] | None = None, **kwargs: Any
    ) -> Any:
        responses = [await self._acall(prompt) for prompt in prompts]
        return MagicMock(generations=[[MagicMock(text=r)] for r in responses])

    @property
    def _llm_type(self) -> str:
        return "fake"

# --- Fixtures ---

@pytest.fixture
def mock_arangograph() -> MagicMock:
    """Fixture for a mocked ArangoGraph instance."""
    graph = MagicMock(spec=ArangoGraph)
    graph.schema_yaml = "mock_schema_yaml"
    graph.query.return_value = [{"result": "mock query result"}]
    graph.explain.return_value = [{"plan": "mock explain plan"}]
    return graph

@pytest.fixture
def fake_llm() -> FakeLLM:
    """Fixture for the FakeLLM."""
    return FakeLLM()

# --- Initialization and Prompt Tests (Adapted from Neo4j) ---

def test_arangograph_qa_chain_default_prompts(mock_arangograph: MagicMock, fake_llm: FakeLLM) -> None:
    """Test initialization with default prompts."""
    chain = ArangoGraphQAChain.from_llm(
        llm=fake_llm,
        graph=mock_arangograph,
        allow_dangerous_requests=True,
    )
    assert isinstance(chain.qa_chain, Runnable) # Check type, specific prompt checked implicitly
    assert isinstance(chain.aql_generation_chain, Runnable)
    assert isinstance(chain.aql_fix_chain, Runnable)
    # Cannot easily assert exact prompt equality for Runnables created with `|`

def test_arangograph_qa_chain_custom_prompts(mock_arangograph: MagicMock, fake_llm: FakeLLM) -> None:
    """Test initialization with custom prompts."""
    qa_prompt = PromptTemplate.from_template("QA: {question} {context}")
    aql_gen_prompt = PromptTemplate.from_template("GenAQL: {question} {schema}")
    aql_fix_prompt = PromptTemplate.from_template("FixAQL: {error} {query}")

    chain = ArangoGraphQAChain.from_llm(
        llm=fake_llm,
        graph=mock_arangograph,
        qa_prompt=qa_prompt,
        aql_generation_prompt=aql_gen_prompt,
        aql_fix_prompt=aql_fix_prompt,
        allow_dangerous_requests=True,
    )
    # Check if the LLM is part of the constructed runnable sequence
    # This is an indirect way to check prompt usage
    assert hasattr(chain.qa_chain, 'middle') and any(isinstance(r, FakeLLM) for r in chain.qa_chain.middle)
    assert hasattr(chain.aql_generation_chain, 'middle') and any(isinstance(r, FakeLLM) for r in chain.aql_generation_chain.middle)
    assert hasattr(chain.aql_fix_chain, 'middle') and any(isinstance(r, FakeLLM) for r in chain.aql_fix_chain.middle)

# --- Core Chain Execution Tests ---

@patch.object(ArangoGraphQAChain, 'aql_generation_chain', new_callable=MagicMock)
@patch.object(ArangoGraphQAChain, 'qa_chain', new_callable=MagicMock)
def test_chain_successful_execution(
    mock_qa_chain: MagicMock,
    mock_aql_gen_chain: MagicMock,
    mock_arangograph: MagicMock,
    fake_llm: FakeLLM,
) -> None:
    """Test a successful run through the QA chain."""
    chain = ArangoGraphQAChain.from_llm(llm=fake_llm, graph=mock_arangograph, allow_dangerous_requests=True)

    # Mock outputs
    generated_aql = "FOR doc IN nodes RETURN doc._key" # Simple valid AQL
    mock_aql_gen_chain.invoke.return_value = AIMessage(content=f"```aql\n{generated_aql}\n```")
    final_answer = "The keys are 123, 456."
    mock_qa_chain.invoke.return_value = AIMessage(content=final_answer)
    mock_query_result = [{"_key": "123"}, {"_key": "456"}]
    mock_arangograph.query.return_value = mock_query_result

    question = "What are the node keys?"
    result = chain.invoke({chain.input_key: question})

    # Assertions
    assert result[chain.output_key] == final_answer
    mock_aql_gen_chain.invoke.assert_called_once()
    call_args, _ = mock_aql_gen_chain.invoke.call_args
    assert call_args[0]["user_input"] == question
    assert call_args[0]["adb_schema"] == mock_arangograph.schema_yaml

    mock_arangograph.query.assert_called_once()
    args, kwargs = mock_arangograph.query.call_args
    assert args[0] == generated_aql
    assert kwargs['params']['limit'] == chain.top_k

    mock_qa_chain.invoke.assert_called_once()
    call_args_qa, _ = mock_qa_chain.invoke.call_args
    assert call_args_qa[0]["user_input"] == question
    assert call_args_qa[0]["aql_result"] == str(mock_query_result)


@patch.object(ArangoGraphQAChain, 'aql_generation_chain', new_callable=MagicMock)
@patch.object(ArangoGraphQAChain, 'aql_fix_chain', new_callable=MagicMock)
@patch.object(ArangoGraphQAChain, 'qa_chain', new_callable=MagicMock)
def test_chain_aql_fix_attempt(
    mock_qa_chain: MagicMock,
    mock_aql_fix_chain: MagicMock,
    mock_aql_gen_chain: MagicMock,
    mock_arangograph: MagicMock,
    fake_llm: FakeLLM,
) -> None:
    """Test the chain when AQL generation fails initially but is fixed."""
    chain = ArangoGraphQAChain.from_llm(llm=fake_llm, graph=mock_arangograph, allow_dangerous_requests=True)

    # Mock outputs
    initial_bad_aql = "FOR doc IN RETURN doc" # Invalid AQL
    fixed_aql = "FOR doc IN nodes RETURN doc" # Fixed AQL
    final_answer = "The documents are ..."
    mock_query_result = [{"_key": "node1"}]

    # Simulate generation -> error -> fix -> success
    mock_aql_gen_chain.invoke.return_value = AIMessage(content=f"```aql\n{initial_bad_aql}\n```")
    mock_arangograph.query.side_effect = [
        AQLQueryExecuteError("Syntax error near RETURN", http_exception=None, error_code=1501),
        mock_query_result # Successful query after fix
    ]
    mock_aql_fix_chain.invoke.return_value = AIMessage(content=f"```aql\n{fixed_aql}\n```")
    mock_qa_chain.invoke.return_value = AIMessage(content=final_answer)

    question = "What are the docs?"
    result = chain.invoke({chain.input_key: question})

    assert result[chain.output_key] == final_answer
    mock_aql_gen_chain.invoke.assert_called_once() # Initial generation
    assert mock_arangograph.query.call_count == 2 # First fails, second succeeds
    mock_aql_fix_chain.invoke.assert_called_once() # Fix attempt
    call_args_fix, _ = mock_aql_fix_chain.invoke.call_args
    assert call_args_fix[0]["aql_query"] == initial_bad_aql
    assert "Syntax error near RETURN" in call_args_fix[0]["aql_error"]
    assert mock_arangograph.query.call_args_list[1][0][0] == fixed_aql # Check fixed query used
    mock_qa_chain.invoke.assert_called_once() # Final QA


@patch.object(ArangoGraphQAChain, 'aql_generation_chain', new_callable=MagicMock)
@patch.object(ArangoGraphQAChain, 'aql_fix_chain', new_callable=MagicMock)
def test_chain_aql_fix_failure(
    mock_aql_fix_chain: MagicMock,
    mock_aql_gen_chain: MagicMock,
    mock_arangograph: MagicMock,
    fake_llm: FakeLLM,
) -> None:
    """Test the chain when AQL fixing fails repeatedly."""
    # Set max attempts low for testing
    chain = ArangoGraphQAChain.from_llm(llm=fake_llm, graph=mock_arangograph, allow_dangerous_requests=True, max_aql_generation_attempts=2)

    # Mock outputs
    bad_aql_1 = "FOR doc IN RETURN doc"
    bad_aql_2 = "FOR doc IN nodes RETURN"

    # Simulate gen -> error -> fix -> error -> fix -> error (max attempts reached)
    mock_aql_gen_chain.invoke.return_value = AIMessage(content=f"```aql\n{bad_aql_1}\n```")
    mock_arangograph.query.side_effect = AQLQueryExecuteError("Syntax error", http_exception=None, error_code=1501)
    mock_aql_fix_chain.side_effect = [
        AIMessage(content=f"```aql\n{bad_aql_2}\n```"), # First fix attempt
        AIMessage(content=f"```aql\n{bad_aql_2}\n```"), # Second fix attempt (no change)
    ]

    question = "What are the docs?"
    with pytest.raises(AQLQueryExecuteError, match="Syntax error"):
        chain.invoke({chain.input_key: question})

    assert mock_aql_gen_chain.invoke.call_count == 1
    assert mock_aql_fix_chain.call_count == 1 # Only called once after first error
    assert mock_arangograph.query.call_count == 2 # Tries initial query, then fixed query


@patch.object(ArangoGraphQAChain, 'aql_generation_chain', new_callable=MagicMock)
@patch.object(ArangoGraphQAChain, 'qa_chain', new_callable=MagicMock)
def test_chain_no_execute_aql(
    mock_qa_chain: MagicMock,
    mock_aql_gen_chain: MagicMock,
    mock_arangograph: MagicMock,
    fake_llm: FakeLLM,
) -> None:
    """Test the chain with execute_aql_query=False."""
    chain = ArangoGraphQAChain.from_llm(llm=fake_llm, graph=mock_arangograph, allow_dangerous_requests=True, execute_aql_query=False)

    generated_aql = "FOR doc IN nodes RETURN doc._key"
    mock_aql_gen_chain.invoke.return_value = AIMessage(content=f"```aql\n{generated_aql}\n```")
    final_answer = "Based on explain plan..."
    mock_qa_chain.invoke.return_value = AIMessage(content=final_answer)
    mock_explain_result = [{"plan": "mock explain plan"}]
    mock_arangograph.explain.return_value = mock_explain_result

    question = "Explain query for node keys?"
    result = chain.invoke({chain.input_key: question})

    assert result[chain.output_key] == final_answer
    mock_aql_gen_chain.invoke.assert_called_once()
    mock_arangograph.query.assert_not_called() # Should not be called
    mock_arangograph.explain.assert_called_once_with(generated_aql, params={'limit': chain.top_k})
    mock_qa_chain.invoke.assert_called_once()
    call_args_qa, _ = mock_qa_chain.invoke.call_args
    assert call_args_qa[0]["aql_result"] == str(mock_explain_result) # QA gets explain plan

@patch.object(ArangoGraphQAChain, 'aql_generation_chain', new_callable=MagicMock)
@patch.object(ArangoGraphQAChain, 'qa_chain', new_callable=MagicMock)
def test_return_aql_options(
    mock_qa_chain: MagicMock,
    mock_aql_gen_chain: MagicMock,
    mock_arangograph: MagicMock,
    fake_llm: FakeLLM,
) -> None:
    """Test return_aql_query and return_aql_result flags."""
    chain = ArangoGraphQAChain.from_llm(
        llm=fake_llm,
        graph=mock_arangograph,
        allow_dangerous_requests=True,
        return_aql_query=True,
        return_aql_result=True
    )

    generated_aql = "FOR doc IN nodes RETURN doc._key"
    mock_aql_gen_chain.invoke.return_value = AIMessage(content=f"```aql\n{generated_aql}\n```")
    final_answer = "The keys are ..."
    mock_qa_chain.invoke.return_value = AIMessage(content=final_answer)
    mock_query_result = [{"_key": "123"}]
    mock_arangograph.query.return_value = mock_query_result

    question = "Return keys"
    result = chain.invoke({chain.input_key: question})

    assert result[chain.output_key] == final_answer
    assert result["aql_query"] == generated_aql
    assert result["aql_result"] == mock_query_result

# --- Security Test ---

def test_allow_dangerous_requests_err(mock_arangograph: MagicMock, fake_llm: FakeLLM) -> None:
    """Test that initialization fails if allow_dangerous_requests is not True."""
    with pytest.raises(ValueError, match="acknowledge that it can make dangerous requests"):
        ArangoGraphQAChain.from_llm(
            llm=fake_llm,
            graph=mock_arangograph,
            allow_dangerous_requests=False # Explicitly False
        )
    # Test default is False
    with pytest.raises(ValueError, match="acknowledge that it can make dangerous requests"):
         ArangoGraphQAChain.from_llm(llm=fake_llm, graph=mock_arangograph)

# Note: More tests could be added for:
# - AQL extraction failure from LLM response.
# - Different schema structures.
# - Usage of `aql_examples`.
# - Edge cases in the retry/fix loop.
# - Truncation logic (`output_list_limit`, `output_string_limit`) in context passed to QA chain. 