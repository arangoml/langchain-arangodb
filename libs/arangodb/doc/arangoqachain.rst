ArangoGraphQAChain
==================

.. currentmodule:: langchain_arangodb.chains.graph_qa.arango_graph_qa

.. autoclass:: ArangoGraphQAChain
    :members:
    :undoc-members:
    :show-inheritance:

Overview
--------

The ``ArangoGraphQAChain`` is a LangChain-compatible class that enables natural language
question answering over a graph database by generating and executing AQL (ArangoDB Query Language)
statements. It combines prompt-based few-shot generation, error recovery, and semantic interpretation
of AQL results.

.. important::

   **Security Warning**: This chain can generate potentially dangerous queries (e.g., deletions or updates).
   It is highly recommended to use database credentials with limited read-only permissions unless explicitly
   allowing mutation operations by setting ``allow_dangerous_requests=True`` and carefully scoping access.

Initialization
--------------

You can create an instance in two ways:

1. Manually by passing preconfigured prompt chains and a graph store.
2. Using the classmethod :meth:`from_llm`.

.. code-block:: python

    from langchain_arangodb.chains.graph_qa import ArangoGraphQAChain
    from langchain_openai import ChatOpenAI
    from langchain_arangodb.graphs import ArangoGraph

    llm = ChatOpenAI(model="gpt-4", temperature=0)
    graph = ArangoGraph.from_connection_args(...)

    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
    )

Attributes
----------

.. attribute:: input_key
   :type: str

   Default input key for question: ``"query"``.

.. attribute:: output_key
   :type: str

   Default output key for answer: ``"result"``.

.. attribute:: top_k
   :type: int

   Number of results to return from the AQL query. Defaults to ``10``.

.. attribute:: return_aql_query
   :type: bool

   Whether to include the generated AQL query in the output. Defaults to ``False``.

.. attribute:: return_aql_result
   :type: bool

   Whether to include the raw AQL query results in the output. Defaults to ``False``.

.. attribute:: max_aql_generation_attempts
   :type: int

   Maximum retries for generating a valid AQL query. Defaults to ``3``.

.. attribute:: execute_aql_query
   :type: bool

   If ``False``, the AQL query is only explained (not executed). Defaults to ``True``.

.. attribute:: output_list_limit
   :type: int

   Limit on the number of list items to include in the response context. Defaults to ``32``.

.. attribute:: output_string_limit
   :type: int

   Limit on string length to include in the response context. Defaults to ``256``.

.. attribute:: force_read_only_query
   :type: bool

   If ``True``, raises an error if the generated AQL query includes write operations.

.. attribute:: allow_dangerous_requests
   :type: bool

   Required to be set ``True`` to acknowledge that write operations may be generated.

Methods
-------

.. method:: from_llm(llm, qa_prompt=AQL_QA_PROMPT, aql_generation_prompt=AQL_GENERATION_PROMPT, aql_fix_prompt=AQL_FIX_PROMPT, **kwargs)

   Create a new QA chain from a language model and default prompts.

   :param llm: A language model (e.g., ChatOpenAI).
   :type llm: BaseLanguageModel
   :param qa_prompt: Prompt template for QA step.
   :param aql_generation_prompt: Prompt template for AQL generation.
   :param aql_fix_prompt: Prompt template for AQL error correction.
   :return: An instance of ArangoGraphQAChain.

.. method:: _call(inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]

   Executes the QA chain: generates AQL, optionally retries on error, and returns an answer.

   :param inputs: Dictionary with key matching ``input_key`` (default: ``"query"``).
   :param run_manager: Optional callback manager.
   :return: Dictionary with key ``output_key`` (default: ``"result"``), and optionally ``aql_query`` and ``aql_result``.

.. method:: _is_read_only_query(aql_query: str) -> Tuple[bool, Optional[str]]

   Checks whether a generated AQL query contains any write operations.

   :param aql_query: The query string.
   :return: Tuple (True/False, operation name if found).

Usage Example
-------------

.. code-block:: python

    from langchain_openai import ChatOpenAI
    from langchain_arangodb.graphs import ArangoGraph
    from langchain_arangodb.chains.graph_qa import ArangoGraphQAChain

    llm = ChatOpenAI(model="gpt-4")
    graph = ArangoGraph.from_connection_args(
        username="readonly",
        password="password",
        db_name="test_db",
        host="localhost",
        port=8529,
        graph_name="my_graph"
    )

    qa_chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        return_aql_query=True,
        return_aql_result=True
    )

    query = "Who are the friends of Alice who live in San Francisco?"

    response = qa_chain.invoke({"query": query})

    print(response["result"])       # Natural language answer
    print(response["aql_query"])    # AQL query generated
    print(response["aql_result"])   # Raw AQL output

Security Considerations
-----------------------

- Always set `allow_dangerous_requests=True` explicitly if write permissions exist.
- Prefer read-only database roles when using this chain.
- Never expose this chain to arbitrary external inputs without sanitization.

API Reference
-------------

.. automodule:: langchain_arangodb.chains.graph_qa.arangodb
   :members: ArangoGraphQAChain
   :undoc-members:
   :show-inheritance:

References
----------

- `LangChain Graph QA Guide <https://python.langchain.com/docs/integrations/vectorstores/arangodb>`_
- `ArangoDB AQL Documentation <https://www.arangodb.com/docs/stable/aql/>`_


