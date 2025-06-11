# 🦜️🔗🥑 LangChain ArangoDB

[![CI](https://github.com/arangoml/langchain-arangodb/actions/workflows/check_diffs.yml/badge.svg?branch=main)](https://github.com/arangoml/langchain-arangodb/actions/workflows/check_diffs.yml)
[![Docs](https://readthedocs.org/projects/langchain-arangodb/badge/?version=latest)](https://langchain-arangodb.readthedocs.io/en/latest/?badge=latest)

[![PyPI version badge](https://img.shields.io/pypi/v/langchain-arangodb?color=3775A9&style=for-the-badge&logo=pypi&logoColor=FFD43B)](https://pypi.org/project/langchain-arangodb/)
[![Python versions badge](https://img.shields.io/pypi/pyversions/langchain-arangodb?color=3776AB&style=for-the-badge&logo=python&logoColor=FFD43B)](https://pypi.org/project/langchain-arangodb/)

[![License](https://img.shields.io/github/license/arangoml/langchain-arangodb?color=9E2165&style=for-the-badge)](https://github.com/arangoml/langchain-arangodb/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/static/v1?style=for-the-badge&label=code%20style&message=black&color=black)](https://github.com/psf/black)
[![Downloads](https://img.shields.io/pepy/dt/langchain-arangodb?style=for-the-badge&color=282661
)](https://pepy.tech/project/langchain-arangodb)

This package contains the ArangoDB integration for LangChain.

## 📦 Installation

```bash
pip install -U langchain-arangodb
```

## 💻 Examples

### ArangoGraph

The `ArangoGraph` class is a wrapper around ArangoDB's Python driver.
It provides a simple interface for interacting with an ArangoDB database.

```python
from arango import ArangoClient
from langchain_arangodb import ArangoGraph

db = ArangoClient(hosts="http://localhost:8529").db(username="root", password="password")

graph = ArangoGraph(db)

graph.query("RETURN 'hello world'")
```

### ArangoChatMessageHistory

The `ArangoChatMessageHistory` class is used to store chat message history in an ArangoDB database.
It stores messages as nodes and creates relationships between them, allowing for easy querying of the conversation history.

```python
from arango import ArangoClient
from langchain_arangodb import ArangoChatMessageHistory

db = ArangoClient(hosts="http://localhost:8529").db(username="root", password="password")

history = ArangoChatMessageHistory(db=db, session_id="session_id_1")
history.add_user_message("hi!")
history.add_ai_message("whats up?")

print(history.messages)
```

### ArangoVector

The `ArangoVector` class provides functionality for managing an ArangoDB Vector Store. It enables you to create new vector indexes, add vectors to existing indexes, and perform queries on indexes.

```python
from arango import ArangoClient

from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_arangodb import ArangoVector

# Create a vector store from some documents and embeddings
docs = [
    Document(
        page_content=(
            "LangChain is a framework to build "
            "with LLMs by chaining interoperable components."
        ),
    )
]
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key="sk-...",  # Replace with your OpenAI API key
)

db = ArangoClient(hosts="http://localhost:8529").db(username="root", password="password")

vector_db = ArangoVector.from_documents(
    docs,
    embeddings,
    database=db,
)

# Query the vector store for similar documents
docs_with_score = vector_db.similarity_search_with_score("What is LangChain?", k=1)
```

### ArangoGraphQAChain

The `ArangoGraphQAChain` class enables natural language interactions with an ArangoDB database.
It uses an LLM and the database's schema to translate a user's question into an AQL query, which is executed against the database.
The resulting data is then sent along with the user's question to the LLM to generate a natural language response.

```python
from arango import ArangoClient

from langchain_openai import ChatOpenAI
from langchain_arangodb import ArangoGraph, ArangoGraphQAChain

llm = ChatOpenAI(
    temperature=0,
    api_key="sk-...",  # Replace with your OpenAI API key
)

db = ArangoClient(hosts="http://localhost:8529").db(username="root", password="password")

graph = ArangoGraph(db)

chain = ArangoGraphQAChain.from_llm(
    llm=llm, graph=graph, allow_dangerous_requests=True
)

chain.run("Who starred in Top Gun?")
```

## 🧪 Tests

Install the test dependencies to run the tests:

```bash
poetry install --with test,test_integration
```

### Unit Tests

Run the unit tests using:

```bash
make tests
```

### Integration Tests

1. Start the ArangoDB instance using Docker:

    ```bash
    cd tests/integration_tests/docker-compose
    docker-compose -f arangodb.yml up
    ```

2. Run the tests:

    ```bash
    make integration_tests
    ```

## 🧹 Code Formatting and Linting

Install the codespell, lint, and typing dependencies to lint and format your code:

```bash
poetry install --with codespell,lint,typing
```

To format your code, run:

```bash
make format
```

To lint it, run:

```bash
make lint
```
