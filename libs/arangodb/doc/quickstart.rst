Quickstart
==========

1. Set up ArangoDB
2. Set environment variables
3. Instantiate a NetworkX-ArangoDB Graph

1. Set up ArangoDB
------------------

**Option A: Local Instance via Docker**

Appears on ``localhost:8529`` with the user ``root`` & password ``openSesame``.

More info: `arangodb.com/download-major <https://arangodb.com/download-major/>`_.

.. code-block:: bash

    docker run -e ARANGO_ROOT_PASSWORD=openSesame -p 8529:8529 arangodb/arangodb

**Option B: ArangoDB Cloud Trial**

`ArangoGraph <https://dashboard.arangodb.cloud/home>`_ is ArangoDB's Cloud offering to use ArangoDB as a managed service.

A 14-day trial is available upon sign up.

**Option C: Temporary Cloud Instance via Python**

A temporary cloud database can be provisioned using the `adb-cloud-connector <https://github.com/arangodb/adb-cloud-connector?tab=readme-ov-file#arangodb-cloud-connector>`_ Python package.


.. code-block:: bash

    pip install adb-cloud-connector

.. code-block:: python

    from adb_cloud_connector import get_temp_credentials

    credentials = get_temp_credentials()

    print(credentials)

2. Set environment variables
----------------------------

Set up your LLM Environment Variables:

.. code-block:: bash

    export OPENAI_API_KEY=sk-proj-....

Or via python:

.. code-block:: python

    import os
    os.environ["OPENAI_API_KEY"] = "sk-proj-...."

3. Instantiate an ArangoGraph
----------------------------------------

4. Instantiate an ArangoGraphQAChain
----------------------------------------

5. Instantiate a VectorStore
----------------------------------------

...