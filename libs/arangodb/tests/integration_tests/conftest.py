import os

import pytest
from arango import ArangoClient

from tests.integration_tests.utils import ArangoCredentials

url = os.environ.get("ARANGO_URL", "http://localhost:8529")
username = os.environ.get("ARANGO_USERNAME", "root")
password = os.environ.get("ARANGO_PASSWORD", "test")

@pytest.fixture
def clear_arangodb_database():
    client = ArangoClient(url)
    db = client.db(username=username, password=password, verify=True)

    for graph in db.graphs():  # type: ignore
        db.delete_graph(graph["name"], drop_collections=True)

    for collection in db.collections():  # type: ignore
        if not collection["system"]:
            db.delete_collection(collection["name"])

    client.close()


@pytest.fixture(scope="session")
def arangodb_credentials():
    return {
        "url": url,
        "username": username,
        "password": password,
    }

@pytest.fixture(scope="session")
def db(arangodb_credentials: ArangoCredentials):
    client = ArangoClient(arangodb_credentials["url"])
    db = client.db(username=arangodb_credentials["username"], password=arangodb_credentials["password"])
    yield db
    client.close()
