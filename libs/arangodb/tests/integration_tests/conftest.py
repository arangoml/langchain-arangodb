import os

from arango import ArangoClient
import pytest

from tests.integration_tests.utils import ArangoCredentials

url = os.environ.get("ARANGODB_URI", "http://localhost:8529")
username = os.environ.get("ARANGODB_USERNAME", "root")
password = os.environ.get("ARANGODB_PASSWORD", "pleaseletmein")
os.environ["ARANGODB_URI"] = url
os.environ["ARANGODB_USERNAME"] = username
os.environ["ARANGODB_PASSWORD"] = password


@pytest.fixture
def clear_arangodb_database() -> None:
    client = ArangoClient(url)
    db = client.db(username=username, password=password, verify=True)

    for collection in db.collections():
        if not collection["system"]:
            db.delete_collection(collection["name"])

    for graph in db.graphs():
        db.delete_graph(graph["name"], drop_collections=True)

    client.close()


@pytest.fixture(scope="session")
def arangodb_credentials() -> ArangoCredentials:
    return {
        "url": url,
        "username": username,
        "password": password,
    }
