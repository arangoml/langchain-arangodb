from importlib import metadata

from langchain_arangodb.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_arangodb.chat_message_histories.arangodb import ArangoChatMessageHistory
from langchain_arangodb.graphs.arangodb_graph import ArangoGraph
from langchain_arangodb.vectorstores.arangodb_vector import ArangoVector

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ArangoGraphQAChain",
    "ArangoChatMessageHistory",
    "ArangoGraph",
    "ArangoVector",
    "__version__",
]
