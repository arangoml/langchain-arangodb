import json
import os
import re
from collections import defaultdict
from math import ceil
from typing import Any, DefaultDict, Dict, List, Optional, Set, Union

import farmhash
import yaml
from arango import ArangoClient
from arango.database import Database, StandardDatabase
from arango.graph import Graph
from langchain_core.embeddings import Embeddings

from langchain_arangodb.graphs.graph_document import (
    Document,
    GraphDocument,
    Node,
    Relationship,
)
from langchain_arangodb.graphs.graph_store import GraphStore

DEFAULT_ANALYZERS = {
    "text_fr",
    "text_pt",
    "identity",
    "text_de",
    "text_zh",
    "text_fi",
    "text_it",
    "text_no",
    "text_nl",
    "text_es",
    "text_ru",
    "text_en",
    "text_sv",
}


def get_arangodb_client(
    url: Optional[str] = None,
    dbname: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Any:
    """Get the Arango DB client from credentials.

    :param url: Arango DB url. Can be passed in as named arg or set as environment
        var ``ARANGODB_URL``. Defaults to "http://localhost:8529".
    :type url: str
    :param dbname: Arango DB name. Can be passed in as named arg or set as
        environment var ``ARANGODB_DBNAME``. Defaults to "_system".
    :type dbname: str
    :param username: Can be passed in as named arg or set as environment var
        ``ARANGODB_USERNAME``. Defaults to "root".
    :type username: str
    :param password: Can be passed in as named arg or set as environment var
        ``ARANGODB_PASSWORD``. Defaults to "".
    :type password: str

    :return: An arango.database.StandardDatabase.
    :rtype: Any
    :raises ArangoClientError: If the ArangoDB client cannot be created.
    :raises ArangoServerError: If the ArangoDB server cannot be reached.
    :raises ArangoCollectionError: If the collection cannot be created.
    """
    _url: str = url or str(os.environ.get("ARANGODB_URL", "http://localhost:8529"))
    _dbname: str = dbname or str(os.environ.get("ARANGODB_DBNAME", "_system"))
    _username: str = username or str(os.environ.get("ARANGODB_USERNAME", "root"))
    _password: str = password or str(os.environ.get("ARANGODB_PASSWORD", ""))

    return ArangoClient(_url).db(_dbname, _username, _password, verify=True)


class ArangoGraph(GraphStore):
    """ArangoDB wrapper for graph operations.

    :param db: The ArangoDB database instance.
    :type db: StandardDatabase
    :param generate_schema_on_init: Whether to generate the graph schema
        on initialization. Defaults to True.
    :type generate_schema_on_init: bool
    :param schema_sample_ratio: The ratio of documents/edges to sample in relation to
        the Collection size to generate each Collection Schema. If 0, one document/edge
        is used per Collection. Defaults to 0.
    :type schema_sample_ratio: float
    :param schema_graph_name: The name of an existing ArangoDB Graph to specifically
        use to generate the schema. If None, the entire database will be used.
        Defaults to None.
    :type schema_graph_name: Optional[str]
    :param schema_include_examples: Whether to include example values fetched from
        a sample documents as part of the schema. Defaults to True. Lists of size
        higher than **schema_list_limit** will be excluded from the schema, even if
        **schema_include_examples** is set to True. Defaults to True.
    :type schema_include_examples: bool
    :param schema_list_limit: The maximum list size the schema will include as part
        of the example values. If the list is longer than this limit, a string
        describing the list will be used in the schema instead. Default is 32.
    :type schema_list_limit: int
    :param schema_string_limit: The maximum number of characters to include
        in a string. If the string is longer than this limit, a string
        describing the string will be used in the schema instead. Default is 256.
    :type schema_string_limit: int
    :param schema_include_views: Whether to include ArangoDB Views and Analyzers as
        part of the schema passed to the AQL Generation prompt. Default is False.
    :type schema_include_views: bool
        :return: None
        :rtype: None
        :raises ArangoClientError: If the ArangoDB client cannot be created.
        :raises ArangoServerError: If the ArangoDB server cannot be reached.
        :raises ArangoCollectionError: If the collection cannot be created.


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

    def __init__(
        self,
        db: StandardDatabase,
        generate_schema_on_init: bool = True,
        schema_sample_ratio: float = 0,
        schema_graph_name: Optional[str] = None,
        schema_include_examples: bool = True,
        schema_list_limit: int = 32,
        schema_string_limit: int = 256,
        schema_include_views: bool = False,
    ) -> None:
        """
        Initializes the ArangoGraph instance.

        """
        self.__db: StandardDatabase = db
        self.__async_db = db.begin_async_execution()

        self.__schema = {}
        if generate_schema_on_init:
            self.__schema = self.generate_schema(
                schema_sample_ratio,
                schema_graph_name,
                schema_include_examples,
                schema_list_limit,
                schema_string_limit,
                schema_include_views,
            )

    @property
    def db(self) -> StandardDatabase:
        return self.__db

    @property
    def schema(self) -> Dict[str, Any]:
        """Returns the schema of the Graph Database as a structured object"""
        return self.__schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the schema of the Graph Database as a structured object"""
        return self.__schema

    @property
    def schema_json(self) -> str:
        """Returns the schema of the Graph Database as a JSON string

        :return: The schema of the Graph Database as a JSON string
        :rtype: str
        """
        return json.dumps(self.__schema)

    @property
    def schema_yaml(self) -> str:
        """Returns the schema of the Graph Database as a YAML string

        :return: The schema of the Graph Database as a YAML string
        :rtype: str
        """
        return yaml.dump(self.__schema, sort_keys=False)

    def set_schema(self, schema: Dict[str, Any]) -> None:
        """Sets a custom schema for the ArangoDB Database.

        :param schema: The schema to set.
        :type schema: Dict[str, Any]
        :return: None
        :rtype: None
        """
        self.__schema = schema

    def refresh_schema(
        self,
        sample_ratio: float = 0,
        graph_name: Optional[str] = None,
        include_examples: bool = True,
        list_limit: int = 32,
    ) -> None:
        """
        Refresh the graph schema information.

        Parameters:

        :param sample_ratio: A float (0 to 1) to determine the
            ratio of documents/edges sampled in relation to the Collection size
            to generate each Collection Schema. If 0, one document/edge
            is used per Collection. Defaults to 0.
        :type sample_ratio: float
        :param graph_name: The name of an existing ArangoDB Graph to specifically
            use to generate the schema. If None, the entire database will be used.
            Defaults to None.
        :type graph_name: Optional[str]
        :param include_examples: Whether to include example values fetched from
            a sample documents as part of the schema. Defaults to True. Lists of size
            higher than **list_limit** will be excluded from the schema, even if
            **schema_include_examples** is set to True. Defaults to True.
        :type include_examples: bool
        :param list_limit: The maximum list size the schema will include as part
            of the example values. If the list is longer than this limit, a string
            describing the list will be used in the schema instead. Default is 32.
        :type list_limit: int

        :return: None
        :rtype: None
        :raises ArangoClientError: If the ArangoDB client cannot be created.
        :raises ArangoServerError: If the ArangoDB server cannot be reached.
        :raises ArangoCollectionError: If the collection cannot be created.
        """
        self.__schema = self.generate_schema(
            sample_ratio, graph_name, include_examples, list_limit
        )

    def generate_schema(
        self,
        sample_ratio: float = 0,
        graph_name: Optional[str] = None,
        include_examples: bool = True,
        list_limit: int = 32,
        schema_string_limit: int = 256,
        schema_include_views: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates the schema of the ArangoDB Database and returns it

        :param sample_ratio: A ratio (0 to 1) to determine the
            ratio of documents/edges used (in relation to the Collection size)
            to render each Collection Schema. If 0, one document/edge
            is used per Collection.
        :type sample_ratio: float
        :param graph_name: The name of the graph to use to generate the schema. If
            None, the entire database will be used.
        :type graph_name: Optional[str]
        :param include_examples: A flag whether to scan the database for
            example values and use them in the graph schema. Default is True.
        :type include_examples: bool
        :param list_limit: The maximum number of elements to include in a list.
            If the list is longer than this limit, a string describing the list
            will be used in the schema instead. Default is 32.
        :type list_limit: int
        :param schema_string_limit: The maximum number of characters to include
            in a string. If the string is longer than this limit, a string
            describing the string will be used in the schema instead. Default is 128.
        :type schema_string_limit: int
        :param schema_include_views: Whether to include ArangoDB Views and Analyzers as
            part of the schema passed to the AQL Generation prompt. Default is False.
        :type schema_include_views: bool
        :return: A dictionary containing the graph schema and collection schema.
        :rtype: Dict[str, List[Dict[str, Any]]]
        :raises ValueError: If the sample ratio is not between 0 and 1.
        :raises ArangoClientError: If the ArangoDB client cannot be created.
        :raises ArangoServerError: If the ArangoDB server cannot be reached.
        :raises ArangoCollectionError: If the collection cannot be created.
        """
        if not 0 <= sample_ratio <= 1:
            raise ValueError("**sample_ratio** value must be in between 0 to 1")

        #####
        # Step 1: Generate Graph Schema
        ####

        graph_schema: List[Dict[str, Any]] = []
        if graph_name:
            # Fetch a single graph
            graph: Graph = self.db.graph(graph_name)
            edge_definitions = graph.edge_definitions()

            graph_schema = [{"name": graph_name, "edge_definitions": edge_definitions}]

            # Fetch graph-specific collections
            collection_names = set(graph.vertex_collections())  # type: ignore
            for edge_definition in edge_definitions:  # type: ignore
                collection_names.add(edge_definition["edge_collection"])

        else:
            # Fetch all graphs
            graph_schema = [
                {"graph_name": g["name"], "edge_definitions": g["edge_definitions"]}
                for g in self.db.graphs()  # type: ignore
            ]

            # Fetch all collections
            collection_names = {
                collection["name"]
                for collection in self.db.collections()  # type: ignore
            }

        #####
        # Step 2: Generate Collection Schema
        ####

        # Stores the schema of every ArangoDB Document/Edge collection
        collection_schema: List[Dict[str, Any]] = []
        for collection in self.db.collections():  # type: ignore
            if collection["system"] or collection["name"] not in collection_names:
                continue

            # Extract collection name, type, and size
            col_name: str = collection["name"]
            col_type: str = collection["type"]
            col_size: int = self.db.collection(col_name).count()  # type: ignore

            # Set number of ArangoDB documents/edges to retrieve
            limit_amount = ceil(sample_ratio * col_size) or 1

            aql = f"""
                FOR doc in @@col_name
                    LIMIT {limit_amount}
                    RETURN doc
            """

            doc: Dict[str, Any]
            properties: List[Dict[str, str]] = []
            cursor = self.db.aql.execute(aql, bind_vars={"@col_name": col_name})
            for doc in cursor:  # type: ignore
                for key, value in doc.items():
                    properties.append({key: type(value).__name__})

            collection_schema_entry = {
                "name": col_name,
                "type": col_type,
                "size": col_size,
                "properties": properties,
            }

            if include_examples and col_size > 0:
                collection_schema_entry["example"] = self._sanitize_input(
                    doc, list_limit, schema_string_limit
                )

            collection_schema.append(collection_schema_entry)

        if not schema_include_views:
            return {
                "graph_schema": graph_schema,
                "collection_schema": collection_schema,
            }

        #####
        # Step 3: Generate View Schema
        #####

        view_schema: List[Dict[str, Any]] = []
        for view in self.db.views():  # type: ignore
            view_name = view["name"]
            view_type = view["type"]
            view_info = self.db.view(view_name)
            key = "links" if view_type == "arangosearch" else "indexes"
            view_schema.append(
                {"name": view_name, "type": view_type, key: view_info.get(key, [])}  # type: ignore
            )

        #####
        # Step 4: Generate Analyzer Schema
        #####

        analyzer_schema: List[Dict[str, Any]] = []
        for a in self.db.analyzers():  # type: ignore
            if a["name"] not in DEFAULT_ANALYZERS:
                analyzer_schema.append({a["name"]: a["properties"]})

        return {
            "graph_schema": graph_schema,
            "collection_schema": collection_schema,
            "view_schema": view_schema,
            "analyzer_schema": analyzer_schema,
        }

    def query(
        self,
        query: str,
        params: dict = {},
    ) -> List[Any]:
        """
        Execute an AQL query and return the results.

        :param query: The AQL query to execute.
        :type query: str
        :param params: Additional arguments piped to the function.
            Defaults to None.
        :type params: dict
        :param list_limit: Removes lists above **list_limit** size
            that have been returned from the AQL query.
        :type list_limit: Optional[int]
        :param string_limit: Removes strings above **string_limit** size
            that have been returned from the AQL query.
        :type string_limit: Optional[int]
        :param remaining_params: Remaining params are passed to the AQL query execution.
            Defaults to None.
        :type remaining_params: Optional[dict]

        :return: A list of dictionaries containing the query results.
        :rtype: List[Any]
        :raises ArangoClientError: If the ArangoDB client cannot be created.
        :raises ArangoServerError: If the ArangoDB server cannot be reached.
        :raises ArangoCollectionError: If the collection cannot be created.
        """
        top_k = params.pop("top_k", None)
        list_limit = params.pop("list_limit", 32)
        string_limit = params.pop("string_limit", 256)
        cursor = self.db.aql.execute(query, **params)

        results = []

        i = 0
        for doc in cursor:  # type: ignore
            results.append(self._sanitize_input(doc, list_limit, string_limit))

            if top_k and i >= top_k:
                break

            i += 1

        return results

    def explain(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """
        Explain an AQL query without executing it.

        :param query: The AQL query to explain.
        :type query: str
        :param params: Additional arguments piped to the function.
            Defaults to None.
        :type params: dict
        :return: A list of dictionaries containing the query explanation.
        :rtype: List[Dict[str, Any]]
        :raises ArangoClientError: If the ArangoDB client cannot be created.
        :raises ArangoServerError: If the ArangoDB server cannot be reached.
        :raises ArangoCollectionError: If the collection cannot be created.
        """
        return self.db.aql.explain(query)  # type: ignore

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        graph_name: Optional[str] = None,
        update_graph_definition_if_exists: bool = False,
        batch_size: int = 1000,
        use_one_entity_collection: bool = True,
        insert_async: bool = False,
        source_collection_name: Union[str, None] = None,
        source_edge_collection_name: Union[str, None] = None,
        entity_collection_name: Union[str, None] = None,
        entity_edge_collection_name: Union[str, None] = None,
        embeddings: Union[Embeddings, None] = None,
        embedding_field: str = "embedding",
        embed_source: bool = False,
        embed_nodes: bool = False,
        embed_relationships: bool = False,
        capitalization_strategy: str = "none",
    ) -> None:
        """
        Constructs nodes & relationships in the graph based on the
        provided GraphDocument objects.

        :param graph_documents: The GraphDocument objects to add to the graph.
        :type graph_documents: List[GraphDocument]
        :param include_source: Whether to include the source document in the graph.
        :type include_source: bool
        :param graph_name: The name of the graph to add the documents to.
        :type graph_name: Optional[str]
        :param update_graph_definition_if_exists: Whether to update the graph definition
            if it already exists.
        :type update_graph_definition_if_exists: bool
        :param batch_size: The number of documents to process in each batch.
        :type batch_size: int
        :param use_one_entity_collection: Whether to use one entity collection
            for all nodes.
        :type use_one_entity_collection: bool
        :param insert_async: Whether to insert the documents asynchronously.
        :type insert_async: bool
        :param source_collection_name: The name of the source collection.
        :type source_collection_name: Union[str, None]
        :param source_edge_collection_name: The name of the source edge collection.
        :type source_edge_collection_name: Union[str, None]
        :param entity_collection_name: The name of the entity collection.
        :type entity_collection_name: Union[str, None]
        :param entity_edge_collection_name: The name of the entity edge collection.
        :type entity_edge_collection_name: Union[str, None]
        :param embeddings: The embeddings model to use.
        :type embeddings: Union[Embeddings, None]
        :param embedding_field: The field to use for the embedding.
        :type embedding_field: str
        :param embed_source: Whether to embed the source document.
        :type embed_source: bool
        :param embed_nodes: Whether to embed the nodes.
        :type embed_nodes: bool
        :param embed_relationships: Whether to embed the relationships.
        :type embed_relationships: bool
        :param capitalization_strategy: The capitalization strategy to use.
        :type capitalization_strategy: str

        :return: None
        :rtype: None
        :raises ValueError: If the capitalization strategy is not 'lower',
            'upper', or 'none'.
        :raises ArangoClientError: If the ArangoDB client cannot be created.
        :raises ArangoServerError: If the ArangoDB server cannot be reached.
        :raises ArangoCollectionError: If the collection cannot be created.

        """
        if not graph_documents:
            return

        if not embeddings and (embed_source or embed_nodes or embed_relationships):
            m = "**embedding** is required to embed source, nodes, or relationships."
            raise ValueError(m)

        def embed_text(text: str) -> list[float]:
            if not embeddings:
                raise ValueError("**embedding** is required to embed text.")

            res: Any = embeddings.embed_documents([text])[0]

            while type(res) is list:
                if type(res[0]) is float:
                    break

                res = res[0]

            return res

        if capitalization_strategy == "none":
            capitalization_fn = lambda x: x  # noqa: E731
        elif capitalization_strategy == "lower":
            capitalization_fn = str.lower
        elif capitalization_strategy == "upper":
            capitalization_fn = str.upper
        else:
            m = "**capitalization_strategy** must be 'lower', 'upper', or 'none'."
            raise ValueError(m)

        #########
        # Setup #
        #########

        suffix = f"{graph_name}_" if graph_name else ""
        if not source_collection_name:
            source_collection_name = f"{suffix}SOURCE"
        if not source_edge_collection_name:
            source_edge_collection_name = f"{suffix}HAS_SOURCE"
        if not entity_collection_name:
            entity_collection_name = f"{suffix}ENTITY"
        if not entity_edge_collection_name:
            entity_edge_collection_name = f"{suffix}LINKS_TO"

        insertion_db = self.__async_db if insert_async else self.__db
        nodes: DefaultDict[str, list[dict[str, Any]]] = defaultdict(list)
        edges: DefaultDict[str, list[dict[str, Any]]] = defaultdict(list)
        edge_definitions_dict: DefaultDict[str, DefaultDict[str, Set[str]]] = (
            defaultdict(lambda: defaultdict(set))
        )

        if include_source:
            self._create_collection(source_collection_name)
            self._create_collection(source_edge_collection_name, is_edge=True)

            from_cols = {entity_collection_name} if use_one_entity_collection else set()

            edge_definitions_dict[source_edge_collection_name][
                "from_vertex_collections"
            ] = from_cols
            edge_definitions_dict[source_edge_collection_name][
                "to_vertex_collections"
            ] = {source_collection_name}

        if use_one_entity_collection:
            self._create_collection(entity_collection_name)
            self._create_collection(entity_edge_collection_name, is_edge=True)

            edge_definitions_dict[entity_edge_collection_name][
                "from_vertex_collections"
            ] = {entity_collection_name}
            edge_definitions_dict[entity_edge_collection_name][
                "to_vertex_collections"
            ] = {entity_collection_name}

        process_node_fn = (
            self._process_node_as_entity
            if use_one_entity_collection
            else self._process_node_as_type
        )

        process_edge_fn = (
            self._process_edge_as_entity
            if use_one_entity_collection
            else self._process_edge_as_type
        )

        #######################
        # Document Processing #
        #######################

        for document in graph_documents:
            source_id_hash = None

            # 1. Process Source Document
            if include_source:
                if not document.source:
                    raise ValueError("Source document is required.")

                source_embedding = (
                    embed_text(document.source.page_content) if embed_source else None
                )

                source_id_hash = self._process_source(
                    document.source,
                    source_collection_name,
                    source_embedding,
                    embedding_field,
                    insertion_db,
                )

            # 2. Process Nodes
            node_key_map = {}
            for i, node in enumerate(document.nodes, 1):
                node.id = str(capitalization_fn(str(node.id)))
                node_key = self._hash(node.id)
                node_key_map[node.id] = node_key

                if embed_nodes:
                    node.properties[embedding_field] = embed_text(node.id)

                node_type = process_node_fn(
                    node_key, node, nodes, entity_collection_name
                )

                # 2.1 Link Source Document to Node
                if include_source:
                    edges[source_edge_collection_name].append(
                        {
                            "_from": f"{node_type}/{node_key}",
                            "_to": f"{source_collection_name}/{source_id_hash}",
                        }
                    )

                    if not use_one_entity_collection:
                        edge_definitions_dict[source_edge_collection_name][
                            "from_vertex_collections"
                        ].add(node_type)

                # 2.2 Batch Insert
                if i % batch_size == 0:
                    self._import_data(insertion_db, nodes, is_edge=False)
                    self._import_data(insertion_db, edges, is_edge=True)

            self._import_data(insertion_db, nodes, is_edge=False)
            self._import_data(insertion_db, edges, is_edge=True)

            # 3. Process Edges
            for i, edge in enumerate(document.relationships, 1):
                source, target = edge.source, edge.target
                source.id = capitalization_fn(str(source.id))
                target.id = capitalization_fn(str(target.id))

                edge_str = f"{source.id} {edge.type} {target.id}"

                if embed_relationships:
                    edge.properties[embedding_field] = embed_text(edge_str)

                if include_source:
                    edge.properties["source_id"] = source_id_hash

                source_key = self._get_node_key(
                    source,
                    nodes,
                    node_key_map,
                    entity_collection_name,
                    process_node_fn,
                )

                target_key = self._get_node_key(
                    target,
                    nodes,
                    node_key_map,
                    entity_collection_name,
                    process_node_fn,
                )

                edge_key = self._hash(edge_str)
                process_edge_fn(
                    edge,
                    edge_str,
                    edge_key,
                    source_key,
                    target_key,
                    edges,
                    entity_collection_name,
                    entity_edge_collection_name,
                    edge_definitions_dict,
                )

                # 3.1 Batch Insert
                if i % batch_size == 0:
                    self._import_data(insertion_db, edges, is_edge=True)
                    self._import_data(insertion_db, nodes, is_edge=False)

            self._import_data(insertion_db, edges, is_edge=True)
            self._import_data(insertion_db, nodes, is_edge=False)

        ##################
        # Graph Creation #
        ##################

        if graph_name:
            edge_definitions: list[dict[str, Union[str, list[str]]]] = [
                {
                    "edge_collection": k,
                    "from_vertex_collections": list(v["from_vertex_collections"]),
                    "to_vertex_collections": list(v["to_vertex_collections"]),
                }
                for k, v in edge_definitions_dict.items()
            ]

            if not self.db.has_graph(graph_name):
                self.db.create_graph(graph_name, edge_definitions)

            elif update_graph_definition_if_exists:
                graph = self.db.graph(graph_name)

                for e_d in edge_definitions:
                    e_c = str(e_d["edge_collection"])
                    if not graph.has_edge_definition(e_c):
                        graph.create_edge_definition(**e_d)  # type: ignore
                    else:
                        graph.replace_edge_definition(**e_d)  # type: ignore

        # Refresh schema after insertions
        self.refresh_schema()

    @classmethod
    def from_db_credentials(
        cls,
        url: Optional[str] = None,
        dbname: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Any:
        """Convenience constructor that builds Arango DB from credentials.

        :param url: Arango DB url. Can be passed in as named arg or set as environment
                var ``ARANGODB_URL``. Defaults to "http://localhost:8529".
        :type url: str
        :param dbname: Arango DB name. Can be passed in as named arg or set as
                environment var ``ARANGODB_DBNAME``. Defaults to "_system".
        :type dbname: str
        :param username: Can be passed in as named arg or set as environment var
                ``ARANGODB_USERNAME``. Defaults to "root".
        :type username: str
        :param password: Can be passed in as named arg or set as environment var
                ``ARANGODB_USERNAME``. Defaults to "root".
        :type  password: str

        :return: An arango.database.StandardDatabase.
        :rtype: Any
        :raises ArangoClientError: If the ArangoDB client cannot be created.
        :raises ArangoServerError: If the ArangoDB server cannot be reached.

        """
        db = get_arangodb_client(
            url=url, dbname=dbname, username=username, password=password
        )
        return cls(db)

    def _import_data(
        self,
        db: Database,
        data: Dict[str, List[Dict[str, Any]]],
        is_edge: bool,
    ) -> None:
        """Imports data into the ArangoDB database in bulk.

        :param db: The ArangoDB database instance.
        :type db: Database
        :param data: The data to import.
        :type data: Dict[str, List[Dict[str, Any]]]
        :param is_edge: Whether the data is an edge.
        :type is_edge: bool

        :return: None
        :rtype: None
        :raises ArangoClientError: If the ArangoDB client cannot be created.
        :raises ArangoServerError: If the ArangoDB server cannot be reached.
        :raises ArangoCollectionError: If the collection cannot be created.
        """
        for collection, batch in data.items():
            self._create_collection(collection, is_edge)
            db.collection(collection).import_bulk(batch, on_duplicate="update")

        data.clear()

    def _create_collection(
        self, collection_name: str, is_edge: bool = False, **kwargs: Any
    ) -> None:
        """Creates a collection in the ArangoDB database if it does not exist.

        :param collection_name: The name of the collection to create.
        :type collection_name: str
        :param is_edge: Whether the collection is an edge.
        :type is_edge: bool

        :return: None
        :rtype: None
        :raises ArangoClientError: If the ArangoDB client cannot be created.
        :raises ArangoServerError: If the ArangoDB server cannot be reached.
        :raises ArangoCollectionError: If the collection cannot be created.
        """
        if not self.db.has_collection(collection_name):
            self.db.create_collection(collection_name, edge=is_edge, **kwargs)

    def _process_node_as_entity(
        self,
        node_key: str,
        node: Node,
        nodes: DefaultDict[str, list],
        entity_collection_name: str,
    ) -> str:
        """Processes a Graph Document Node into ArangoDB as a unanimous Entity.

        :param node_key: The key of the node.
        :type node_key: str
        :param node: The node to process.
        :type node: Node
        :param nodes: The nodes to process.
        :type nodes: DefaultDict[str, list]
        :param entity_collection_name: The name of the entity collection.
        :type entity_collection_name: str

        :return: The name of the entity collection.
        :rtype: str
        """
        nodes[entity_collection_name].append(
            {
                "_key": node_key,
                "text": node.id,
                "type": node.type,
                **node.properties,
            }
        )
        return entity_collection_name

    def _process_node_as_type(
        self, node_key: str, node: Node, nodes: DefaultDict[str, list], _: str
    ) -> str:
        """Processes a Graph Document Node into ArangoDB based on its Node Type.

        :param node_key: The key of the node.
        :type node_key: str
        :param node: The node to process.
        :type node: Node
        :param nodes: The nodes to process.
        :type nodes: DefaultDict[str, list]
        :param _: The name of the node type.
        :type _: str

        :return: The name of the node type.
        :rtype: str
        """
        node_type = self._sanitize_collection_name(node.type)
        nodes[node_type].append({"_key": node_key, "text": node.id, **node.properties})
        return node_type

    def _process_edge_as_entity(
        self,
        edge: Relationship,
        edge_str: str,
        edge_key: str,
        source_key: str,
        target_key: str,
        edges: DefaultDict[str, list],
        entity_collection_name: str,
        entity_edge_collection_name: str,
        _: DefaultDict[str, DefaultDict[str, set[str]]],
    ) -> None:
        """Processes a Graph Document Edge into ArangoDB as a unanimous Entity.

        :param edge: The edge to process.
        :type edge: Relationship
        :param edge_str: The string representation of the edge.
        :type edge_str: str
        :param edge_key: The key of the edge.
        :type edge_key: str
        :param source_key: The key of the source node.
        :type source_key: str
        :param target_key: The key of the target node.
        :type target_key: str
        :param edges: The edges to process.
        :type edges: DefaultDict[str, list]
        :param entity_collection_name: The name of the entity collection.
        :type entity_collection_name: str
        :param entity_edge_collection_name: The name of the entity edge collection.
        :type entity_edge_collection_name: str
        :param _: The name of the edge type.
        :type _: DefaultDict[str, DefaultDict[str, set[str]]]

        :return: None
        :rtype: None
        :raises ArangoClientError: If the ArangoDB client cannot be created.
        :raises ArangoServerError: If the ArangoDB server cannot be reached.
        :raises ArangoCollectionError: If the collection cannot be created.

        """
        edges[entity_edge_collection_name].append(
            {
                "_key": edge_key,
                "_from": f"{entity_collection_name}/{source_key}",
                "_to": f"{entity_collection_name}/{target_key}",
                "type": edge.type,
                "text": edge_str,
                **edge.properties,
            }
        )

    def _process_edge_as_type(
        self,
        edge: Relationship,
        edge_str: str,
        edge_key: str,
        source_key: str,
        target_key: str,
        edges: DefaultDict[str, list],
        _1: str,
        _2: str,
        edge_definitions_dict: DefaultDict[str, DefaultDict[str, set[str]]],
    ) -> None:
        """Processes a Graph Document Edge into ArangoDB based on its Edge Type.

        :param edge: The edge to process.
        :type edge: Relationship
        :param edge_str: The string representation of the edge.
        :type edge_str: str
        :param edge_key: The key of the edge.
        :type edge_key: str
        :param source_key: The key of the source node.
        :type source_key: str
        :param target_key: The key of the target node.
        :type target_key: str
        :param edges: The edges to process.
        :type edges: DefaultDict[str, list]
        :param edge_definitions_dict: The edge definitions dictionary.
        :type edge_definitions_dict: DefaultDict[str, DefaultDict[str, set[str]]]

        :return: None
        :rtype: None
        :raises ArangoClientError: If the ArangoDB client cannot be created.
        :raises ArangoServerError: If the ArangoDB server cannot be reached.
        :raises ArangoCollectionError: If the collection cannot be created.
        """
        source: Node = edge.source
        target: Node = edge.target

        edge_type = self._sanitize_collection_name(edge.type)
        source_type = self._sanitize_collection_name(source.type)
        target_type = self._sanitize_collection_name(target.type)

        edge_definitions_dict[edge_type]["from_vertex_collections"].add(source_type)
        edge_definitions_dict[edge_type]["to_vertex_collections"].add(target_type)

        edges[edge_type].append(
            {
                "_key": edge_key,
                "_from": f"{source_type}/{source_key}",
                "_to": f"{target_type}/{target_key}",
                "text": edge_str,
                **edge.properties,
            }
        )

    def _get_node_key(
        self,
        node: Node,
        nodes: DefaultDict[str, list],
        node_key_map: Dict[str, str],
        entity_collection_name: str,
        process_node_fn: Any,
    ) -> str:
        """Gets the key of a node and processes it if it doesn't exist.

        :param node: The node to process.
        :type node: Node
        :param nodes: The nodes to process.
        :type nodes: DefaultDict[str, list]
        :param node_key_map: The node key map.
        :type node_key_map: Dict[str, str]
        :param entity_collection_name: The name of the entity collection.
        :type entity_collection_name: str
        :param process_node_fn: The function to process the node.
        :type process_node_fn: Any

        :return: The key of the node.
        :rtype: str
        :raises ArangoClientError: If the ArangoDB client cannot be created.
        :raises ArangoServerError: If the ArangoDB server cannot be reached.
        :raises ArangoCollectionError: If the collection cannot be created.
        """
        node.id = str(node.id)
        if node.id in node_key_map:
            return node_key_map[node.id]

        node_key = self._hash(node.id)
        node_key_map[node.id] = node_key
        process_node_fn(node_key, node, nodes, entity_collection_name)

        return node_key

    def _process_source(
        self,
        source: Document,
        source_collection_name: str,
        source_embedding: Union[list[float], None],
        embedding_field: str,
        insertion_db: Database,
    ) -> str:
        """Processes a Graph Document Source into ArangoDB.

        :param source: The source to process.
        :type source: Document
        :param source_collection_name: The name of the source collection.
        :type source_collection_name: str
        :param source_embedding: The embedding of the source.
        :type source_embedding: Union[list[float], None]
        :param embedding_field: The field name to store the embedding.
        :type embedding_field: str
        :param insertion_db: The database to insert the source into.
        :type insertion_db: Database

        :return: The key of the source.
        :rtype: str
        :raises ArangoClientError: If the ArangoDB client cannot be created.
        :raises ArangoServerError: If the ArangoDB server cannot be reached.
        :raises ArangoCollectionError: If the collection cannot be created.
        """
        source_id = self._hash(
            source.id if source.id else source.page_content.encode("utf-8")
        )

        doc = {
            **source.metadata,
            "_key": source_id,
            "text": source.page_content,
            "type": source.type,
        }

        if source_embedding:
            doc[embedding_field] = source_embedding

        insertion_db.collection(source_collection_name).insert(doc, overwrite=True)

        return source_id

    def _hash(self, value: Any) -> str:
        """Applies the Farmhash hash function to a value.

        :param value: The value to hash.
        :type value: Any

        :return: The hashed value.
        :rtype: str
        :raises ValueError: If the value is not a string or has no
            string representation.
        """
        try:
            value_str = str(value)
        except Exception:
            raise ValueError("Value must be a string or have a string representation.")

        return str(farmhash.Fingerprint64(value_str))  # type: ignore

    def _sanitize_collection_name(self, name: str) -> str:
        """
        Modifies a string to adhere to ArangoDB collection name rules.

        - Trims the name to 256 characters if it's too long.
        - Replaces invalid characters with underscores (_).
        - Ensures the name starts with a letter (prepends 'a' if needed).

        :param name: The name to sanitize.
        :type name: str

        :return: The sanitized name.
        :rtype: str
        :raises ValueError: If the collection name is empty.
        """
        if not name:
            raise ValueError("Collection name cannot be empty.")

        name = name[:256]

        # Replace invalid characters with underscores
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        # Ensure the name starts with a letter; prepend 'a' if not
        if not re.match(r"^[a-zA-Z]", name):
            name = f"Collection_{name}"

        return name

    def _sanitize_input(self, d: Any, list_limit: int, string_limit: int) -> Any:
        """Sanitize the input dictionary or list.

        Sanitizes the input by removing embedding-like values,
        lists with more than **list_limit** elements, that are mostly irrelevant for
        generating answers in a LLM context. These properties, if left in
        results, can occupy significant context space and detract from
        the LLM's performance by introducing unnecessary noise and cost.

        :param d: The input dictionary or list to sanitize.
        :type d: Any
        :param list_limit: The limit for the number of elements in a list.
        :type list_limit: int
        :param string_limit: The limit for the number of characters in a string.
        :type string_limit: int

        :return: The sanitized dictionary or list.
        :rtype: Any
        :raises ValueError: If the input is not a dictionary or list.
        :raises ValueError: If the list limit is less than 0.
        :raises ValueError: If the string limit is less than 0.

        """

        if isinstance(d, dict):
            new_dict = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    sanitized_value = self._sanitize_input(
                        value, list_limit, string_limit
                    )
                    if sanitized_value is not None:
                        new_dict[key] = sanitized_value
                elif isinstance(value, list):
                    if len(value) < list_limit:
                        sanitized_value = self._sanitize_input(
                            value, list_limit, string_limit
                        )
                        if sanitized_value is not None:
                            new_dict[key] = sanitized_value
                    else:
                        new_dict[key] = (
                            f"List of {len(value)} elements of type {type(value[0])}"
                        )
                elif isinstance(value, str):
                    if len(value) > string_limit:
                        new_dict[key] = f"String of {len(value)} characters"
                    else:
                        new_dict[key] = value
                else:
                    new_dict[key] = value
            return new_dict
        elif isinstance(d, list):
            if len(d) == 0:
                return d
            elif len(d) < list_limit:
                arr = []
                for item in d:
                    sanitized_item = self._sanitize_input(
                        item, list_limit, string_limit
                    )
                    if sanitized_item is not None:
                        arr.append(sanitized_item)
                return arr
            else:
                return f"List of {len(d)} elements of type {type(d[0])}"
        else:
            return d
