from arango import ArangoClient
from arango_datasets import Datasets
from langchain_arangodb import ArangoGraphQAChain, ArangoGraph
from langchain_openai import ChatOpenAI
import json

# 1. DB 
db = ArangoClient(hosts="http://localhost:8529").db(
    "_system",
    username="root",
    password="test",
)

# 2. Loading a sample into ArangoDB
if not db.has_graph("GAME_OF_THRONES"):
    Datasets(db).load("GAME_OF_THRONES")

graph = ArangoGraph(db, schema_include_views=True)  # <---------

# print(db.analyzers()[1]["properties"])
print(json.dumps(graph.schema, indent=4))
breakpoint() 

llm = ChatOpenAI(model="gpt-4o", temperature=0)

chain = ArangoGraphQAChain.from_llm(
    llm=llm, graph=graph, allow_dangerous_requests=True, verbose=True
)

# del chain.graph.schema["analyzer_schema"]
# del chain.graph.schema["view_schema"]


chain.invoke({"query": "Find all characters whose surname includes 'Stark'. From those characters, fetch me the neighbors."})
# chain.invoke({"query": "Find all characters whose name contains the word 'Arya'."})
# chain.invoke({"query": "Show characters with the surname including 'Lanister'."}) # Fuzzy search
# chain.invoke({"query": "List characters whose name includes 'San'."}) # 
# chain.invoke({"query": "Query the ArangoSearch view 'CharactersSearchView' to return all documents where the 'surname' field contains the token 'Stark', using the 'text_en' analyzer."})