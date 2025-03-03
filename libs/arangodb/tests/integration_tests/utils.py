from typing import TypedDict


class ArangoCredentials(TypedDict):
    url: str
    username: str
    password: str
