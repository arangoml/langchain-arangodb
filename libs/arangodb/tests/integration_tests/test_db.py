from arango.database import Database


def test_db(db: Database) -> None:
    db.version()
