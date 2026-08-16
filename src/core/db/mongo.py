from pymongo import MongoClient

from ragify.utils.settings import settings


def build_mongo_uri() -> str:
    return (
        f"mongodb://{settings.mongo_user}:{settings.mongo_password}"
        f"@{settings.mongo_host}:{settings.mongo_port}/"
        f"{settings.mongo_db}?authSource={settings.mongo_auth_source}"
    )


# Singleton client (important)
_client = None


def get_mongo_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(build_mongo_uri())
    return _client


def get_database():
    return get_mongo_client()[settings.mongo_db]
