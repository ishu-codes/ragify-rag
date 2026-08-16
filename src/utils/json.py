import json
from typing import Any


def load_json_file(path: str) -> list[dict[str, Any] | dict[str, Any]]:
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

def save_to_json(path: str, data: list[dict[str, Any]] | dict[str, Any]) -> None:
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
