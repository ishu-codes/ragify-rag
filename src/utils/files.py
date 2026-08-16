from pathlib import Path

from .colors import colorize


def save_to_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def append_to_file(path: str, content: str, reverse=False):
    if reverse:
        prev_content = get_file_content(path)
        save_to_file(path, f"{content}\n{prev_content}")
        return

    with open(path, "a", encoding="utf-8") as file:
        file.write(content)


def get_file_content(path: str, default: str | None = ""):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(colorize(f"File not found! {path}", "RED"))
        return default


def ensure_dir(path: str):
    p = Path(path)
    target_dir = p.parent if p.suffix else p
    target_dir.mkdir(parents=True, exist_ok=True)


def remove_dir(path: str):
    dir = Path(path)
    if not dir.exists():
        return

    for child in dir.iterdir():
        if child.is_file():
            child.unlink()
    dir.rmdir()


def remove_file(path: str) -> bool:
    """
    Deletes a file if it exists.
    Returns True if deleted, False if file did not exist or if it's not a file.
    """
    p = Path(path)

    if not p.exists() or not p.is_file():
        return False

    p.unlink()
    return True
