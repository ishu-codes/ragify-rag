import json
import os
from datetime import datetime
from typing import Any, Dict

# ---------------------------
# Helpers
# ---------------------------

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Run ID Generator
# ---------------------------

def create_run_id(experiment_name: str) -> str:
    return f"{experiment_name}_{get_timestamp()}"


# ---------------------------
# Summary Saver
# ---------------------------

def save_summary(results: Dict[str, Any], run_id: str, base_dir: str = "benchmarking/results"):
    summary_dir = os.path.join(base_dir, "summaries")
    ensure_dir(summary_dir)

    path = os.path.join(summary_dir, f"{run_id}.json")

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[✔] Summary saved: {path}")


# ---------------------------
# Per-query Logger (JSONL)
# ---------------------------

class QueryLogger:
    def __init__(self, run_id: str, base_dir: str = "benchmarking/results"):
        self.run_id = run_id
        self.log_dir = os.path.join(base_dir, "runs")
        ensure_dir(self.log_dir)

        self.path = os.path.join(self.log_dir, f"{run_id}.jsonl")

    def log(self, record: Dict[str, Any]):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_path(self):
        return self.path
