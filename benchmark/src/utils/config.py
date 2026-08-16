import argparse
from dataclasses import dataclass, field
from typing import Optional

import yaml

# ---------------------------
# Config Schema (STRICT)
# ---------------------------

@dataclass
class Config:
    # retrieval
    embedding_model: str
    top_k: int = 5

    # reranking
    reranker: Optional[str] = None

    # llm
    llm: str = "openai"

    # evaluation
    use_llm_judge: bool = False

    # experiment metadata
    experiment_name: str = "default"
    save_results: bool = True


# ---------------------------
# Load YAML
# ---------------------------

def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------
# Validate + Map to Dataclass
# ---------------------------

def parse_config(raw: dict) -> Config:
    try:
        return Config(**raw)
    except TypeError as e:
        raise ValueError(f"Invalid config keys: {e}")


# ---------------------------
# CLI Override Support
# ---------------------------

def override_with_args(config: Config) -> Config:
    parser = argparse.ArgumentParser()

    parser.add_argument("--top_k", type=int)
    parser.add_argument("--reranker", type=str)
    parser.add_argument("--llm", type=str)
    parser.add_argument("--experiment_name", type=str)

    args = parser.parse_args()

    # override only if provided
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

    return config


# ---------------------------
# Main Loader
# ---------------------------

def load_config(config_path: str) -> Config:
    raw = load_yaml_config(config_path)
    config = parse_config(raw)
    config = override_with_args(config)
    return config
