from pathlib import Path

import yaml


class Prompts:
    def __init__(self, config_file: str | None = None):
        base_path = Path(__file__).parent
        config_path = (
            base_path / "prompts.yaml" if (config_file is None) else Path(config_file)
        )
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str) -> str:
        return self._config["prompts"][key]

    @property
    def system_prompt(self) -> str:
        return self.get("system_prompt")

    @property
    def classify_prompt(self) -> str:
        return self.get("classify_prompt")

    @property
    def grading_prompt(self) -> str:
        return self.get("grading_prompt")

    @property
    def rewrite_prompt(self) -> str:
        return self.get("rewrite_prompt")

    @property
    def generate_prompt(self) -> str:
        return self.get("generate_prompt")

    @property
    def verify_prompt(self) -> str:
        return self.get("verify_prompt")


prompts = Prompts()
