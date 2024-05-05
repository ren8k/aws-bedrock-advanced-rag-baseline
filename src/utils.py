from typing import Any

import yaml


def load_yaml(config_path: str) -> Any:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
