import os
import yaml
import json


def load_config(config_path):
    with open(os.path.join(config_path)) as file:
        config = yaml.safe_load(file)
    return config


def get_neptune_api_token():
    if not os.path.exists(".neptune_api_token.json"):
        raise SystemExit("Please obtain the token first.")
    with open(".neptune_api_token.json") as f:
        return json.load(f)
