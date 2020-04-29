# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun 2020/04/28

import json


def load_json(path):
    """Loads json"""
    with open(path) as json_file:
        return json.load(json_file)
