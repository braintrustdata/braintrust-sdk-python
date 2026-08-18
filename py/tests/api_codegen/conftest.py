import copy
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from openapi_codegen import CONFIG_PATH, load_config  # noqa: E402


@pytest.fixture
def codegen_config():
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["endpoint_generator"]["generated_tags"] = ["Widgets"]
    config["endpoint_generator"]["idempotent_writes"] = []
    return config


@pytest.fixture
def minimal_spec():
    return {
        "openapi": "3.0.3",
        "info": {"title": "Test", "version": "1"},
        "paths": {
            "/widgets/{widget_id}": {
                "get": {
                    "operationId": "getWidget",
                    "tags": ["Widgets"],
                    "parameters": [{"$ref": "#/components/parameters/WidgetId"}],
                    "responses": {
                        "200": {
                            "description": "OK",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Widget"}}},
                        }
                    },
                }
            }
        },
        "components": {
            "parameters": {
                "WidgetId": {
                    "name": "widget_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            },
            "schemas": {
                "Widget": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
        },
    }
