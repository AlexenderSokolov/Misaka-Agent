"""Flatten a provider's generated catalog into the models of one type.

Port of pi-ai ``src/model-catalog.ts``. A provider's data file (``providers/data``) groups its
models by api and, since schema 6, holds every model type in one file; a typed catalog keeps the
models of its type, keyed by id, in file order.

Only the chat flattener is ported. pi's ``flattenImageModelCatalog`` and
``flattenClassifierModelCatalog`` have no misaka consumer yet: the image catalog is still
``image_models_generated``, and misaka runs no classifier models.
"""

from __future__ import annotations

from typing import Any


def flatten_model_catalog(groups: dict[str, dict[str, dict[str, Any]]], type: str) -> dict[str, dict[str, Any]]:
    return {
        model["id"]: model
        for models in groups.values()
        for model in models.values()
        if model.get("type") == type
    }


def flatten_chat_model_catalog(_provider: str, groups: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return flatten_model_catalog(groups, "chat")
