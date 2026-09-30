"""The built-in model catalog is pi-ai's published data, read the way pi reads it."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from misaka.ai import models
from misaka.ai.model_catalog import flatten_chat_model_catalog
from misaka.ai.models_generated import MODEL_DATA_MANIFEST, MODELS
from misaka.ai.types import Model
from misaka.core.model_resolver import defaultModelPerProvider

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "misaka" / "ai" / "providers" / "data"


def _script():
    spec = importlib.util.spec_from_file_location("refresh_model_catalog", ROOT / "scripts" / "refresh_model_catalog.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provider_files():
    return sorted(path for path in DATA.glob("*.json") if not path.name.startswith("."))


def test_the_data_is_pi_s_as_published():
    script = _script()
    assert MODEL_DATA_MANIFEST["schemaVersion"] == script.SCHEMA_VERSION
    assert {path.name for path in _provider_files()} == set(MODEL_DATA_MANIFEST["files"])
    for name, digest in MODEL_DATA_MANIFEST["files"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name


def test_the_provider_list_is_what_the_script_writes():
    script = _script()
    assert (ROOT / "misaka" / "ai" / "models_generated.py").read_text(encoding="utf-8") == script.render(MODEL_DATA_MANIFEST)


def test_every_chat_model_is_loaded_exactly_as_pi_ships_it():
    for path in _provider_files():
        chat = flatten_chat_model_catalog(path.stem, json.loads(path.read_text(encoding="utf-8")))
        if not chat:
            assert path.stem not in MODELS  # classifier-only providers (pi 0.99 "typesafe") stay out
            continue
        assert list(MODELS[path.stem]) == list(chat), path.stem  # pi's file order
        for model_id, raw in chat.items():
            assert MODELS[path.stem][model_id].model_dump(mode="json", exclude_none=True) == raw


def test_every_default_model_is_in_the_catalog():
    """pi drops models between releases; a default that went with them fails every sign-in."""
    missing = {provider: model_id for provider, model_id in defaultModelPerProvider.items()
               if models.get_model(provider, model_id) is None}
    assert missing == {}


def test_the_generation_stamp_comes_from_the_manifest():
    assert models.get_builtin_model_data_generated_at() is not None


def test_a_model_s_type_is_chat_or_unset():
    base = next(iter(MODELS["anthropic"].values())).model_dump(exclude_none=True)
    assert Model.model_validate({**base, "type": "chat"}).type == "chat"
    assert Model.model_validate({key: value for key, value in base.items() if key != "type"}).type is None
    with pytest.raises(ValidationError):
        Model.model_validate({**base, "type": "image"})
