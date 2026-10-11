"""models.json loading (``misaka/core/model_config.py``), after pi's ``model-config.ts``."""

from __future__ import annotations

import json

from misaka.core.model_config import ModelConfig
from misaka.utils.json import stripJsonComments


def _write(tmp_path, value) -> str:
    path = tmp_path / "models.json"
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return str(path)


async def test_missing_path_or_file_is_an_empty_config_without_error(tmp_path):
    for path in (None, "", str(tmp_path / "missing.json")):
        config = await ModelConfig.load(path)
        assert config.getProviderIds() == [] and config.getError() is None


async def test_accepts_comments_trailing_commas_and_a_bom(tmp_path):
    text = '﻿{\n  // providers\n  "providers": {"demo": {"baseUrl": "https://x", "models": [{"id": "m"},],},},\n}'
    config = await ModelConfig.load(_write(tmp_path, text))

    assert config.getError() is None
    assert config.getProviderIds() == ["demo"]
    assert config.getProvider("demo").models[0].id == "m"


async def test_thinking_level_map_takes_null_and_compat_is_a_lenient_union(tmp_path):
    config = await ModelConfig.load(_write(tmp_path, {"providers": {"demo": {
        "api": "openai-completions",
        "compat": {"supportsEagerToolInputStreaming": False, "notYetKnown": 1},
        "models": [{
            "id": "m",
            "thinkingLevelMap": {"minimal": None, "high": "max"},
            "compat": {"thinkingFormat": "chat-template", "chatTemplateKwargs": {"thinking": {"$var": "thinking.enabled"}}},
        }],
    }}}))

    assert config.getError() is None
    model = config.getProvider("demo").models[0]
    assert model.thinkingLevelMap.minimal is None and model.thinkingLevelMap.high == "max"
    assert config.getProvider("demo").compat == {"supportsEagerToolInputStreaming": False, "notYetKnown": 1}


async def test_any_schema_error_discards_the_whole_file_with_paths(tmp_path):
    path = _write(tmp_path, {"providers": {
        "good": {"baseUrl": "https://x"},
        "bad": {"baseUrl": None, "models": [{"name": "no id"}]},
    }})
    config = await ModelConfig.load(path)

    assert config.getProviderIds() == []
    error = config.getError()
    assert error.startswith("Invalid models.json schema:\n")
    assert "  - providers.bad.baseUrl: " in error and "  - providers.bad.models.0.id: " in error
    assert error.endswith(f"\n\nFile: {path}")


async def test_parse_errors_name_the_file(tmp_path):
    path = _write(tmp_path, "{bad")
    config = await ModelConfig.load(path)

    assert config.getError().startswith("Failed to parse models.json: ")
    assert config.getError().endswith(f"\n\nFile: {path}")


def test_strip_json_comments_leaves_strings_alone():
    assert stripJsonComments('{"a": "x // y", // z\n "b": [1,],}') == '{"a": "x // y", \n "b": [1]}'


async def test_values_are_never_coerced(tmp_path):
    # TypeBox `Check` does not convert: a string is not a boolean or a number.
    for model in ({"id": "m", "reasoning": "yes"}, {"id": "m", "maxTokens": "4096"},
                  {"id": "m", "cost": {"input": True, "output": 0, "cacheRead": 0, "cacheWrite": 0}}):
        config = await ModelConfig.load(_write(tmp_path, {"providers": {"p": {"models": [model]}}}))
        assert config.getProviderIds() == [] and "Invalid models.json schema" in config.getError()

    config = await ModelConfig.load(_write(tmp_path, {"providers": {"p": {"models": [
        {"id": "m", "contextWindow": 128000, "inputLimits": {"maxRequestBytes": 2.0}},
    ]}}}))
    assert config.getError() is None


async def test_compat_arms_are_upstreams_three_schemas(tmp_path):
    # No arm declares supportsMidConvoSystemMessages, the Anthropic arm has no
    # sessionAffinityFormat, and Type.Optional rejects an explicit null.
    for compat in ({"sessionAffinityFormat": "bogus"}, {"supportsMidConvoSystemMessages": 5},
                   {"openRouterRouting": {"order": ["a"], "notYetKnown": 1}}, {"vllmPriority": 1.5}):
        config = await ModelConfig.load(_write(tmp_path, {"providers": {"p": {"compat": compat}}}))
        assert config.getError() is None, compat

    config = await ModelConfig.load(_write(tmp_path, {"providers": {"p": {"compat": {"supportsLongCacheRetention": None}}}}))
    assert config.getProviderIds() == [] and "compat" in config.getError()


async def test_invalid_utf8_is_replaced_as_node_reads_it(tmp_path):
    path = tmp_path / "models.json"
    path.write_bytes(b'{"providers":{"p":{"name":"caf\xe9"}}}')

    config = await ModelConfig.load(str(path))

    assert config.getError() is None
    assert config.getProvider("p").name == "caf�"
