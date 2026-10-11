"""LCM streams through the model's provider, except where a task's api_mode put the model on an
API that provider does not serve (``misaka_lcm/host/llm.py`` ``_provider_stream``)."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from misaka.ai.compat import getModel
from misaka.extensions.misaka_lcm.host import llm


def _registry(provider):
    return SimpleNamespace(
        getProvider=lambda _id: provider,
        find=lambda provider_id, model_id: getModel(provider_id, model_id),
    )


def test_a_catalog_model_streams_through_its_provider(monkeypatch):
    calls = []
    provider = SimpleNamespace(streamSimple=lambda model, context, options: calls.append(("provider", model.api)))
    monkeypatch.setattr(importlib.import_module("misaka.ai.compat"), "streamSimple", lambda *a: calls.append(("api", a[0].api)))
    model = getModel("zai", "glm-4.7")

    llm._provider_stream(_registry(provider), model, {"messages": []}, None)

    assert calls == [("provider", "openai-completions")]


def test_an_api_mode_override_goes_by_the_api_it_names(monkeypatch):
    calls = []
    provider = SimpleNamespace(streamSimple=lambda model, context, options: calls.append(("provider", model.api)))
    monkeypatch.setattr(importlib.import_module("misaka.ai.compat"), "streamSimple", lambda *a: calls.append(("api", a[0].api)))
    model = getModel("zai", "glm-4.7").model_copy(update={"api": "anthropic-messages"})

    llm._provider_stream(_registry(provider), model, {"messages": []}, None)

    assert calls == [("api", "anthropic-messages")]
