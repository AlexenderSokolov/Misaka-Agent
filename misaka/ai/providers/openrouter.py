"""Translated from pi's ``packages/ai/src/providers/openrouter.ts`` (v1.1.0)."""

from __future__ import annotations

from misaka.ai.api.anthropic_messages_lazy import anthropicMessagesApi
from misaka.ai.api.openai_completions_lazy import openAICompletionsApi
from misaka.ai.api.openrouter_images_lazy import openrouterImagesApi
from misaka.ai.api.typesafe_system_one_lazy import typesafeSystemOneApi
from misaka.ai.auth.helpers import envApiKeyAuth, lazyOAuth
from misaka.ai.auth.oauth.load import loadOpenRouterOAuth
from misaka.ai.auth.types import ProviderAuth
from misaka.ai.models import CreateProviderOptions, Provider, createProvider
from misaka.ai.providers.openrouter_models import (
    OPENROUTER_CLASSIFIER_MODELS,
    OPENROUTER_IMAGE_MODELS,
    OPENROUTER_MODELS,
)


def openrouterProvider() -> Provider:
    return createProvider(
        CreateProviderOptions(
            id="openrouter",
            name="OpenRouter",
            baseUrl="https://openrouter.ai/api/v1",
            auth=ProviderAuth(
                apiKey=envApiKeyAuth("OpenRouter API key", ["OPENROUTER_API_KEY"]),
                oauth=lazyOAuth(
                    name="OpenRouter OAuth",
                    loginLabel="Sign in with OpenRouter",
                    load=loadOpenRouterOAuth,
                ),
            ),
            models=[
                *OPENROUTER_MODELS.values(),
                *OPENROUTER_IMAGE_MODELS.values(),
                *OPENROUTER_CLASSIFIER_MODELS.values(),
            ],
            api={
                "anthropic-messages": anthropicMessagesApi(),
                "openai-completions": openAICompletionsApi(),
            },
            images={"openrouter-images": openrouterImagesApi()},
            # OpenRouter serves TypeSafe's System One protocol at /api/v1/systemone.
            classifiers={"typesafe-system-one": typesafeSystemOneApi()},
        )
    )


__all__ = ["openrouterProvider"]
