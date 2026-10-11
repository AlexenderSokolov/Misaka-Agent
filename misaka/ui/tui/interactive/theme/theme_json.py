"""Translated from pi's ``modes/interactive/theme/theme-json.ts`` (v1.1.0).

Theme JSON validation, kept out of ``theme.py`` on purpose.

Validating user-authored theme files needs typebox, which costs ~17 MB of module graph to import.
Palette lookup does not, so a presentation that only uses built-in themes should never pay for it.
``interactive-mode.ts`` installs this validator; anything that does not simply skips validation, as
built-in themes already do.

MISAKA forks: TypeBox's ``Compile`` is a jsonschema Draft 2020-12 validator, so the texts under "Other errors" come
from jsonschema (as in ``misaka/ai/utils/validation.py``). The colors also accept MISAKA's optional
``sectionTitle``, ``appTitle`` and ``appTitleOnLight`` tokens.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict, cast

from jsonschema import Draft202012Validator

ColorValueSchema: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},  # hex, var ref "primary", or empty ""
        {"type": "integer", "minimum": 0, "maximum": 255},  # 256-color index
    ]
}

ThemeJsonSchema: dict[str, Any] = {
    "type": "object",
    "required": ["name", "colors"],
    "properties": {
        "$schema": {"type": "string"},
        "name": {"type": "string"},
        # Background the theme is designed for. Detected from the theme colors when omitted.
        "appearance": {"anyOf": [{"const": "dark"}, {"const": "light"}]},
        "vars": {"type": "object", "additionalProperties": ColorValueSchema},
        "colors": {
            "type": "object",
            "required": [
                # Core UI (11 colors)
                "accent",
                "border",
                "borderAccent",
                "borderMuted",
                "success",
                "error",
                "warning",
                "muted",
                "dim",
                "text",
                "thinkingText",
                # Backgrounds & Content Text (11 required, 2 optional)
                "selectedBg",
                "userMessageBg",
                "userMessageText",
                "customMessageBg",
                "customMessageText",
                "customMessageLabel",
                "toolPendingBg",
                "toolSuccessBg",
                "toolErrorBg",
                "toolTitle",
                "toolOutput",
                # Markdown (10 colors)
                "mdHeading",
                "mdLink",
                "mdLinkUrl",
                "mdCode",
                "mdCodeBlock",
                "mdCodeBlockBorder",
                "mdQuote",
                "mdQuoteBorder",
                "mdHr",
                "mdListBullet",
                # Tool Diffs (3 colors)
                "toolDiffAdded",
                "toolDiffRemoved",
                "toolDiffContext",
                # Syntax Highlighting (9 colors)
                "syntaxComment",
                "syntaxKeyword",
                "syntaxFunction",
                "syntaxVariable",
                "syntaxString",
                "syntaxNumber",
                "syntaxType",
                "syntaxOperator",
                "syntaxPunctuation",
                # Thinking Level Borders (6 colors)
                "thinkingOff",
                "thinkingMinimal",
                "thinkingLow",
                "thinkingMedium",
                "thinkingHigh",
                "thinkingXhigh",
                # Bash Mode (1 color)
                "bashMode",
            ],
            "properties": {
                # Core UI (11 colors)
                "accent": ColorValueSchema,
                "border": ColorValueSchema,
                "borderAccent": ColorValueSchema,
                "borderMuted": ColorValueSchema,
                "success": ColorValueSchema,
                "error": ColorValueSchema,
                "warning": ColorValueSchema,
                "muted": ColorValueSchema,
                "dim": ColorValueSchema,
                "text": ColorValueSchema,
                "thinkingText": ColorValueSchema,
                # Scrollbar (2 optional colors)
                "scrollbarTrack": ColorValueSchema,
                "scrollbarThumb": ColorValueSchema,
                # Backgrounds & Content Text (11 required, 2 optional)
                "selectedBg": ColorValueSchema,
                "searchMatchBg": ColorValueSchema,
                "searchMatchText": ColorValueSchema,
                "userMessageBg": ColorValueSchema,
                "userMessageText": ColorValueSchema,
                "customMessageBg": ColorValueSchema,
                "customMessageText": ColorValueSchema,
                "customMessageLabel": ColorValueSchema,
                "toolPendingBg": ColorValueSchema,
                "toolSuccessBg": ColorValueSchema,
                "toolErrorBg": ColorValueSchema,
                "toolTitle": ColorValueSchema,
                "toolOutput": ColorValueSchema,
                # Markdown (10 colors)
                "mdHeading": ColorValueSchema,
                "mdLink": ColorValueSchema,
                "mdLinkUrl": ColorValueSchema,
                "mdCode": ColorValueSchema,
                "mdCodeBlock": ColorValueSchema,
                "mdCodeBlockBorder": ColorValueSchema,
                "mdQuote": ColorValueSchema,
                "mdQuoteBorder": ColorValueSchema,
                "mdHr": ColorValueSchema,
                "mdListBullet": ColorValueSchema,
                # Tool Diffs (3 colors)
                "toolDiffAdded": ColorValueSchema,
                "toolDiffRemoved": ColorValueSchema,
                "toolDiffContext": ColorValueSchema,
                # Syntax Highlighting (9 colors)
                "syntaxComment": ColorValueSchema,
                "syntaxKeyword": ColorValueSchema,
                "syntaxFunction": ColorValueSchema,
                "syntaxVariable": ColorValueSchema,
                "syntaxString": ColorValueSchema,
                "syntaxNumber": ColorValueSchema,
                "syntaxType": ColorValueSchema,
                "syntaxOperator": ColorValueSchema,
                "syntaxPunctuation": ColorValueSchema,
                # Thinking Level Borders (6 colors)
                "thinkingOff": ColorValueSchema,
                "thinkingMinimal": ColorValueSchema,
                "thinkingLow": ColorValueSchema,
                "thinkingMedium": ColorValueSchema,
                "thinkingHigh": ColorValueSchema,
                "thinkingXhigh": ColorValueSchema,
                "thinkingMax": ColorValueSchema,
                # Bash Mode (1 color)
                "bashMode": ColorValueSchema,
                # MISAKA fork: startup section titles and the app title (optional).
                "sectionTitle": ColorValueSchema,
                "appTitle": ColorValueSchema,
                "appTitleOnLight": ColorValueSchema,
            },
        },
        "export": {
            "type": "object",
            "properties": {
                "pageBg": ColorValueSchema,
                "cardBg": ColorValueSchema,
                "infoBg": ColorValueSchema,
            },
        },
    },
}

compiledThemeSchema = Draft202012Validator(ThemeJsonSchema)

type ThemeColorValue = str | int


class _ThemeJsonExport(TypedDict, total=False):
    pageBg: ThemeColorValue
    cardBg: ThemeColorValue
    infoBg: ThemeColorValue


ValidatedThemeJson = TypedDict(
    "ValidatedThemeJson",
    {
        "$schema": NotRequired[str],
        "name": str,
        "appearance": NotRequired[Literal["dark", "light"]],
        "vars": NotRequired[dict[str, ThemeColorValue]],
        "colors": dict[str, ThemeColorValue],
        "export": NotRequired[_ThemeJsonExport],
    },
)


def validateThemeJson(label: str, json: Any) -> ValidatedThemeJson:
    """Validate one theme document, throwing a message that names the offending tokens."""
    errors = list(compiledThemeSchema.iter_errors(json))
    if errors:
        missingColors: set[str] = set()
        otherErrors: list[str] = []

        for error in errors:
            if error.validator == "required" and list(error.absolute_path) == ["colors"]:
                for requiredProperty in cast(list[str], error.validator_value):
                    if requiredProperty not in error.instance:
                        missingColors.add(requiredProperty)
                continue

            path = "".join(f"/{part}" for part in error.absolute_path) or "/"
            otherErrors.append(f"  - {path}: {error.message}")

        errorMessage = f'Invalid theme "{label}":\n'
        if len(missingColors) > 0:
            errorMessage += "\nMissing required color tokens:\n"
            errorMessage += "\n".join(f"  - {color}" for color in sorted(missingColors))
            errorMessage += "\n\nPlease add these colors to your theme's \"colors\" object."
            errorMessage += "\nSee the built-in themes (dark.json, light.json) for reference values."
        if len(otherErrors) > 0:
            errorMessage += f"\n\nOther errors:\n{chr(10).join(otherErrors)}"

        raise ValueError(errorMessage)

    themeJson = cast(ValidatedThemeJson, json)
    if "/" in themeJson["name"]:
        raise ValueError(
            f'Invalid theme name "{themeJson["name"]}": theme names cannot contain "/" because it is reserved for '
            "automatic light/dark theme settings."
        )
    return themeJson
