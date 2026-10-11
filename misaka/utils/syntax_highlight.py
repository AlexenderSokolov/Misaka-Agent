"""Syntax highlighting helpers built on Pygments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pygments import lex
from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer
from pygments.token import Comment, Keyword, Literal, Name, Operator, Token
from pygments.util import ClassNotFound

HighlightFormatter = Callable[[str], str]
HighlightTheme = dict[str, HighlightFormatter]

@dataclass(slots=True)
class HighlightOptions:
    language: str | None = None
    # Accepted for source compatibility with upstream's highlight.js options and inert
    # here: highlight.js aborts a highlight on an illegal match unless told not to,
    # while Pygments never aborts -- it emits `Token.Error` and keeps going. So this
    # side always behaves as `ignoreIllegals: True`, which is what both in-repo callers
    # ask for; nothing reads the field.
    ignoreIllegals: bool | None = None
    # Only consulted when `language` is unset. Both in-repo callers gate on
    # `supportsLanguage()` and always pass `language`, so the scoring loop below and
    # the `guess_lexer` fallback are reached only through this module's public
    # `highlight()` from outside the repo (an extension, a test).
    languageSubset: Sequence[str] | None = None
    theme: HighlightTheme | None = None


def highlight(code: str, options: HighlightOptions | dict[str, Any] | None = None) -> str:
    resolved = _resolve_options(options)
    lexer = _resolve_lexer(code, resolved)
    output: list[str] = []
    theme = resolved.theme or {}

    for token, value in lex(code, lexer):
        scope = _scope_for_token(token)
        formatter = getScopeFormatter(scope, theme) if scope else None
        if formatter is None:
            formatter = theme.get("default")
        # Format each line on its own, so a multi-line token (a docstring, a block comment) keeps
        # its colour after highlightCode splits the result into lines (pi #10143).
        if formatter is None:
            output.append(value)
            continue
        output.append("\n".join(formatter(line) if line else line for line in value.split("\n")))

    return "".join(output)


def supportsLanguage(name: str) -> bool:
    try:
        get_lexer_by_name(name)
    except ClassNotFound:
        return False
    return True


def _resolve_options(options: HighlightOptions | dict[str, Any] | None) -> HighlightOptions:
    if isinstance(options, HighlightOptions):
        return options
    if isinstance(options, dict):
        return HighlightOptions(
            language=options.get("language"),
            ignoreIllegals=options.get("ignoreIllegals"),
            languageSubset=options.get("languageSubset"),
            theme=options.get("theme"),
        )
    return HighlightOptions()


# MISAKA fork: Pygments strips leading blank lines and appends a newline by default; highlight.js keeps
# the text line for line, so lexers are made without either.
_LEXER_OPTIONS: dict[str, Any] = {"stripnl": False, "ensurenl": False}


def _resolve_lexer(code: str, options: HighlightOptions):
    if options.language:
        try:
            return get_lexer_by_name(options.language, **_LEXER_OPTIONS)
        except ClassNotFound:
            return TextLexer(**_LEXER_OPTIONS)
    if options.languageSubset:
        best = None
        best_score: tuple[float, int, int] | None = None
        for name in options.languageSubset:
            try:
                candidate = get_lexer_by_name(name, **_LEXER_OPTIONS)
            except ClassNotFound:
                continue
            analyse = getattr(candidate, "analyse_text", None)
            analyse_score = float(analyse(code)) if callable(analyse) else 0.0
            error_chars = 0
            non_text_chars = 0
            for token, value in lex(code, candidate):
                size = len(value)
                if token in Token.Error:
                    error_chars += size
                elif not (token is Token.Text or token in Token.Text):
                    non_text_chars += size
            score = (analyse_score, -error_chars, non_text_chars)
            if best_score is None or score > best_score:
                best = candidate
                best_score = score
        if best is not None:
            return best
    try:
        return guess_lexer(code, **_LEXER_OPTIONS)
    except ClassNotFound:
        return TextLexer(**_LEXER_OPTIONS)


def getScopeFormatter(scope: str | None, theme: HighlightTheme) -> HighlightFormatter | None:
    if scope is None:
        return None
    exact = theme.get(scope)
    if exact is not None:
        return exact
    for separator in (".", "-"):
        index = scope.find(separator)
        if index != -1:
            formatter = theme.get(scope[:index])
            if formatter is not None:
                return formatter
    return None


def _scope_for_token(token: Any) -> str | None:
    if token in Keyword:
        return "keyword"
    if token in Literal.Number:
        return "number"
    if token in Literal.String:
        return "string"
    if token in Comment:
        return "comment"
    if token in Name.Function or token in Name.Class:
        return "title"
    if token in Name.Tag:
        return "tag"
    if token in Name.Attribute:
        return "attr"
    if token in Name.Builtin:
        return "built_in"
    if token in Name.Variable:
        return "variable"
    if token in Operator:
        return "operator"
    if token is Token.Text:
        return None
    return None


__all__ = [
    "HighlightFormatter",
    "HighlightOptions",
    "HighlightTheme",
    "highlight",
    ]
