"""Non-executing LaTeX scanner with balanced groups and environment stacks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

COMMAND = re.compile(r"\\([A-Za-z@]+|.)", re.S)
LITERAL_ENVS = {"verbatim", "verbatim*", "Verbatim", "lstlisting", "minted", "comment"}
PANEL_ENVS = {"subfigure", "subcaptionblock"}


def issue(code: str, message: str, offset: int | None = None, **extra: Any) -> dict[str, Any]:
    return {"severity": "warning", "code": code, "message": message, "offset": offset, **extra}


def group(text: str, pos: int, opening: str = "{") -> tuple[str, int, int]:
    """Read one balanced argument, respecting escaped delimiters and brace nesting."""
    while pos < len(text) and text[pos].isspace():
        pos += 1
    closing = "}" if opening == "{" else "]"
    if pos >= len(text) or text[pos] != opening:
        raise ValueError(f"expected {opening} at offset {pos}")
    start = pos + 1
    stack = [closing]
    pos += 1
    while pos < len(text):
        char = text[pos]
        if char == "\\":
            match = COMMAND.match(text, pos)
            pos = match.end() if match else pos + 1
            continue
        if char == "{" or (char == "[" and stack[-1] == "]"):
            stack.append("}" if char == "{" else "]")
        elif char == stack[-1]:
            stack.pop()
            if not stack:
                return text[start:pos], start, pos + 1
        pos += 1
    raise ValueError(f"unclosed {opening} at offset {start - 1}")


def _space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def mask_noncode(raw: str) -> tuple[str, list[dict[str, Any]]]:
    """Keep offsets while excluding comments, literal examples and macro definitions."""
    chars = list(raw)
    warnings = []

    def blank(start: int, end: int) -> None:
        chars[start:end] = ["\n" if c == "\n" else " " for c in raw[start:end]]

    pos = 0
    while pos < len(raw):
        if raw[pos] == "%":
            end = raw.find("\n", pos)
            end = len(raw) if end < 0 else end
            blank(pos, end)
            pos = end
            continue
        match = COMMAND.match(raw, pos)
        if not match:
            pos += 1
            continue
        name = match[1]
        end = match.end()
        if name == "verb":
            if end < len(raw) and raw[end] == "*":
                end += 1
            if end < len(raw):
                close = raw.find(raw[end], end + 1)
                end = close + 1 if close >= 0 else len(raw)
                blank(pos, end)
        elif name == "begin":
            try:
                env, _, body = group(raw, end)
                if env in LITERAL_ENVS:
                    marker = "\\end{" + env + "}"
                    close = raw.find(marker, body)
                    end = close + len(marker) if close >= 0 else len(raw)
                    blank(pos, end)
            except ValueError:
                pass
        pos = end

    text = "".join(chars)
    pos = 0
    definitions = {"newcommand", "renewcommand", "providecommand", "DeclareRobustCommand",
                   "newenvironment", "renewenvironment", "def", "gdef", "edef", "xdef"}
    while match := COMMAND.search(text, pos):
        pos = match.end()
        if match[1] not in definitions:
            continue
        try:
            if match[1] in {"def", "gdef", "edef", "xdef"}:
                body = text.find("{", pos)
                if body < 0:
                    raise ValueError("missing definition body")
                _, _, end = group(text, body)
            else:
                pos = _space(text, pos + (text[pos:pos + 1] == "*"))
                if text[pos:pos + 1] == "{":
                    _, _, pos = group(text, pos)
                else:
                    macro = COMMAND.match(text, pos)
                    if macro is None:
                        raise ValueError("missing macro name")
                    pos = macro.end()
                while text[_space(text, pos):_space(text, pos) + 1] == "[":
                    _, _, pos = group(text, pos, "[")
                _, _, end = group(text, pos)
                if match[1] in {"newenvironment", "renewenvironment"}:
                    _, _, end = group(text, end)
            warnings.append(issue("macro_definition_skipped", "Macro definitions are not expanded.", match.start()))
            blank(match.start(), end)
            pos = end
        except ValueError as exc:
            warnings.append(issue("malformed_definition", str(exc), match.start()))
    return "".join(chars), warnings


@dataclass
class Command:
    name: str
    start: int
    end: int
    value: str = ""
    options: list[str] = field(default_factory=list)
    starred: bool = False
    graphicspath: tuple[str, ...] = ()
    body_span: tuple[int, int] | None = None
    value_span: tuple[int, int] | None = None
    option_spans: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Environment:
    name: str
    start: int
    body_start: int
    end: int
    closed: bool = True


@dataclass
class Document:
    raw: str
    text: str
    commands: list[Command]
    environments: list[Environment]
    warnings: list[dict[str, Any]]


def scan(raw: str) -> Document:
    text, warnings = mask_noncode(raw)
    commands: list[Command] = []
    environments: list[Environment] = []
    env_stack: list[Environment] = []
    known = {"begin", "end", "includegraphics", "graphicspath", "caption", "label",
             "subcaption", "subfigure", "subfloat", "subcaptionbox", "input", "include",
             "section", "subsection", "subsubsection"}

    def walk(start: int, stop: int, paths: tuple[str, ...], depth: int = 0) -> None:
        if depth > 128:
            warnings.append(issue("nesting_limit", "Group nesting exceeds the inventory limit of 128.", start))
            return
        scopes: list[tuple[str, ...]] = []
        pos = start
        while pos < stop:
            if text[pos] == "{":
                try:
                    _, body, end = group(text, pos)
                except ValueError as exc:
                    warnings.append(issue("unbalanced_group", str(exc), pos))
                    pos += 1
                    continue
                walk(body, min(end - 1, stop), paths, depth + 1)
                pos = end
                continue
            match = COMMAND.match(text, pos)
            if match is None:
                pos += 1
                continue
            name = match[1]
            if name not in known:
                if name.startswith("if") or name in {"else", "fi"}:
                    warnings.append(issue("conditional_not_evaluated", "Conditional branches are inventoried literally.", pos))
                pos = match.end()
                continue
            cursor = match.end()
            starred = text[cursor:cursor + 1] == "*"
            cursor += starred
            options = []
            option_spans = []
            try:
                while text[_space(text, cursor):_space(text, cursor) + 1] == "[":
                    option, option_start, cursor = group(text, cursor, "[")
                    options.append(option)
                    option_spans.append((option_start, cursor - 1))
                value, body, end = group(text, cursor)
                value_span = (body, end - 1)
                body_span = None
                if name == "subcaptionbox":
                    cursor = end
                    while text[_space(text, cursor):_space(text, cursor) + 1] == "[":
                        option, option_start, cursor = group(text, cursor, "[")
                        options.append(option)
                        option_spans.append((option_start, cursor - 1))
                    _, body, end = group(text, cursor)
                    body_span = (body, end - 1)
                elif name in {"subfigure", "subfloat"}:
                    body_span = (body, end - 1)
                command = Command(name, pos, end, value, options, starred, paths, body_span, value_span, option_spans)
                commands.append(command)
                if name == "begin":
                    env = Environment(value, pos, end, stop, False)
                    env_stack.append(env)
                    environments.append(env)
                    scopes.append(paths)
                elif name == "end":
                    index = next((i for i in range(len(env_stack) - 1, -1, -1)
                                  if env_stack[i].name == value), None)
                    if index is None:
                        warnings.append(issue("unmatched_environment_end", value, pos))
                    else:
                        closing_count = len(env_stack) - index
                        for unclosed in env_stack[index + 1:]:
                            unclosed.end = pos
                        env_stack[index].end = end
                        env_stack[index].closed = True
                        del env_stack[index:]
                        for _ in range(min(closing_count, len(scopes))):
                            paths = scopes.pop()
                elif name == "graphicspath":
                    entries = []
                    entry_pos = 0
                    while _space(value, entry_pos) < len(value):
                        entry, _, entry_pos = group(value, entry_pos)
                        entries.append(entry.strip())
                    paths = tuple(entries)
                elif body_span:
                    if name == "subcaptionbox":
                        walk(*value_span, paths, depth + 1)
                    walk(*body_span, paths, depth + 1)
                elif name in {"caption", "subcaption"}:
                    # Labels inside captions remain discoverable without executing markup.
                    walk(body, end - 1, paths, depth + 1)
                pos = end
            except ValueError as exc:
                warnings.append(issue("malformed_command", str(exc), pos, command=name))
                if name == "includegraphics":
                    commands.append(Command(name, pos, match.end(), graphicspath=paths))
                pos = match.end()

    walk(0, len(text), ())
    for env in environments:
        if not env.closed:
            warnings.append(issue("unclosed_environment", env.name, env.start))
    return Document(raw, text, sorted(commands, key=lambda c: c.start),
                    sorted(environments, key=lambda e: e.start), warnings)


def option_values(raw: str) -> dict[str, str | bool]:
    """Split graphics options only at commas outside balanced braces."""
    parts = []
    start = depth = 0
    for index, char in enumerate(raw):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(raw[start:index])
            start = index + 1
    parts.append(raw[start:])
    result: dict[str, str | bool] = {}
    for part in parts:
        key, separator, value = part.partition("=")
        if key.strip():
            result[key.strip()] = value.strip() if separator else True
    return result
