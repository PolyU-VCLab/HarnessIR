#!/usr/bin/env python3
"""Small shared helpers: JSON I/O, prompt loading, task-type parsing.

The block parser in ``load_blocks`` is the only place that knows the
``### [[<type>]] ###`` format used by prompts/compose_blocks.txt,
prompts/verify_blocks.txt and prompts/df/df_types.txt.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .settings import PROMPTS_DIR

# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def write_json(path: str | Path, obj: Any) -> None:
    """Write JSON with a stable indent, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def strip_comments(text: str) -> str:
    """Drop ``#`` comment lines, but keep block markers (``# [[...]]``).

    Comment lines are the authoring notes at the top of every prompt file. Any
    line whose content inside the hashes starts with ``[[`` is a block marker
    and must survive.
    """
    keep = [
        line for line in text.splitlines()
        if not (line.startswith("#")
                and not line.lstrip("#").strip().startswith("[["))
    ]
    return "\n".join(keep).strip()


def load_prompt(name: str, *, subdir: str = "") -> str:
    """Read one prompt file from prompts/ (or prompts/<subdir>/)."""
    path = (PROMPTS_DIR / subdir / name) if subdir else (PROMPTS_DIR / name)
    return strip_comments(path.read_text(encoding="utf-8"))


def load_blocks(name: str, *, subdir: str = "") -> dict[str, str]:
    """Parse a block file into ``{type: body}``.

    Blocks are delimited by ``### [[<type>]] ###`` lines. Everything before the
    first marker is ignored, which is what makes the authoring header harmless.
    """
    blocks: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for line in load_prompt(name, subdir=subdir).splitlines():
        s = line.strip()
        if s.startswith("### [[") and s.endswith("]] ###"):
            if cur:
                blocks[cur] = "\n".join(buf).strip()
            cur, buf = s[6:-6], []
        elif cur is not None:
            buf.append(line)
    if cur:
        blocks[cur] = "\n".join(buf).strip()
    return blocks


# ---------------------------------------------------------------------------
# Task types
# ---------------------------------------------------------------------------

# An image may carry more than one degradation type. The separator is ``|``
# rather than a comma because the annotation table itself is a CSV.
TYPE_SEP = "|"


def parse_types(raw: Any) -> list[str]:
    """Normalise a type field into a de-duplicated, order-preserving list.

    Accepts all three spellings that occur in practice::

        "rain"                          single type
        "rain|lowlight"                 several, separated by ``|``
        ["rain", "lowlight"]            a JSON array

    An empty input returns ``[]`` - the caller decides what to fall back to.
    """
    if raw is None:
        return []
    items = raw if isinstance(raw, (list, tuple)) else str(raw).split(TYPE_SEP)
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        t = str(it).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def types_label(types: list[str]) -> str:
    """Human-readable type list: ``"rain + lowlight"``."""
    return " + ".join(types) if types else "(unspecified)"


def types_key(types: list[str]) -> str:
    """Machine-readable type key, round-trippable through ``parse_types``."""
    return TYPE_SEP.join(types)
