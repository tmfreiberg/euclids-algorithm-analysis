"""Every cross-reference and in-page link in the book has a target.

Quarto reports an unresolved ``@sec-...`` reference only as a warning, and a
raw ``[text](#id)`` link to a missing id renders as a dead link with no warning
at all. Both exit with status 0, so this test does the checking instead. It
reads the book's sources and needs neither Quarto nor the package.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "index.qmd", *sorted((ROOT / "book").glob("*.qmd"))]

# {#id} on a heading, a fenced div or an equation's closing $$.
DEFINITION = re.compile(r"\{#([A-Za-z][\w-]*)")
# @sec-..., @eq-..., @thm-... and the other cross-reference prefixes.
REFERENCE = re.compile(r"@((?:sec|eq|thm|lem|prp|cor|cnj|def|rem|fig|tbl)-[\w-]*\w)")
# [text](#id), the raw in-page form.
ANCHOR = re.compile(r"\]\(#([\w-]+)\)")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _defined() -> set[str]:
    return {m for path in SOURCES for m in DEFINITION.findall(_read(path))}


def test_sources_found():
    assert len(SOURCES) >= 2
    assert all(path.exists() for path in SOURCES)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_every_reference_has_a_target(path):
    defined = _defined()
    text = _read(path)
    used = REFERENCE.findall(text) + ANCHOR.findall(text)
    missing = sorted({ref for ref in used if ref not in defined})
    assert not missing, f"{path.name} refers to undefined ids: {missing}"


def test_no_id_is_defined_twice():
    ids = [m for path in SOURCES for m in DEFINITION.findall(_read(path))]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, f"ids defined more than once: {duplicates}"
