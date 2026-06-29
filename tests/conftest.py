"""Shared fixtures: a tiny GO DAG spanning all three namespaces."""

from textwrap import dedent

import pytest

from viral_annotation.ontology import GoDag

# MF chain: 0003 -> 0002 -> 0003674(root MF); 0004 -> 0002 (iea sibling)
# BP: 0011 -> 0008150(root BP).  CC: 0021 -> 0005575(root CC).
TINY_OBO = dedent(
    """
    format-version: 1.2

    [Term]
    id: GO:0003674
    name: molecular_function
    namespace: molecular_function

    [Term]
    id: GO:0000002
    name: mid mf
    namespace: molecular_function
    is_a: GO:0003674

    [Term]
    id: GO:0000003
    name: leaf mf (manual)
    namespace: molecular_function
    is_a: GO:0000002

    [Term]
    id: GO:0000004
    name: leaf mf (iea)
    namespace: molecular_function
    is_a: GO:0000002

    [Term]
    id: GO:0008150
    name: biological_process
    namespace: biological_process

    [Term]
    id: GO:0000011
    name: leaf bp
    namespace: biological_process
    is_a: GO:0008150

    [Term]
    id: GO:0005575
    name: cellular_component
    namespace: cellular_component

    [Term]
    id: GO:0000021
    name: leaf cc
    namespace: cellular_component
    is_a: GO:0005575
    """
).strip()


@pytest.fixture
def tiny_dag(tmp_path) -> GoDag:
    obo = tmp_path / "tiny.obo"
    obo.write_text(TINY_OBO, encoding="utf-8")
    return GoDag.from_obo(obo)


@pytest.fixture
def tiny_obo_path(tmp_path):
    """The tiny OBO written to disk; for tools/valib that take an --obo path."""
    obo = tmp_path / "tiny.obo"
    obo.write_text(TINY_OBO, encoding="utf-8")
    return obo
