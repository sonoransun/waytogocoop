"""Common test fixtures."""

from __future__ import annotations

import pytest

from waytogocoop.materials.database import Material, get_material


@pytest.fixture
def fete() -> Material:
    """Return the FeTe substrate material."""
    return get_material("FeTe")


@pytest.fixture
def sb2te3() -> Material:
    """Return the Sb2Te3 overlayer material."""
    return get_material("Sb2Te3")


@pytest.fixture
def bi2te3() -> Material:
    """Return the Bi2Te3 overlayer material."""
    return get_material("Bi2Te3")


@pytest.fixture
def sb2te() -> Material:
    """Return the Sb2Te overlayer material."""
    return get_material("Sb2Te")
