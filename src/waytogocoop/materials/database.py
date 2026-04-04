"""Material database for topological insulator / superconductor heterostructures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Immutable record for a crystalline material.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. "Iron Telluride").
    formula : str
        Chemical formula (e.g. "FeTe").
    lattice_type : str
        "hexagonal" or "square".
    a : float
        In-plane lattice constant in Angstrom.
    c : float
        Out-of-plane lattice constant in Angstrom (0 if not applicable).
    space_group : str
        Hermann-Mauguin space group symbol.
    description : str
        Short description of the material's role.
    role : str
        "substrate" or "overlayer".
    """

    name: str
    formula: str
    lattice_type: str
    a: float
    c: float
    space_group: str
    description: str
    role: str

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError(f"Lattice constant 'a' must be positive, got {self.a}")
        if self.c < 0:
            raise ValueError(f"Out-of-plane constant 'c' must be non-negative, got {self.c}")
        if self.lattice_type not in ("hexagonal", "square"):
            raise ValueError(
                f"lattice_type must be 'hexagonal' or 'square', got {self.lattice_type!r}"
            )
        if self.role not in ("substrate", "overlayer"):
            raise ValueError(
                f"role must be 'substrate' or 'overlayer', got {self.role!r}"
            )


# ---------------------------------------------------------------------------
# Pre-populated materials dictionary
# ---------------------------------------------------------------------------
MATERIALS: dict[str, Material] = {
    "FeTe": Material(
        name="Iron Telluride",
        formula="FeTe",
        lattice_type="square",
        a=3.82,
        c=6.27,
        space_group="P4/nmm",
        description="Superconducting substrate with square lattice",
        role="substrate",
    ),
    "Sb2Te3": Material(
        name="Antimony Telluride",
        formula="Sb2Te3",
        lattice_type="hexagonal",
        a=4.264,
        c=30.458,
        space_group="R-3m",
        description="Topological insulator overlayer (quintuple layer)",
        role="overlayer",
    ),
    "Bi2Te3": Material(
        name="Bismuth Telluride",
        formula="Bi2Te3",
        lattice_type="hexagonal",
        a=4.386,
        c=30.497,
        space_group="R-3m",
        description="Topological insulator overlayer (quintuple layer)",
        role="overlayer",
    ),
    "Sb2Te": Material(
        name="Antimony Ditelluride",
        formula="Sb2Te",
        lattice_type="hexagonal",
        a=4.272,
        c=17.633,
        space_group="R-3m",
        description="Topological insulator overlayer variant",
        role="overlayer",
    ),
}


def get_material(name: str) -> Material:
    """Return a Material by its formula key.

    Raises ``KeyError`` if the material is not in the database.
    """
    if name not in MATERIALS:
        raise KeyError(
            f"Material '{name}' not found. Available: {list(MATERIALS.keys())}"
        )
    return MATERIALS[name]


def list_materials(role: str | None = None) -> list[Material]:
    """Return a list of materials, optionally filtered by *role*.

    Parameters
    ----------
    role : str or None
        If given, only return materials whose ``role`` matches (e.g.
        ``"substrate"`` or ``"overlayer"``).
    """
    if role is None:
        return list(MATERIALS.values())
    return [m for m in MATERIALS.values() if m.role == role]
