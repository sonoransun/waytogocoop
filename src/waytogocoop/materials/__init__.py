"""Materials database and lattice utilities."""

from waytogocoop.materials.database import Material, get_material, list_materials, MATERIALS
from waytogocoop.materials.lattice import (
    square_lattice,
    hexagonal_lattice,
    apply_rotation,
    lattice_vectors,
    reciprocal_vectors,
)
from waytogocoop.materials.isotopes import (
    Isotope,
    ElementData,
    ELEMENTS,
    MATERIAL_COMPOSITION,
    get_element,
    get_composition,
    natural_average_mass,
    formula_unit_avg_mass,
)

__all__ = [
    "Material",
    "get_material",
    "list_materials",
    "MATERIALS",
    "square_lattice",
    "hexagonal_lattice",
    "apply_rotation",
    "lattice_vectors",
    "reciprocal_vectors",
    "Isotope",
    "ElementData",
    "ELEMENTS",
    "MATERIAL_COMPOSITION",
    "get_element",
    "get_composition",
    "natural_average_mass",
    "formula_unit_avg_mass",
]
