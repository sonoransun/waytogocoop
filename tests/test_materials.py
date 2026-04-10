"""Tests for the materials database and lattice utilities."""

from __future__ import annotations

import numpy as np
import pytest

from waytogocoop.materials.database import (
    MATERIALS,
    Material,
    get_material,
    list_materials,
)
from waytogocoop.materials.lattice import (
    apply_rotation,
    hexagonal_lattice,
    lattice_vectors,
    reciprocal_vectors,
    square_lattice,
)

# -----------------------------------------------------------------------
# Material registry
# -----------------------------------------------------------------------

class TestMaterialRegistry:
    def test_materials_count(self):
        # 4 TI/chalcogenide materials + Graphene
        assert len(MATERIALS) == 5

    def test_fete_properties(self, fete):
        assert fete.formula == "FeTe"
        assert fete.lattice_type == "square"
        assert fete.a == pytest.approx(3.82)
        assert fete.role == "substrate"
        assert fete.space_group == "P4/nmm"

    def test_sb2te3_properties(self, sb2te3):
        assert sb2te3.formula == "Sb2Te3"
        assert sb2te3.lattice_type == "hexagonal"
        assert sb2te3.a == pytest.approx(4.264)
        assert sb2te3.role == "overlayer"
        assert sb2te3.space_group == "R-3m"

    def test_bi2te3_properties(self, bi2te3):
        assert bi2te3.formula == "Bi2Te3"
        assert bi2te3.a == pytest.approx(4.386)
        assert bi2te3.role == "overlayer"

    def test_graphene_registered(self, graphene):
        assert graphene.formula == "Graphene"
        assert graphene.lattice_type == "hexagonal"
        assert graphene.a == pytest.approx(2.46)
        assert graphene.c == pytest.approx(3.35)
        assert graphene.space_group == "P6/mmm"
        assert graphene.role == "both"

    def test_get_material_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_material("UnknownMaterial")

    def test_list_materials_all(self):
        all_mats = list_materials()
        assert len(all_mats) == 5

    def test_list_materials_substrates(self):
        # FeTe ("substrate") and Graphene ("both") both qualify.
        substrates = list_materials(role="substrate")
        formulas = {m.formula for m in substrates}
        assert "FeTe" in formulas
        assert "Graphene" in formulas
        assert len(substrates) == 2

    def test_list_materials_overlayers(self):
        # 3 TI overlayers plus Graphene ("both") = 4.
        overlayers = list_materials(role="overlayer")
        formulas = {m.formula for m in overlayers}
        assert "Graphene" in formulas
        assert len(overlayers) == 4

    def test_list_materials_both_role(self, graphene):
        """Graphene (role='both') must appear in both filtered lists."""
        assert graphene in list_materials(role="substrate")
        assert graphene in list_materials(role="overlayer")

    def test_material_is_frozen(self, fete):
        with pytest.raises(AttributeError):
            fete.a = 999.0  # type: ignore[misc]


# -----------------------------------------------------------------------
# Lattice generation
# -----------------------------------------------------------------------

class TestLatticeGeneration:
    def test_square_lattice_shape(self):
        pts = square_lattice(3.82, 5, 5)
        assert pts.shape == (25, 2)

    def test_square_lattice_spacing(self):
        pts = square_lattice(3.82, 2, 1)
        # Two points: (0,0) and (3.82, 0)
        assert pts.shape == (2, 2)
        dist = np.linalg.norm(pts[1] - pts[0])
        assert dist == pytest.approx(3.82)

    def test_hexagonal_lattice_shape(self):
        pts = hexagonal_lattice(4.264, 4, 4)
        assert pts.shape == (16, 2)

    def test_hexagonal_lattice_first_neighbour(self):
        pts = hexagonal_lattice(4.264, 2, 2)
        # Check that minimum nonzero distance matches lattice constant
        dists = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dists.append(np.linalg.norm(pts[i] - pts[j]))
        assert min(dists) == pytest.approx(4.264, abs=1e-6)


# -----------------------------------------------------------------------
# Rotation
# -----------------------------------------------------------------------

class TestRotation:
    def test_identity_rotation(self):
        pts = np.array([[1.0, 0.0], [0.0, 1.0]])
        rotated = apply_rotation(pts, 0.0)
        np.testing.assert_allclose(rotated, pts, atol=1e-12)

    def test_90_degree_rotation(self):
        pts = np.array([[1.0, 0.0]])
        rotated = apply_rotation(pts, 90.0)
        np.testing.assert_allclose(rotated, [[0.0, 1.0]], atol=1e-12)

    def test_360_returns_to_start(self):
        pts = np.array([[3.5, -2.1]])
        rotated = apply_rotation(pts, 360.0)
        np.testing.assert_allclose(rotated, pts, atol=1e-12)


# -----------------------------------------------------------------------
# Reciprocal lattice
# -----------------------------------------------------------------------

class TestReciprocalLattice:
    def test_square_reciprocal_orthogonality(self):
        a1, a2 = lattice_vectors("square", 3.82)
        b1, b2 = reciprocal_vectors(a1, a2)
        assert np.dot(b1, a1) == pytest.approx(2 * np.pi)
        assert np.dot(b2, a2) == pytest.approx(2 * np.pi)
        assert np.dot(b1, a2) == pytest.approx(0.0, abs=1e-10)
        assert np.dot(b2, a1) == pytest.approx(0.0, abs=1e-10)

    def test_hexagonal_reciprocal_orthogonality(self):
        a1, a2 = lattice_vectors("hexagonal", 4.264)
        b1, b2 = reciprocal_vectors(a1, a2)
        assert np.dot(b1, a1) == pytest.approx(2 * np.pi)
        assert np.dot(b2, a2) == pytest.approx(2 * np.pi)
        assert np.dot(b1, a2) == pytest.approx(0.0, abs=1e-10)
        assert np.dot(b2, a1) == pytest.approx(0.0, abs=1e-10)

    def test_unknown_lattice_type_raises(self):
        with pytest.raises(ValueError, match="Unknown lattice_type"):
            lattice_vectors("triangular", 1.0)


class TestMaterialValidation:
    """Edge case and input validation tests for materials module."""

    def test_material_negative_a_raises(self):
        with pytest.raises(ValueError):
            Material(
                name="Test",
                formula="Xx",
                lattice_type="square",
                a=-1.0,
                c=1.0,
                space_group="P1",
                description="test",
                role="substrate",
            )

    def test_material_zero_a_raises(self):
        with pytest.raises(ValueError):
            Material(
                name="Test",
                formula="Xx",
                lattice_type="square",
                a=0.0,
                c=1.0,
                space_group="P1",
                description="test",
                role="substrate",
            )

    def test_material_negative_c_raises(self):
        with pytest.raises(ValueError):
            Material(
                name="Test",
                formula="Xx",
                lattice_type="square",
                a=1.0,
                c=-1.0,
                space_group="P1",
                description="test",
                role="substrate",
            )

    def test_material_invalid_lattice_type_raises(self):
        with pytest.raises(ValueError):
            Material(
                name="Test",
                formula="Xx",
                lattice_type="cubic",
                a=1.0,
                c=1.0,
                space_group="P1",
                description="test",
                role="substrate",
            )

    def test_material_invalid_role_raises(self):
        with pytest.raises(ValueError):
            Material(
                name="Test",
                formula="Xx",
                lattice_type="square",
                a=1.0,
                c=1.0,
                space_group="P1",
                description="test",
                role="insulator",
            )

    def test_square_lattice_zero_a_raises(self):
        with pytest.raises(ValueError):
            square_lattice(0.0, 5, 5)

    def test_hexagonal_lattice_zero_a_raises(self):
        with pytest.raises(ValueError):
            hexagonal_lattice(0.0, 5, 5)

    def test_reciprocal_parallel_vectors_raises(self):
        with pytest.raises(ValueError):
            reciprocal_vectors(np.array([1.0, 0.0]), np.array([2.0, 0.0]))
