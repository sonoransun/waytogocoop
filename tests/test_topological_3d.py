"""Cross-stack LUT parity + 3D Majorana density tests.

The cross-stack dual-stack rule requires both Python and Rust renderers to
produce visually identical screenshots. That requirement is enforced here by
asserting specific endpoint colors of the shared colormap JSON. If the Rust
colormap tables are edited without regenerating the JSON
(``cargo run -p moire-core --bin dump_lut``), this test will fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from waytogocoop.components.colormaps import (
    LIGHT_POSITION,
    LIGHTING_PRESET,
    available_colormaps,
    get_plotly_colorscale,
)


class TestColormapLutParity:
    """Endpoints hardcoded from crates/moire-core/src/colormap.rs tables."""

    def test_all_names_present(self):
        assert available_colormaps() == ["coolwarm", "inferno", "plasma", "viridis"]

    @pytest.mark.parametrize(
        "name,first_rgb,last_rgb",
        [
            ("viridis", (68, 1, 84), (253, 231, 37)),
            ("inferno", (0, 0, 4), (252, 255, 252)),
            ("coolwarm", (59, 76, 192), (180, 4, 38)),
            ("plasma", (13, 8, 135), (240, 249, 33)),
        ],
    )
    def test_endpoint_colors(self, name, first_rgb, last_rgb):
        scale = get_plotly_colorscale(name)
        assert len(scale) == 256
        assert scale[0][0] == pytest.approx(0.0)
        assert scale[-1][0] == pytest.approx(1.0)
        assert scale[0][1] == f"rgb({first_rgb[0]}, {first_rgb[1]}, {first_rgb[2]})"
        assert scale[-1][1] == f"rgb({last_rgb[0]}, {last_rgb[1]}, {last_rgb[2]})"

    def test_reversed_scale_swaps_endpoints(self):
        forward = get_plotly_colorscale("coolwarm")
        reverse = get_plotly_colorscale("coolwarm_r")
        assert forward[0][1] == reverse[-1][1]
        assert forward[-1][1] == reverse[0][1]

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            get_plotly_colorscale("notacolormap")


class TestLightingPreset:
    def test_required_keys(self):
        assert set(LIGHTING_PRESET) == {
            "ambient",
            "diffuse",
            "specular",
            "roughness",
            "fresnel",
        }

    def test_values_in_range(self):
        for v in LIGHTING_PRESET.values():
            assert 0.0 <= v <= 1.0

    def test_light_position_xyz(self):
        assert set(LIGHT_POSITION) == {"x", "y", "z"}


class TestMajoranaDensity3D:
    """Shape + decay checks for the new 3D Majorana probability density."""

    @pytest.fixture
    def grid(self):
        x = np.linspace(-100.0, 100.0, 32)
        y = np.linspace(-100.0, 100.0, 32)
        z = np.linspace(-50.0, 200.0, 25)
        vortex_pos = np.array([[0.0, 0.0], [50.0, 50.0]])
        return x, y, z, vortex_pos

    def test_shape(self, grid):
        pytest.importorskip("waytogocoop.computation.topological")
        from waytogocoop.computation.topological import majorana_probability_density_3d

        x, y, z, vortex_pos = grid
        d = majorana_probability_density_3d(
            x, y, z, vortex_pos, xi_M=50.0, k_F=0.1, xi_prox=100.0
        )
        assert d.shape == (z.size, y.size, x.size)

    def test_nonnegative(self, grid):
        from waytogocoop.computation.topological import majorana_probability_density_3d

        x, y, z, vortex_pos = grid
        d = majorana_probability_density_3d(
            x, y, z, vortex_pos, xi_M=50.0, k_F=0.1, xi_prox=100.0
        )
        assert d.min() >= 0.0

    def test_decay_with_distance_from_interface(self, grid):
        from waytogocoop.computation.topological import majorana_probability_density_3d

        x, y, z, vortex_pos = grid
        d = majorana_probability_density_3d(
            x, y, z, vortex_pos, xi_M=50.0, k_F=0.1, xi_prox=100.0
        )
        z0_idx = int(np.argmin(np.abs(z)))
        # Integrated density should be maximal at z=0 (the interface).
        planar_sums = d.sum(axis=(1, 2))
        assert planar_sums[z0_idx] == pytest.approx(planar_sums.max())
        # And decay monotonically moving away in either direction.
        for direction in (slice(z0_idx, None), slice(None, z0_idx + 1)):
            slab = planar_sums[direction]
            if direction.start is None:
                slab = slab[::-1]
            diffs = np.diff(slab)
            assert (diffs <= 1e-9).all(), f"non-monotone decay: {diffs}"

    def test_peak_near_vortex_cores(self, grid):
        from waytogocoop.computation.topological import majorana_probability_density_3d

        x, y, z, vortex_pos = grid
        d = majorana_probability_density_3d(
            x, y, z, vortex_pos, xi_M=50.0, k_F=0.1, xi_prox=100.0
        )
        z0 = int(np.argmin(np.abs(z)))
        planar = d[z0]  # (ny, nx)
        ix = int(np.argmin(np.abs(x - 0.0)))
        iy = int(np.argmin(np.abs(y - 0.0)))
        assert planar[iy, ix] > 0.5 * planar.max()
