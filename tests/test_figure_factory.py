"""Tests for the Plotly figure builders and the web-page wiring they depend on.

Regression tests for six live web-app fixes:

- U1: a 3D clip plane drops the z-layers above it. NaN-masking them made
  plotly.js draw nothing, which blanked the /proximity3d clip slider.
- U2: the phase-diagram boundary is a lines-only contour, so the template's
  default colorscale no longer floods the /phase map.
- U3: the /magnetic page fills its whole ±extent window with vortices.
- U4: "RdBu_r" resolves to the shared coolwarm LUT, so every signed map reads
  low = blue, high = red, like the Rust desktop app.
- U5: the /proximity3d "Interface" and "ξ_prox" arrows point at the stack's
  vertical edge nearest the camera, and a marker above the clip plane is
  left out instead of stretching the z-axis over an empty box.
- U6: the /phase "Delta_avg" line label is white on a dark box, legible over
  both phase colors in both themes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from waytogocoop.components.colormaps import get_plotly_colorscale
from waytogocoop.components.figure_factory import (
    _resolve_colorscale,
    create_2d_contour,
    create_3d_cone_field,
    create_3d_isosurface,
    create_3d_surface,
    create_3d_volume,
    create_gap_heatmap,
    create_phase_colormap,
    create_pseudo_field_heatmap,
    create_quiver_field,
    create_signed_field_heatmap,
    create_vortex_overlay_heatmap,
)
from waytogocoop.computation.magnetic import generate_vortex_positions, vortex_lattice_period
from waytogocoop.computation.topological import phase_diagram_sweep

# Endpoints of the coolwarm LUT in crates/moire-core/src/colormap.rs.
COOLWARM_LOW = "rgb(59, 76, 192)"  # blue
COOLWARM_HIGH = "rgb(180, 4, 38)"  # red

# The /proximity3d z grid: 30 layers from -50 to 300 Å.
Z_LAYERS = np.linspace(-50.0, 300.0, 30)


def _stops(colorscale) -> list[list[object]]:
    """Plotly stores a list colorscale as tuples; compare it as lists."""
    return [[float(t), color] for t, color in colorscale]


def _linear_field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """f = x + 2y + 3z laid out (nz, ny, nx) like topological.gap_3d.

    Every grid point gets a distinct value, so a mis-ordered flatten shows.
    """
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return xx + 2.0 * yy + 3.0 * zz


@pytest.fixture
def volume_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small (nz, ny, nx) field with nx != ny, on the page's z layers."""
    x = np.linspace(-30.0, 30.0, 7)
    y = np.linspace(-20.0, 20.0, 5)
    return x, y, Z_LAYERS, _linear_field(x, y, Z_LAYERS)


# ---------------------------------------------------------------------------
# U1: clip plane
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", [create_3d_isosurface, create_3d_volume])
class TestClipPlane:
    def test_clip_drops_layers_instead_of_nan(self, builder, volume_grid):
        x, y, z, values = volume_grid
        trace = builder(x, y, z, values, clip_z=100.0).data[0]
        value = np.asarray(trace.value)
        assert value.size > 0
        assert np.isfinite(value).all()
        assert np.asarray(trace.z).max() <= 100.0

    def test_keeps_every_layer_at_or_below_the_plane(self, builder, volume_grid):
        x, y, z, values = volume_grid
        trace = builder(x, y, z, values, clip_z=100.0).data[0]
        kept = z[z <= 100.0]
        np.testing.assert_allclose(np.unique(trace.z), kept)
        assert np.asarray(trace.value).size == x.size * y.size * kept.size

    def test_values_stay_aligned_with_coordinates(self, builder, volume_grid):
        x, y, z, values = volume_grid
        trace = builder(x, y, z, values, clip_z=100.0).data[0]
        expected = np.asarray(trace.x) + 2.0 * np.asarray(trace.y) + 3.0 * np.asarray(trace.z)
        np.testing.assert_allclose(trace.value, expected)

    @pytest.mark.parametrize("clip_z", [None, 300.0, 1000.0])
    def test_plane_at_or_above_the_top_keeps_everything(self, builder, volume_grid, clip_z):
        x, y, z, values = volume_grid
        trace = builder(x, y, z, values, clip_z=clip_z).data[0]
        np.testing.assert_allclose(np.unique(trace.z), z)
        assert np.isfinite(np.asarray(trace.value)).all()

    @pytest.mark.parametrize("clip_z", [-50.0, -45.0, -100.0])
    def test_plane_below_the_second_layer_keeps_the_bottom_two(self, builder, volume_grid, clip_z):
        # A 3D trace needs two layers to draw anything.
        x, y, z, values = volume_grid
        trace = builder(x, y, z, values, clip_z=clip_z).data[0]
        np.testing.assert_allclose(np.unique(trace.z), z[:2])
        assert np.isfinite(np.asarray(trace.value)).all()

    def test_input_field_is_not_modified(self, builder, volume_grid):
        x, y, z, values = volume_grid
        before = values.copy()
        builder(x, y, z, values, clip_z=100.0)
        np.testing.assert_array_equal(values, before)


class TestClipPlaneLevels:
    def test_isosurface_levels_come_from_the_whole_field(self, volume_grid):
        x, y, z, values = volume_grid
        full = create_3d_isosurface(x, y, z, values).data[0]
        clipped = create_3d_isosurface(x, y, z, values, clip_z=100.0).data[0]
        assert full.isomin == pytest.approx(np.percentile(values, 20))
        assert full.isomax == pytest.approx(np.percentile(values, 80))
        assert (clipped.isomin, clipped.isomax) == (full.isomin, full.isomax)

    def test_volume_range_comes_from_the_whole_field(self, volume_grid):
        x, y, z, values = volume_grid
        clipped = create_3d_volume(x, y, z, values, clip_z=100.0).data[0]
        assert clipped.isomin == pytest.approx(values.min())
        assert clipped.isomax == pytest.approx(values.max())


# ---------------------------------------------------------------------------
# U2: phase-diagram boundary
# ---------------------------------------------------------------------------


class TestPhaseBoundary:
    @staticmethod
    def _figure(dark: bool = True):
        b = np.linspace(0.0, 8.0, 40)
        delta = np.linspace(0.0, 6.0, 30)
        return create_phase_colormap(b, delta, phase_diagram_sweep(b, delta), dark=dark)

    @staticmethod
    def _contour(fig):
        contours = [t for t in fig.data if t.type == "contour"]
        assert len(contours) == 1
        return contours[0]

    def test_boundary_draws_lines_only(self):
        contour = self._contour(self._figure())
        assert contour.contours.coloring == "lines"

    def test_boundary_sits_between_the_two_phases(self):
        contour = self._contour(self._figure())
        assert contour.contours.start == 0.5
        assert contour.contours.end == 0.5

    def test_phase_heatmap_is_drawn_under_the_boundary(self):
        fig = self._figure()
        assert [t.type for t in fig.data] == ["heatmap", "contour"]

    @pytest.mark.parametrize(("dark", "color"), [(True, "white"), (False, "black")])
    def test_boundary_line_color(self, dark, color):
        # In "lines" mode Plotly colors the line from the colorscale and
        # ignores line.color, so every stop must be the boundary color.
        contour = self._contour(self._figure(dark=dark))
        assert {stop[1] for stop in contour.colorscale} == {color}
        assert contour.showscale is False


# ---------------------------------------------------------------------------
# U4: colormap orientation
# ---------------------------------------------------------------------------


class TestColormapOrientation:
    @pytest.mark.parametrize("name", ["RdBu_r", "rdbu_r"])
    def test_rdbu_r_is_coolwarm(self, name):
        assert _resolve_colorscale(name) == get_plotly_colorscale("coolwarm")

    @pytest.mark.parametrize("name", ["RdBu", "rdbu"])
    def test_rdbu_is_coolwarm_r(self, name):
        assert _resolve_colorscale(name) == get_plotly_colorscale("coolwarm_r")

    def test_rdbu_r_runs_blue_low_to_red_high(self):
        scale = _resolve_colorscale("RdBu_r")
        assert scale[0] == [0.0, COOLWARM_LOW]
        assert scale[-1] == [1.0, COOLWARM_HIGH]

    def test_plotly_builtin_names_pass_through(self):
        assert _resolve_colorscale("Hot") == "Hot"


def _xy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-20.0, 20.0, 9)
    y = np.linspace(-10.0, 10.0, 5)
    return x, y, np.add.outer(y, x)


def _signed_traces():
    """Every builder that draws a signed or gap-valued map, as (id, trace factory)."""
    x, y, field = _xy()
    jx, jy = np.ones_like(field), np.zeros_like(field)
    z = np.linspace(-50.0, 300.0, 4)
    volume = _linear_field(x, y, z)
    cores = np.zeros((1, 2))
    return [
        ("gap-heatmap", lambda: create_gap_heatmap(x, y, field).data[0]),
        ("vortex-overlay", lambda: create_vortex_overlay_heatmap(x, y, field, cores).data[0]),
        ("quiver-backdrop", lambda: create_quiver_field(x, y, jx, jy, field, skip=2).data[0]),
        (
            "cone-backdrop",
            lambda: create_3d_cone_field(x, y, 0.0, jx, jy, base_surface=field, skip=2).data[0],
        ),
        ("isosurface", lambda: create_3d_isosurface(x, y, z, volume).data[0]),
        ("volume", lambda: create_3d_volume(x, y, z, volume).data[0]),
        (
            "surface-gap-colored",
            lambda: create_3d_surface(
                x, y, field, "t", colorscale="RdBu_r", surfacecolor=field
            ).data[0],
        ),
        (
            "surface-mesh3d",
            lambda: create_3d_surface(
                x, y, field, "t", colorscale="RdBu_r", high_density=True
            ).data[0],
        ),
        ("contour", lambda: create_2d_contour(x, y, field, "t", colorscale="RdBu_r").data[0]),
        ("pseudo-field", lambda: create_pseudo_field_heatmap(x, y, field).data[0]),
        ("pdw-signed-field", lambda: create_signed_field_heatmap(x, y, field, "t", "z").data[0]),
    ]


class TestSignedMapsReadHighRed:
    @pytest.mark.parametrize(
        "make_trace",
        [pytest.param(make, id=name) for name, make in _signed_traces()],
    )
    def test_uses_coolwarm(self, make_trace):
        trace = make_trace()
        assert _stops(trace.colorscale) == get_plotly_colorscale("coolwarm")


# ---------------------------------------------------------------------------
# U3: vortex window on the /magnetic page
# ---------------------------------------------------------------------------


def _page_module(stem: str):
    """Return a page module that create_app() imported (Dash names it pages.<stem>)."""
    suffix = f"waytogocoop/pages/{stem}.py"
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).as_posix().endswith(suffix):
            return module
    raise LookupError(f"create_app() did not import the {stem!r} page")


def _app_page(stem: str):
    # dash.register_page raises unless an app exists, so the page modules
    # can only be imported through create_app().
    from waytogocoop.app import create_app

    create_app()
    return _page_module(stem)


@pytest.fixture(scope="module")
def magnetic_page():
    return _app_page("magnetic_field")


def _vortex_cores(page, extent: float, bz: float, grid_size: int = 60) -> np.ndarray:
    """Vortex-core markers from the page's main callback in its "vortex" view."""
    fig = page.update_magnetic(
        substrate_key="FeTe",
        overlayer_key="Sb2Te3",
        twist=0.0,
        grid_size=grid_size,
        extent=extent,
        Bz=bz,
        Bx=0.0,
        By=0.0,
        viz_mode="vortex",
        xi_prox=100.0,
        g_factor=30.0,
        bc2=47.0,
        theme="dark",
    )[0]
    markers = [t for t in fig.data if t.type == "scatter"]
    assert len(markers) == 1, f"no vortex markers; figure title: {fig.layout.title.text!r}"
    return np.column_stack([markers[0].x, markers[0].y])


# Importing the pages builds their dash_table.DataTable components, which makes
# Dash 4 announce the component's future removal. Unrelated to these tests.
_DATATABLE_NOTICE = r"ignore:(?s).*dash_table\.DataTable will be removed:DeprecationWarning"


@pytest.mark.filterwarnings(_DATATABLE_NOTICE)
class TestMagneticPageVortexWindow:
    def test_page_passes_the_full_side_length(self, magnetic_page):
        # The web extent is a half-width; the helper takes the full side.
        cores = _vortex_cores(magnetic_page, extent=300.0, bz=4.0)
        np.testing.assert_allclose(cores, generate_vortex_positions(4.0, 600.0, 60))

    def test_four_tesla_over_300_angstrom_shows_seven_cores(self, magnetic_page):
        cores = _vortex_cores(magnetic_page, extent=300.0, bz=4.0)
        assert len(cores) == 7  # the center plus a hexagon of six
        radii = np.sort(np.hypot(cores[:, 0], cores[:, 1]))
        assert radii[0] == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(radii[1:], vortex_lattice_period(4.0))

    @pytest.mark.parametrize(("extent", "bz"), [(300.0, 4.0), (300.0, 20.0), (200.0, 20.0)])
    def test_cores_span_the_whole_window(self, magnetic_page, extent, bz):
        cores = _vortex_cores(magnetic_page, extent=extent, bz=bz)
        a_v = vortex_lattice_period(bz)
        row = a_v * np.sqrt(3.0) / 2.0
        assert np.all(np.abs(cores) <= extent)
        # The lattice reaches within one spacing of every edge of ±extent,
        # beyond the ±extent/2 square that the half-width bug filled.
        assert cores[:, 0].max() >= extent - a_v
        assert cores[:, 0].min() <= -(extent - a_v)
        assert cores[:, 1].max() >= extent - row
        assert cores[:, 1].min() <= -(extent - row)


# ---------------------------------------------------------------------------
# U5: height markers on the /proximity3d iso and volume views
# ---------------------------------------------------------------------------

_PROXIMITY_VIEWS = ("iso", "volume")


@pytest.fixture(scope="module")
def proximity_page():
    return _app_page("proximity_3d")


def _proximity_figure(
    page,
    view: str = "iso",
    clip_z: float = 300.0,
    xi_prox: float = 100.0,
    theme: str = "dark",
):
    """The /proximity3d main figure from the page callback (page defaults otherwise).

    clip_z = 300, the top of the slider, turns clipping off.
    """
    fig = page.update_proximity(
        substrate_key="FeTe",
        overlayer_key="Sb2Te3",
        twist=0.0,
        grid_size=200,
        extent=100.0,
        xi_prox=xi_prox,
        transparency=0.8,
        z_slice_idx=15,
        view_mode=view,
        iso_range=[0.2, 0.8],
        clip_z=clip_z,
        theme=theme,
    )[0]
    assert "Computation error" not in str(fig.layout.title.text)
    return fig


def _gap_trace(fig):
    """The isosurface or volume trace that draws Δ(x, y, z)."""
    return next(t for t in fig.data if t.type in ("isosurface", "volume"))


def _markers(fig) -> list:
    return list(fig.layout.scene.annotations)


def _edge_lines(fig) -> list:
    return [t for t in fig.data if t.type == "scatter3d"]


def _nearest_vertical_edge(fig) -> tuple[float, float]:
    """(x, y) of the drawn stack's vertical edge nearest the scene camera.

    Plotly centres the box on the scene origin with the sides set by the
    aspect ratio, and the camera eye is given in those units.
    """
    trace = _gap_trace(fig)
    xs, ys = np.asarray(trace.x, dtype=float), np.asarray(trace.y, dtype=float)
    scene = fig.layout.scene
    eye, aspect = scene.camera.eye, scene.aspectratio

    def box_units(value, values, side):
        return (value - (values.min() + values.max()) / 2.0) / np.ptp(values) * side

    def eye_distance(corner):
        cx, cy = corner
        return np.hypot(box_units(cx, xs, aspect.x) - eye.x, box_units(cy, ys, aspect.y) - eye.y)

    corners = [(cx, cy) for cx in (xs.min(), xs.max()) for cy in (ys.min(), ys.max())]
    return min(corners, key=eye_distance)


def _scene_z_span(fig) -> tuple[float, float]:
    """The z span plotly.js autoranges the scene over.

    plotly.js takes the bounds of every 3D trace and of every scene
    annotation, then pads the span by 1/32 on each side.
    """
    zs = [np.asarray(t.z, dtype=float) for t in fig.data]
    zs.append(np.asarray([a.z for a in _markers(fig)], dtype=float))
    zs = np.concatenate(zs)
    return float(zs.min()), float(zs.max())


@pytest.mark.filterwarnings(_DATATABLE_NOTICE)
class TestProximityHeightMarkers:
    @pytest.mark.parametrize("view", _PROXIMITY_VIEWS)
    def test_arrows_point_at_the_vertical_edge_nearest_the_camera(self, proximity_page, view):
        fig = _proximity_figure(proximity_page, view)
        edge_x, edge_y = _nearest_vertical_edge(fig)
        markers = _markers(fig)
        assert len(markers) == 2
        for marker in markers:
            assert (marker.x, marker.y) == (edge_x, edge_y)

    @pytest.mark.parametrize("view", _PROXIMITY_VIEWS)
    @pytest.mark.parametrize("xi_prox", [100.0, 250.0])
    def test_arrows_mark_the_interface_and_xi(self, proximity_page, view, xi_prox):
        fig = _proximity_figure(proximity_page, view, xi_prox=xi_prox)
        heights = {marker.z: marker.text for marker in _markers(fig)}
        assert sorted(heights) == [0.0, xi_prox]
        assert "Interface" in heights[0.0]
        assert heights[xi_prox] == f"ξ_prox ≈ {xi_prox:.0f} Å"

    @pytest.mark.parametrize("view", _PROXIMITY_VIEWS)
    @pytest.mark.parametrize("clip_z", [50.0, 95.0])
    def test_no_xi_marker_when_the_clip_plane_is_below_xi(self, proximity_page, view, clip_z):
        fig = _proximity_figure(proximity_page, view, clip_z=clip_z, xi_prox=100.0)
        assert [marker.z for marker in _markers(fig)] == [0.0]

    @pytest.mark.parametrize("view", _PROXIMITY_VIEWS)
    def test_xi_marker_stays_when_the_clip_plane_is_at_xi(self, proximity_page, view):
        fig = _proximity_figure(proximity_page, view, clip_z=100.0, xi_prox=100.0)
        assert sorted(marker.z for marker in _markers(fig)) == [0.0, 100.0]

    @pytest.mark.parametrize("view", _PROXIMITY_VIEWS)
    def test_xi_beyond_the_z_grid_gets_no_marker(self, proximity_page, view):
        # The slider reaches 500 Å; the z grid stops at 300 Å.
        fig = _proximity_figure(proximity_page, view, xi_prox=400.0)
        assert [marker.z for marker in _markers(fig)] == [0.0]

    @pytest.mark.parametrize("view", _PROXIMITY_VIEWS)
    def test_z_range_ends_at_the_top_drawn_layer_below_the_interface(self, proximity_page, view):
        # At clip -45 the builder keeps the bottom two layers, -50 and -37.9 Å.
        fig = _proximity_figure(proximity_page, view, clip_z=-45.0)
        top_layer = float(np.asarray(_gap_trace(fig).z, dtype=float).max())
        assert top_layer == pytest.approx(Z_LAYERS[1])
        assert _markers(fig) == []
        assert _scene_z_span(fig)[1] == top_layer

    @pytest.mark.parametrize("view", _PROXIMITY_VIEWS)
    def test_z_range_ends_at_the_top_drawn_layer_below_xi(self, proximity_page, view):
        fig = _proximity_figure(proximity_page, view, clip_z=50.0)
        top_layer = float(np.asarray(_gap_trace(fig).z, dtype=float).max())
        assert top_layer == pytest.approx(Z_LAYERS[Z_LAYERS <= 50.0][-1])
        assert _scene_z_span(fig)[1] == top_layer

    @pytest.mark.parametrize("view", _PROXIMITY_VIEWS)
    def test_edge_line_joins_the_two_marked_heights(self, proximity_page, view):
        fig = _proximity_figure(proximity_page, view, xi_prox=250.0)
        (line,) = _edge_lines(fig)
        edge_x, edge_y = _nearest_vertical_edge(fig)
        np.testing.assert_array_equal(line.x, [edge_x, edge_x])
        np.testing.assert_array_equal(line.y, [edge_y, edge_y])
        np.testing.assert_array_equal(line.z, [0.0, 250.0])
        assert line.mode == "lines"
        assert line.hoverinfo == "skip"

    @pytest.mark.parametrize("view", _PROXIMITY_VIEWS)
    @pytest.mark.parametrize("clip_z", [-45.0, 50.0])
    def test_no_edge_line_without_both_markers(self, proximity_page, view, clip_z):
        fig = _proximity_figure(proximity_page, view, clip_z=clip_z)
        assert _edge_lines(fig) == []

    @pytest.mark.parametrize(("theme", "ink"), [("dark", "white"), ("light", "black")])
    def test_markers_follow_the_theme(self, proximity_page, theme, ink):
        fig = _proximity_figure(proximity_page, theme=theme)
        for marker in _markers(fig):
            assert marker.font.color == ink
            assert marker.arrowcolor == ink
        assert _edge_lines(fig)[0].line.color == ink


# ---------------------------------------------------------------------------
# U6: the Delta_avg label on the /phase map
# ---------------------------------------------------------------------------

# The CSS color names the /phase page and its light-theme map use.
_CSS_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "steelblue": (70, 130, 180),
    "firebrick": (178, 34, 34),
}


def _rgba(color: str) -> tuple[tuple[float, float, float], float]:
    """A CSS color as ((r, g, b), alpha) in 0-255 channels."""
    color = color.strip().lower()
    if color in _CSS_COLORS:
        return _CSS_COLORS[color], 1.0
    if color.startswith("#"):
        return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5)), 1.0
    parts = [float(p) for p in color[color.index("(") + 1 : -1].split(",")]
    return tuple(parts[:3]), (parts[3] if len(parts) == 4 else 1.0)


def _luminance(rgb) -> float:
    """WCAG 2 relative luminance of an sRGB color."""
    channels = np.asarray(rgb, dtype=float) / 255.0
    linear = np.where(channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)
    return float(np.dot([0.2126, 0.7152, 0.0722], linear))


def _contrast(a, b) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.fixture(scope="module")
def phase_page():
    return _app_page("phase_diagram")


def _avg_label(fig):
    (label,) = [a for a in fig.layout.annotations if a.text.startswith("Delta_avg")]
    return label


@pytest.mark.filterwarnings(_DATATABLE_NOTICE)
class TestPhaseAverageGapLabel:
    @staticmethod
    def _figure(page, theme: str):
        fig = page.update_phase("FeTe", "Sb2Te3", 0, 8, 0.1, 6, 0, 30, 200, theme)[0]
        assert "Computation error" not in str(fig.layout.title.text)
        return fig

    @pytest.mark.parametrize("theme", ["dark", "light"])
    def test_label_is_white_on_a_dark_box(self, phase_page, theme):
        label = _avg_label(self._figure(phase_page, theme))
        assert label.bgcolor == "rgba(0, 0, 0, 0.6)"
        assert label.font.color == "white"

    @pytest.mark.parametrize("theme", ["dark", "light"])
    def test_label_reads_over_both_phase_colors(self, phase_page, theme):
        fig = self._figure(phase_page, theme)
        label = _avg_label(fig)
        ink, _ = _rgba(label.font.color or fig.layout.template.layout.font.color)
        box, alpha = _rgba(label.bgcolor or "rgba(0, 0, 0, 0)")
        for _stop, region in fig.data[0].colorscale:
            under, _ = _rgba(region)
            backdrop = alpha * np.asarray(box) + (1.0 - alpha) * np.asarray(under)
            assert _contrast(ink, backdrop) >= 4.5, region
