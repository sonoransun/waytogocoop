"""Checks that scripts/capture_screenshots.py stays faithful to the Dash pages.

The capture script cannot import ``waytogocoop.pages.*``: ``dash.register_page``
raises ``PageError`` until an app exists. It therefore copies the page presets
and the callback logic its scenes mirror. These tests import the pages the one
legal way, through ``waytogocoop.app.create_app()``, and check

- every copied preset against the page's ``_PRESETS`` entry;
- the "app" scenes against the figures the page callbacks build at the same
  settings (same data arrays; the docs styling may relabel);
- the scene registry and its command line;
- that every PNG in docs/images is fully opaque.
"""

from __future__ import annotations

import importlib.util
import re
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from waytogocoop.computation.fourier import identify_peaks

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "capture_screenshots.py"
IMAGES = REPO / "docs" / "images"

# Importing the pages builds their dash_table.DataTable components, which makes
# Dash 4 announce the component's future removal. Unrelated to these tests.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:(?s).*dash_table\.DataTable will be removed:DeprecationWarning"
)


@pytest.fixture(scope="module")
def capture():
    """The capture script, imported as a module (its main() does not run)."""
    spec = importlib.util.spec_from_file_location("capture_screenshots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


def _page_module(stem: str):
    """A page module that create_app() imported (Dash names it pages.<stem>)."""
    suffix = f"waytogocoop/pages/{stem}.py"
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).as_posix().endswith(suffix):
            return module
    raise LookupError(f"create_app() did not import the {stem!r} page")


@pytest.fixture(scope="module")
def pages():
    from waytogocoop.app import create_app

    create_app()
    stems = (
        "moire_viewer",
        "parameter_sweep",
        "fourier_analysis",
        "graphene",
        "magnetic_field",
        "proximity_3d",
        "phase_diagram",
        "spin_fluctuation",
        "pair_density_wave",
    )
    return {stem: _page_module(stem) for stem in stems}


def _first(fig, trace_type: str):
    return next(t for t in fig.data if t.type == trace_type)


def _array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


# ---------------------------------------------------------------------------
# Copied presets
# ---------------------------------------------------------------------------


class TestCopiedPresets:
    def test_every_copy_matches_its_page(self, capture, pages):
        assert capture.PAGE_PRESET_COPIES, "the script declares no preset copies"
        for (stem, key), copy in capture.PAGE_PRESET_COPIES.items():
            assert pages[stem]._PRESETS[key] == copy, f"{stem} preset {key!r} drifted"

    def test_viewer_paper_default_is_the_page_default(self, capture, pages):
        assert pages["moire_viewer"]._DEFAULTS == capture._VIEWER_PAPER_DEFAULT

    def test_pdw_base_preset_matches(self, capture, pages):
        assert pages["pair_density_wave"]._BASE_PRESET == capture._PDW_BASE_PRESET

    def test_sf_base_preset_matches(self, capture, pages):
        assert pages["spin_fluctuation"]._BASE_PRESET == capture._SF_BASE_PRESET

    def test_comparison_overlayers_match(self, capture):
        # substrate_comparison.py is not in `pages` above, so import it the same way.
        from waytogocoop.app import create_app

        create_app()
        page = _page_module("substrate_comparison")
        assert tuple(page._OVERLAYER_FORMULAS) == capture._COMPARISON_OVERLAYERS


# ---------------------------------------------------------------------------
# App scenes against the page callbacks
# ---------------------------------------------------------------------------


def _pdw_figures(pages, key: str, **override):
    """The page callback's figures at the page's own preset (not the script's copy)."""
    page = pages["pair_density_wave"]
    values = {**page._PRESETS[key], **override}
    args = [values[key] for _cid, _prop, key in page._PRESET_BINDINGS]
    figures = page._update_pdw(*args, "dark")
    assert "Computation error" not in str(figures[0].layout.title.text)
    return figures


def _same_lines(page_fig, scene_fig) -> None:
    page_lines = [t for t in page_fig.data if t.type == "scatter"]
    scene_lines = [t for t in scene_fig.data if t.type == "scatter"]
    assert len(page_lines) == len(scene_lines)
    for page_trace, scene_trace in zip(page_lines, scene_lines, strict=True):
        np.testing.assert_array_equal(_array(scene_trace.x), _array(page_trace.x))
        np.testing.assert_array_equal(_array(scene_trace.y), _array(page_trace.y))


def _same_marker_lines(page_fig, scene_fig) -> None:
    def positions(fig):
        return [(s.type, s.x0, s.x1) for s in fig.layout.shapes]

    assert positions(scene_fig) == positions(page_fig)


class TestPdwScenes:
    def test_tsweep(self, capture, pages):
        _fig, t_fig, _b, _r, _info = _pdw_figures(pages, "UTe2 300 mK, zero field")
        scene = capture.SCENE_BY_NAME["pdw-tsweep"].build()
        _same_lines(t_fig, scene)
        _same_marker_lines(t_fig, scene)

    def test_bsweep(self, capture, pages):
        _fig, _t, b_fig, _r, _info = _pdw_figures(pages, "UTe2 field-induced p2 (Bz ≈ b* = 6 T)")
        scene = capture.SCENE_BY_NAME["pdw-bsweep"].build()
        _same_lines(b_fig, scene)
        _same_marker_lines(b_fig, scene)

    def test_regime_map(self, capture, pages):
        _fig, _t, _b, r_fig, _info = _pdw_figures(pages, "UTe2 4.2 K, By = a = 0.2 T (q fading)")
        scene = capture.SCENE_BY_NAME["pdw-regime-map"].build()
        for attr in ("x", "y", "z"):
            np.testing.assert_array_equal(
                _array(getattr(_first(scene, "heatmap"), attr)),
                _array(getattr(_first(r_fig, "heatmap"), attr)),
            )
        _same_lines(r_fig, scene)  # the current-state marker

    def test_fft(self, capture, pages):
        fig, *_rest = _pdw_figures(pages, "UTe2 3 K — Cheshire-cat window")
        scene = capture.SCENE_BY_NAME["pdw-fft"].build()
        np.testing.assert_array_equal(
            _array(_first(scene, "heatmap").z), _array(_first(fig, "heatmap").z)
        )
        _same_lines(fig, scene)  # the peak rings

    def test_moire_whatif_pair_view(self, capture, pages):
        fig, *_rest = _pdw_figures(pages, "Moire what-if: intrinsic PDW above Tc", view_mode="pair")
        scene = capture.SCENE_BY_NAME["pdw-moire-whatif"].build()
        np.testing.assert_array_equal(
            _array(_first(scene, "heatmap").z), _array(_first(fig, "heatmap").z)
        )


_SF_PAPER_FIT = "LCCO x = 0.159 — paper fit (Duffy Fig. 4)"
_SF_IMPRINT = "Moire glue what-if: imprint vs operating Tc"


def _sf_figures(pages, key: str):
    """The page callback's (main, FS, coupling) figures at the page's own preset."""
    page = pages["spin_fluctuation"]
    values = page._PRESETS[key]
    args = [values[name] for _cid, _prop, name in page._PRESET_BINDINGS]
    figures = page._update_sf(*args, "dark")[:3]
    for fig in figures:
        assert "Computation error" not in str(fig.layout.title.text)
    return figures


class TestSfScenes:
    @pytest.mark.parametrize("key", [_SF_PAPER_FIT, _SF_IMPRINT])
    def test_configs_mirror_the_page(self, capture, pages, key):
        from waytogocoop.computation.spin_fluctuation import SfPairingMap

        page = pages["spin_fluctuation"]
        s = page._clamp(page._PRESETS[key])
        config, glue = capture._sf_config(page._PRESETS[key])
        assert config == page._sf_config(s)
        assert glue == page._glue_state(s, SfPairingMap(form=s["pairing_form"])).glue

    def test_magnetoresistance(self, capture, pages):
        fig, _fs, _coupling = _sf_figures(pages, _SF_PAPER_FIT)
        _same_lines(fig, capture.SCENE_BY_NAME["sf-magnetoresistance"].build())

    def test_fermi_surface(self, capture, pages):
        _fig, fs_fig, _coupling = _sf_figures(pages, _SF_PAPER_FIT)
        scene = capture.SCENE_BY_NAME["sf-fermi-surface"].build()
        _same_lines(fs_fig, scene)  # zone boundary, FS, CVC partner segments, hot spots

        def gamma(fig) -> np.ndarray:
            return _array(next(t for t in fig.data if t.name == "Fermi surface").marker.color)

        np.testing.assert_array_equal(gamma(scene), gamma(fs_fig))

    def test_coupling(self, capture, pages):
        _fig, _fs, coupling_fig = _sf_figures(pages, _SF_PAPER_FIT)
        scene = capture.SCENE_BY_NAME["sf-coupling"].build()
        _same_lines(coupling_fig, scene)  # both Tc(λ) maps, λ_tr, λ_ρ, current x, the star
        _same_marker_lines(coupling_fig, scene)  # the λ_c threshold

    def test_glue_imprint(self, capture, pages):
        fig, _fs, _coupling = _sf_figures(pages, _SF_IMPRINT)
        scene = capture.SCENE_BY_NAME["sf-glue-imprint"].build()
        _same_lines(fig, scene)  # dressed, local, linear, the 0.165 line and the star
        assert not scene.layout.shapes and not fig.layout.shapes  # clean: no shading


def _graphene_figures(pages, key: str, **override):
    """The page callback's figures at the page's own preset (not the script's copy)."""
    values = {**pages["graphene"]._PRESETS[key], **override}
    figures = pages["graphene"]._update_graphene(
        stack=values["stack"],
        twist=values["twist"],
        filling=values["filling"],
        honeycomb=values["honeycomb"],
        valley=values["valley"],
        strain_pct=values["strain_pct"],
        strain_ang=values["strain_ang"],
        overlayer=values["overlayer"],
        iface_twist=values["iface_twist"],
        curv_geom=values["curvature"],
        amp=values["amp"],
        feature=values["feature"],
        phi=values["phi"],
        warp=values["warp"],
        grid_size=values["grid_size"],
        extent=values["extent"],
        view_mode=values["view_mode"],
        theme="dark",
    )
    assert "Computation error" not in str(figures[0].layout.title.text)
    return figures


class TestGrapheneScenes:
    def test_flatband_curve(self, capture, pages):
        _fig, flat_fig, _info, _notes = _graphene_figures(pages, "Magic-angle TBG")
        _same_lines(flat_fig, capture.SCENE_BY_NAME["graphene-flatband"].build())

    def test_ripple_pseudo_field(self, capture, pages):
        fig, *_rest = _graphene_figures(pages, "Armchair ripple pseudo-field")
        scene = capture.SCENE_BY_NAME["graphene-pseudo-field"].build()
        np.testing.assert_array_equal(
            _array(_first(scene, "heatmap").z), _array(_first(fig, "heatmap").z)
        )

    def test_bump_pseudo_field(self, capture, pages):
        fig, *_rest = _graphene_figures(
            pages,
            "Gaussian bump (curved 3D)",
            extent=100.0,
            grid_size=201,
            view_mode="pseudo_field",
        )
        scene = capture.SCENE_BY_NAME["graphene-bump-field"].build()
        np.testing.assert_array_equal(
            _array(_first(scene, "heatmap").z), _array(_first(fig, "heatmap").z)
        )

    def test_supermoire_zoom(self, capture, pages):
        fig, *_rest = _graphene_figures(pages, "Supermoire on Sb2Te3", extent=30.0, grid_size=300)
        scene = capture.SCENE_BY_NAME["graphene-supermoire-zoom"].build()
        np.testing.assert_array_equal(
            _array(_first(scene, "heatmap").z), _array(_first(fig, "heatmap").z)
        )

    def test_dos(self, capture, pages):
        fig, *_rest = _graphene_figures(pages, "Magic-angle flat bands", view_mode="dos")
        _same_lines(fig, capture.SCENE_BY_NAME["graphene-dos"].build())

    def test_bands_report_the_page_bandwidth(self, capture, pages):
        # The scene samples 48 k-points per segment, the page 16; W agrees.
        fig, *_rest = _graphene_figures(pages, "Magic-angle flat bands")
        scene = capture.SCENE_BY_NAME["graphene-bands"].build()
        assert scene.layout.annotations[0].text == fig.layout.annotations[0].text


class TestCoreScenes:
    def _viewer(self, pages):
        preset = pages["moire_viewer"]._PRESETS["Sb₂Te₃ / FeTe (paper default)"]
        return pages["moire_viewer"]._update_viewer(
            preset["substrate"],
            preset["overlayer"],
            preset["twist"],
            preset["grid_size"],
            preset["extent"],
            preset["view_mode"],
            [],
            None,
            None,
            None,
            None,
            None,
            [],
            "dark",
        )

    def test_viewer_maps(self, capture, pages):
        moire_fig, gap_fig, _info, _iso = self._viewer(pages)
        for name, page_fig in (("viewer-2d", moire_fig), ("viewer-gap-2d", gap_fig)):
            scene = capture.SCENE_BY_NAME[name].build()
            np.testing.assert_array_equal(
                _array(_first(scene, "heatmap").z), _array(_first(page_fig, "heatmap").z)
            )

    def test_fourier_and_its_detected_peaks(self, capture, pages):
        fft_fig, peaks = pages["fourier_analysis"]._update_fourier(
            "FeTe", "Sb2Te3", 0.0, 200, 100.0, 0.3, "dark"
        )
        for name in ("fourier", "fourier-peaks"):
            scene = capture.SCENE_BY_NAME[name].build()
            np.testing.assert_array_equal(
                _array(_first(scene, "heatmap").z), _array(_first(fft_fig, "heatmap").z)
            )
        # The rings are every peak identify_peaks finds; the page's table lists
        # the strongest 20 of them.
        rings = _first(capture.SCENE_BY_NAME["fourier-peaks"].build(), "scatter")
        ring_set = {(f"{x:.4f}", f"{y:.4f}") for x, y in zip(rings.x, rings.y, strict=True)}
        table_set = {(p["kx"], p["ky"]) for p in peaks}
        assert table_set <= ring_set
        spectrum = _first(fft_fig, "heatmap")
        detected = identify_peaks(
            _array(spectrum.z), _array(spectrum.x), _array(spectrum.y), threshold_fraction=0.3
        )
        assert len(ring_set) == len(detected)

    def test_sweep_twist(self, capture, pages):
        fig, _markers = pages["parameter_sweep"]._run_sweep(
            None, "twist_angle", 3.82, 0.5, 5.0, 100, "dark"
        )
        _same_lines(fig, capture.SCENE_BY_NAME["sweep-twist"].build())


class TestSpeculativeScenes:
    def test_phase_diagram(self, capture, pages):
        fig, _comm, _extras = pages["phase_diagram"].update_phase(
            "FeTe", "Sb2Te3", 0, 8, 0.1, 6, 0, 30, 200, "dark"
        )
        scene = capture.SCENE_BY_NAME["phase-diagram"].build()
        for attr in ("x", "y", "z"):
            np.testing.assert_array_equal(
                _array(getattr(_first(scene, "heatmap"), attr)),
                _array(getattr(_first(fig, "heatmap"), attr)),
            )

    def test_majorana_3d(self, capture, pages):
        fig = pages["magnetic_field"].update_magnetic(
            substrate_key="FeTe",
            overlayer_key="Sb2Te3",
            twist=0.0,
            grid_size=81,
            extent=300.0,
            Bz=4.0,
            Bx=0.0,
            By=0.0,
            viz_mode="majorana3d",
            xi_prox=100.0,
            g_factor=30.0,
            bc2=47.0,
            theme="dark",
        )[0]
        scene = capture.SCENE_BY_NAME["magnetic-majorana-3d"].build()
        np.testing.assert_array_equal(
            _array(_first(scene, "isosurface").value), _array(_first(fig, "isosurface").value)
        )
        cores = _first(scene, "scatter3d")
        assert len(cores.x) == 7
        np.testing.assert_array_equal(_array(cores.x), _array(_first(fig, "scatter3d").x))

    def _proximity(
        self, pages, extent: float, grid_size: int, view_mode: str, clip_z: float = 300.0
    ):
        return pages["proximity_3d"].update_proximity(
            "FeTe",
            "Sb2Te3",
            0.0,
            grid_size,
            extent,
            100.0,
            0.8,
            15,
            view_mode,
            [0.2, 0.8],
            clip_z,
            "dark",
        )

    def test_proximity_decay(self, capture, pages):
        _main, decay_fig, _dirac, _info = self._proximity(pages, 100.0, 200, "iso")
        _same_lines(decay_fig, capture.SCENE_BY_NAME["proximity-decay"].build())

    def test_proximity_volume_data(self, capture, pages):
        main, *_rest = self._proximity(pages, 30.0, 200, "volume")
        scene = capture.SCENE_BY_NAME["proximity-3d-volume"].build()
        page_volume, scene_volume = _first(main, "volume"), _first(scene, "volume")
        np.testing.assert_array_equal(_array(scene_volume.value), _array(page_volume.value))
        assert scene_volume.surface.count == page_volume.surface.count
        assert scene_volume.opacity == pytest.approx(0.3)
        assert page_volume.opacity == pytest.approx(0.1)

    @staticmethod
    def _height_marks(fig):
        """The arrow anchors and the edge line of a proximity iso/volume figure."""
        anchors = sorted((a.x, a.y, a.z) for a in fig.layout.scene.annotations)
        edges = [(tuple(t.x), tuple(t.y), tuple(t.z)) for t in fig.data if t.type == "scatter3d"]
        return anchors, edges

    @pytest.mark.parametrize(
        ("name", "view_mode", "extent", "clip_z"),
        [
            ("proximity-3d", "iso", 100.0, 300.0),
            ("proximity-3d-clipped", "iso", 100.0, 100.0),
            ("proximity-3d-volume", "volume", 30.0, 300.0),
        ],
    )
    def test_proximity_height_marks(self, capture, pages, name, view_mode, extent, clip_z):
        # The page's camera and the docs eye both look from the +x, +y side,
        # so both mark the same edge: the corner (extent, extent).
        main, *_rest = self._proximity(pages, extent, 200, view_mode, clip_z)
        scene = capture.SCENE_BY_NAME[name].build()
        anchors, edges = self._height_marks(scene)
        assert (anchors, edges) == self._height_marks(main)
        assert anchors == [(extent, extent, 0.0), (extent, extent, 100.0)]
        assert edges == [((extent, extent), (extent, extent), (0.0, 100.0))]

    @pytest.mark.parametrize("z_top", [-45.0, 0.0, 50.0, 100.0, 300.0])
    def test_proximity_marker_rule_matches_the_page(self, capture, pages, z_top):
        # Which markers survive a given drawn top, where they sit, and when
        # the edge line is drawn.
        x = np.linspace(-100.0, 100.0, 81)
        page_marks, page_edge = pages["proximity_3d"]._height_markers(x, x, z_top, 100.0, True)
        scene_marks, scene_edge = capture._prox_markers(x, x, z_top)

        def points(marks):
            return [(m["x"], m["y"], m["z"]) for m in marks]

        assert points(scene_marks) == points(page_marks)
        assert (scene_edge is None) == (page_edge is None)
        if page_edge is not None:
            for axis in ("x", "y", "z"):
                assert tuple(scene_edge[axis]) == tuple(page_edge[axis])


# ---------------------------------------------------------------------------
# Registry and command line
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_names_and_files_are_unique_pngs(self, capture):
        names = [scene.name for scene in capture.SCENES]
        files = [scene.filename for scene in capture.SCENES]
        assert len(set(names)) == len(names)
        assert len(set(files)) == len(files)
        for scene in capture.SCENES:
            assert scene.filename.endswith(".png")
            assert scene.name == Path(scene.filename).stem

    def test_frames_kinds_and_steps(self, capture):
        assert capture.FRAMES == {"wide": (800, 500, 2.0), "square": (720, 720, 2.0)}
        for scene in capture.SCENES:
            assert scene.frame in capture.FRAMES
            assert scene.kind in ("app", "presentation", "docs-only")
            assert bool(scene.step) == (scene.kind == "presentation"), scene.name

    def test_filenames_are_literal_strings(self, capture):
        # scripts/check_docs.py finds the scenes by their literal "*.png" strings.
        literals = set(re.findall(r'"([A-Za-z0-9_.-]+\.png)"', SCRIPT.read_text()))
        assert literals == {scene.filename for scene in capture.SCENES}

    def test_no_python_scene_uses_the_rust_prefix(self, capture):
        assert not [s for s in capture.SCENES if s.filename.startswith("rust-")]

    def test_list_prints_the_registry(self, capture, capsys):
        assert capture.main(["--list"]) == 0
        rows = [line.split("\t") for line in capsys.readouterr().out.splitlines()]
        assert [row[0] for row in rows] == [scene.name for scene in capture.SCENES]
        for row, scene in zip(rows, capture.SCENES, strict=True):
            width, height, scale = capture.FRAMES[scene.frame]
            size = f"{round(width * scale)}x{round(height * scale)}"
            assert row == [scene.name, scene.filename, scene.frame, scene.kind, size]

    def test_only_takes_comma_lists_and_repeats(self, capture, capsys):
        assert capture.main(["--list", "--only", "pdw-fft,viewer-2d", "--only", "fourier"]) == 0
        listed = [line.split("\t")[0] for line in capsys.readouterr().out.splitlines()]
        assert listed == ["viewer-2d", "fourier", "pdw-fft"]  # registry order

    def test_unknown_scene_is_an_error(self, capture, capsys):
        with pytest.raises(SystemExit) as exit_info:
            capture.main(["--list", "--only", "viewer-2d,no-such-scene"])
        assert exit_info.value.code == 2
        assert "no-such-scene" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Opaque PNGs
# ---------------------------------------------------------------------------


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _encode_png(pixels: np.ndarray, filter_type: int) -> bytes:
    """A minimal 8-bit PNG writer that applies one row filter to every row."""
    height, width, channels = pixels.shape
    color_type = {3: 2, 4: 6}[channels]
    raw = bytearray()
    previous = [0] * (width * channels)
    for row in pixels.reshape(height, width * channels).tolist():
        raw.append(filter_type)
        for i, value in enumerate(row):
            a = row[i - channels] if i >= channels else 0
            b = previous[i]
            c = previous[i - channels] if i >= channels else 0
            predictor = (0, a, b, (a + b) // 2, _paeth(a, b, c))[filter_type]
            raw.append((value - predictor) % 256)
        previous = row

    def chunk(tag: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(tag + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )


class TestOpaquePngs:
    @pytest.fixture
    def rgba(self) -> np.ndarray:
        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 256, size=(5, 6, 4), dtype=np.uint8)
        pixels[..., 3] = 255
        return pixels

    @pytest.mark.parametrize("filter_type", range(5))
    def test_opaque_rgba_passes(self, capture, rgba, filter_type):
        assert capture.png_is_opaque(_encode_png(rgba, filter_type))

    @pytest.mark.parametrize("filter_type", range(5))
    @pytest.mark.parametrize("where", [(0, 0), (0, 3), (2, 0), (4, 5)])
    def test_one_translucent_pixel_fails(self, capture, rgba, filter_type, where):
        rgba[where][3] = 254
        assert not capture.png_is_opaque(_encode_png(rgba, filter_type))

    def test_rgb_without_alpha_is_opaque(self, capture, rgba):
        assert capture.png_is_opaque(_encode_png(np.ascontiguousarray(rgba[..., :3]), 1))

    def test_rgb_with_trns_is_not(self, capture, rgba):
        data = _encode_png(np.ascontiguousarray(rgba[..., :3]), 0)
        body = b"\x00\x00\x00\x00\x00\x00"
        trns = struct.pack(">I", 6) + b"tRNS" + body
        trns += struct.pack(">I", zlib.crc32(b"tRNS" + body) & 0xFFFFFFFF)
        end = data.index(b"IDAT") - 4
        assert not capture.png_is_opaque(data[:end] + trns + data[end:])

    def test_every_docs_png_is_opaque(self, capture):
        pngs = sorted(IMAGES.glob("*.png"))
        assert pngs, f"no PNGs in {IMAGES}"
        translucent = [p.name for p in pngs if not capture.png_is_opaque(p.read_bytes())]
        assert not translucent


# ---------------------------------------------------------------------------
# Writing PNGs (no browser: kaleido is stubbed out and each render replaced)
# ---------------------------------------------------------------------------


def _png_of_size(size: int, alpha: int = 255) -> bytes:
    """A small 8-bit RGBA PNG padded with a tEXt chunk to exactly ``size`` bytes."""
    pixels = np.full((4, 4, 4), 128, dtype=np.uint8)
    pixels[..., 3] = alpha
    png = _encode_png(pixels, 0)
    body = b"Comment\x00"
    body += b"x" * (size - len(png) - 12 - len(body))
    crc = struct.pack(">I", zlib.crc32(b"tEXt" + body) & 0xFFFFFFFF)
    end = png.index(b"IEND") - 4
    padded = png[:end] + struct.pack(">I", len(body)) + b"tEXt" + body + crc + png[end:]
    assert len(padded) == size
    return padded


class TestCaptureWrites:
    @pytest.fixture
    def render(self, capture, monkeypatch):
        """Stub the browser; main() then writes whatever render["png"] holds."""
        import kaleido

        monkeypatch.setattr(kaleido, "start_sync_server", lambda **_kwargs: None)
        monkeypatch.setattr(kaleido, "stop_sync_server", lambda **_kwargs: None)
        rendered: dict[str, bytes] = {}
        monkeypatch.setattr(capture, "_render_png", lambda _fig, _scene: rendered["png"])
        return rendered

    def test_a_transparent_render_is_not_written(self, capture, render, tmp_path):
        render["png"] = _png_of_size(2048, alpha=254)
        existing = tmp_path / "proximity-decay.png"
        existing.write_bytes(b"the committed figure")
        with pytest.raises(RuntimeError, match="transparent"):
            capture.main(["--only", "proximity-decay", "--out-dir", str(tmp_path)])
        assert existing.read_bytes() == b"the committed figure"

    def test_run_and_folder_totals_agree(self, capture, render, tmp_path, capsys):
        # Two 1.5 KB files: floored one by one they would make 2 KB, summed 3 KB.
        render["png"] = _png_of_size(1536)
        args = ["--only", "proximity-decay,sweep-twist", "--out-dir", str(tmp_path)]
        assert capture.main(args) == 0
        assert [p.read_bytes() for p in sorted(tmp_path.glob("*.png"))] == [render["png"]] * 2
        summary = capsys.readouterr().out.splitlines()[-1]
        assert summary.startswith("wrote 2 PNGs, 3 KB;"), summary
        assert "holds 2 PNGs, 3 KB of the" in summary
