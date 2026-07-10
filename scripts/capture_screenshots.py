"""Kaleido-based reproducible screenshot generator for docs/images/.

Each scene is a closure that performs the same physics computation used by
the Dash page, then builds a figure via :mod:`waytogocoop.components.figure_factory`
and exports it as a PNG. No Dash server is started — this runs headless, so
it's CI-safe (though kaleido pulls a ~80 MB Chromium on first install).

Usage
-----
::

    python scripts/capture_screenshots.py                  # all scenes
    python scripts/capture_screenshots.py --only viewer-3d
    python scripts/capture_screenshots.py --out-dir /tmp/imgs
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Kaleido's default timeout is ~90s; 3D traces with 50k+ vertices need more
# time for Chrome's JS engine to stringify. Bump it once at module load so
# every write_image call benefits.
with contextlib.suppress(AttributeError):
    pio.kaleido.scope.default_timeout = 180  # legacy-api safeguard, no-op on v1
with contextlib.suppress(AttributeError, ImportError):
    # Kaleido v1 moved the knob to the per-call TIMEOUT; set the module default.
    import kaleido as _kaleido
    _kaleido.TIMEOUT = 180

from waytogocoop.components.figure_factory import (
    create_3d_cone_field,
    create_3d_isosurface,
    create_3d_majorana_isosurface,
    create_3d_surface,
    create_3d_volume,
    create_band_structure_plot,
    create_dos_plot,
    create_fft_heatmap,
    create_moire_heatmap,
    create_phase_colormap,
    create_pseudo_field_heatmap,
    create_sweep_plot,
    create_vortex_overlay_heatmap,
)
from waytogocoop.computation.bm_model import BMConfig, compute_band_structure, compute_dos
from waytogocoop.computation.curvature import CurvatureConfig, compute_curvature_effects
from waytogocoop.computation.fourier import fft_2d
from waytogocoop.computation.graphene import (
    compute_flat_band_sc,
    generate_stack_pattern_v2,
    generate_supermoire_pattern,
)
from waytogocoop.computation.magnetic import (
    combined_gap_with_vortices,
    generate_vortex_positions,
    screening_currents,
    vortex_suppression_field,
)
from waytogocoop.computation.moire import generate_moire_pattern, moire_periodicity_with_twist
from waytogocoop.computation.superconducting import cpdm_amplitude, gap_modulation
from waytogocoop.computation.topological import (
    ProximityConfig,
    gap_3d,
    majorana_probability_density_3d,
    phase_diagram_sweep,
)
from waytogocoop.config import DEFAULT_COHERENCE_LENGTH, DELTA_AMPLITUDE, DELTA_AVG
from waytogocoop.materials.database import get_material

# ---- Shared physics: Sb2Te3 / FeTe at twist=0 (README's canonical preset) ---

def _moire(grid_size: int = 200) -> dict:
    substrate = get_material("FeTe")
    overlayer = get_material("Sb2Te3")
    result = generate_moire_pattern(
        substrate_a=substrate.a,
        overlayer_a=overlayer.a,
        overlayer_lattice_type=overlayer.lattice_type,
        substrate_lattice_type=substrate.lattice_type,
        twist_angle_deg=0.0,
        grid_size=grid_size,
        physical_extent=100.0,
    )
    return result


def _gap_field(pattern: np.ndarray) -> np.ndarray:
    return gap_modulation(pattern, DELTA_AVG, DELTA_AMPLITUDE)


# ---- Scene builders -------------------------------------------------------

def scene_viewer_2d() -> go.Figure:
    result = _moire()
    return create_moire_heatmap(
        result["x"], result["y"], result["pattern"],
        title="Moire Pattern — Sb₂Te₃ / FeTe (paper default)", dark=True,
    )


def scene_viewer_3d() -> go.Figure:
    # 160² ≈ 25k vertices / 50k triangles: above the high_density threshold
    # (grid >= 150) so we exercise the WebGL Mesh3d path without blowing out
    # kaleido's Chrome JSON stringify timeout.
    result = _moire(grid_size=160)
    return create_3d_surface(
        result["x"], result["y"], result["pattern"],
        title="3D Moire Surface — WebGL mesh3d (Sb₂Te₃ / FeTe)",
        colorscale="viridis", z_label="Intensity",
        dark=True, high_density=True,
    )


def scene_fourier() -> go.Figure:
    result = _moire()
    dx = float(result["x"][1] - result["x"][0])
    ft = fft_2d(result["pattern"], dx)
    return create_fft_heatmap(
        ft["kx"], ft["ky"], ft["power_spectrum"],
        title="FFT log₁₀(|F|²) — moire reciprocal vectors",
        dark=True,
    )


def _proximity_volume() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 64² × 24 ≈ 98k voxels — enough for a readable iso/volume render while
    # staying inside kaleido's Chrome JSON serialisation budget.
    result = _moire(grid_size=64)
    gap_field = _gap_field(result["pattern"])
    cfg = ProximityConfig(
        xi_prox=100.0, n_z_layers=24, z_min=-50.0, z_max=300.0,
        interface_transparency=0.8,
    )
    prox = gap_3d(gap_field, cfg)
    return result["x"], result["y"], prox.z_coords, prox.gap_3d


def _prox_annotations(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[dict]:
    x_mid = float((x[0] + x[-1]) / 2.0)
    y_mid = float((y[0] + y[-1]) / 2.0)
    return [
        dict(x=x_mid, y=y_mid, z=0.0,
             text="Interface (z=0)",
             showarrow=True, arrowhead=2, ax=30, ay=-30,
             font=dict(size=11, color="white")),
        dict(x=x_mid, y=y_mid, z=min(100.0, float(z[-1])),
             text="ξ_prox ≈ 100 Å",
             showarrow=True, arrowhead=2, ax=40, ay=-40,
             font=dict(size=11, color="white")),
    ]


def scene_proximity_3d() -> go.Figure:
    x, y, z, vol = _proximity_volume()
    return create_3d_isosurface(
        x, y, z, vol,
        title="3D Cooper Surface — Sb₂Te₃/FeTe (isosurface)",
        surface_count=5,
        annotations=_prox_annotations(x, y, z),
        dark=True,
    )


def scene_proximity_3d_volume() -> go.Figure:
    x, y, z, vol = _proximity_volume()
    # viridis (monotonic) is readable as a volume — diverging RdBu_r maps the
    # middle of the gap range to near-transparent, which kills the rendering.
    return create_3d_volume(
        x, y, z, vol,
        title="3D Cooper Volume — Sb₂Te₃/FeTe (true volumetric render)",
        opacity=0.3, surface_count=20, colorscale="viridis",
        annotations=_prox_annotations(x, y, z),
        dark=True,
    )


def scene_proximity_3d_clipped() -> go.Figure:
    x, y, z, vol = _proximity_volume()
    return create_3d_isosurface(
        x, y, z, vol,
        title="3D Cooper Surface — clipped at z=100 Å",
        surface_count=5,
        annotations=_prox_annotations(x, y, z),
        clip_z=100.0,
        dark=True,
    )


def scene_magnetic_currents_3d() -> go.Figure:
    result = _moire(grid_size=160)
    gap_field = _gap_field(result["pattern"])
    Bz = 4.0
    vortex_pos = generate_vortex_positions(Bz, physical_extent=100.0, grid_size=160)
    suppression = vortex_suppression_field(
        result["x"], result["y"], vortex_pos, DEFAULT_COHERENCE_LENGTH,
    )
    combined = combined_gap_with_vortices(gap_field, suppression)
    Jx, Jy = screening_currents(result["x"], result["y"], vortex_pos)
    return create_3d_cone_field(
        result["x"], result["y"], 0.0, Jx, Jy, jz=None,
        base_surface=combined,
        title="Screening Currents at z=0 — Bz=4 T (3D cones)",
        skip=12, dark=True,
    )


def scene_magnetic_majorana_3d() -> go.Figure:
    result = _moire(grid_size=80)
    Bz = 4.0
    vortex_pos = generate_vortex_positions(Bz, physical_extent=100.0, grid_size=80)
    z3d = np.linspace(-40.0, 200.0, 24)
    density_3d = majorana_probability_density_3d(
        result["x"], result["y"], z3d, vortex_pos, xi_prox=100.0,
    )
    return create_3d_majorana_isosurface(
        result["x"], result["y"], z3d, density_3d, vortex_pos,
        title="Majorana ZM 3D Density (SPECULATIVE) — Bz=4 T",
        dark=True,
    )


def scene_phase_diagram() -> go.Figure:
    b = np.linspace(0.0, 20.0, 60)
    delta = np.linspace(0.0, 5.0, 60)
    phase = phase_diagram_sweep(b, delta, g_factor=30.0)
    return create_phase_colormap(b, delta, phase, dark=True)


# ---- Graphene scenes: "Magic-angle TBG" preset of pages/graphene.py --------
# Twisted bilayer at theta = 1.08 deg, filling nu = 2.4, grid 200, extent 200 Å.

_TBG_TWIST_DEG = 1.08
_TBG_FILLING = 2.4
_GRAPHENE_GRID = 200
_GRAPHENE_EXTENT = 200.0


def _fmt_period(period: float) -> str:
    return "∞" if np.isinf(period) else f"{period:.1f} Å"


def scene_graphene_pattern() -> go.Figure:
    result = generate_stack_pattern_v2(
        "twisted_bilayer", _TBG_TWIST_DEG,
        grid_size=_GRAPHENE_GRID, physical_extent=_GRAPHENE_EXTENT,
    )
    return create_moire_heatmap(
        result["x"], result["y"], result["pattern"],
        title=(
            f"Moire Pattern: Twisted bilayer — θ = {_TBG_TWIST_DEG}° (magic angle), "
            f"L = {_fmt_period(result['moire_period'])}"
        ),
        dark=True,
    )


def scene_graphene_bands() -> go.Figure:
    bands = compute_band_structure(BMConfig(twist_angle_deg=_TBG_TWIST_DEG))
    return create_band_structure_plot(
        bands.k_distances,
        bands.energies_mev,
        bands.tick_positions,
        bands.tick_labels,
        bands.flat_bandwidth_mev,
        title=f"Moire Band Structure (BM model) — θ = {_TBG_TWIST_DEG}°",
        dark=True,
    )


def scene_graphene_dos() -> go.Figure:
    dos_result = compute_dos(BMConfig(twist_angle_deg=_TBG_TWIST_DEG))
    return create_dos_plot(
        dos_result["energies_mev"], dos_result["dos"],
        title=f"Density of States (BM model) — θ = {_TBG_TWIST_DEG}°",
        dark=True,
    )


def _graphene_grid() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-_GRAPHENE_EXTENT, _GRAPHENE_EXTENT, _GRAPHENE_GRID)
    y = np.linspace(-_GRAPHENE_EXTENT, _GRAPHENE_EXTENT, _GRAPHENE_GRID)
    return x, y


def scene_graphene_pseudo_field() -> go.Figure:
    # "Armchair ripple pseudo-field" preset: 2 Å ripple, 100 Å feature size,
    # oriented 30 deg from zigzag (= armchair), K valley.  The page feeds the
    # single feature-size slider into sigma, wavelength, and radius alike.
    x, y = _graphene_grid()
    cfg = CurvatureConfig(
        geometry="sinusoidal_ripple",
        amplitude=2.0,
        sigma=100.0,
        wavelength=100.0,
        radius=100.0,
        orientation_deg=30.0,
        valley=1,
    )
    curv = compute_curvature_effects(cfg, x, y)
    return create_pseudo_field_heatmap(
        x, y, curv.pseudo_field,
        title="Pseudo-Magnetic Field (SPECULATIVE) — armchair ripple, K valley",
        dark=True,
    )


def scene_graphene_curved_3d() -> go.Figure:
    # "Gaussian bump (curved 3D)" preset: 5 Å bump, sigma = 50 Å.  Height is
    # the bump; the surface color is the SPECULATIVE flat-band gap times the
    # pseudo-field suppression, exactly as the page's curved3d view builds it.
    x, y = _graphene_grid()
    cfg = CurvatureConfig(
        geometry="gaussian_bump",
        amplitude=5.0,
        sigma=50.0,
        wavelength=50.0,
        radius=50.0,
        orientation_deg=30.0,
        valley=1,
    )
    curv = compute_curvature_effects(cfg, x, y)
    result = generate_stack_pattern_v2(
        "twisted_bilayer", _TBG_TWIST_DEG,
        grid_size=_GRAPHENE_GRID, physical_extent=_GRAPHENE_EXTENT,
    )
    fb = compute_flat_band_sc(_TBG_TWIST_DEG, 2, _TBG_FILLING)
    gap = (
        gap_modulation(result["pattern"], fb.delta_mev, 0.5 * fb.delta_mev)
        * curv.gap_suppression
    )
    # Same [::2] downsampling the graphene page applies for grid_size > 150.
    return create_3d_surface(
        x[::2], y[::2], curv.height[::2, ::2],
        title="Curved Sheet - Gap Overlay (SPECULATIVE)",
        colorscale="RdBu_r", z_label="h (A)",
        dark=True,
        surfacecolor=gap[::2, ::2],
        color_label="Delta (meV)",
    )


def scene_graphene_supermoire() -> go.Figure:
    # "Supermoire on Sb2Te3" preset: magic-angle TBG on an untwisted Sb2Te3
    # overlayer — intra-stack, interface, and beat periods in the title.
    overlayer = get_material("Sb2Te3")
    result = generate_supermoire_pattern(
        "twisted_bilayer", _TBG_TWIST_DEG,
        overlayer_a=overlayer.a,
        overlayer_lattice_type=overlayer.lattice_type,
        interface_twist_deg=0.0,
        grid_size=_GRAPHENE_GRID,
        physical_extent=_GRAPHENE_EXTENT,
    )
    return create_moire_heatmap(
        result["x"], result["y"], result["pattern"],
        title=(
            f"Supermoire — TBG θ = {_TBG_TWIST_DEG}° on Sb₂Te₃ "
            f"(stack {_fmt_period(result['stack_period'])}, "
            f"interface {_fmt_period(result['interface_period'])}, "
            f"beat {_fmt_period(result['supermoire_period'])})"
        ),
        dark=True,
    )


# ---- Sweep + magnetic 2D scenes ---------------------------------------------


def scene_sweep_twist() -> go.Figure:
    # Twist branch of pages/parameter_sweep.py: FeTe homo-bilayer (a = 3.82 Å,
    # the page's default substrate), 0.5-5.0 deg, 100 points.
    substrate_a = 3.82
    param_values = np.linspace(0.5, 5.0, 100)
    periods = moire_periodicity_with_twist(substrate_a, param_values)
    amplitudes = cpdm_amplitude(periods)
    # Cap infinite periods for plotting, exactly like the page.
    finite_mask = np.isfinite(periods)
    max_period = np.max(periods[finite_mask]) * 1.1 if finite_mask.any() else 1000.0
    periods = np.where(finite_mask, periods, max_period)
    return create_sweep_plot(
        param_values, periods, amplitudes, "Twist angle (deg)", dark=True
    )


def scene_magnetic_vortex_2d() -> go.Figure:
    # The magnetic page's "vortex" view: moire gap heatmap with the Abrikosov
    # vortex core positions overlaid as markers.
    result = _moire()
    gap_field = _gap_field(result["pattern"])
    vortex_pos = generate_vortex_positions(4.0, physical_extent=100.0, grid_size=200)
    return create_vortex_overlay_heatmap(
        result["x"], result["y"], gap_field, vortex_pos,
        title="Gap + Vortex Lattice — Bz = 4 T",
        dark=True,
    )


# ---- Scene registry -------------------------------------------------------

SCENES: dict[str, tuple[str, Callable[[], go.Figure]]] = {
    "viewer-2d":              ("viewer-2d.png",              scene_viewer_2d),
    "viewer-3d":              ("viewer-3d.png",              scene_viewer_3d),
    "fourier":                ("fourier.png",                scene_fourier),
    "proximity-3d":           ("proximity-3d.png",           scene_proximity_3d),
    "proximity-3d-volume":    ("proximity-3d-volume.png",    scene_proximity_3d_volume),
    "proximity-3d-clipped":   ("proximity-3d-clipped.png",   scene_proximity_3d_clipped),
    "magnetic-currents-3d":   ("magnetic-currents-3d.png",   scene_magnetic_currents_3d),
    "magnetic-majorana-3d":   ("magnetic-majorana-3d.png",   scene_magnetic_majorana_3d),
    "phase-diagram":          ("phase-diagram.png",          scene_phase_diagram),
    "graphene-pattern":       ("graphene-pattern.png",       scene_graphene_pattern),
    "graphene-bands":         ("graphene-bands.png",         scene_graphene_bands),
    "graphene-dos":           ("graphene-dos.png",           scene_graphene_dos),
    "graphene-pseudo-field":  ("graphene-pseudo-field.png",  scene_graphene_pseudo_field),
    "graphene-curved-3d":     ("graphene-curved-3d.png",     scene_graphene_curved_3d),
    "graphene-supermoire":    ("graphene-supermoire.png",    scene_graphene_supermoire),
    "sweep-twist":            ("sweep-twist.png",            scene_sweep_twist),
    "magnetic-vortex-2d":     ("magnetic-vortex-2d.png",     scene_magnetic_vortex_2d),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture docs/images screenshots.")
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "images",
        help="Output directory (default: docs/images).",
    )
    parser.add_argument(
        "--only", action="append", default=[],
        choices=list(SCENES.keys()),
        help="Restrict to the given scene name(s); repeatable.",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=1.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = args.only or list(SCENES.keys())

    for name in targets:
        filename, builder = SCENES[name]
        out_path = args.out_dir / filename
        print(f"[{name}] building…", flush=True)
        fig = builder()
        fig.update_layout(width=args.width, height=args.height)
        fig.write_image(str(out_path), scale=args.scale)
        print(f"[{name}] wrote {out_path} ({out_path.stat().st_size // 1024} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
