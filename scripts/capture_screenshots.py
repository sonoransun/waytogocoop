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
    create_fft_heatmap,
    create_moire_heatmap,
    create_phase_colormap,
)
from waytogocoop.computation.fourier import fft_2d
from waytogocoop.computation.magnetic import (
    combined_gap_with_vortices,
    generate_vortex_positions,
    screening_currents,
    vortex_suppression_field,
)
from waytogocoop.computation.moire import generate_moire_pattern
from waytogocoop.computation.superconducting import gap_modulation
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
