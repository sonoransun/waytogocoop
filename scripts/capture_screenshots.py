"""Reproducible docs figures for docs/images/: the web-app (Plotly) scenes.

Each scene runs the same physics as a page of the Dash app, builds its figure
with the page's own builders from :mod:`waytogocoop.components.figure_factory`,
applies the shared docs styling (:func:`_docs_style`) and exports a PNG
through kaleido. No Dash server runs, so this works headless; kaleido drives
its own Chromium (one-time setup: ``.venv/bin/plotly_get_chrome -y``).

The desktop scenes (``rust-*.png``) come from
``crates/moire-desktop/src/bin/capture.rs``. The Figure gallery
(``docs/images/README.md``) holds the canonical captions.

Registry
--------
``SCENES`` lists every figure in the canonical page order (core pipeline,
graphene track, speculative extensions) as ``Scene(filename, build, frame,
kind[, step])``. The scene name is the file stem.

* ``frame``: ``"wide"`` is an 800 x 500 Plotly layout at 2x scale
  (1600 x 1000 PNG) for line plots, sweeps, (B, T) maps and 3D scenes;
  ``"square"`` is 720 x 720 at 2x (1440 x 1440 PNG) for equal-aspect real-
  and k-space maps.
* ``kind``: ``"app"`` is the app's own figure at settings reachable in its
  UI (captions state any non-default setting); ``"presentation"`` is app
  physics plus exactly one named display ``step``, which the caption names;
  ``"docs-only"`` plots app physics in a chart the apps do not draw.

Web extents are half-widths: a "±100 Å" window spans -100..+100 Å. The Rust
extents are full widths.

The page modules cannot be imported here: ``dash.register_page`` raises
``PageError`` unless an app exists. The presets and callback logic the
scenes need are therefore copied, each with a comment naming its source;
``tests/test_capture_parity.py`` checks the copies against the pages.

Usage
-----
::

    python scripts/capture_screenshots.py                          # every scene
    python scripts/capture_screenshots.py --only pdw-fft,fourier   # some (repeatable)
    python scripts/capture_screenshots.py --list                   # the registry
    python scripts/capture_screenshots.py --out-dir /tmp/figs      # another folder

The script prints each file's size and render time, warns above the 800 KB
soft cap, and prints the folder total against the 12 MB budget. It checks
that each rendered PNG is opaque before writing it: a render with
transparent pixels stops the run and leaves the existing file untouched.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import time
import warnings
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter

from waytogocoop.components.colormaps import get_plotly_colorscale
from waytogocoop.components.figure_factory import (
    _line_colors,
    _pdw_family_colors,
    create_3d_cone_field,
    create_3d_isosurface,
    create_3d_majorana_isosurface,
    create_3d_surface,
    create_3d_volume,
    create_band_structure_plot,
    create_dos_plot,
    create_fft_heatmap,
    create_flat_band_plot,
    create_gap_heatmap,
    create_moire_heatmap,
    create_pdw_peak_map,
    create_pdw_regime_map,
    create_pdw_sweep_plot,
    create_phase_colormap,
    create_pseudo_field_heatmap,
    create_sf_coupling_plot,
    create_sf_fermi_surface,
    create_sf_imprint_plot,
    create_sf_mr_plot,
    create_sweep_plot,
    create_vortex_overlay_heatmap,
    create_z_decay_profile,
)
from waytogocoop.components.pdw_panel import FIELD_AXES, T_PDW_RATIO_UTE2
from waytogocoop.components.sf_panel import COHERENCE_PLOT_LABELS, doping_value
from waytogocoop.computation import fourier
from waytogocoop.computation.bm_model import BMConfig, compute_band_structure, compute_dos
from waytogocoop.computation.curvature import CurvatureConfig, compute_curvature_effects
from waytogocoop.computation.fourier import fft_2d, identify_peaks
from waytogocoop.computation.graphene import (
    compute_flat_band_sc,
    dirac_velocity_ratio,
    generate_stack_pattern_v2,
    generate_supermoire_pattern,
    magic_angle_deg,
)
from waytogocoop.computation.isotope_effects import compute_isotope_effects
from waytogocoop.computation.magnetic import (
    combined_gap_with_vortices,
    generate_vortex_positions,
    screening_currents,
    vortex_lattice_period,
    vortex_suppression_field,
)
from waytogocoop.computation.moire import (
    estimate_moire_period,
    generate_moire_pattern,
    moire_periodicity_with_twist,
)
from waytogocoop.computation.pdw import (
    PdwConfig,
    charge_spectrum,
    compute_pdw_state,
    critical_field_along,
    field_sweep,
    moire_locked_config,
    moire_locked_pdw_ratio,
    moire_locked_period,
    onset_field_along,
    pdw_peaks,
    pdw_real_space,
    reduced_field,
    regime_map,
    temperature_sweep,
)
from waytogocoop.computation.spin_fluctuation import (
    SfConfig,
    SfGlueConfig,
    SfPairingMap,
    bcs_route_relative_modulation,
    coherence_factor,
    cvc_centres,
    cvc_partner_angle,
    doping_sweep,
    fermi_surface,
    fermi_surface_at,
    fragmentation_tc,
    hotspot_angles,
    imprint_curve,
    lambda_from_tc,
    lambda_threshold,
    local_imprint,
    magnetoresistance,
    scattering_rate,
    snap_doping,
    tc_from_lambda,
)
from waytogocoop.computation.superconducting import cpdm_amplitude, gap_modulation
from waytogocoop.computation.topological import (
    ProximityConfig,
    gap_3d,
    majorana_probability_density_3d,
    phase_diagram_sweep,
    proximity_decay_profile,
)
from waytogocoop.config import (
    A_LCCO,
    ALPHA1_LCCO,
    ALPHA2_LCCO,
    DEFAULT_COHERENCE_LENGTH,
    DEFAULT_FFT_THRESHOLD_FRACTION,
    DELTA_2,
    DELTA_AMPLITUDE,
    DELTA_AVG,
    G_FACTOR_TSS,
    GAMMA_IMP_LCCO,
    HOTSPOT_PEAK_LCCO,
    HOTSPOT_WIDTH_LCCO,
    MU_B_MEV_T,
    PDW_KAPPA_H,
    PDW_KAPPA_HARM,
    PDW_KAPPA_P,
    PDW_KAPPA_Q,
    PHI_HS_DEG_LCCO,
    RIPPLE_HEIGHT_DEFAULT,
    SF_CVC_EPSILON,
    SF_DOPING_REF,
    SF_FIELD_MAX,
    SF_FIT_TEMPERATURE,
    SF_GLUE_COHERENCE,
    SF_GLUE_ETA,
    SF_MR_TEMPERATURES,
    SF_NU,
    TC_FETE_FILM,
    W_AA_TBG,
    W_AB_TBG,
    W_INTERLAYER_TBG,
    XI_PROXIMITY_DEFAULT,
)
from waytogocoop.materials.database import get_material
from waytogocoop.materials.isotopes import ELEMENTS

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"

# ---- Frames, kinds and the registry entry ----------------------------------

# name -> (Plotly layout width, layout height, export scale). Fonts are sized
# against the layout, so labels stay legible when GitHub shows a PNG at half
# width in a two-up card.
FRAMES: dict[str, tuple[int, int, float]] = {
    "wide": (800, 500, 2.0),  # 1600 x 1000 PNG: line plots, sweeps, maps, 3D
    "square": (720, 720, 2.0),  # 1440 x 1440 PNG: equal-aspect real/k-space maps
}
KINDS = ("app", "presentation", "docs-only")

SOFT_CAP_KB = 800  # warn above this per PNG
FOLDER_BUDGET_KB = 12 * 1024  # docs/images as a whole (Python + Rust PNGs)


@dataclass(frozen=True)
class Scene:
    """One docs figure: its file, builder, frame, kind and display step."""

    filename: str
    build: Callable[[], go.Figure]
    frame: str
    kind: str
    step: str | None = None  # the one named display step of a presentation scene

    def __post_init__(self) -> None:
        if not self.filename.endswith(".png"):
            raise ValueError(f"{self.filename}: scene files are PNGs")
        if self.frame not in FRAMES:
            raise ValueError(f"{self.filename}: unknown frame {self.frame!r}")
        if self.kind not in KINDS:
            raise ValueError(f"{self.filename}: unknown kind {self.kind!r}")
        if (self.kind == "presentation") != bool(self.step):
            raise ValueError(f"{self.filename}: exactly the presentation scenes name a step")

    @property
    def name(self) -> str:
        return self.filename.removesuffix(".png")

    @property
    def pixels(self) -> tuple[int, int]:
        width, height, scale = FRAMES[self.frame]
        return round(width * scale), round(height * scale)


# ---- Shared docs style -------------------------------------------------------

_3D_TRACES = frozenset({"surface", "mesh3d", "isosurface", "volume", "cone", "scatter3d"})
_DOCS_EYE = dict(x=1.25, y=1.25, z=0.85)  # ~17 % closer than the app's iso eye
# Looking slightly below the box center lifts the box in the frame, so the
# front-corner tick labels clear the bottom edge; the eye stays where it is.
_DOCS_CENTER = dict(x=0.0, y=0.0, z=-0.12)


def _docs_style(fig: go.Figure, frame: str) -> go.Figure:
    """Presentation-only polish shared by every docs PNG; never changes data."""
    width, height, _scale = FRAMES[frame]
    fig.update_layout(
        width=width,
        height=height,
        template="plotly_dark",  # opaque #111111 paper: the PNGs read in both themes
        font=dict(size=14),
        title=dict(font=dict(size=17), x=0.01, xanchor="left"),
    )
    # The Iso/Top/Side camera buttons are live-app UI. Assign, because
    # update_layout(updatemenus=[]) leaves an existing list in place.
    fig.layout.updatemenus = ()
    if fig.layout.title.subtitle.text:
        fig.update_layout(
            title_subtitle_font=dict(size=14),
            margin_t=max(fig.layout.margin.t or 0, 95),
        )
    if any(trace.type in _3D_TRACES for trace in fig.data):
        fig.update_layout(scene_camera_eye=_DOCS_EYE, scene_camera_center=_DOCS_CENTER)
    else:
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
    return fig


def _sub(formula: str) -> str:
    """Chemical formula with Plotly subscripts: Sb2Te3 -> Sb<sub>2</sub>Te<sub>3</sub>."""
    return re.sub(r"(\d+)", r"<sub>\1</sub>", formula)


def _fmt_period(period: float) -> str:
    return "∞" if np.isinf(period) else f"{period:.1f} Å"


def _minus(value: float, fmt: str = "g") -> str:
    """Format a number with a typographic minus sign."""
    return format(value, fmt).replace("-", "−")


def _inside_legend(x: float, y: float, xanchor: str, yanchor: str) -> dict:
    """A legend inside the plot area, clear of the title and subtitle."""
    return dict(
        orientation="v",
        x=x,
        y=y,
        xanchor=xanchor,
        yanchor=yanchor,
        bgcolor="rgba(17, 17, 17, 0.75)",
        bordercolor="#444",
        borderwidth=1,
    )


def _plain_secondary_axis(fig: go.Figure) -> go.Figure:
    """Round ticks on a secondary y-axis instead of ticks synced to the primary grid."""
    fig.update_yaxes(tickmode="auto", nticks=7, showgrid=False, secondary_y=True)
    return fig


# ---- Local averages for multi-scale fields (presentation steps) -------------
# A field with atomic-scale texture is computed on a fine grid (dx <= 0.5 Å)
# with a pad around the window, averaged, then cropped and decimated, so the
# displayed map neither aliases the atomic rows nor carries edge bias.

_PAD = 20.0  # Å computed beyond each window edge


def _padded_grid(extent: float, dx: float) -> tuple[int, float, int]:
    """Grid size, padded half-width and pad (in samples) for a ±extent window."""
    padded = extent + _PAD
    return int(round(2.0 * padded / dx)) + 1, padded, int(round(_PAD / dx))


def _envelope(field: np.ndarray, dx: float, sigma: float) -> np.ndarray:
    """Gaussian local average over sigma Å (the step captions must name)."""
    return gaussian_filter(field, sigma / dx, mode="nearest")


def _local_rms(field: np.ndarray, dx: float, sigma: float) -> np.ndarray:
    """Local RMS contrast: sqrt of the sigma-Å average of (field - local mean)²."""
    mean = _envelope(field, dx, sigma)
    return np.sqrt(np.maximum(_envelope((field - mean) ** 2, dx, sigma), 0.0))


def _crop(a: np.ndarray, pad: int, step: int) -> np.ndarray:
    """Drop the pad and keep every step-th sample along each axis."""
    if a.ndim == 2:
        return a[pad:-pad:step, pad:-pad:step]
    return a[pad:-pad:step]


# ---- Copied page presets (the pages cannot be imported; see the docstring) --

# pages/moire_viewer.py _PRESETS["Sb₂Te₃ / FeTe (paper default)"] (lines 40-47),
# the viewer's paper-default preset; it equals the page's _DEFAULTS.
_VIEWER_PAPER_DEFAULT = {
    "substrate": "FeTe",
    "overlayer": "Sb2Te3",
    "twist": 0.0,
    "grid_size": 200,
    "extent": 100.0,
    "view_mode": "heatmap",
}

# pages/graphene.py _V2_NEUTRAL (lines 67-75), shared by every graphene preset.
_GRAPHENE_V2_NEUTRAL = {
    "honeycomb": False,
    "valley": 1,
    "strain_pct": 0,
    "strain_ang": 0,
    "overlayer": "none",
    "iface_twist": 0,
    "warp": False,
}

# pages/graphene.py _PRESETS (lines 77-171): the five presets the graphene
# scenes start from, keyed as on the page.
_GRAPHENE_PRESETS = {
    "Magic-angle TBG": {
        "stack": "twisted_bilayer",
        "twist": 1.08,
        "filling": 2.4,
        "curvature": "flat",
        "amp": 5,
        "feature": 100,
        "phi": 30,
        "grid_size": 200,
        "extent": 200.0,
        "view_mode": "pattern",
        **_GRAPHENE_V2_NEUTRAL,
    },
    "Armchair ripple pseudo-field": {
        "stack": "twisted_bilayer",
        "twist": 1.08,
        "filling": 2.4,
        "curvature": "sinusoidal_ripple",
        "amp": RIPPLE_HEIGHT_DEFAULT,
        "feature": 100,
        "phi": 30,
        "grid_size": 200,
        "extent": 200.0,
        "view_mode": "pseudo_field",
        **_GRAPHENE_V2_NEUTRAL,
    },
    "Gaussian bump (curved 3D)": {
        "stack": "twisted_bilayer",
        "twist": 1.08,
        "filling": 2.4,
        "curvature": "gaussian_bump",
        "amp": 5,
        "feature": 50,
        "phi": 30,
        "grid_size": 200,
        "extent": 200.0,
        "view_mode": "curved3d",
        **_GRAPHENE_V2_NEUTRAL,
    },
    "Supermoire on Sb2Te3": {
        "stack": "twisted_bilayer",
        "twist": 1.08,
        "filling": 2.4,
        "curvature": "flat",
        "amp": 5,
        "feature": 100,
        "phi": 30,
        "grid_size": 200,
        "extent": 200.0,
        "view_mode": "pattern",
        **_GRAPHENE_V2_NEUTRAL,
        "overlayer": "Sb2Te3",
        "iface_twist": 0,
    },
    "Magic-angle flat bands": {
        "stack": "twisted_bilayer",
        "twist": 1.08,
        "filling": 2.4,
        "curvature": "flat",
        "amp": 5,
        "feature": 100,
        "phi": 30,
        "grid_size": 200,
        "extent": 200.0,
        "view_mode": "bands",
        **_GRAPHENE_V2_NEUTRAL,
    },
}

# pages/spin_fluctuation.py _BASE_PRESET (lines 155-179).
_SF_BASE_PRESET = {
    "doping": doping_value(SF_DOPING_REF),
    "temperature": SF_FIT_TEMPERATURE,
    "field_max": SF_FIELD_MAX,
    "inelastic": "afm_boundary",
    "hotspots": True,
    "gamma_imp": GAMMA_IMP_LCCO,
    "alpha1": ALPHA1_LCCO,
    "alpha2": ALPHA2_LCCO,
    "hotspot_peak": HOTSPOT_PEAK_LCCO,
    "hotspot_width": HOTSPOT_WIDTH_LCCO,
    "nu": SF_NU,
    "phi_hs": PHI_HS_DEG_LCCO,
    "cvc_epsilon": SF_CVC_EPSILON,
    "pairing_form": "power_law",
    "eta": SF_GLUE_ETA,
    "tc_target": TC_FETE_FILM,
    "substrate": "FeTe",
    "overlayer": "Sb2Te3",
    "twist": 0.0,
    "grid_size": 200,
    "extent": 100.0,
    "view_mode": "mr",
    "coherence": SF_GLUE_COHERENCE,
}

# pages/spin_fluctuation.py _PRESETS (lines 182-212): presets 1 and 11, the two
# the SF scenes start from, keyed as on the page.
_SF_PRESETS = {
    "LCCO x = 0.159 — paper fit (Duffy Fig. 4)": dict(_SF_BASE_PRESET),
    "Moire glue what-if: imprint vs operating Tc": {
        **_SF_BASE_PRESET,
        "tc_target": 5.0,
        "view_mode": "imprint",
    },
}

# pages/pair_density_wave.py _BASE_PRESET (lines 100-119).
_PDW_BASE_PRESET = {
    "geometry": "ute2",
    "temperature": 0.3,
    "tpdw_ratio": T_PDW_RATIO_UTE2,
    "bx": 0.0,
    "by": 0.0,
    "bz": 0.0,
    "sweep_axis": "z",
    "kappa_p": PDW_KAPPA_P,
    "kappa_q": PDW_KAPPA_Q,
    "kappa_h": PDW_KAPPA_H,
    "kappa_harm": PDW_KAPPA_HARM,
    "floor": -2.0,
    "substrate": "FeTe",
    "overlayer": "Sb2Te3",
    "twist": 0.0,
    "grid_size": 200,
    "extent": 40.0,
    "view_mode": "charge",
}

# pages/pair_density_wave.py _PRESETS (lines 121-156): the five presets the PDW
# scenes start from, keyed as on the page.
_PDW_PRESETS = {
    "UTe2 300 mK, zero field": dict(_PDW_BASE_PRESET),
    "UTe2 3 K — Cheshire-cat window": {
        **_PDW_BASE_PRESET,
        "temperature": 3.0,
        "view_mode": "fft",
    },
    "UTe2 field-induced p2 (Bz ≈ b* = 6 T)": {
        **_PDW_BASE_PRESET,
        "bz": 6.0,
        "view_mode": "peaks",
    },
    "UTe2 4.2 K, By = a = 0.2 T (q fading)": {
        **_PDW_BASE_PRESET,
        "temperature": 4.2,
        "by": 0.2,
        "sweep_axis": "y",
        "view_mode": "peaks",
    },
    "Moire what-if: intrinsic PDW above Tc": {
        **_PDW_BASE_PRESET,
        "geometry": "moire",
        "temperature": 16.0,
        "tpdw_ratio": 1.5,
        "extent": 100.0,
        "view_mode": "charge",
    },
}

# (page module stem, preset key) -> copy; tests/test_capture_parity.py checks
# each copy against that page's _PRESETS entry.
PAGE_PRESET_COPIES: dict[tuple[str, str], dict] = {
    ("moire_viewer", "Sb₂Te₃ / FeTe (paper default)"): _VIEWER_PAPER_DEFAULT,
    **{("graphene", key): preset for key, preset in _GRAPHENE_PRESETS.items()},
    **{("spin_fluctuation", key): preset for key, preset in _SF_PRESETS.items()},
    **{("pair_density_wave", key): preset for key, preset in _PDW_PRESETS.items()},
}

# ---- Core pipeline: moire -> gap -> Fourier ----------------------------------

_FETE = get_material("FeTe")
_SB2TE3 = get_material("Sb2Te3")
_SYSTEM = f"{_sub('Sb2Te3')} on FeTe"
_VIEWER_SETTINGS = "θ = 0°, ±100 Å, 200 × 200 grid"


def _moire(grid_size: int, extent: float, overlayer=_SB2TE3) -> dict:
    """Overlayer on FeTe at zero twist, as the viewer computes it."""
    return generate_moire_pattern(
        substrate_a=_FETE.a,
        overlayer_a=overlayer.a,
        overlayer_lattice_type=overlayer.lattice_type,
        substrate_lattice_type=_FETE.lattice_type,
        twist_angle_deg=0.0,
        grid_size=grid_size,
        physical_extent=extent,
    )


def _viewer_moire() -> dict:
    preset = _VIEWER_PAPER_DEFAULT
    return _moire(preset["grid_size"], preset["extent"])


def _gap_field(pattern: np.ndarray) -> np.ndarray:
    """The viewer's gap map: Δ(r) = Δ_avg + δΔ cos(π + π m(r))."""
    return gap_modulation(pattern, DELTA_AVG, DELTA_AMPLITUDE)


def scene_viewer_2d() -> go.Figure:
    r = _viewer_moire()
    fig = create_moire_heatmap(r["x"], r["y"], r["pattern"], title=f"Moire pattern, {_SYSTEM}")
    fig.update_layout(title_subtitle_text=f"Moire Viewer defaults: {_VIEWER_SETTINGS}")
    return fig


def scene_viewer_3d() -> go.Figure:
    # Viewer "3D Surface" view at extent 25 Å and grid 160: grids >= 150 take
    # the WebGL Mesh3d path (moire_viewer.py line 324), and 3.2 samples per Å
    # (dx 0.31 Å) resolve the 3.7 Å rows that a ±100 Å window aliases.
    r = _moire(160, 25.0)
    fig = create_3d_surface(
        r["x"],
        r["y"],
        r["pattern"],
        title=f"Moire pattern as a 3D surface, {_SYSTEM}",
        colorscale="viridis",
        z_label="Intensity",
        high_density=True,
    )
    fig.update_layout(
        title_subtitle_text="Moire Viewer, 3D Surface view: θ = 0°, ±25 Å, 160 × 160 grid"
    )
    return fig


# pages/substrate_comparison.py _OVERLAYER_FORMULAS (line 33).
_COMPARISON_OVERLAYERS = ("Sb2Te3", "Bi2Te3", "Sb2Te")


def _beat_period(overlayer) -> float:
    """Collinear beat of FeTe's (1, 0) shell with the overlayer's 0° hexagonal shell (Å)."""
    g_square = 2.0 * np.pi / _FETE.a
    g_hex = 4.0 * np.pi / (np.sqrt(3.0) * overlayer.a)
    return 2.0 * np.pi / abs(g_hex - g_square)


def scene_substrate_comparison() -> go.Figure:
    """The /comparison page's six heatmaps in one figure, averaged over 2 Å.

    Mirrors _update_comparison (substrate_comparison.py lines 185-281) in its
    heatmap view at twist 0 and ±100 Å. Each pattern is computed at
    dx = 0.25 Å, its gap from the full-resolution pattern, both averaged over
    σ = 2 Å and shown every 2 Å; the color scales are shared across columns.
    """
    dx, sigma, step = 0.25, 2.0, 8
    n, padded, pad = _padded_grid(100.0, dx)  # 961 points, 120 Å, 80
    titles, maps = [], []
    for formula in _COMPARISON_OVERLAYERS:
        overlayer = get_material(formula)
        period = estimate_moire_period(_FETE.a, overlayer.a, 0.0)
        titles.append(
            f"{_sub(formula)}: L<sub>m</sub> {period:.1f} Å, beat {_beat_period(overlayer):.0f} Å"
        )
        r = _moire(n, padded, overlayer)
        x = _crop(r["x"], pad, step)  # 101 points, 2 Å apart
        moire = _crop(_envelope(r["pattern"], dx, sigma), pad, step)
        gap = _crop(_envelope(_gap_field(r["pattern"]), dx, sigma), pad, step)
        maps.append((x, moire, gap))
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=titles + ["", "", ""],
        horizontal_spacing=0.03,
        vertical_spacing=0.07,
    )
    for col, (x, moire, gap) in enumerate(maps, start=1):
        fig.add_trace(go.Heatmap(z=moire, x=x, y=x, coloraxis="coloraxis"), row=1, col=col)
        fig.add_trace(go.Heatmap(z=gap, x=x, y=x, coloraxis="coloraxis2"), row=2, col=col)
    moire_all = np.concatenate([m[1].ravel() for m in maps])
    gap_all = np.concatenate([m[2].ravel() for m in maps])
    fig.update_layout(
        title="Three tellurides on FeTe, averaged over σ = 2 Å",
        title_subtitle_text=(
            "Substrate Comparison page: θ = 0°, ±100 Å · top: moire intensity, "
            "bottom: gap Δ · shared color scales"
        ),
        coloraxis=dict(
            colorscale=get_plotly_colorscale("viridis"),
            cmin=float(moire_all.min()),
            cmax=float(moire_all.max()),
            colorbar=dict(title="⟨I⟩", y=0.755, len=0.43, thickness=14),
        ),
        coloraxis2=dict(
            colorscale=get_plotly_colorscale("coolwarm"),
            cmin=float(gap_all.min()),
            cmax=float(gap_all.max()),
            colorbar=dict(title="⟨Δ⟩ (meV)", y=0.215, len=0.43, thickness=14),
        ),
        margin=dict(l=60, r=20, t=125, b=40),
    )
    for row in (1, 2):
        for col in (1, 2, 3):
            index = (row - 1) * 3 + col
            anchor = "x" if index == 1 else f"x{index}"
            fig.update_yaxes(
                scaleanchor=anchor,
                scaleratio=1,
                constrain="domain",
                tickvals=[-100, 0, 100],
                showticklabels=col == 1,
                row=row,
                col=col,
            )
            fig.update_xaxes(
                constrain="domain",
                tickvals=[-100, 0, 100],
                showticklabels=row == 2,
                row=row,
                col=col,
            )
    fig.update_xaxes(title_text="x (Å)", row=2)
    fig.update_yaxes(title_text="y (Å)", col=1)
    fig.update_annotations(font_size=12)
    return fig


def scene_sweep_twist() -> go.Figure:
    # Twist branch of pages/parameter_sweep.py _run_sweep (lines 141-170): the
    # page's default substrate (a = 3.82 Å, an FeTe homo-bilayer), 0.5-5.0°,
    # 100 points. Every period is finite here, so the page's inf cap is a no-op.
    substrate_a = 3.82
    twist = np.linspace(0.5, 5.0, 100)
    periods = moire_periodicity_with_twist(substrate_a, twist)
    fig = create_sweep_plot(twist, periods, cpdm_amplitude(periods), "Twist angle θ (°)")
    fig.update_traces(mode="lines", line_width=2.5)
    fig.update_layout(
        title="Moire period and CPDM amplitude vs twist",
        title_subtitle_text=(
            "Parameter Sweep, twist angle: FeTe homo-bilayer (a = 3.82 Å), "
            "L = a / (2 sin(θ/2)), A = exp(−ξ/L), ξ = 20 Å"
        ),
        legend=_inside_legend(0.6, 0.97, "left", "top"),
    )
    return _plain_secondary_axis(fig)


def scene_viewer_gap_2d() -> go.Figure:
    r = _viewer_moire()
    fig = create_gap_heatmap(
        r["x"], r["y"], _gap_field(r["pattern"]), title=f"Gap map Δ(r), {_SYSTEM}"
    )
    fig.update_layout(
        title_subtitle_text=(
            f"Moire Viewer defaults: {_VIEWER_SETTINGS} · "
            f"Δ<sub>avg</sub> = {DELTA_AVG:.2f}, δΔ = {DELTA_AMPLITUDE:.2f} meV"
        )
    )
    return fig


def _fourier_figure(title: str) -> tuple[go.Figure, dict]:
    # pages/fourier_analysis.py _update_fourier (lines 140-189) at the defaults.
    r = _viewer_moire()
    dx = (r["x"][-1] - r["x"][0]) / (len(r["x"]) - 1)
    ft = fft_2d(r["pattern"], dx)
    fig = create_fft_heatmap(ft["kx"], ft["ky"], ft["power_spectrum"], title=title)
    fig.update_traces(colorbar_title_text="log<sub>10</sub>|F|²", selector=dict(type="heatmap"))
    fig.update_layout(
        xaxis_title="k<sub>x</sub> (Å<sup>−1</sup>)",
        yaxis_title="k<sub>y</sub> (Å<sup>−1</sup>)",
    )
    return fig, {**ft, "dx": dx}


def scene_fourier() -> go.Figure:
    fig, _ft = _fourier_figure(f"FFT power spectrum, {_SYSTEM}")
    fig.update_layout(title_subtitle_text=f"Fourier Analysis defaults: {_VIEWER_SETTINGS}")
    return fig


def _sum_difference_vectors() -> np.ndarray:
    """Every G_sub ± G_over of the two first shells: the only lines in the spectrum."""
    g_sq = 2.0 * np.pi / _FETE.a
    g_sub = g_sq * np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    g_hex = 4.0 * np.pi / (np.sqrt(3.0) * _SB2TE3.a)
    angles = np.arange(6) * np.pi / 3.0
    g_over = g_hex * np.column_stack([np.cos(angles), np.sin(angles)])
    return np.array([s + sign * o for s in g_sub for o in g_over for sign in (1.0, -1.0)])


def scene_fourier_peaks() -> go.Figure:
    fig, ft = _fourier_figure(f"Detected FFT peaks, {_SYSTEM}")
    threshold = DEFAULT_FFT_THRESHOLD_FRACTION
    peaks = identify_peaks(ft["power_spectrum"], ft["kx"], ft["ky"], threshold_fraction=threshold)

    # A detected peak is an alias when the sum/difference vector it samples
    # lies beyond the Nyquist wavenumber and folds back into the window.
    k_nyq = np.pi / ft["dx"]
    true_k = _sum_difference_vectors()
    folded = (true_k + k_nyq) % (2.0 * k_nyq) - k_nyq
    xs, ys, labels, positions = [], [], [], []
    for p in peaks:
        k = np.array([p["kx"], p["ky"]])
        xs.append(k[0])
        ys.append(k[1])
        # Label one member of each ± pair: the one with kx > 0 (or ky > 0 on kx = 0).
        if k[0] < -1e-9 or (abs(k[0]) <= 1e-9 and k[1] < 0):
            labels.append("")
            positions.append("top center")
            continue
        source = true_k[np.argmin(np.linalg.norm(folded - k, axis=1))]
        if np.any(np.abs(source) > k_nyq):
            labels.append(f"alias of {2.0 * np.pi / np.linalg.norm(source):.2f} Å")
        else:
            labels.append(f"{2.0 * np.pi / np.linalg.norm(k):.2f} Å")
        if abs(k[1]) > 2.6:
            positions.append("middle right")
        elif abs(k[1]) < 1e-9:
            positions.append("top left")
        else:
            positions.append("top center" if k[1] > 0 else "bottom center")
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=dict(size=20, symbol="circle-open", color="white", line=dict(width=2)),
            text=labels,
            textposition=positions,
            textfont=dict(size=13, color="white", shadow="auto"),
            cliponaxis=False,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Keep the axes on the spectrum: the rings must not pad the autorange.
    dk = float(ft["kx"][1] - ft["kx"][0])
    fig.update_xaxes(range=[float(ft["kx"][0]) - dk / 2, float(ft["kx"][-1]) + dk / 2])
    fig.update_yaxes(range=[float(ft["ky"][0]) - dk / 2, float(ft["ky"][-1]) + dk / 2])
    # identify_peaks masks a (2 margin + 1)² block of bins around DC (fourier.py
    # lines 92-95); the square is that block's outer edge.
    shape = ft["power_spectrum"].shape
    size = max(fourier._MIN_NEIGHBOURHOOD_SIZE, min(shape) // fourier._NEIGHBOURHOOD_DIVISOR)
    half = (max(2, size // 2) + 0.5) * dk
    fig.add_shape(
        type="rect",
        x0=-half,
        x1=half,
        y0=-half,
        y1=half,
        line=dict(color="cyan", width=2, dash="dot"),
    )
    fig.update_layout(
        title_subtitle_text=(
            f"rings: peaks the page detects (threshold {threshold:g}); one label per ± pair, "
            "λ = 2π/|k| · dotted: DC mask"
        )
    )
    return fig


# ---- Graphene track: stacks -> BM model -> curvature --------------------------

_TBG = _GRAPHENE_PRESETS["Magic-angle TBG"]
_GRAPHENE_EXTENT = 200.0  # Å half-width of the enveloped TBG maps (the page default)
_TBG_LABEL = "Twisted bilayer graphene"


def _graphene_envelope(strain: float) -> tuple[np.ndarray, np.ndarray, float]:
    """TBG pattern at dx = 0.25 Å over ±220 Å, averaged over σ = 1.5 Å, cropped to ±200 Å."""
    dx, sigma = 0.25, 1.5
    n, padded, pad = _padded_grid(_GRAPHENE_EXTENT, dx)  # 1761 points, 220 Å, 80
    r = generate_stack_pattern_v2(
        _TBG["stack"],
        _TBG["twist"],
        grid_size=n,
        physical_extent=padded,
        heterostrain_percent=strain,
        heterostrain_angle_deg=0.0,
    )
    x = _crop(r["x"], pad, 8)  # 201 points, 2 Å apart: ample for a 1.5 Å average
    return x, _crop(_envelope(r["pattern"], dx, sigma), pad, 8), r["moire_period"]


def _graphene_pattern_figure(strain: float) -> go.Figure:
    x, envelope, period = _graphene_envelope(strain)
    what = f"θ = {_TBG['twist']:g}°" + (f", {strain:g} % heterostrain" if strain else "")
    fig = create_moire_heatmap(x, x, envelope, title=f"{_TBG_LABEL}, {what}")
    length = "longest period" if strain else "L"
    fig.update_layout(
        title_subtitle_text=(
            f"moire envelope: pattern averaged over σ = 1.5 Å · {length} = {_fmt_period(period)}"
            " · ±200 Å"
        )
    )
    fig.update_traces(colorbar_title_text="⟨I⟩, σ = 1.5 Å", selector=dict(type="heatmap"))
    return fig


def scene_graphene_pattern() -> go.Figure:
    return _graphene_pattern_figure(strain=0.0)


def scene_graphene_heterostrain() -> go.Figure:
    # 0.5 % uniaxial heterostrain along zigzag (0°) on the top layer: the
    # page's strain controls on the "Magic-angle TBG" preset.
    return _graphene_pattern_figure(strain=0.5)


_SUPERMOIRE = _GRAPHENE_PRESETS["Supermoire on Sb2Te3"]


def _supermoire(grid_size: int, extent: float) -> dict:
    # pages/graphene.py _update_graphene, supermoire branch (lines 388-401).
    over = get_material(_SUPERMOIRE["overlayer"])
    return generate_supermoire_pattern(
        _SUPERMOIRE["stack"],
        _SUPERMOIRE["twist"],
        overlayer_a=over.a,
        overlayer_lattice_type=over.lattice_type,
        interface_twist_deg=_SUPERMOIRE["iface_twist"],
        grid_size=grid_size,
        physical_extent=extent,
        honeycomb=_SUPERMOIRE["honeycomb"],
    )


def scene_graphene_supermoire() -> go.Figure:
    dx, sigma = 0.25, 3.0
    n, padded, pad = _padded_grid(_GRAPHENE_EXTENT, dx)
    r = _supermoire(n, padded)
    x = _crop(r["x"], pad, 8)  # 201 points, 2 Å apart
    rms = _crop(_local_rms(r["pattern"], dx, sigma), pad, 8)
    fig = create_moire_heatmap(
        x, x, rms, title=f"TBG θ = 1.08° on {_sub('Sb2Te3')}: local contrast"
    )
    fig.update_layout(
        title_subtitle_text=(
            f"RMS over σ = 3 Å · stack {_fmt_period(r['stack_period'])} · interface "
            f"{_fmt_period(r['interface_period'])} · beat {_fmt_period(r['supermoire_period'])}"
        )
    )
    fig.update_traces(colorbar_title_text="RMS, σ = 3 Å", selector=dict(type="heatmap"))
    return fig


def scene_graphene_supermoire_zoom() -> go.Figure:
    # The "Supermoire on Sb2Te3" preset with extent 30 Å and grid 300
    # (dx = 0.2 Å, five samples per 1.07 Å period of the finest product term)
    # in the Pattern view (graphene.py lines 508-511).
    r = _supermoire(300, 30.0)
    fig = create_moire_heatmap(
        r["x"],
        r["y"],
        r["pattern"],
        title=f"TBG θ = 1.08° on {_sub('Sb2Te3')}: raw pattern, ±30 Å",
    )
    fig.update_layout(
        title_subtitle_text=(
            f"preset “Supermoire on Sb2Te3” at extent 30 Å, grid 300 · interface "
            f"{_fmt_period(r['interface_period'])}, beat {_fmt_period(r['supermoire_period'])}"
        )
    )
    return fig


def scene_graphene_flatband() -> go.Figure:
    # pages/graphene.py _update_graphene lower figure (lines 513-526) on a
    # twisted-bilayer preset: 2-layer curve at filling 2.4, marker at 1.08°.
    thetas = np.linspace(0.3, 3.0, 200)
    deltas = np.array([compute_flat_band_sc(t, 2, _TBG["filling"]).delta_mev for t in thetas])
    fig = create_flat_band_plot(thetas, dirac_velocity_ratio(thetas), deltas, _TBG["twist"])
    # Docs typography for the builder's ASCII labels.
    fig.update_traces(name="v*/v<sub>F</sub>", selector=dict(name="v*/v_F"))
    fig.update_traces(
        name="Δ<sub>max</sub> (SPECULATIVE)", selector=dict(name="Delta_max (meV) (SPECULATIVE)")
    )
    fig.update_layout(
        title="Flat-band velocity and gap dome vs twist (SPECULATIVE)",
        title_subtitle_text=(
            f"first-order v*/v<sub>F</sub>, zero at θ<sub>m</sub> = {magic_angle_deg(2):.3f}° · "
            f"Δ<sub>max</sub>: speculative dome, {_TBG['filling']:g} electrons per moire cell"
        ),
        xaxis_title="Twist angle θ (°)",
        # make_subplots(secondary_y=True) ends the x-domain at 0.94 of the paper.
        legend=_inside_legend(0.92, 0.42, "right", "middle"),
    )
    fig.update_yaxes(title_text="v*/v<sub>F</sub>", secondary_y=False)
    fig.update_yaxes(title_text="Δ<sub>max</sub> (meV)", secondary_y=True)
    fig.update_annotations(text=f"θ = {_TBG['twist']:.2f}°")
    return _plain_secondary_axis(fig)


_BAND_WINDOW_MEV = 150.0  # the DOS window (compute_dos e_window_mev)


def scene_graphene_bands() -> go.Figure:
    # "Magic-angle flat bands" preset, Band structure view (graphene.py lines
    # 443-498), with 48 k-points per path segment instead of 16 (W unchanged),
    # zoomed to the ±150 meV DOS window.
    preset = _GRAPHENE_PRESETS["Magic-angle flat bands"]
    b = compute_band_structure(
        BMConfig(twist_angle_deg=preset["twist"], valley=preset["valley"]), n_k_per_segment=48
    )
    fig = create_band_structure_plot(
        b.k_distances,
        b.energies_mev,
        b.tick_positions,
        [label.replace("'", "′") for label in b.tick_labels],
        b.flat_bandwidth_mev,
        title=f"BM bands along K–Γ–M–K′, θ = {preset['twist']:g}°",
    )
    fig.update_yaxes(range=[-_BAND_WINDOW_MEV, _BAND_WINDOW_MEV], title_text="E (meV)")
    fig.update_layout(
        title_subtitle_text=(
            f"flat bands in red · W = {b.flat_bandwidth_mev:.1f} meV · "
            f"{b.flat_gap_mev:.1f} meV gap to the remote bands · ±150 meV window"
        )
    )
    return fig


def scene_graphene_dos() -> go.Figure:
    preset = _GRAPHENE_PRESETS["Magic-angle flat bands"]
    dos = compute_dos(BMConfig(twist_angle_deg=preset["twist"], valley=preset["valley"]))
    fig = create_dos_plot(
        dos["energies_mev"], dos["dos"], title=f"BM density of states, θ = {preset['twist']:g}°"
    )
    fig.update_layout(
        title_subtitle_text="DOS view: 12 × 12 k-grid, 2 meV Gaussian broadening, ±150 meV"
    )
    return fig


def _bm_width_gap(thetas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bands = [compute_band_structure(BMConfig(twist_angle_deg=float(t))) for t in thetas]
    return (
        np.array([b.flat_bandwidth_mev for b in bands]),
        np.array([b.flat_gap_mev for b in bands]),
    )


def scene_graphene_bandwidth() -> go.Figure:
    """Docs-only: BM flat-band width and flat-remote gap against twist.

    0.005° steps over 0.8-1.4°, refined to 0.0001° around the minimum: W has
    a cusp there, so a coarse grid misses its depth.
    """
    thetas = np.round(np.arange(0.800, 1.4001, 0.005), 4)
    width, gap = _bm_width_gap(thetas)
    for step, half in ((0.001, 0.005), (0.0001, 0.001)):
        center = float(thetas[int(np.argmin(width))])
        fine = np.round(np.arange(center - half, center + half + step / 2, step), 4)
        fine = fine[~np.isin(fine, thetas)]
        fine_width, fine_gap = _bm_width_gap(fine)
        thetas = np.concatenate([thetas, fine])
        width = np.concatenate([width, fine_width])
        gap = np.concatenate([gap, fine_gap])
        order = np.argsort(thetas)
        thetas, width, gap = thetas[order], width[order], gap[order]
    primary, secondary, muted = _line_colors(True)
    i_min = int(np.argmin(width))
    i_preset = int(np.argmin(np.abs(thetas - 1.08)))
    theta_m = magic_angle_deg(2)
    fig = go.Figure(
        [
            go.Scatter(
                x=thetas,
                y=width,
                name="flat-band width W",
                mode="lines",
                line=dict(color=primary, width=2.5),
            ),
            go.Scatter(
                x=thetas,
                y=gap,
                name="flat–remote gap",
                mode="lines",
                line=dict(color=secondary, width=2.5, dash="dash"),
            ),
            go.Scatter(
                x=[thetas[i_min], thetas[i_preset]],
                y=[width[i_min], width[i_preset]],
                mode="markers+text",
                marker=dict(size=10, color=primary),
                text=[
                    f"minimum W = {width[i_min]:.2f} meV at {thetas[i_min]:.3f}°",
                    f"1.08° preset: W = {width[i_preset]:.1f} meV",
                ],
                textposition=["bottom right", "middle right"],
                textfont=dict(size=13),
                showlegend=False,
            ),
        ]
    )
    fig.add_hline(y=0.0, line=dict(color=muted, width=1))
    fig.add_vline(
        x=theta_m,
        line=dict(color=muted, width=1, dash="dot"),
        annotation_text=(
            f"first-order θ<sub>m</sub> = {theta_m:.3f}° (w = {1e3 * W_INTERLAYER_TBG:.0f} meV)"
        ),
        annotation_position="top right",
        annotation_font_size=13,
    )
    fig.update_layout(
        title="BM flat-band width vs twist angle",
        title_subtitle_text=(
            f"default tunnelling w<sub>AA</sub> = {1e3 * W_AA_TBG:.1f}, "
            f"w<sub>AB</sub> = {1e3 * W_AB_TBG:.1f} meV · K–Γ–M–K′ path · 0.005° steps, "
            "0.0001° at the minimum"
        ),
        xaxis_title="Twist angle θ (°)",
        yaxis=dict(title="Energy (meV)", range=[-14.0, 1.06 * float(width.max())]),
        template="plotly_dark",
        legend=_inside_legend(0.02, 0.97, "left", "top"),
        margin=dict(l=60, r=20, t=95, b=50),
    )
    return fig


def _curvature(preset: dict, x: np.ndarray):
    # pages/graphene.py _update_graphene: one feature-size slider feeds sigma,
    # wavelength and radius alike (lines 377-386).
    cfg = CurvatureConfig(
        geometry=preset["curvature"],
        amplitude=preset["amp"],
        sigma=preset["feature"],
        wavelength=preset["feature"],
        radius=preset["feature"],
        orientation_deg=preset["phi"],
        valley=preset["valley"],
    )
    return compute_curvature_effects(cfg, x, x)


def _pseudo_field_figure(x: np.ndarray, curv, title: str, subtitle: str) -> go.Figure:
    fig = create_pseudo_field_heatmap(x, x, curv.pseudo_field, title=title)
    fig.update_traces(colorbar_title_text="B<sub>ps</sub> (T)", selector=dict(type="heatmap"))
    fig.update_layout(title_subtitle_text=subtitle)
    return fig


def scene_graphene_pseudo_field() -> go.Figure:
    preset = _GRAPHENE_PRESETS["Armchair ripple pseudo-field"]
    x = np.linspace(-preset["extent"], preset["extent"], preset["grid_size"])
    curv = _curvature(preset, x)
    return _pseudo_field_figure(
        x,
        curv,
        f"Pseudo-magnetic field (SPECULATIVE): {preset['amp']:g} Å armchair ripple",
        (
            f"preset “Armchair ripple pseudo-field”: λ = {preset['feature']:g} Å, K valley, "
            f"±{preset['extent']:g} Å · max |B<sub>ps</sub>| = {curv.max_abs_field:.1f} T"
        ),
    )


_BUMP = _GRAPHENE_PRESETS["Gaussian bump (curved 3D)"]
_BUMP_EXTENT = 100.0


def scene_graphene_bump_field() -> go.Figure:
    # The bump preset in the Pseudo-field view at extent 100 Å and grid 201.
    x = np.linspace(-_BUMP_EXTENT, _BUMP_EXTENT, 201)
    curv = _curvature(_BUMP, x)
    return _pseudo_field_figure(
        x,
        curv,
        f"Pseudo-magnetic field (SPECULATIVE): {_BUMP['amp']:g} Å Gaussian bump",
        (
            f"σ = {_BUMP['feature']:g} Å, K valley, ±{_BUMP_EXTENT:g} Å, 201 × 201 grid · "
            f"max |B<sub>ps</sub>| = {curv.max_abs_field:.1f} T"
        ),
    )


def scene_graphene_curved_3d() -> go.Figure:
    # The bump preset's Curved 3D sheet view (graphene.py lines 432, 462-476) at
    # extent 100 Å: height = bump, color = flat-band gap x pseudo-field
    # suppression. The gap is computed at dx = 0.25 Å and averaged over 1.5 Å.
    dx, sigma = 0.25, 1.5
    n, padded, pad = _padded_grid(_BUMP_EXTENT, dx)  # 961 points, 120 Å, 80
    r = generate_stack_pattern_v2(
        _BUMP["stack"], _BUMP["twist"], grid_size=n, physical_extent=padded
    )
    fb = compute_flat_band_sc(_BUMP["twist"], 2, _BUMP["filling"])
    gap = gap_modulation(r["pattern"], fb.delta_mev, 0.5 * fb.delta_mev)
    gap = _crop(_envelope(gap, dx, sigma), pad, 8)  # 101 points, 2 Å apart
    x = _crop(r["x"], pad, 8)
    curv = _curvature(_BUMP, x)
    fig = create_3d_surface(
        x,
        x,
        curv.height,
        title=f"Curved sheet: flat-band gap on a {_BUMP['amp']:g} Å bump (SPECULATIVE)",
        colorscale="RdBu_r",
        z_label="h (Å)",
        surfacecolor=gap * curv.gap_suppression,
        color_label="Δ (meV)",
    )
    fig.update_layout(
        title_subtitle_text=(
            f"TBG θ = 1.08°, bump σ = {_BUMP['feature']:g} Å, ±{_BUMP_EXTENT:g} Å · color: "
            "gap averaged over σ = 1.5 Å × pseudo-field suppression"
        )
    )
    return fig


# ---- Speculative extensions: magnetic field ---------------------------------

_VORTEX_BZ = 4.0
_VORTEX_EXTENT = 300.0  # ±300 Å: the center core and its six neighbors


def _vortex_positions(extent: float, grid_size: int) -> np.ndarray:
    # generate_vortex_positions fills a square of side physical_extent
    # ([-e/2, e/2]); web extents are half-widths, so pass the full side, as
    # pages/magnetic_field.py does (line 213).
    return generate_vortex_positions(_VORTEX_BZ, 2.0 * extent, grid_size)


def _vortex_fields(step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combined gap of the /magnetic page at 4 T over ±300 Å, gap averaged over 2 Å."""
    dx, sigma = 0.5, 2.0
    n, padded, pad = _padded_grid(_VORTEX_EXTENT, dx)  # 1281 points, 320 Å, 40
    r = _moire(n, padded)
    x = _crop(r["x"], pad, step)
    gap = _crop(_envelope(_gap_field(r["pattern"]), dx, sigma), pad, step)
    positions = _vortex_positions(_VORTEX_EXTENT, x.size)
    suppression = vortex_suppression_field(x, x, positions, DEFAULT_COHERENCE_LENGTH)
    return x, combined_gap_with_vortices(gap, suppression), positions


def scene_magnetic_vortex_2d() -> go.Figure:
    x, combined, positions = _vortex_fields(step=4)  # 301 points, 2 Å apart
    fig = create_vortex_overlay_heatmap(
        x, x, combined, positions, title="Vortex lattice at B<sub>z</sub> = 4 T (SPECULATIVE)"
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=1)), selector=dict(type="scatter"))
    fig.update_layout(
        showlegend=False,
        title_subtitle_text=(
            f"a<sub>v</sub> = {vortex_lattice_period(_VORTEX_BZ):.0f} Å, × = cores · "
            "combined gap averaged over σ = 2 Å · ±300 Å"
        ),
    )
    return fig


def scene_magnetic_currents_3d() -> go.Figure:
    x, combined, positions = _vortex_fields(step=8)  # 151 points, 4 Å apart
    jx, jy = screening_currents(x, x, positions)
    fig = create_3d_cone_field(
        x,
        x,
        0.0,
        jx,
        jy,
        jz=None,
        base_surface=combined,
        title="Screening currents at B<sub>z</sub> = 4 T (SPECULATIVE)",
        skip=6,
    )
    # The builder stacks its two colorbars almost on top of each other.
    fig.update_traces(colorbar=dict(x=0.86, len=0.8, thickness=14), selector=dict(type="surface"))
    fig.update_traces(
        colorbar=dict(x=1.0, len=0.8, thickness=14, title="|J| (arb.)"),
        selector=dict(type="cone"),
    )
    fig.update_layout(
        scene_domain=dict(x=[0.0, 0.8]),
        title_subtitle_text=(
            "cones: current at z = 0 around the 7 cores · surface: combined gap "
            "averaged over σ = 2 Å · ±300 Å"
        ),
    )
    return fig


# ---- Speculative extensions: topological proximity -------------------------

_PROX = dict(xi_prox=XI_PROXIMITY_DEFAULT, transparency=0.8, n_z=30, z_min=-50.0, z_max=300.0)


def scene_proximity_decay() -> go.Figure:
    # The /proximity3d decay graph at its defaults (proximity_3d.py lines
    # 335-337): f(z) on the page's 30 z-layers from -50 to 300 Å.
    z = np.linspace(_PROX["z_min"], _PROX["z_max"], _PROX["n_z"])
    profile = proximity_decay_profile(z, _PROX["xi_prox"], _PROX["transparency"])
    fig = create_z_decay_profile(z, profile, title="Proximity decay f(z) (SPECULATIVE)")
    xi, t = _PROX["xi_prox"], _PROX["transparency"]
    fig.update_traces(line_width=2.5)
    fig.add_annotation(
        x=xi,
        y=t / np.e,
        text="z = ξ<sub>prox</sub>: f = T/e",
        showarrow=True,
        arrowhead=2,
        ax=60,
        ay=-50,
        font=dict(size=13),
    )
    fig.update_layout(
        title_subtitle_text=(
            f"f = 1 in FeTe (z < 0), T·exp(−z/ξ<sub>prox</sub>) in the TI · "
            f"ξ<sub>prox</sub> = {xi:g} Å, T = {t:g} · the page's 30 z-layers"
        ),
        showlegend=False,
    )
    return fig


def _proximity_volume() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Δ(x, y)·f(z) on the page's z grid, the in-plane gap averaged over 2 Å."""
    dx, sigma = 0.25, 2.0
    n, padded, pad = _padded_grid(100.0, dx)  # 961 points, 120 Å, 80
    r = _moire(n, padded)
    x = _crop(r["x"], pad, 10)  # 81 points, 2.5 Å apart (the page caps at 80)
    gap = _crop(_envelope(_gap_field(r["pattern"]), dx, sigma), pad, 10)
    config = ProximityConfig(
        xi_prox=_PROX["xi_prox"],
        n_z_layers=_PROX["n_z"],
        z_min=_PROX["z_min"],
        z_max=_PROX["z_max"],
        interface_transparency=_PROX["transparency"],
    )
    prox = gap_3d(gap, config)
    return x, prox.z_coords, prox.gap_3d


def _prox_markers(
    x: np.ndarray, y: np.ndarray, z_top: float
) -> tuple[list[dict], go.Scatter3d | None]:
    # The page's height markers (proximity_3d.py _front_edge and
    # _height_markers, lines 53-105; used at lines 356-361 and 383-384):
    # arrows to z = 0 and z = ξ on the stack's vertical edge nearest the
    # camera (here the docs eye), a marker above the drawn top z_top left
    # out, and the edge between the two heights drawn as a line. The label
    # text, type size, backing and arrow offsets are docs styling, set for
    # the docs frame's larger type.
    edge_x = float(np.max(x)) if _DOCS_EYE["x"] >= 0 else float(np.min(x))
    edge_y = float(np.max(y)) if _DOCS_EYE["y"] >= 0 else float(np.min(y))
    xi = _PROX["xi_prox"]
    markers = (
        # height, label, arrow-tail offset (px), side of the tail for the label
        (0.0, "Interface<br>(z = 0)", (55, 30), "left"),
        (xi, f"ξ<sub>prox</sub> ≈ {xi:g} Å", (-45, -40), "right"),
    )
    annotations = [
        dict(
            x=edge_x,
            y=edge_y,
            z=z,
            text=text,
            showarrow=True,
            arrowhead=2,
            arrowwidth=2,
            arrowcolor="white",
            ax=ax,
            ay=ay,
            xanchor=xanchor,
            bgcolor="rgba(17, 17, 17, 0.8)",
            font=dict(size=13, color="white"),
        )
        for z, text, (ax, ay), xanchor in markers
        if z <= z_top
    ]
    edge = None
    if len(annotations) == len(markers):
        edge = go.Scatter3d(
            x=[edge_x, edge_x],
            y=[edge_y, edge_y],
            z=[0.0, xi],
            mode="lines",
            line=dict(color="white", width=4),
            hoverinfo="skip",
            showlegend=False,
        )
    return annotations, edge


def _prox_ticks(fig: go.Figure, extent: float, z_ticks: list[float]) -> go.Figure:
    """Fewer 3D ticks, so no two labels meet at the box's front corners.

    With the default ticks, z's lowest label sat on y's first label at the
    left corner. These scenes keep x and y at -extent, 0 and extent, and
    label only the z heights the caption needs.
    """
    xy_ticks = [-extent, 0.0, extent]
    fig.update_scenes(xaxis_tickvals=xy_ticks, yaxis_tickvals=xy_ticks, zaxis_tickvals=z_ticks)
    return fig


_PROX_STEP = "in-plane gap averaged over σ = 2 Å"


def _proximity_iso(clip_z: float | None) -> go.Figure:
    # /proximity3d iso view (proximity_3d.py lines 362-372, 383-384): iso
    # range [0.2, 0.8] × peak Δ, 5 surfaces; clip_z drops the layers above
    # the plane.
    x, z, vol = _proximity_volume()
    peak = float(np.nanmax(vol))
    clipped = f", clipped at z = {clip_z:g} Å" if clip_z is not None else ""
    annotations, edge = _prox_markers(x, x, clip_z if clip_z is not None else float(z[-1]))
    fig = create_3d_isosurface(
        x,
        x,
        z,
        vol,
        title=f"3D Cooper surface, {_SYSTEM}{clipped} (SPECULATIVE)",
        iso_min=0.2 * peak,
        iso_max=0.8 * peak,
        surface_count=5,
        annotations=annotations,
        clip_z=clip_z,
    )
    if edge is not None:
        fig.add_trace(edge)
    fig.update_layout(
        title_subtitle_text=(
            f"Δ(x, y)·f(z), ξ<sub>prox</sub> = 100 Å, T = 0.8 · iso levels 0.2–0.8 × peak · "
            f"{_PROX_STEP}"
        )
    )
    z_ticks = [0.0, clip_z / 2.0, clip_z] if clip_z is not None else [0.0, 100.0, 200.0, 300.0]
    return _prox_ticks(fig, 100.0, z_ticks)


def scene_proximity_3d() -> go.Figure:
    return _proximity_iso(clip_z=None)


def scene_proximity_3d_clipped() -> go.Figure:
    return _proximity_iso(clip_z=100.0)


_VOLUME_EXTENT = 30.0  # Å: the page's 80-point grid cap resolves the 3.7 Å rows here
_VOLUME_OPACITY = 0.3  # the page draws 0.1, which leaves the volume nearly invisible


def scene_proximity_3d_volume() -> go.Figure:
    # /proximity3d Volume view (proximity_3d.py lines 306-332, 373-384) at
    # extent 30 Å: the page caps the grid at 80 (dx = 0.76 Å, so the raw gap
    # needs no average), keeps its coolwarm colors and 17 surfaces; only the
    # opacity is raised.
    r = _moire(80, _VOLUME_EXTENT)
    config = ProximityConfig(
        xi_prox=_PROX["xi_prox"],
        n_z_layers=_PROX["n_z"],
        z_min=_PROX["z_min"],
        z_max=_PROX["z_max"],
        interface_transparency=_PROX["transparency"],
    )
    prox = gap_3d(_gap_field(r["pattern"]), config)
    annotations, edge = _prox_markers(r["x"], r["y"], float(prox.z_coords[-1]))
    fig = create_3d_volume(
        r["x"],
        r["y"],
        prox.z_coords,
        prox.gap_3d,
        title=f"3D Cooper volume, {_SYSTEM}, ±30 Å (SPECULATIVE)",
        opacity=_VOLUME_OPACITY,
        surface_count=17,
        annotations=annotations,
    )
    if edge is not None:
        fig.add_trace(edge)
    fig.update_layout(
        title_subtitle_text=(
            "Δ(x, y)·f(z), ξ<sub>prox</sub> = 100 Å, T = 0.8 · Volume view at extent 30 Å, "
            "grid 80 · opacity 0.3 (page: 0.1)"
        )
    )
    return _prox_ticks(fig, _VOLUME_EXTENT, [0.0, 100.0, 200.0, 300.0])


def scene_phase_diagram() -> go.Figure:
    # pages/phase_diagram.py update_phase (lines 215-233) with B 0-8 T,
    # Δ 0.1-6 meV and the maximum resolution 200; g = 30, μ = 0 (defaults).
    b = np.linspace(0.0, 8.0, 200)
    delta = np.linspace(0.1, 6.0, 200)
    phase = phase_diagram_sweep(b, delta, G_FACTOR_TSS, 0.0)
    fig = create_phase_colormap(
        b,
        delta,
        phase,
        title=f"Topological phase diagram, {_SYSTEM} (SPECULATIVE)",
        x_label="B (T)",
        y_label="Δ (meV)",
    )
    fig.add_hline(
        y=DELTA_AVG,
        line=dict(dash="dash", color="lime", width=2),
        annotation_text=f"Δ<sub>avg</sub> = {DELTA_AVG:.2f} meV",
        annotation_bgcolor="rgba(0, 0, 0, 0.6)",
        annotation_font=dict(color="white", size=14),
    )
    b_cross = DELTA_AVG / (G_FACTOR_TSS * MU_B_MEV_T)
    fig.update_layout(
        title_subtitle_text=(
            f"Fu–Kane: topological where g μ<sub>B</sub> B > √(Δ² + μ²) · g = {G_FACTOR_TSS:g}, "
            f"μ = 0 · crosses Δ<sub>avg</sub> at {b_cross:.2f} T"
        )
    )
    return fig


def scene_magnetic_majorana_3d() -> go.Figure:
    # /magnetic "Majorana 3D" view (magnetic_field.py lines 248-255) at
    # B_z = 4 T, extent 300 Å, grid 81; the page's 32 z-layers from -40 to 200 Å.
    x = np.linspace(-_VORTEX_EXTENT, _VORTEX_EXTENT, 81)
    positions = _vortex_positions(_VORTEX_EXTENT, x.size)
    z = np.linspace(-40.0, 200.0, 32)
    density = majorana_probability_density_3d(x, x, z, positions, xi_prox=XI_PROXIMITY_DEFAULT)
    fig = create_3d_majorana_isosurface(
        x,
        x,
        z,
        density,
        positions,
        title="Majorana zero-mode density at B<sub>z</sub> = 4 T (SPECULATIVE)",
    )
    fig.update_layout(
        showlegend=False,
        title_subtitle_text=(
            "isosurfaces of |ψ|² at the 7 vortex cores (cyan ×) · ±300 Å, 81 × 81 × 32, "
            "ξ<sub>prox</sub> = 100 Å"
        ),
    )
    return fig


# ---- Speculative extensions: isotope effects --------------------------------


def scene_isotope_gap_shift() -> go.Figure:
    """Docs-only: Δ₂ shift for fully enriched FeTe against the isotope exponent."""
    alphas = np.linspace(-0.5, 1.0, 151)
    masses = {iso.mass_number: iso.atomic_mass for iso in ELEMENTS["Fe"].isotopes}
    colors = list(_pdw_family_colors(True).values())
    _primary, _secondary, muted = _line_colors(True)
    fig = go.Figure()
    for (mass_number, label), color in zip(
        ((54, "<sup>54</sup>Fe"), (57, "<sup>57</sup>Fe"), (58, "<sup>58</sup>Fe")),
        colors,
        strict=True,
    ):
        shift = [
            100.0
            * (
                compute_isotope_effects(
                    "FeTe",
                    "Sb2Te3",
                    _FETE.a,
                    _SB2TE3.a,
                    _SB2TE3.lattice_type,
                    _FETE.lattice_type,
                    mass_overrides={"Fe": masses[mass_number]},
                    alpha=float(alpha),
                ).delta_2_modified
                / DELTA_2
                - 1.0
            )
            for alpha in alphas
        ]
        fig.add_trace(
            go.Scatter(
                x=alphas,
                y=shift,
                name=f"100 % {label}",
                mode="lines",
                line=dict(color=color, width=2.5),
            )
        )
    fig.add_hline(y=0.0, line=dict(color=muted, width=1))
    for alpha, who, position in (
        (-0.18, "(Ba,K)Fe<sub>2</sub>As<sub>2</sub>", "top left"),
        (0.4, "iron-based consensus", "top left"),
        (0.5, "BCS", "bottom right"),
        (0.81, "FeSe", "top right"),
    ):
        fig.add_vline(
            x=alpha,
            line=dict(dash="dot", color=muted, width=1),
            annotation_text=f"α = {_minus(alpha)}: {who}",
            annotation_position=position,
            annotation_font_size=13,
        )
    fig.update_layout(
        title="Isotope shift of the gap vs exponent α (SPECULATIVE)",
        title_subtitle_text=(
            "FeTe fully enriched in one Fe isotope · Δ ∝ M<sup>−α</sup> with M the "
            "formula-unit mass · Δ<sub>2</sub> = 3.60 meV"
        ),
        xaxis_title="Isotope exponent α",
        yaxis_title="Δ/Δ<sub>0</sub> − 1 (%)",
        template="plotly_dark",
        legend=_inside_legend(0.02, 0.03, "left", "bottom"),
        margin=dict(l=60, r=20, t=95, b=50),
    )
    return fig


# ---- Speculative extensions: spin-fluctuation pairing ------------------------

_SF_PAPER_FIT = _SF_PRESETS["LCCO x = 0.159 — paper fit (Duffy Fig. 4)"]
_SF_IMPRINT = _SF_PRESETS["Moire glue what-if: imprint vs operating Tc"]
_SF_IMPRINT_TARGETS = np.linspace(0.5, 25.0, 80)  # the page's _IMPRINT_TARGETS (line 249)
_SF_COUPLING_LAMBDA = np.linspace(0.0, 3.0, 301)  # the page's _COUPLING_LAMBDA (line 250)


def _sf_config(preset: dict) -> tuple[SfConfig, SfGlueConfig]:
    """Mirror of _clamp, _sf_config and _glue_state (spin_fluctuation.py lines 426-478, 641-675).

    Both presets sit inside every clamp, so _clamp only snaps the doping string to
    the tabulated x. The glue config carries the layers' shared two-lattice period.
    """
    config = SfConfig(
        doping=snap_doping(preset["doping"]),
        inelastic=preset["inelastic"],
        hotspots=preset["hotspots"],
        gamma_imp=preset["gamma_imp"],
        alpha1=preset["alpha1"],
        alpha2=preset["alpha2"],
        hotspot_peak=preset["hotspot_peak"],
        hotspot_width=preset["hotspot_width"],
        nu=preset["nu"],
        phi_hs_deg=preset["phi_hs"],
        cvc_epsilon=preset["cvc_epsilon"],
    )
    substrate = get_material(preset["substrate"])
    overlayer = get_material(preset["overlayer"])
    glue = SfGlueConfig(
        eta=preset["eta"],
        tc_target=preset["tc_target"],
        coherence=preset["coherence"],
        pairing=SfPairingMap(form=preset["pairing_form"]),
        moire_period=moire_locked_period(substrate.a, overlayer.a, preset["twist"]),
    )
    return config, glue


# Docs typography for the SF builders' plain-text labels (Tc is replaced separately).
_SF_TYPOGRAPHY = (
    ("λ_tr", "λ<sub>tr</sub>"),
    ("λ_ρ", "λ<sub>ρ</sub>"),
    ("λ_c", "λ<sub>c</sub>"),
    ("Γ_HS", "Γ<sub>HS</sub>"),
    ("L_m", "L<sub>m</sub>"),
    ("k_x", "k<sub>x</sub>"),
    ("k_y", "k<sub>y</sub>"),
    ("(1/Å)", "(Å<sup>−1</sup>)"),
)


def _sf_label(text: str | None) -> str | None:
    if not text:
        return text
    text = re.sub(r"\bTc\b", "T<sub>c</sub>", text)
    for old, new in _SF_TYPOGRAPHY:
        text = text.replace(old, new)
    return text


def _sf_typography(fig: go.Figure) -> go.Figure:
    """Docs typography in the title, axis titles, legend entries and annotations."""
    fig.update_layout(title_text=_sf_label(fig.layout.title.text))
    fig.update_xaxes(title_text=_sf_label(fig.layout.xaxis.title.text))
    fig.update_yaxes(title_text=_sf_label(fig.layout.yaxis.title.text))
    for trace in fig.data:
        trace.name = _sf_label(trace.name)
    for annotation in fig.layout.annotations:
        annotation.text = _sf_label(annotation.text)
    return fig


def scene_sf_magnetoresistance() -> go.Figure:
    # Preset 1, the "mr" main graph (_main_figure, lines 737-748): the 20-50 K
    # family on the 0-35 T grid. The preset's 40 K is one of the family, so the
    # page draws no extra current-T curve (_current_curve, lines 604-610).
    config, _glue = _sf_config(_SF_PAPER_FIT)
    mr = magnetoresistance(config, SF_MR_TEMPERATURES, _SF_PAPER_FIT["field_max"])
    fig = create_sf_mr_plot(
        mr.fields,
        mr.temperatures,
        mr.mr,
        title=f"Magnetoresistance, LCCO x = {config.doping:.3f} (SPECULATIVE)",
    )
    fig.update_traces(line_width=2.5)
    at_20t = mr.mr[:, int(np.argmin(np.abs(mr.fields - 20.0)))]
    fig.update_layout(
        title_subtitle_text=(
            "preset “LCCO x = 0.159 — paper fit (Duffy Fig. 4)” · at 20 T: "
            f"{100.0 * at_20t[0]:.2f} % at 20 K → {100.0 * at_20t[-1]:.2f} % at 50 K"
        )
    )
    return _sf_typography(fig)


def scene_sf_fermi_surface() -> go.Figure:
    # Preset 1, the Fermi-surface graph (_fermi_surface_figure, lines 829-852)
    # at the preset's 40 K.
    config, _glue = _sf_config(_SF_PAPER_FIT)
    t = _SF_PAPER_FIT["temperature"]
    fs = fermi_surface(config)

    def points(phi: np.ndarray) -> np.ndarray:
        kf = fermi_surface_at(phi, config)[0] * 1e-10  # 1/m -> 1/Å
        return np.column_stack([kf * np.cos(phi), kf * np.sin(phi)])

    centres = cvc_centres(config)
    hot = config.hotspots and config.hotspot_peak > 0.0
    fig = create_sf_fermi_surface(
        fs.kf * 1e-10 * np.cos(fs.phi),
        fs.kf * 1e-10 * np.sin(fs.phi),
        scattering_rate(fs.phi, t, config),
        points(hotspot_angles(config)) if hot else None,
        points(centres),
        points(cvc_partner_angle(centres, config)),
        np.pi / A_LCCO,
        title=f"Fermi-surface Γ(φ) at {t:g} K (SPECULATIVE)",
    )
    fig.update_layout(
        title_subtitle_text=(
            f"LCCO x = {config.doping:.3f} about M · rings: hot spots · lines: CVC centers "
            "→ k + Q partners"
        )
    )
    return _sf_typography(fig)


def scene_sf_coupling() -> go.Figure:
    # Preset 1, the coupling graph (_coupling_figure, lines 855-890): the doping
    # sweep keyed on SF_DOPING_REF (line 1210), both pairing maps, the current
    # doping and the glue operating point at the preset's Tc* = 13.5 K.
    config, glue = _sf_config(_SF_PAPER_FIT)
    sweep = doping_sweep(replace(config, doping=SF_DOPING_REF))
    selected = glue.pairing.form
    lam = _SF_COUPLING_LAMBDA
    curves = [
        (
            "Tc(λ): empirical power law",
            lam,
            tc_from_lambda(lam, SfPairingMap(form="power_law")),
            "solid",
            selected == "power_law",
        ),
        (
            "Tc(λ): McMillan best fit (fails)",
            lam,
            tc_from_lambda(lam, SfPairingMap(form="mcmillan")),
            "dot",
            selected == "mcmillan",
        ),
    ]
    i = int(np.argmin(np.abs(sweep.dopings - config.doping)))
    fig = create_sf_coupling_plot(
        sweep.lambda_tr,
        sweep.lambda_rho,
        sweep.tc,
        sweep.dopings,
        curves,
        current=(float(sweep.lambda_tr[i]), float(sweep.tc[i])),
        operating_point=(lambda_from_tc(glue.tc_target, glue.pairing), glue.tc_target),
        lambda_c=lambda_threshold(glue.pairing),
        title="Coupling vs Tc across the LCCO dome (SPECULATIVE)",
    )
    fig.update_layout(
        title_subtitle_text=_sf_label(
            f"{sweep.dopings.size} LCCO dopings at their table Tc · ring: x = "
            f"{config.doping:.3f} · star: glue operating point, Tc* = {glue.tc_target:g} K"
        )
    )
    return _sf_typography(fig)


def scene_sf_glue_imprint() -> go.Figure:
    # Preset 11, the "imprint" main graph (_main_figure, lines 793-811): clean
    # coherence, so no shading (the page shades only in the local limit); the
    # star is the dressed imprint F·(local exact) at the preset's Tc* = 5 K.
    _config, glue = _sf_config(_SF_IMPRINT)
    targets = _SF_IMPRINT_TARGETS
    dressed, local, linear = imprint_curve(glue, targets)
    fragment = fragmentation_tc(glue) if glue.coherence == "local" else 0.0
    local_exact, _local_linear = local_imprint(glue.eta, glue.tc_target, glue.pairing)
    star = coherence_factor(glue) * local_exact
    fig = create_sf_imprint_plot(
        targets,
        dressed,
        local,
        linear,
        bcs_route_relative_modulation(),
        coherence_label=COHERENCE_PLOT_LABELS[glue.coherence],
        fragment_below=fragment if fragment > targets[0] else None,
        current=(glue.tc_target, star),
        title=f"Glue imprint vs operating Tc*, {_sub('Sb2Te3')}/FeTe, θ = 0.0° (SPECULATIVE)",
    )
    fig.update_layout(
        title_subtitle_text=_sf_label(
            f"η = {glue.eta:g}, {COHERENCE_PLOT_LABELS[glue.coherence]}, L_m = "
            f"{glue.moire_period:.1f} Å · star: {star:.4f} at Tc* = {glue.tc_target:g} K · "
            f"amplitude route: {bcs_route_relative_modulation():.3f}"
        )
    )
    return _sf_typography(fig)


# ---- Speculative extensions: pair density waves -----------------------------


def _pdw_config(preset: dict) -> PdwConfig:
    """Mirror of _build_config and _update_pdw (pair_density_wave.py lines 344-360, 417-442)."""
    if preset["geometry"] == "moire":
        substrate = get_material(preset["substrate"])
        overlayer = get_material(preset["overlayer"])
        period = moire_locked_period(substrate.a, overlayer.a, preset["twist"])
        base = moire_locked_config(
            period, preset["twist"], DELTA_AVG, moire_locked_pdw_ratio(period)
        )
    else:
        base = PdwConfig()
    return replace(
        base,
        temperature=max(preset["temperature"], 0.0),
        t_pdw=preset["tpdw_ratio"] * base.tc,
        field=(max(preset["bx"], 0.0), max(preset["by"], 0.0), max(preset["bz"], 0.0)),
        kappa_p=preset["kappa_p"],
        kappa_q=preset["kappa_q"],
        kappa_h=preset["kappa_h"],
        kappa_harm=preset["kappa_harm"],
        floor=10.0 ** preset["floor"],
    )


@dataclass(frozen=True)
class _PdwAxis:
    """The field geometry _update_pdw derives for its sweeps (lines 480-508)."""

    direction: np.ndarray
    b_along: float
    on_axis: bool
    hc2_now: float
    hpdw_now: float
    onset: float
    field_induced: bool
    b_max: float
    t_max: float


def _pdw_axis(config: PdwConfig, sweep_axis: str) -> _PdwAxis:
    field = np.asarray(config.field, dtype=float)
    direction = np.asarray(FIELD_AXES[sweep_axis], dtype=float)
    b_norm = float(np.linalg.norm(field))
    b_along = float(abs(np.dot(field, direction)))
    t_now = config.temperature
    return _PdwAxis(
        direction=direction,
        b_along=b_along,
        on_axis=abs(b_norm - b_along) < 1e-9,
        hc2_now=critical_field_along(config.tc, config.hc2, t_now, direction),
        hpdw_now=critical_field_along(config.t_pdw, config.h_pdw, t_now, direction),
        onset=onset_field_along(config.h_rot, direction),
        field_induced=bool(np.any(np.asarray(config.overlap0) < 1.0)),
        b_max=1.15
        * max(
            critical_field_along(config.tc, config.hc2, 0.0, direction),
            critical_field_along(config.t_pdw, config.h_pdw, 0.0, direction),
            b_along,
        ),
        t_max=1.25 * max(config.tc, config.t_pdw),
    )


# Docs typography for create_pdw_sweep_plot's legend entries.
_PDW_CURVE_NAMES = {
    "p₁,₂ (field-induced)": "p<sub>1,2</sub> (field-induced)",
    "p₃": "p<sub>3</sub>",
    "q₁,₂": "q<sub>1,2</sub>",
    "q₃": "q<sub>3</sub>",
    "h₁,₂": "h<sub>1,2</sub>",
}
_UTE2 = _sub("UTe2")


def _pdw_sweep_typography(fig: go.Figure) -> go.Figure:
    for old, new in _PDW_CURVE_NAMES.items():
        fig.update_traces(name=new, selector=dict(name=old))
    fig.update_yaxes(title_text="|ρ<sub>k</sub>|² (arb.)")
    fig.update_shapes(opacity=0.25, selector=dict(type="rect"))  # the app's 0.12 barely shows
    return fig


def scene_pdw_tsweep() -> go.Figure:
    # "UTe2 300 mK, zero field" preset: the temperature-sweep graph of
    # _update_pdw (lines 505-523).
    config = _pdw_config(_PDW_PRESETS["UTe2 300 mK, zero field"])
    field = np.asarray(config.field)
    tc_b = config.tc * max(0.0, 1.0 - reduced_field(field, config.hc2))
    tpdw_b = config.t_pdw * max(0.0, 1.0 - reduced_field(field, config.h_pdw))
    temps = np.linspace(0.0, 1.25 * max(config.tc, config.t_pdw), 241)
    sweep = temperature_sweep(config, temps)
    fig = create_pdw_sweep_plot(
        temps,
        sweep.intensity_p,
        sweep.intensity_q,
        sweep.intensity_h,
        x_label="T (K)",
        title=f"{_UTE2} composite-CDW peaks vs temperature at B = 0 (SPECULATIVE)",
        floor=config.floor,
        current=config.temperature,
        markers=[(tc_b, "T<sub>c</sub>"), (tpdw_b, "T<sub>PDW</sub>")],
        shade=(tc_b, tpdw_b, "PDW only"),
    )
    return _pdw_sweep_typography(fig)


def scene_pdw_fft() -> go.Figure:
    # "UTe2 3 K — Cheshire-cat window" preset, FFT view (lines 462-469):
    # charge_spectrum over the preset's ±40 Å.
    preset = _PDW_PRESETS["UTe2 3 K — Cheshire-cat window"]
    config = _pdw_config(preset)
    state = compute_pdw_state(config)
    spectrum = charge_spectrum(config, state, preset["extent"])
    fig = create_pdw_peak_map(
        pdw_peaks(config, state),
        title=f"{_UTE2} at 3 K: STM-style FFT of ρ(r) (SPECULATIVE)",
        fft=(spectrum["kx"], spectrum["ky"], spectrum["power_spectrum"]),
    )
    top = float(np.max(spectrum["power_spectrum"]))
    fig.update_traces(
        zmin=top - 4.0,
        zmax=top,
        colorbar_title_text="log<sub>10</sub>|F|²",
        selector=dict(type="heatmap"),
    )
    for trace in fig.data:
        if trace.type == "scatter":
            trace.text = [re.sub(r"(\d)$", r"<sub>\1</sub>", t) for t in trace.text]
    fig.update_traces(textfont=dict(size=15, shadow="auto"), selector=dict(type="scatter"))
    fig.update_layout(
        title_subtitle_text=(
            "PDW only (Cheshire cat): q and h peaks, empty p rings · ±40 Å, Hann taper · "
            "top 4 decades shown"
        ),
        xaxis_title="k<sub>x</sub> (Å<sup>−1</sup>)",
        yaxis_title="k<sub>y</sub> (Å<sup>−1</sup>)",
    )
    return fig


def scene_pdw_bsweep() -> go.Figure:
    # "UTe2 field-induced p2 (Bz ≈ b* = 6 T)" preset: the field-sweep graph of
    # _update_pdw (lines 525-543), along z ≈ b*.
    preset = _PDW_PRESETS["UTe2 field-induced p2 (Bz ≈ b* = 6 T)"]
    config = _pdw_config(preset)
    axis = _pdw_axis(config, preset["sweep_axis"])
    mags = np.linspace(0.0, axis.b_max, 241)
    sweep = field_sweep(config, mags, axis.direction)
    markers = [(axis.hc2_now, "H<sub>c2</sub>(T)"), (axis.hpdw_now, "H<sub>PDW</sub>(T)")]
    if axis.field_induced:
        markers.append((axis.onset, "d-vector onset"))
    fig = create_pdw_sweep_plot(
        mags,
        sweep.intensity_p,
        sweep.intensity_q,
        sweep.intensity_h,
        x_label="|B| along z ≈ b* (T)",
        title=f"{_UTE2} peaks vs field along b* at {config.temperature:g} K (SPECULATIVE)",
        floor=config.floor,
        current=axis.b_along if axis.on_axis else None,
        markers=markers,
        shade=(axis.hc2_now, axis.hpdw_now, "PDW only"),
    )
    return _pdw_sweep_typography(fig)


def scene_pdw_regime_map() -> go.Figure:
    # "UTe2 4.2 K, By = a = 0.2 T (q fading)" preset: the regime map of
    # _update_pdw (lines 545-556), 121 × 121 over (|B| along y = a, T).
    preset = _PDW_PRESETS["UTe2 4.2 K, By = a = 0.2 T (q fading)"]
    config = _pdw_config(preset)
    axis = _pdw_axis(config, preset["sweep_axis"])
    temps = np.linspace(0.0, axis.t_max, 121)
    mags = np.linspace(0.0, axis.b_max, 121)
    fig = create_pdw_regime_map(
        mags,
        temps,
        regime_map(config, temps, mags, axis.direction),
        field_label="|B| along y = a",
        title=f"{_UTE2} PDW regimes, field along a (SPECULATIVE)",
        current=(axis.b_along, config.temperature) if axis.on_axis else None,
    )
    fig.update_layout(
        title_subtitle_text=(
            f"× = the preset: T = {config.temperature:g} K, B<sub>y</sub> = "
            f"{config.field[1]:g} T · PDW only = Cheshire cat"
        ),
        margin_t=95,
    )
    return fig


def scene_pdw_moire_whatif() -> go.Figure:
    # "Moire what-if: intrinsic PDW above Tc" preset (FeTe/Sb2Te3, 16 K,
    # T_PDW = 1.5 Tc, ±100 Å, grid 200) in the Pair amplitude view
    # (lines 451-454), which plots |d(r)| with the gap builder.
    preset = _PDW_PRESETS["Moire what-if: intrinsic PDW above Tc"]
    config = _pdw_config(preset)
    state = compute_pdw_state(config)
    x = np.linspace(-preset["extent"], preset["extent"], preset["grid_size"])
    fields = pdw_real_space(config, state, x, x)
    fig = create_gap_heatmap(
        x,
        x,
        fields.pair_amplitude,
        title="Moire-locked PDW what-if above T<sub>c</sub> (SPECULATIVE)",
    )
    fig.update_traces(colorbar_title_text="|d| (meV)", selector=dict(type="heatmap"))
    period = moire_locked_period(_FETE.a, _SB2TE3.a, preset["twist"])
    fig.update_layout(
        title_subtitle_text=(
            f"pair amplitude |d(r)|, {_SYSTEM}, θ = 0° (L<sub>m</sub> = {period:.1f} Å) · "
            f"T = {config.temperature:g} K, T<sub>PDW</sub> = 1.5 T<sub>c</sub>"
        )
    )
    return fig


# ---- Registry: canonical page order ------------------------------------------
# Core pipeline (moire -> gap -> Fourier), graphene track (stacks -> BM model
# -> curvature), speculative extensions (magnetic -> topological -> isotope ->
# spin-fluctuation pairing -> PDW). Filenames are literal strings:
# scripts/check_docs.py reads them.

_AVG_2 = "gap averaged over σ = 2 Å (Gaussian local average)"

SCENES: tuple[Scene, ...] = (
    # Core pipeline
    Scene("viewer-2d.png", scene_viewer_2d, "square", "app"),
    Scene("viewer-3d.png", scene_viewer_3d, "wide", "app"),
    Scene(
        "substrate-comparison.png",
        scene_substrate_comparison,
        "wide",
        "presentation",
        step="moire pattern and gap averaged over σ = 2 Å (Gaussian local average)",
    ),
    Scene("sweep-twist.png", scene_sweep_twist, "wide", "app"),
    Scene("viewer-gap-2d.png", scene_viewer_gap_2d, "square", "app"),
    Scene("fourier.png", scene_fourier, "square", "app"),
    Scene(
        "fourier-peaks.png",
        scene_fourier_peaks,
        "square",
        "presentation",
        step="detected peaks (threshold 0.3) overlaid as rings with λ labels, plus the DC mask",
    ),
    # Graphene track
    Scene(
        "graphene-pattern.png",
        scene_graphene_pattern,
        "square",
        "presentation",
        step="pattern averaged over σ = 1.5 Å (moire envelope)",
    ),
    Scene(
        "graphene-heterostrain.png",
        scene_graphene_heterostrain,
        "square",
        "presentation",
        step="pattern averaged over σ = 1.5 Å (moire envelope)",
    ),
    Scene(
        "graphene-supermoire.png",
        scene_graphene_supermoire,
        "square",
        "presentation",
        step="local RMS contrast over σ = 3 Å",
    ),
    Scene("graphene-supermoire-zoom.png", scene_graphene_supermoire_zoom, "square", "app"),
    Scene("graphene-flatband.png", scene_graphene_flatband, "wide", "app"),
    Scene("graphene-bands.png", scene_graphene_bands, "wide", "app"),
    Scene("graphene-dos.png", scene_graphene_dos, "wide", "app"),
    Scene("graphene-bandwidth.png", scene_graphene_bandwidth, "wide", "docs-only"),
    Scene("graphene-pseudo-field.png", scene_graphene_pseudo_field, "square", "app"),
    Scene("graphene-bump-field.png", scene_graphene_bump_field, "square", "app"),
    Scene(
        "graphene-curved-3d.png",
        scene_graphene_curved_3d,
        "wide",
        "presentation",
        step="gap averaged over σ = 1.5 Å before coloring the sheet",
    ),
    # Speculative extensions: magnetic field
    Scene(
        "magnetic-vortex-2d.png",
        scene_magnetic_vortex_2d,
        "square",
        "presentation",
        step=_AVG_2,
    ),
    Scene(
        "magnetic-currents-3d.png",
        scene_magnetic_currents_3d,
        "wide",
        "presentation",
        step=_AVG_2,
    ),
    # Topological proximity
    Scene("proximity-decay.png", scene_proximity_decay, "wide", "app"),
    Scene("proximity-3d.png", scene_proximity_3d, "wide", "presentation", step=_AVG_2),
    Scene(
        "proximity-3d-clipped.png",
        scene_proximity_3d_clipped,
        "wide",
        "presentation",
        step=_AVG_2,
    ),
    Scene(
        "proximity-3d-volume.png",
        scene_proximity_3d_volume,
        "wide",
        "presentation",
        step="volume opacity raised from the page's 0.1 to 0.3",
    ),
    Scene("phase-diagram.png", scene_phase_diagram, "wide", "app"),
    Scene("magnetic-majorana-3d.png", scene_magnetic_majorana_3d, "wide", "app"),
    # Isotope effects
    Scene("isotope-gap-shift.png", scene_isotope_gap_shift, "wide", "docs-only"),
    # Spin-fluctuation pairing
    Scene("sf-magnetoresistance.png", scene_sf_magnetoresistance, "wide", "app"),
    Scene("sf-fermi-surface.png", scene_sf_fermi_surface, "square", "app"),
    Scene("sf-coupling.png", scene_sf_coupling, "wide", "app"),
    Scene("sf-glue-imprint.png", scene_sf_glue_imprint, "wide", "app"),
    # Pair density waves
    Scene("pdw-tsweep.png", scene_pdw_tsweep, "wide", "app"),
    Scene(
        "pdw-fft.png",
        scene_pdw_fft,
        "square",
        "presentation",
        step="color range clipped to the top 4 decades",
    ),
    Scene("pdw-bsweep.png", scene_pdw_bsweep, "wide", "app"),
    Scene("pdw-regime-map.png", scene_pdw_regime_map, "wide", "app"),
    Scene("pdw-moire-whatif.png", scene_pdw_moire_whatif, "square", "app"),
)
SCENE_BY_NAME: dict[str, Scene] = {scene.name: scene for scene in SCENES}


# ---- PNG checks ---------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_is_opaque(data: bytes) -> bool:
    """True when every pixel of an 8-bit, non-interlaced PNG is fully opaque.

    Formats without an alpha channel count as opaque unless a tRNS chunk adds
    transparency. For RGBA and gray+alpha, the alpha channel is checked
    without decoding the image: PNG filters predict each alpha byte from
    alpha bytes only, so an all-255 alpha channel leaves one known filtered
    byte per pixel (255, 128 or 0, by filter type and position), and the
    image is opaque exactly when every stored alpha byte matches it.
    """
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    pos, ihdr, idat, has_trns = len(_PNG_SIGNATURE), None, [], False
    while pos + 8 <= len(data):
        length, tag = struct.unpack(">I4s", data[pos : pos + 8])
        body = data[pos + 8 : pos + 8 + length]
        if tag == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"tRNS":
            has_trns = True
        elif tag == b"IEND":
            break
        pos += 12 + length
    if ihdr is None:
        raise ValueError("PNG without an IHDR chunk")
    width, height, depth, color_type, _compression, _filter, interlace = ihdr
    if color_type in (0, 2, 3):  # gray, RGB, palette: no alpha channel
        return not has_trns
    if depth != 8 or interlace != 0 or color_type not in (4, 6):
        raise ValueError(f"unsupported PNG layout: depth {depth}, color type {color_type}")
    channels = 4 if color_type == 6 else 2
    rows = np.frombuffer(zlib.decompress(b"".join(idat)), dtype=np.uint8)
    rows = rows.reshape(height, 1 + width * channels)
    filters = rows[:, :1]
    if np.any(filters > 4):
        raise ValueError("PNG with an unknown row filter")
    alpha = rows[:, channels::channels]  # the stored (filtered) alpha bytes
    first_row = (np.arange(height) == 0)[:, None]
    first_col = (np.arange(width) == 0)[None, :]
    expected = np.zeros((height, width), dtype=np.uint8)  # Sub, Up, Average, Paeth inside
    expected[np.broadcast_to(filters == 0, expected.shape)] = 255  # None: the value itself
    expected[np.broadcast_to((filters == 1) & first_col, expected.shape)] = 255
    expected[np.broadcast_to((filters == 2) & first_row, expected.shape)] = 255
    average = filters == 3
    expected[np.broadcast_to(average & (first_row ^ first_col), expected.shape)] = 128
    expected[np.broadcast_to(average & first_row & first_col, expected.shape)] = 255
    expected[np.broadcast_to((filters == 4) & first_row & first_col, expected.shape)] = 255
    return bool(np.array_equal(alpha, expected))


# ---- Command line --------------------------------------------------------------


def _selected(parser: argparse.ArgumentParser, only: Iterable[str]) -> list[Scene]:
    names = [name.strip() for arg in only for name in arg.split(",") if name.strip()]
    unknown = sorted(set(names) - SCENE_BY_NAME.keys())
    if unknown:
        parser.error(f"unknown scene(s): {', '.join(unknown)} (see --list)")
    if not names:
        return list(SCENES)
    return [scene for scene in SCENES if scene.name in set(names)]


def _render_png(fig: go.Figure, scene: Scene) -> bytes:
    width, height, scale = FRAMES[scene.frame]
    return fig.to_image(format="png", width=width, height=height, scale=scale)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the web-app figures in docs/images (see docs/images/README.md).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output folder (default: the repository's docs/images)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="SCENE[,SCENE...]",
        help="render only these scenes; comma lists, repeatable",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print name, file, frame, kind and PNG size of each scene, then exit",
    )
    args = parser.parse_args(argv)
    scenes = _selected(parser, args.only)

    if args.list:
        for scene in scenes:
            width, height = scene.pixels
            print(f"{scene.name}\t{scene.filename}\t{scene.frame}\t{scene.kind}\t{width}x{height}")
        return 0

    import kaleido  # deferred: --list and the tests need no browser

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # One Chromium for the whole run. MathJax is off (no figure uses LaTeX),
    # so rendering needs no network; plotly.js comes from the plotly package.
    kaleido.start_sync_server(timeout=300, mathjax=False, silence_warnings=True)
    written_bytes = 0
    try:
        with warnings.catch_warnings():
            # plotly passes kaleido options that a running server ignores.
            warnings.filterwarnings("ignore", "The kopts argument is ignored", UserWarning)
            for scene in scenes:
                start = time.perf_counter()
                path = args.out_dir / scene.filename
                print(f"[{scene.name}] building ({scene.frame}, {scene.kind})…", flush=True)
                data = _render_png(_docs_style(scene.build(), scene.frame), scene)
                # Check before writing, so a bad render never replaces a good file.
                if not png_is_opaque(data):
                    raise RuntimeError(f"{scene.name}: the PNG has transparent pixels; not written")
                path.write_bytes(data)
                written_bytes += len(data)
                kb = len(data) // 1024
                warn = f"  WARNING: above the {SOFT_CAP_KB} KB soft cap" if kb > SOFT_CAP_KB else ""
                print(
                    f"[{scene.name}] wrote {path} ({kb} KB, "
                    f"{time.perf_counter() - start:.1f} s){warn}",
                    flush=True,
                )
    finally:
        kaleido.stop_sync_server(silence_warnings=True)

    # Both totals sum bytes and floor once, so they agree on the same files.
    pngs = sorted(args.out_dir.glob("*.png"))
    total_kb = sum(p.stat().st_size for p in pngs) // 1024
    print(
        f"wrote {len(scenes)} PNGs, {written_bytes // 1024} KB; {args.out_dir} holds "
        f"{len(pngs)} PNGs, {total_kb} KB of the {FOLDER_BUDGET_KB} KB (12 MB) budget"
    )
    if total_kb > FOLDER_BUDGET_KB:
        print("WARNING: the folder is over the 12 MB docs figure budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
