"""Plotly figure constructors for moire visualisation.

Colormap convention (must stay in sync with crates/moire-core/src/colormap.rs).
Colormap data itself lives in :mod:`waytogocoop.components.colormaps`, which
loads a 256-stop LUT generated from the Rust side — the two stacks produce
pixel-identical screenshots so long as that JSON is regenerated after Rust-side
edits.

- ``viridis``  — unsigned scalar fields (moire pattern, surface plots) and the
  scattering rate Γ(φ) on the spin-fluctuation Fermi surface
- ``coolwarm`` — signed and gap-valued fields: gap modulation Δ (meV) and the
  gap-colored overlays, the PDW signed fields, and the signed pseudo-magnetic
  field (T). It runs low = blue to high = red, identical to the Rust LUT.
  Builders ask for it as ``"RdBu_r"`` or ``"coolwarm"``; see
  :func:`_resolve_colorscale`.
- ``inferno``  — FFT power spectrum (log10 scaled), Majorana density
- ``plasma``   — susceptibility (arbitrary units); fixed plasma stops also give
  the temperature ramps of the spin-fluctuation sweeps
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from waytogocoop.components.colormaps import (
    CAMERA_PRESETS,
    LIGHT_POSITION,
    LIGHTING_PRESET,
    get_plotly_colorscale,
)
from waytogocoop.config import K_F_TSS


def _resolve_colorscale(name: str) -> list[list[object]]:
    """Accept either a shared-LUT name or a Plotly built-in name.

    Shared-LUT names ('viridis', 'inferno', 'coolwarm', 'plasma', and their
    ``_r`` reversals) are resolved via the cross-stack JSON. 'RdBu_r' and
    'RdBu' are aliases for 'coolwarm' and 'coolwarm_r' (see below). Anything
    else passes through unchanged so Plotly's built-in scales ('Hot',
    'Blues', ...) still work for figure types we haven't unified yet.
    """
    key = name.lower()
    base = key[:-2] if key.endswith("_r") else key
    if base in ("viridis", "inferno", "coolwarm", "plasma"):
        return get_plotly_colorscale(key)
    # "RdBu_r" is the conventional diverging map with low = blue and
    # high = red (Plotly's and matplotlib's RdBu_r), so it resolves to the
    # shared coolwarm LUT, which runs the same way and is the map the Rust app
    # uses for signed fields: both stacks then show high values in red.
    # "RdBu" is its reversal (low = red, high = blue).
    if key in ("rdbu_r", "rdbu"):
        return get_plotly_colorscale("coolwarm" if key.endswith("_r") else "coolwarm_r")
    return name  # type: ignore[return-value]  # Plotly accepts str or list


# ---------------------------------------------------------------------------
# Theme + shared layout helpers
# ---------------------------------------------------------------------------


def _template(dark: bool = True) -> str:
    return "plotly_dark" if dark else "plotly_white"


def _line_colors(dark: bool = True) -> tuple[str, str, str]:
    if dark:
        return "#5dade2", "#ec7063", "#888"
    return "steelblue", "firebrick", "gray"


def _marker_color(dark: bool = True) -> str:
    return "white" if dark else "black"


def _equal_aspect_axes() -> dict:
    """Strict 1:1 pixel aspect for real-space heatmaps."""
    return dict(scaleanchor="x", scaleratio=1, constrain="domain")


def _hover_scalar(x_unit: str, y_unit: str, value_label: str, value_fmt: str = ".3g") -> str:
    return (
        f"x: %{{x:.2f}} {x_unit}"
        f"<br>y: %{{y:.2f}} {y_unit}"
        f"<br>{value_label}: %{{z:{value_fmt}}}"
        "<extra></extra>"
    )


# ---------------------------------------------------------------------------
# Real-space 2D heatmaps
# ---------------------------------------------------------------------------


def create_moire_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    pattern: np.ndarray,
    title: str = "Moire Pattern",
    dark: bool = True,
) -> go.Figure:
    """Heatmap of the real-space moire pattern."""
    fig = go.Figure(
        data=go.Heatmap(
            z=pattern,
            x=x,
            y=y,
            colorscale=_resolve_colorscale("viridis"),
            colorbar=dict(title="Intensity"),
            hovertemplate=_hover_scalar("Å", "Å", "intensity"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x (Å)",
        yaxis_title="y (Å)",
        yaxis=_equal_aspect_axes(),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_gap_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    gap_field: np.ndarray,
    title: str = "Gap Modulation",
    dark: bool = True,
) -> go.Figure:
    """Heatmap of the spatially varying superconducting gap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=gap_field,
            x=x,
            y=y,
            colorscale=_resolve_colorscale("RdBu_r"),
            colorbar=dict(title="Δ (meV)"),
            hovertemplate=_hover_scalar("Å", "Å", "Δ (meV)"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x (Å)",
        yaxis_title="y (Å)",
        yaxis=_equal_aspect_axes(),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_fft_heatmap(
    kx: np.ndarray,
    ky: np.ndarray,
    power_spectrum: np.ndarray,
    title: str = "FFT Power Spectrum",
    dark: bool = True,
) -> go.Figure:
    """Heatmap of the log-scaled FFT power spectrum."""
    fig = go.Figure(
        data=go.Heatmap(
            z=power_spectrum,
            x=kx,
            y=ky,
            colorscale=_resolve_colorscale("inferno"),
            colorbar=dict(title="log₁₀(|F|²)"),
            hovertemplate=(
                "kx: %{x:.3f} 1/Å"
                "<br>ky: %{y:.3f} 1/Å"
                "<br>log₁₀(|F|²): %{z:.2f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="kx (1/Å)",
        yaxis_title="ky (1/Å)",
        yaxis=_equal_aspect_axes(),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_2d_contour(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    colorscale: str = "viridis",
    z_label: str = "Intensity",
    dark: bool = True,
) -> go.Figure:
    """2D contour/topographic map projection of a real-space field."""
    fig = go.Figure(
        data=go.Contour(
            z=z,
            x=x,
            y=y,
            colorscale=_resolve_colorscale(colorscale),
            colorbar=dict(title=z_label),
            contours=dict(showlabels=True, labelfont=dict(size=10)),
            hovertemplate=_hover_scalar("Å", "Å", z_label),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x (Å)",
        yaxis_title="y (Å)",
        yaxis=_equal_aspect_axes(),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_pseudo_field_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    b_ps: np.ndarray,
    title: str = "Pseudo-Magnetic Field (SPECULATIVE)",
    dark: bool = True,
) -> go.Figure:
    """Heatmap of the strain-induced pseudo-magnetic field (K valley)."""
    fig = go.Figure(
        data=go.Heatmap(
            z=b_ps,
            x=x,
            y=y,
            colorscale=_resolve_colorscale("RdBu_r"),
            zmid=0,
            colorbar=dict(title="B_ps (T)"),
            hovertemplate=_hover_scalar("Å", "Å", "B_ps (T)"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x (Å)",
        yaxis_title="y (Å)",
        yaxis=_equal_aspect_axes(),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


# ---------------------------------------------------------------------------
# Line plots
# ---------------------------------------------------------------------------


def create_sweep_plot(
    param_values: np.ndarray,
    periods: np.ndarray,
    amplitudes: np.ndarray,
    param_name: str = "Parameter",
    dark: bool = True,
) -> go.Figure:
    """Dual-axis line plot for parameter sweeps (moire period + CPDM amplitude)."""
    primary, secondary, _muted = _line_colors(dark)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=param_values,
            y=periods,
            name="Moire period (Å)",
            mode="lines+markers",
            line=dict(color=primary),
            hovertemplate=(
                f"{param_name}: %{{x:.3g}}"
                "<br>Moire period: %{y:.2f} Å"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=param_values,
            y=amplitudes,
            name="CPDM amplitude",
            mode="lines+markers",
            line=dict(color=secondary),
            hovertemplate=(
                f"{param_name}: %{{x:.3g}}"
                "<br>CPDM amplitude: %{y:.3g}"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Parameter Sweep: {param_name}",
        xaxis_title=param_name,
        margin=dict(l=60, r=60, t=50, b=50),
        template=_template(dark),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Moire period (Å)", secondary_y=False)
    fig.update_yaxes(title_text="CPDM amplitude", secondary_y=True)
    return fig


def create_flat_band_plot(
    theta_values: np.ndarray,
    velocity_ratios: np.ndarray,
    delta_values: np.ndarray,
    theta_current: float,
    title: str = "Flat Band vs Twist Angle",
    dark: bool = True,
) -> go.Figure:
    """Dual-axis plot: renormalized Dirac velocity and flat-band gap vs twist."""
    primary, secondary, muted = _line_colors(dark)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=theta_values,
            y=velocity_ratios,
            name="v*/v_F",
            mode="lines",
            line=dict(color=primary),
            hovertemplate=(
                "θ: %{x:.3f}°"
                "<br>v*/v_F: %{y:.3g}"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=theta_values,
            y=delta_values,
            name="Delta_max (meV) (SPECULATIVE)",
            mode="lines",
            line=dict(color=secondary),
            hovertemplate=(
                "θ: %{x:.3f}°"
                "<br>Delta_max: %{y:.3g} meV"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.add_vline(
        x=theta_current,
        line=dict(dash="dash", color=muted),
        annotation_text=f"theta = {theta_current:.2f} deg",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Twist angle (deg)",
        margin=dict(l=60, r=60, t=50, b=50),
        template=_template(dark),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="v*/v_F", secondary_y=False)
    fig.update_yaxes(title_text="Delta_max (meV)", secondary_y=True)
    return fig


def create_band_structure_plot(
    k_distances: np.ndarray,
    energies_mev: np.ndarray,
    tick_positions: list[float],
    tick_labels: list[str],
    flat_bandwidth_mev: float,
    title: str = "Moire Band Structure (BM model)",
    dark: bool = True,
) -> go.Figure:
    """Band structure along the moire high-symmetry path, flat bands highlighted.

    One line per band; the two middle bands (the flat bands) are drawn in an
    accent color with a heavier line. Ticks/gridlines sit at the
    high-symmetry-point path distances, and an annotation reports the
    flat-band width W.
    """
    _primary, secondary, muted = _line_colors(dark)
    n_bands = energies_mev.shape[1]
    flat_indices = {n_bands // 2 - 1, n_bands // 2}

    fig = go.Figure()
    for band in range(n_bands):
        if band in flat_indices:
            trace_kwargs = dict(
                line=dict(color=secondary, width=3),
                hovertemplate="k: %{x:.4f} 1/Å<br>E: %{y:.2f} meV<extra>flat band</extra>",
            )
        else:
            trace_kwargs = dict(line=dict(color=muted, width=1), hoverinfo="skip")
        fig.add_trace(
            go.Scatter(
                x=k_distances,
                y=energies_mev[:, band],
                mode="lines",
                showlegend=False,
                **trace_kwargs,
            )
        )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        text=f"W = {flat_bandwidth_mev:.1f} meV",
        showarrow=False,
    )
    fig.update_layout(
        title=title,
        xaxis=dict(
            tickvals=tick_positions,
            ticktext=tick_labels,
            showgrid=True,  # vertical gridlines land exactly on the ticks
            range=[float(k_distances[0]), float(k_distances[-1])],
        ),
        yaxis_title="E (meV)",
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_dos_plot(
    energies_mev: np.ndarray,
    dos: np.ndarray,
    title: str = "Density of States (BM model)",
    dark: bool = True,
) -> go.Figure:
    """Density of states versus energy with a shaded fill and E = 0 marker."""
    primary, _secondary, muted = _line_colors(dark)
    fig = go.Figure(
        data=go.Scatter(
            x=energies_mev,
            y=dos,
            mode="lines",
            line=dict(color=primary, width=2),
            fill="tozeroy",
            name="DOS",
            hovertemplate="E: %{x:.1f} meV<br>DOS: %{y:.3g}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(dash="dash", color=muted))
    fig.update_layout(
        title=title,
        xaxis_title="E (meV)",
        yaxis_title="DOS (states / meV per moire cell, a.u.)",
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 3D views
# ---------------------------------------------------------------------------


def _camera_buttons(pos_x: float = 0.02, pos_y: float = 0.98) -> dict:
    return dict(
        type="buttons",
        direction="right",
        x=pos_x,
        y=pos_y,
        xanchor="left",
        yanchor="top",
        showactive=False,
        pad=dict(t=2, r=2),
        buttons=[
            dict(label="Iso",  method="relayout", args=[{"scene.camera": CAMERA_PRESETS["iso"]}]),
            dict(label="Top",  method="relayout", args=[{"scene.camera": CAMERA_PRESETS["top"]}]),
            dict(label="Side", method="relayout", args=[{"scene.camera": CAMERA_PRESETS["side"]}]),
        ],
    )


def _scene_dict(
    *,
    z_label: str,
    aspect_z: float,
    annotations: list[dict] | None = None,
) -> dict:
    scene = dict(
        xaxis_title="x (Å)",
        yaxis_title="y (Å)",
        zaxis_title=z_label,
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=aspect_z),
        camera=CAMERA_PRESETS["iso"],
    )
    if annotations:
        scene["annotations"] = annotations
    return scene


def _apply_3d_scene(
    fig: go.Figure,
    *,
    dark: bool,
    title: str,
    z_label: str = "Intensity",
    aspect_z: float = 0.4,
    annotations: list[dict] | None = None,
    camera_buttons: bool = True,
) -> go.Figure:
    """Apply the shared 3D scene layout used by every 3D builder."""
    fig.update_layout(
        title=title,
        scene=_scene_dict(z_label=z_label, aspect_z=aspect_z, annotations=annotations),
        margin=dict(l=20, r=20, t=50, b=20),
        template=_template(dark),
        updatemenus=[_camera_buttons()] if camera_buttons else None,
    )
    return fig


def _apply_surface_lighting(trace: go.Surface) -> go.Surface:
    trace.update(lighting=LIGHTING_PRESET, lightposition=LIGHT_POSITION)
    return trace


def create_3d_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    colorscale: str = "viridis",
    z_label: str = "Intensity",
    dark: bool = True,
    high_density: bool = False,
    surfacecolor: np.ndarray | None = None,
    color_label: str | None = None,
) -> go.Figure:
    """3D rotatable surface.

    When ``high_density`` is True the surface is drawn as a triangulated
    :class:`plotly.graph_objects.Mesh3d` with per-vertex intensity and no
    hover-interpolation — that path stays responsive at grid sizes >= 150
    where ``go.Surface`` would stutter. Otherwise we use ``go.Surface`` with
    the shared PBR-ish lighting preset.

    When ``surfacecolor`` is given, the surface is colored by that field
    instead of height; ``color_label`` titles the colorbar (falls back to
    ``z_label``).
    """
    if high_density:
        return create_surface_mesh3d(
            x, y, z, title=title, colorscale=colorscale, z_label=z_label, dark=dark
        )
    cbar_label = color_label or z_label
    if surfacecolor is not None:
        value_line = f"<br>{cbar_label}: %{{surfacecolor:.3g}}"
    else:
        value_line = f"<br>{z_label}: %{{z:.3g}}"
    surface_kwargs = dict(
        z=z,
        x=x,
        y=y,
        colorscale=_resolve_colorscale(colorscale),
        colorbar=dict(title=cbar_label),
        hovertemplate=(
            "x: %{x:.2f} Å"
            "<br>y: %{y:.2f} Å"
            f"{value_line}"
            "<extra></extra>"
        ),
    )
    if surfacecolor is not None:
        surface_kwargs["surfacecolor"] = surfacecolor
    surface = _apply_surface_lighting(go.Surface(**surface_kwargs))
    fig = go.Figure(data=surface)
    return _apply_3d_scene(fig, dark=dark, title=title, z_label=z_label, aspect_z=0.4)


def create_surface_mesh3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str = "3D Surface (WebGL mesh)",
    colorscale: str = "viridis",
    z_label: str = "Intensity",
    dark: bool = True,
) -> go.Figure:
    """WebGL-friendly triangulated mesh of a regular-grid heightmap.

    ``z`` has shape (ny, nx). The function emits (ny-1)*(nx-1)*2 triangles
    with per-vertex intensity so Plotly's Mesh3d renderer interpolates the
    colormap smoothly without needing the full Surface-trace hover pipeline.
    """
    z_arr = np.asarray(z)
    ny, nx = z_arr.shape
    X, Y = np.meshgrid(x, y)  # both (ny, nx)
    verts_x = X.ravel()
    verts_y = Y.ravel()
    verts_z = z_arr.ravel()

    # Build triangle indices: for each cell (i,j) emit two triangles.
    idx = np.arange(ny * nx).reshape(ny, nx)
    tl = idx[:-1, :-1].ravel()
    tr = idx[:-1, 1:].ravel()
    bl = idx[1:, :-1].ravel()
    br = idx[1:, 1:].ravel()
    i_tris = np.concatenate([tl, tl])
    j_tris = np.concatenate([tr, br])
    k_tris = np.concatenate([br, bl])

    mesh = go.Mesh3d(
        x=verts_x,
        y=verts_y,
        z=verts_z,
        i=i_tris,
        j=j_tris,
        k=k_tris,
        intensity=verts_z,
        colorscale=_resolve_colorscale(colorscale),
        colorbar=dict(title=z_label),
        flatshading=False,
        hoverinfo="skip",
        lighting=LIGHTING_PRESET,
        lightposition=LIGHT_POSITION,
    )
    fig = go.Figure(data=mesh)
    return _apply_3d_scene(fig, dark=dark, title=title, z_label=z_label, aspect_z=0.4)


def _clip_z_layers(
    z: np.ndarray, values: np.ndarray, clip_z: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the z-layers above ``clip_z`` from a ``(nz, ny, nx)`` field.

    A clip plane has to remove layers rather than NaN-mask them: plotly.js
    isosurface and volume traces draw nothing at all once the value array
    holds NaN. A 3D trace needs at least two layers, so a plane below the
    second-lowest layer keeps the bottom two.
    """
    z = np.asarray(z, dtype=float)
    values = np.asarray(values, dtype=float)
    if clip_z is None:
        return z, values
    keep = z <= clip_z
    if np.count_nonzero(keep) < 2:
        keep[np.argsort(z)[:2]] = True
    return z[keep], values[keep]


def create_3d_isosurface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    title: str = "3D Cooper Surface",
    iso_min: float | None = None,
    iso_max: float | None = None,
    surface_count: int = 5,
    colorscale: str = "RdBu_r",
    dark: bool = True,
    annotations: list[dict] | None = None,
    clip_z: float | None = None,
) -> go.Figure:
    """3D isosurface of Δ(x,y,z).

    ``values`` has shape ``(nz, ny, nx)``, the layout returned by
    :func:`waytogocoop.computation.topological.gap_3d`. ``surface_count`` is
    caller-driven so an animation loop in ``pages/proximity_3d.py`` can sweep
    it. ``iso_min`` and ``iso_max`` default to the 20th and 80th percentiles
    of the whole field.

    ``clip_z`` cuts the view at a z-plane by dropping every z-layer above it
    (see :func:`_clip_z_layers`). The iso levels still come from the whole
    field, so moving the plane cuts the same surfaces instead of re-leveling
    them.
    """
    values = np.asarray(values, dtype=float)
    if iso_min is None:
        iso_min = float(np.percentile(values, 20))
    if iso_max is None:
        iso_max = float(np.percentile(values, 80))

    z, values = _clip_z_layers(z, values, clip_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    vals = values.transpose(2, 1, 0)

    fig = go.Figure(
        data=go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=vals.flatten(),
            isomin=iso_min,
            isomax=iso_max,
            surface_count=surface_count,
            colorscale=_resolve_colorscale(colorscale),
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(title="Δ (meV)"),
            hovertemplate=(
                "x: %{x:.2f} Å"
                "<br>y: %{y:.2f} Å"
                "<br>z: %{z:.2f} Å"
                "<br>Δ: %{value:.3g} meV"
                "<extra></extra>"
            ),
        )
    )
    return _apply_3d_scene(
        fig, dark=dark, title=title, z_label="z (Å)", aspect_z=0.5, annotations=annotations
    )


def create_3d_volume(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    title: str = "3D Cooper-Pair Volumetric Density",
    opacity: float = 0.25,
    surface_count: int = 20,
    colorscale: str = "RdBu_r",
    dark: bool = True,
    annotations: list[dict] | None = None,
    clip_z: float | None = None,
) -> go.Figure:
    """True volumetric rendering of Δ(x,y,z) via :class:`go.Volume`.

    ``values`` has shape ``(nz, ny, nx)``. The opacity ramp is tuned so
    low-gap regions stay partially transparent and strong-gap bulk lights up,
    while still showing enough mid-range contrast that the moire modulation
    remains visible through the slab. The tuple
    ``[[0, 0], [0.5, 0.25], [1, 0.85]]`` works for both signed
    (coolwarm-like) gap fields and unsigned densities.

    ``clip_z`` cuts the view at a z-plane by dropping every z-layer above it
    (see :func:`_clip_z_layers`). ``isomin`` and ``isomax`` span the whole
    field, so clipping removes layers without re-stretching the colors and
    opacities of the rest.
    """
    values = np.asarray(values, dtype=float)
    isomin, isomax = float(np.nanmin(values)), float(np.nanmax(values))

    z, values = _clip_z_layers(z, values, clip_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    vals = values.transpose(2, 1, 0)

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=vals.flatten(),
            isomin=isomin,
            isomax=isomax,
            opacity=opacity,
            surface_count=surface_count,
            colorscale=_resolve_colorscale(colorscale),
            opacityscale=[[0.0, 0.0], [0.5, 0.25], [1.0, 0.85]],
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(title="Δ (meV)"),
            hovertemplate=(
                "x: %{x:.2f} Å"
                "<br>y: %{y:.2f} Å"
                "<br>z: %{z:.2f} Å"
                "<br>Δ: %{value:.3g} meV"
                "<extra></extra>"
            ),
        )
    )
    return _apply_3d_scene(
        fig, dark=dark, title=title, z_label="z (Å)", aspect_z=0.5, annotations=annotations
    )


def create_3d_cone_field(
    x: np.ndarray,
    y: np.ndarray,
    z_plane: float,
    jx: np.ndarray,
    jy: np.ndarray,
    jz: np.ndarray | None = None,
    base_surface: np.ndarray | None = None,
    title: str = "Screening Currents (3D)",
    skip: int = 8,
    dark: bool = True,
) -> go.Figure:
    """Native ``go.Cone`` quiver lifted to ``z_plane`` with optional backdrop.

    ``jx``, ``jy`` are 2D (ny, nx); ``jz`` is optional and defaults to zero so
    the cones lie tangent to the superconductor surface. ``base_surface`` is a
    translucent :class:`go.Surface` rendered at ``z_plane`` as a gap-field
    backdrop so the current flow is readable against the underlying gap
    modulation.
    """
    xs, ys = x[::skip], y[::skip]
    jxs = jx[::skip, ::skip]
    jys = jy[::skip, ::skip]
    jzs = np.zeros_like(jxs) if jz is None else jz[::skip, ::skip]

    Xs, Ys = np.meshgrid(xs, ys)
    Zs = np.full_like(Xs, float(z_plane))

    traces: list[go.Scatter3d | go.Surface | go.Cone] = []
    if base_surface is not None:
        traces.append(
            _apply_surface_lighting(
                go.Surface(
                    x=x,
                    y=y,
                    z=np.full((y.size, x.size), float(z_plane)),
                    surfacecolor=base_surface,
                    colorscale=_resolve_colorscale("RdBu_r"),
                    showscale=True,
                    colorbar=dict(title="Δ (meV)"),
                    opacity=0.6,
                    hoverinfo="skip",
                )
            )
        )

    traces.append(
        go.Cone(
            x=Xs.ravel(),
            y=Ys.ravel(),
            z=Zs.ravel(),
            u=jxs.ravel(),
            v=jys.ravel(),
            w=jzs.ravel(),
            sizemode="scaled",
            sizeref=0.8,
            anchor="tail",
            colorscale=_resolve_colorscale("viridis"),
            colorbar=dict(title="|J|", x=1.07),
            showscale=True,
            hovertemplate=(
                "x: %{x:.1f} Å"
                "<br>y: %{y:.1f} Å"
                "<br>|J|: %{u:.2g}"
                "<extra></extra>"
            ),
        )
    )

    fig = go.Figure(data=traces)
    return _apply_3d_scene(fig, dark=dark, title=title, z_label="z (Å)", aspect_z=0.4)


def create_3d_majorana_isosurface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    density_3d: np.ndarray,
    vortex_positions: np.ndarray,
    title: str = "Majorana ZM 3D Density (SPECULATIVE)",
    iso_min: float | None = None,
    iso_max: float | None = None,
    dark: bool = True,
) -> go.Figure:
    """3D isosurface of the Majorana zero-mode probability density.

    ``density_3d`` is (nz, ny, nx). Vortex core positions are overlaid as
    cyan cross markers at the z=0 interface plane.
    """
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    peak = float(np.nanmax(density_3d)) or 1.0
    if iso_min is None:
        iso_min = 0.08 * peak
    if iso_max is None:
        iso_max = 0.6 * peak

    vals = np.transpose(density_3d, (2, 1, 0)).astype(float)

    iso = go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=vals.flatten(),
        isomin=iso_min,
        isomax=iso_max,
        surface_count=4,
        colorscale=_resolve_colorscale("inferno"),
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title="|ψ|²"),
        opacity=0.7,
        hovertemplate=(
            "x: %{x:.1f} Å"
            "<br>y: %{y:.1f} Å"
            "<br>z: %{z:.1f} Å"
            "<br>|ψ|²: %{value:.3g}"
            "<extra></extra>"
        ),
    )

    traces: list[go.Isosurface | go.Scatter3d] = [iso]
    if vortex_positions is not None and vortex_positions.size > 0:
        traces.append(
            go.Scatter3d(
                x=vortex_positions[:, 0],
                y=vortex_positions[:, 1],
                z=np.zeros(vortex_positions.shape[0]),
                mode="markers",
                marker=dict(size=6, color="cyan", symbol="x"),
                name="Vortex cores",
                hovertemplate=(
                    "Vortex core<br>x: %{x:.1f} Å<br>y: %{y:.1f} Å<extra></extra>"
                ),
            )
        )

    fig = go.Figure(data=traces)
    return _apply_3d_scene(fig, dark=dark, title=title, z_label="z (Å)", aspect_z=0.5)


def create_z_decay_profile(
    z: np.ndarray,
    profile: np.ndarray,
    title: str = "Proximity Decay Profile",
    dark: bool = True,
) -> go.Figure:
    """Line plot of the proximity decay f(z) with the interface marked."""
    primary, _secondary, muted = _line_colors(dark)
    fig = go.Figure(
        data=go.Scatter(
            x=z,
            y=profile,
            mode="lines",
            line=dict(color=primary, width=2),
            name="f(z)",
            hovertemplate="z: %{x:.2f} Å<br>f(z): %{y:.3g}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(dash="dash", color=muted), annotation_text="Interface")
    fig.update_layout(
        title=title,
        xaxis_title="z (Å)",
        yaxis_title="f(z)",
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Magnetic / topological figure constructors
# ---------------------------------------------------------------------------


def create_vortex_overlay_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    gap_field: np.ndarray,
    vortex_positions: np.ndarray,
    title: str = "Gap + Vortex Lattice",
    dark: bool = True,
) -> go.Figure:
    """Gap heatmap with vortex core positions overlaid as markers."""
    marker_col = _marker_color(dark)
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=gap_field, x=x, y=y,
            colorscale=_resolve_colorscale("RdBu_r"),
            colorbar=dict(title="Δ (meV)"),
            hovertemplate=_hover_scalar("Å", "Å", "Δ (meV)"),
        )
    )
    if vortex_positions.size > 0:
        fig.add_trace(
            go.Scatter(
                x=vortex_positions[:, 0],
                y=vortex_positions[:, 1],
                mode="markers",
                marker=dict(size=6, color=marker_col, symbol="x"),
                name="Vortex cores",
                hovertemplate="Vortex core<br>x: %{x:.1f} Å<br>y: %{y:.1f} Å<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="x (Å)",
        yaxis_title="y (Å)",
        yaxis=_equal_aspect_axes(),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_majorana_density_map(
    x: np.ndarray,
    y: np.ndarray,
    density: np.ndarray,
    vortex_positions: np.ndarray,
    title: str = "Majorana ZM Density (SPECULATIVE)",
    dark: bool = True,
) -> go.Figure:
    """Heatmap of Majorana zero-mode probability density with vortex cores."""
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=density, x=x, y=y,
            colorscale=_resolve_colorscale("inferno"),
            colorbar=dict(title="|ψ|²"),
            hovertemplate=_hover_scalar("Å", "Å", "|ψ|²"),
        )
    )
    if vortex_positions.size > 0:
        fig.add_trace(
            go.Scatter(
                x=vortex_positions[:, 0],
                y=vortex_positions[:, 1],
                mode="markers",
                marker=dict(size=8, color="cyan", symbol="circle-open", line=dict(width=2)),
                name="Vortex cores",
                hovertemplate="Vortex core<br>x: %{x:.1f} Å<br>y: %{y:.1f} Å<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="x (Å)",
        yaxis_title="y (Å)",
        yaxis=_equal_aspect_axes(),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_quiver_field(
    x: np.ndarray,
    y: np.ndarray,
    jx: np.ndarray,
    jy: np.ndarray,
    base_field: np.ndarray,
    title: str = "Screening Currents",
    skip: int = 8,
    dark: bool = True,
) -> go.Figure:
    """Background gap heatmap with arrow overlay for current density."""
    arrow_color = "white" if dark else "black"
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=base_field, x=x, y=y,
            colorscale=_resolve_colorscale("RdBu_r"),
            colorbar=dict(title="Δ (meV)"),
            opacity=0.6,
            hovertemplate=_hover_scalar("Å", "Å", "Δ (meV)"),
        )
    )

    xs, ys = x[::skip], y[::skip]
    jxs, jys = jx[::skip, ::skip], jy[::skip, ::skip]

    scale = (x[-1] - x[0]) / max(len(xs), 1) * 0.8
    mag = np.sqrt(jxs**2 + jys**2)
    mag_max = mag.max() if mag.size > 0 else 1.0
    if mag_max > 1e-30:
        jxs = jxs / mag_max * scale
        jys = jys / mag_max * scale

    Xs, Ys = np.meshgrid(xs, ys)
    for i in range(Xs.shape[0]):
        for j in range(Xs.shape[1]):
            x0, y0 = Xs[i, j], Ys[i, j]
            dx, dy = jxs[i, j], jys[i, j]
            if abs(dx) + abs(dy) < 1e-10:
                continue
            fig.add_annotation(
                x=x0 + dx, y=y0 + dy,
                ax=x0, ay=y0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1, arrowwidth=1.5,
                arrowcolor=arrow_color,
            )

    fig.update_layout(
        title=title,
        xaxis_title="x (Å)",
        yaxis_title="y (Å)",
        yaxis=_equal_aspect_axes(),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_susceptibility_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    chi: np.ndarray,
    title: str = "Local Susceptibility (SPECULATIVE)",
    dark: bool = True,
) -> go.Figure:
    """Heatmap of local magnetic susceptibility."""
    fig = go.Figure(
        data=go.Heatmap(
            z=chi, x=x, y=y,
            colorscale=_resolve_colorscale("plasma"),
            colorbar=dict(title="χ (a.u.)"),
            hovertemplate=_hover_scalar("Å", "Å", "χ"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x (Å)",
        yaxis_title="y (Å)",
        yaxis=_equal_aspect_axes(),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_phase_colormap(
    param_x: np.ndarray,
    param_y: np.ndarray,
    phase_index: np.ndarray,
    title: str = "Topological Phase Diagram (SPECULATIVE)",
    x_label: str = "B (Tesla)",
    y_label: str = "Δ (meV)",
    dark: bool = True,
) -> go.Figure:
    """2D colormap of topological phase index with boundary overlay.

    The boundary is a lines-only contour at phase index 0.5, so the heatmap's
    two phase colors stay visible underneath it.
    """
    boundary_color = "white" if dark else "black"
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=phase_index,
            x=param_x,
            y=param_y,
            colorscale=[[0, "#5dade2" if dark else "steelblue"],
                        [1, "#ec7063" if dark else "firebrick"]],
            colorbar=dict(title="Phase", tickvals=[0, 1], ticktext=["Trivial", "Topological"]),
            zmin=0,
            zmax=1,
            hovertemplate=(
                f"{x_label}: %{{x:.3g}}"
                f"<br>{y_label}: %{{y:.3g}}"
                "<br>Phase: %{z}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Contour(
            z=phase_index.astype(float),
            x=param_x,
            y=param_y,
            # Without coloring="lines" the contour fills both sides of the
            # level with the template's default colorscale and hides the
            # heatmap. In "lines" mode Plotly colors the line from the
            # colorscale and ignores line.color, hence the one-color scale.
            contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
            colorscale=[[0.0, boundary_color], [1.0, boundary_color]],
            line=dict(color=boundary_color, width=2),
            showscale=False,
            name="Phase boundary",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_commensuration_sweep(
    B_values: np.ndarray,
    a_v_values: np.ndarray,
    moire_period: float,
    title: str = "Vortex-Moire Commensuration",
    dark: bool = True,
) -> go.Figure:
    """Dual-axis plot: a_v(B) and a_v/L_m ratio vs B."""
    primary, secondary, muted = _line_colors(dark)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=B_values, y=a_v_values,
            name="a_v (Å)",
            mode="lines",
            line=dict(color=primary),
            hovertemplate="B: %{x:.2f} T<br>a_v: %{y:.2f} Å<extra></extra>",
        ),
        secondary_y=False,
    )

    ratio = a_v_values / moire_period if moire_period > 0 else np.zeros_like(a_v_values)
    fig.add_trace(
        go.Scatter(
            x=B_values, y=ratio,
            name="a_v / L_m",
            mode="lines",
            line=dict(color=secondary),
            hovertemplate="B: %{x:.2f} T<br>a_v/L_m: %{y:.3g}<extra></extra>",
        ),
        secondary_y=True,
    )

    for n in [1, 2, 3, 4]:
        fig.add_hline(
            y=n, secondary_y=True,
            line=dict(dash="dot", color=muted, width=1),
            annotation_text=f"n={n}",
        )

    fig.update_layout(
        title=title,
        xaxis_title="B (Tesla)",
        margin=dict(l=60, r=60, t=50, b=50),
        template=_template(dark),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Vortex period (Å)", secondary_y=False)
    fig.update_yaxes(title_text="a_v / L_m", secondary_y=True)
    return fig


def create_field_cpdm_sweep(
    B_values: np.ndarray,
    amplitudes: np.ndarray,
    Bz_current: float,
    Bc2: float,
    title: str = "Field-Tunable CPDM (SPECULATIVE)",
    dark: bool = True,
) -> go.Figure:
    """Line plot of the speculative field-tunable CPDM amplitude A_CPDM(B)."""
    primary, _secondary, muted = _line_colors(dark)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=B_values, y=amplitudes,
            name="A_CPDM",
            mode="lines",
            line=dict(color=primary),
            hovertemplate="B: %{x:.2f} T<br>A_CPDM: %{y:.3g}<extra></extra>",
        )
    )
    fig.add_vline(
        x=Bz_current,
        line=dict(dash="dash", color=muted, width=1),
        annotation_text="Bz",
    )
    fig.add_vline(
        x=Bc2,
        line=dict(dash="dot", color=muted, width=1),
        annotation_text="Bc2",
    )
    fig.update_layout(
        title=title,
        xaxis_title="B (Tesla)",
        yaxis_title="CPDM amplitude",
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
        hovermode="x unified",
    )
    return fig


def create_pinning_energy_sweep(
    B_values: np.ndarray,
    pinning: np.ndarray,
    Bz_current: float,
    title: str = "Commensuration Pinning Energy (SPECULATIVE)",
    dark: bool = True,
) -> go.Figure:
    """Line plot of the speculative vortex-moire commensuration pinning energy."""
    primary, _secondary, muted = _line_colors(dark)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=B_values, y=pinning,
            name="E_pin",
            mode="lines",
            line=dict(color=primary),
            hovertemplate="B: %{x:.2f} T<br>E_pin: %{y:.3g}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0.0,
        line=dict(dash="dot", color=muted, width=1),
    )
    fig.add_vline(
        x=Bz_current,
        line=dict(dash="dash", color=muted, width=1),
        annotation_text="Bz",
    )
    fig.update_layout(
        title=title,
        xaxis_title="B (Tesla)",
        yaxis_title="E_pin (arb. units)",
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
        hovermode="x unified",
    )
    return fig


def create_topological_extras_sweep(
    B_values: np.ndarray,
    chern_values: np.ndarray,
    polarization_values: np.ndarray,
    title: str = "Chern Number & Magnetoelectric Polarization (SPECULATIVE)",
    dark: bool = True,
) -> go.Figure:
    """Dual-axis plot: Chern number C(B) (step) and surface polarization P(B)."""
    primary, secondary, muted = _line_colors(dark)
    chern_values = np.asarray(chern_values, dtype=float)
    B_values = np.asarray(B_values, dtype=float)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=B_values, y=chern_values,
            name="Chern C",
            mode="lines",
            line=dict(color=primary, shape="hv"),
            hovertemplate="B: %{x:.2f} T<br>C: %{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=B_values, y=polarization_values,
            name="P (C/m²)",
            mode="lines",
            line=dict(color=secondary),
            hovertemplate="B: %{x:.2f} T<br>P: %{y:.3g} C/m²<extra></extra>",
        ),
        secondary_y=True,
    )

    # Mark the field where the Chern number flips sign (topological onset).
    neg = np.where(chern_values < 0)[0]
    pos = np.where(chern_values > 0)[0]
    if len(neg) and len(pos):
        B_star = B_values[pos[0]]
        fig.add_vline(
            x=B_star,
            line=dict(dash="dash", color=muted, width=1),
            annotation_text="C flips",
        )

    fig.update_layout(
        title=title,
        xaxis_title="B (Tesla)",
        margin=dict(l=60, r=60, t=50, b=50),
        template=_template(dark),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Chern number C", secondary_y=False)
    fig.update_yaxes(title_text="Polarization P (C/m²)", secondary_y=True)
    return fig


def create_dirac_cone_plot(
    k_values: np.ndarray,
    E_upper: np.ndarray,
    induced_gap_meV: float,
    title: str = "TI Surface Dirac Cone",
    dark: bool = True,
) -> go.Figure:
    """Plot the ±E(k) Dirac branches with the induced proximity gap marked."""
    primary, secondary, muted = _line_colors(dark)
    E_upper = np.asarray(E_upper, dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k_values, y=E_upper,
            name="+E(k)",
            mode="lines",
            line=dict(color=primary),
            hovertemplate="k: %{x:.3f} 1/Å<br>E: %{y:.2f} meV<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=k_values, y=-E_upper,
            name="−E(k)",
            mode="lines",
            line=dict(color=secondary),
            hovertemplate="k: %{x:.3f} 1/Å<br>E: %{y:.2f} meV<extra></extra>",
        )
    )
    for sign in (1.0, -1.0):
        fig.add_hline(
            y=sign * induced_gap_meV,
            line=dict(dash="dash", color=muted, width=1),
            annotation_text="induced Δ" if sign > 0 else None,
        )
    for sign in (1.0, -1.0):
        fig.add_vline(
            x=sign * K_F_TSS,
            line=dict(dash="dot", color=muted, width=1),
            annotation_text="k_F" if sign > 0 else None,
        )
    fig.update_layout(
        title=title,
        xaxis_title="k (1/Å)",
        yaxis_title="E (meV)",
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


# ---------------------------------------------------------------------------
# Pair-density-wave figures (SPECULATIVE)
# ---------------------------------------------------------------------------

# Composite-CDW family colours: the first three slots of the validated
# categorical palette (all-pairs CVD-safe in both themes), stepped per theme.
_PDW_FAMILY_LABELS: dict[str, str] = {
    "p": "p (PDW × uniform SC)",
    "q": "q = 2p (PDW alone)",
    "h": "h (harmonic)",
}

# Regime names in PDW_REGIMES order: normal, pdw_only, uniform_only, coexistence.
_PDW_REGIME_LABELS: tuple[str, ...] = (
    "Normal",
    "PDW only (Cheshire cat)",
    "Uniform SC only",
    "PDW + uniform SC",
)


def _pdw_family_colors(dark: bool = True) -> dict[str, str]:
    if dark:
        return {"q": "#3987e5", "p": "#d95926", "h": "#199e70"}
    return {"q": "#2a78d6", "p": "#eb6834", "h": "#1baf7a"}


def _pdw_regime_colors(dark: bool = True) -> tuple[str, ...]:
    """Neutral grays where no CDW shows; the q / p hues where they do."""
    family = _pdw_family_colors(dark)
    if dark:
        return ("#2b2b29", family["q"], "#6b6a65", family["p"])
    return ("#e8e7e3", family["q"], "#a3a29d", family["p"])


def create_signed_field_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    title: str,
    z_label: str,
    dark: bool = True,
) -> go.Figure:
    """Real-space heatmap of a signed field on the shared coolwarm LUT, centred at 0."""
    fig = go.Figure(
        data=go.Heatmap(
            z=field,
            x=x,
            y=y,
            colorscale=_resolve_colorscale("coolwarm"),
            zmid=0,
            colorbar=dict(title=z_label),
            hovertemplate=_hover_scalar("Å", "Å", z_label),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x (Å)",
        yaxis_title="y (Å)",
        yaxis=_equal_aspect_axes(),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_pdw_peak_map(
    peaks: list,
    title: str = "Composite CDW Peaks (SPECULATIVE)",
    dark: bool = True,
    fft: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> go.Figure:
    """k-space map of the composite CDW peaks at ±k, sized by intensity.

    ``peaks`` are :class:`~waytogocoop.computation.pdw.PdwPeak` records.
    Filled markers sit above the visibility floor, open markers below it;
    only the +k member of each visible pair carries a text label.  With
    ``fft=(kx, ky, power)`` the log power spectrum of ρ(r) is drawn
    underneath and every peak is ringed and labelled in white instead: the
    inferno background is dark in both themes, so white stays legible there,
    and the labels replace the legend (white swatches vanish on plotly_white).
    """
    colors = _pdw_family_colors(dark)
    k_max = 1.15 * max((np.hypot(p.kx, p.ky) for p in peaks), default=1.0)
    i_max = max((p.intensity for p in peaks), default=0.0)

    fig = go.Figure()
    if fft is not None:
        kx, ky, power = fft
        fig.add_trace(
            go.Heatmap(
                z=power,
                x=kx,
                y=ky,
                colorscale=_resolve_colorscale("inferno"),
                colorbar=dict(title="log₁₀(|F|²)"),
                hovertemplate=(
                    "kx: %{x:.3f} 1/Å<br>ky: %{y:.3f} 1/Å"
                    "<br>log₁₀(|F|²): %{z:.2f}<extra></extra>"
                ),
            )
        )

    for family, family_label in _PDW_FAMILY_LABELS.items():
        members = [p for p in peaks if p.family == family]
        if not members:
            continue
        xs, ys, sizes, symbols, texts, custom = [], [], [], [], [], []
        for p in members:
            k = float(np.hypot(p.kx, p.ky))
            wavelength = 2.0 * np.pi / k if k > 0 else float("inf")
            if p.visible and i_max > 0:
                size = 9.0 + 21.0 * float(np.sqrt(p.intensity / i_max))
            else:
                size = 9.0
            symbol = "circle" if p.visible and fft is None else "circle-open"
            for sign in (1.0, -1.0):
                xs.append(sign * p.kx)
                ys.append(sign * p.ky)
                sizes.append(size)
                symbols.append(symbol)
                labelled = p.visible or fft is not None
                texts.append(p.label if labelled and sign > 0 else "")
                custom.append([p.label, k, wavelength, p.intensity])
        if fft is None:
            marker = dict(
                size=sizes,
                symbol=symbols,
                color=colors[family],
                line=dict(width=2, color=colors[family]),
            )
        else:
            marker = dict(
                size=sizes, symbol=symbols, color="white", line=dict(width=2, color="white")
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=marker,
                text=texts,
                textposition="top center",
                textfont=dict(color="white") if fft is not None else None,
                name=family_label,
                showlegend=fft is None,
                customdata=custom,
                hovertemplate=(
                    "%{customdata[0]}<br>kx: %{x:.3f} 1/Å<br>ky: %{y:.3f} 1/Å"
                    "<br>|k|: %{customdata[1]:.3f} 1/Å"
                    "<br>λ = 2π/|k|: %{customdata[2]:.1f} Å"
                    "<br>|ρ|²: %{customdata[3]:.3g}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(title="kx (1/Å)", range=[-k_max, k_max], zeroline=True),
        yaxis=dict(title="ky (1/Å)", range=[-k_max, k_max], zeroline=True, **_equal_aspect_axes()),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=20, t=80, b=50),
        template=_template(dark),
    )
    return fig


def _log_intensity(values: np.ndarray) -> np.ndarray:
    """Zero intensity drops off a log axis; NaN leaves a clean gap instead."""
    values = np.asarray(values, dtype=float)
    return np.where(values > 0.0, values, np.nan)


def create_pdw_sweep_plot(
    values: np.ndarray,
    intensity_p: np.ndarray,
    intensity_q: np.ndarray,
    intensity_h: np.ndarray,
    x_label: str,
    title: str,
    floor: float,
    current: float | None = None,
    markers: list[tuple[float, str]] | None = None,
    shade: tuple[float, float, str] | None = None,
    dark: bool = True,
) -> go.Figure:
    """Composite-peak intensities |ρ_k|² along a temperature or field sweep.

    One log-scaled axis.  Members 1 and 2 of each family are mirror partners
    with equal intensity on the page's presets, so they share one dashed
    curve (the larger of the two); member 3 is solid.  ``markers`` adds
    dotted reference lines (Tc, T_PDW, Hc2, ...), ``shade`` a labelled band
    (the PDW-only window), ``floor`` the detection floor.
    """
    colors = _pdw_family_colors(dark)
    _primary, _secondary, muted = _line_colors(dark)
    # A zero floor is valid model input but has no place on a log axis.
    floor = max(floor, 1e-12)
    curves = [
        ("p₁,₂ (field-induced)", "p", np.maximum(intensity_p[0], intensity_p[1]), "dash"),
        ("p₃", "p", intensity_p[2], "solid"),
        ("q₁,₂", "q", np.maximum(intensity_q[0], intensity_q[1]), "dash"),
        ("q₃", "q", intensity_q[2], "solid"),
        ("h₁,₂", "h", np.maximum(intensity_h[0], intensity_h[1]), "solid"),
    ]
    fig = go.Figure()
    if shade is not None:
        x0, x1, shade_label = shade
        if x1 > x0:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=_pdw_regime_colors(dark)[1],
                opacity=0.12,
                line_width=0,
                annotation_text=shade_label,
                annotation_position="bottom left",
            )
    for name, family, ys, dash in curves:
        fig.add_trace(
            go.Scatter(
                x=values,
                y=_log_intensity(ys),
                name=name,
                mode="lines",
                line=dict(color=colors[family], width=2, dash=dash),
                hovertemplate=f"{x_label}: %{{x:.3g}}<br>|ρ|²: %{{y:.3g}}<extra>{name}</extra>",
            )
        )
    fig.add_hline(y=floor, line=dict(dash="dot", color=muted, width=1))
    # Annotation y on a log axis is in log10 units (shapes take data units).
    fig.add_annotation(
        x=1.0,
        xref="paper",
        xanchor="right",
        y=np.log10(floor),
        yref="y",
        yanchor="bottom",
        text="detection floor",
        showarrow=False,
    )
    for x_mark, label in markers or []:
        if np.isfinite(x_mark) and x_mark > 0:
            fig.add_vline(
                x=x_mark, line=dict(dash="dot", color=muted, width=1), annotation_text=label
            )
    if current is not None:
        fig.add_vline(x=current, line=dict(dash="dash", color=muted, width=1))

    peak = max(float(np.nanmax(np.concatenate([c[2] for c in curves]))), floor)
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis=dict(
            title="|ρ_k|² (arb.)",
            type="log",
            dtick=1,
            exponentformat="power",
            range=[np.log10(floor) - 2.0, np.log10(peak) + 0.3],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=20, t=80, b=50),
        template=_template(dark),
        hovermode="x unified",
    )
    return fig


def create_pdw_regime_map(
    magnitudes: np.ndarray,
    temperatures: np.ndarray,
    regimes: np.ndarray,
    field_label: str,
    title: str = "PDW Regimes (SPECULATIVE)",
    current: tuple[float, float] | None = None,
    dark: bool = True,
) -> go.Figure:
    """Categorical (|B|, T) map of which orders exist, with the current point."""
    colors = _pdw_regime_colors(dark)
    n = len(colors)
    colorscale = []
    for i, color in enumerate(colors):
        colorscale += [[i / n, color], [(i + 1) / n, color]]
    names = np.array(_PDW_REGIME_LABELS, dtype=object)[np.asarray(regimes, dtype=int)]

    fig = go.Figure(
        data=go.Heatmap(
            z=regimes,
            x=magnitudes,
            y=temperatures,
            zmin=-0.5,
            zmax=n - 0.5,
            colorscale=colorscale,
            customdata=names,
            colorbar=dict(
                title="Regime",
                tickvals=list(range(n)),
                ticktext=list(_PDW_REGIME_LABELS),
            ),
            hovertemplate=(
                f"{field_label}: %{{x:.2f}} T<br>T: %{{y:.2f}} K"
                "<br>%{customdata}<extra></extra>"
            ),
        )
    )
    if current is not None:
        fig.add_trace(
            go.Scatter(
                x=[current[0]],
                y=[current[1]],
                mode="markers",
                marker=dict(
                    size=12, symbol="x", color=_marker_color(dark), line=dict(width=1)
                ),
                name="Current (|B|, T)",
                hovertemplate="Current<br>|B|: %{x:.2f} T<br>T: %{y:.2f} K<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=f"{field_label} (T)",
        yaxis_title="T (K)",
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


# ---------------------------------------------------------------------------
# Spin fluctuations (SPECULATIVE)
# ---------------------------------------------------------------------------

# Temperature ramps of the SF sweeps sample the shared plasma LUT at evenly
# spaced stops: dark themes skip the near-black purple end, light themes the
# pale yellow end that vanishes on white.  The Rust tab samples the same rows.
_SF_TEMPERATURE_STOPS: dict[bool, tuple[float, float]] = {
    True: (0.35, 0.92),
    False: (0.05, 0.78),
}

_SF_SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

# The imprint plot's reference line: the measured gaps of the amplitude route.
_SF_AMPLITUDE_ROUTE_LABEL = "amplitude route, measured gaps (13.5 K)"


def _sf_temperature_colors(n: int, dark: bool = True) -> list[str]:
    """``n`` colours for a temperature ramp, coldest first.

    Evenly spaced stops on [0.35, 0.92] (dark) or [0.05, 0.78] (light), each
    read from the 256-row plasma LUT at row ``floor(255 t + 0.5)``.
    """
    lo, hi = _SF_TEMPERATURE_STOPS[dark]
    lut = get_plotly_colorscale("plasma")
    return [lut[int(np.floor(255.0 * t + 0.5))][1] for t in np.linspace(lo, hi, n)]


def _sf_sci(value: float, digits: int = 2) -> str:
    """``value`` as "5.05×10⁻⁴", with a true minus sign."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}×10{str(int(exponent)).translate(_SF_SUPERSCRIPTS)}".replace("-", "−")


def create_sf_fermi_surface(
    kx: np.ndarray,
    ky: np.ndarray,
    gamma: np.ndarray,
    hotspot_xy: np.ndarray | None,
    cvc_xy: np.ndarray,
    partner_xy: np.ndarray,
    zone_boundary: float,
    *,
    title: str,
    dark: bool = True,
) -> go.Figure:
    """Scattering rate Γ(φ) on the LCCO hole pocket, in k-space about M = (π/a, π/a).

    ``kx``, ``ky`` (1/Å) trace the Fermi surface and ``gamma`` (ps⁻¹) colours
    it on the viridis LUT.  The dashed diamond |u_x| + |u_y| = ``zone_boundary``
    (π/a, 1/Å) is the AFM zone boundary.  ``hotspot_xy`` (8, 2) rings the Γ_HS
    centres (None when the hot spots are off); thin segments run from each
    CVC centre in ``cvc_xy`` (8, 2), a Fermi-surface × zone-boundary crossing,
    to its Q = (π, π) partner in ``partner_xy``.  "N" and "AN" mark the node
    (φ = 45°) and the antinode (φ = 0).
    """
    colors = _pdw_family_colors(dark)
    _primary, _secondary, muted = _line_colors(dark)
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    phi_deg = np.degrees(np.mod(np.arctan2(ky, kx), 2.0 * np.pi))
    zb = float(zone_boundary)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[zb, 0.0, -zb, 0.0, zb],
            y=[0.0, zb, 0.0, -zb, 0.0],
            name="AFM zone boundary",
            mode="lines",
            line=dict(color=muted, width=1, dash="dash"),
            hoverinfo="skip",
        )
    )
    closed = np.append(np.arange(kx.size), 0)
    fig.add_trace(
        go.Scatter(
            x=kx[closed],
            y=ky[closed],
            name="Fermi surface",
            mode="lines+markers",
            line=dict(color=muted, width=1),
            marker=dict(
                size=5,
                color=gamma[closed],
                colorscale=_resolve_colorscale("viridis"),
                colorbar=dict(title="Γ (ps⁻¹)"),
                showscale=True,
            ),
            customdata=np.column_stack([phi_deg[closed], gamma[closed]]),
            hovertemplate=(
                "φ: %{customdata[0]:.1f}°<br>Γ: %{customdata[1]:.1f} ps⁻¹<extra></extra>"
            ),
            showlegend=False,
        )
    )
    seg_x: list[float | None] = []
    seg_y: list[float | None] = []
    for (cx, cy), (px, py) in zip(np.asarray(cvc_xy), np.asarray(partner_xy), strict=True):
        seg_x += [float(cx), float(px), None]
        seg_y += [float(cy), float(py), None]
    fig.add_trace(
        go.Scatter(
            x=seg_x,
            y=seg_y,
            name="CVC partners (k + Q)",
            mode="lines+markers",
            line=dict(color=colors["p"], width=1),
            marker=dict(size=4, color=colors["p"]),
            hoverinfo="skip",
        )
    )
    if hotspot_xy is not None and len(hotspot_xy):
        hot = np.asarray(hotspot_xy, dtype=float)
        fig.add_trace(
            go.Scatter(
                x=hot[:, 0],
                y=hot[:, 1],
                name="hot spots (Γ_HS)",
                mode="markers",
                marker=dict(
                    symbol="circle-open", size=15, color=_marker_color(dark), line=dict(width=2)
                ),
                hoverinfo="skip",
            )
        )
    for label, angle in (("N", 45.0), ("AN", 0.0)):
        i = int(np.argmin(np.abs((phi_deg - angle + 180.0) % 360.0 - 180.0)))
        fig.add_annotation(
            x=1.16 * kx[i], y=1.16 * ky[i], text=label, showarrow=False, font=dict(size=13)
        )

    lim = 1.12 * zb
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),  # the page's narrow column
        xaxis=dict(title="k_x − π/a (1/Å)", range=[-lim, lim], zeroline=False),
        yaxis=dict(
            title="k_y − π/a (1/Å)", range=[-lim, lim], zeroline=False, **_equal_aspect_axes()
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.17, xanchor="center", x=0.5),
        margin=dict(l=60, r=20, t=50, b=90),
        template=_template(dark),
    )
    return fig


def create_sf_mr_plot(
    fields: np.ndarray,
    temperatures: np.ndarray,
    mr: np.ndarray,
    *,
    current: tuple[float, np.ndarray] | None = None,
    title: str,
    dark: bool = True,
) -> go.Figure:
    """Magnetoresistance Δρ/ρ(0) against μ₀H, one line per temperature, coldest first.

    ``mr`` is (n_T, n_H) on ``fields``.  ``current = (T, mr_T)`` adds the current
    temperature as a dashed line (the page passes it when T is not in the family).
    """
    colors = _sf_temperature_colors(len(temperatures), dark)
    fig = go.Figure()
    for t, row, color in zip(temperatures, mr, colors, strict=True):
        name = f"{t:g} K"
        fig.add_trace(
            go.Scatter(
                x=fields,
                y=row,
                name=name,
                mode="lines",
                line=dict(color=color, width=2),
                hovertemplate=f"μ₀H: %{{x:.1f}} T<br>Δρ/ρ(0): %{{y:.4f}}<extra>{name}</extra>",
            )
        )
    if current is not None:
        t, row = current
        name = f"{t:g} K (current T)"
        fig.add_trace(
            go.Scatter(
                x=fields,
                y=row,
                name=name,
                mode="lines",
                line=dict(color=_marker_color(dark), width=1.5, dash="dash"),
                hovertemplate=f"μ₀H: %{{x:.1f}} T<br>Δρ/ρ(0): %{{y:.4f}}<extra>{name}</extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="μ₀H (T)",
        yaxis=dict(title="Δρ/ρ(0)", rangemode="tozero"),
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_sf_derivative_plot(
    h_over_t: np.ndarray,
    drho_dh: np.ndarray,
    temperatures: np.ndarray,
    quad_h_over_t: np.ndarray,
    quad_drho: np.ndarray,
    *,
    current: tuple[float, np.ndarray, np.ndarray] | None = None,
    title: str,
    dark: bool = True,
) -> go.Figure:
    """dρ/dμ₀H against μ₀H/T: the LCCO model beside the hole-doped quadrature form.

    Left, the model's (n_T, n_H) family, which does not collapse; right, the
    Ayres et al. (2021) quadrature form at the same temperatures with its
    illustrative Tl2201 parameters, which collapses by construction.  Both in
    μΩ cm/T.  ``current = (T, h_over_t, drho_dh)`` adds the current temperature
    dashed on the left.
    """
    colors = _sf_temperature_colors(len(temperatures), dark)
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        subplot_titles=(
            "LCCO model: no H/T collapse",
            "Hole-doped quadrature (illustrative Tl2201): collapse",
        ),
    )
    hover = "μ₀H/T: %{{x:.3f}} T/K<br>dρ/dμ₀H: %{{y:.4f}} μΩ cm/T<extra>{name}</extra>"
    n = len(temperatures)
    for i, (t, color) in enumerate(zip(temperatures, colors, strict=True)):
        name = f"{t:g} K"
        # The quadrature curves coincide, so they are nested (coldest widest) to
        # show every temperature on the one collapsed curve.
        for col, xs, ys, width in (
            (1, h_over_t, drho_dh, 2.0),
            (2, quad_h_over_t, quad_drho, 2.0 + 2.0 * (n - 1 - i)),
        ):
            fig.add_trace(
                go.Scatter(
                    x=xs[i],
                    y=ys[i],
                    name=name,
                    legendgroup=name,
                    showlegend=col == 1,
                    mode="lines",
                    line=dict(color=color, width=width),
                    hovertemplate=hover.format(name=name),
                ),
                row=1,
                col=col,
            )
    if current is not None:
        t, xs, ys = current
        name = f"{t:g} K (current T)"
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name=name,
                mode="lines",
                line=dict(color=_marker_color(dark), width=1.5, dash="dash"),
                hovertemplate=hover.format(name=name),
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(title_text="μ₀H/T (T/K)", rangemode="tozero")
    fig.update_yaxes(title_text="dρ/dμ₀H (μΩ cm/T)", rangemode="tozero")
    fig.update_layout(
        title=title,
        legend=dict(x=0.99, y=0.03, xanchor="right", yanchor="bottom"),
        margin=dict(l=60, r=20, t=80, b=50),
        template=_template(dark),
    )
    return fig


def create_sf_resistivity_plot(
    temperatures: np.ndarray,
    rho: np.ndarray,
    fit,
    *,
    current: float | None = None,
    title: str,
    dark: bool = True,
) -> go.Figure:
    """Zero-field ρ(0, T) with its ρ₀ + A₁T + A₂T² fit, in μΩ cm.

    ``fit`` is a :class:`~waytogocoop.computation.spin_fluctuation.SfResistivityFit`
    (rho0, a1, a2, t_min, t_max).  The fit is dashed over its shaded window and
    its coefficients are annotated; ``current`` marks the current temperature.
    """
    colors = _pdw_family_colors(dark)
    _primary, _secondary, muted = _line_colors(dark)
    fig = go.Figure()
    fig.add_vrect(
        x0=fit.t_min,
        x1=fit.t_max,
        fillcolor=muted,
        opacity=0.15,
        line_width=0,
        annotation_text=f"fit window {fit.t_min:g}–{fit.t_max:g} K",
        annotation_position="bottom right",
    )
    fig.add_trace(
        go.Scatter(
            x=temperatures,
            y=rho,
            name="ρ(0, T), model",
            mode="lines",
            line=dict(color=colors["q"], width=2.5),
            hovertemplate="T: %{x:.1f} K<br>ρ(0, T): %{y:.3f} μΩ cm<extra></extra>",
        )
    )
    t_fit = np.linspace(fit.t_min, fit.t_max, 91)
    fig.add_trace(
        go.Scatter(
            x=t_fit,
            y=fit.rho0 + fit.a1 * t_fit + fit.a2 * t_fit**2,
            name="ρ₀ + A₁T + A₂T² fit",
            mode="lines",
            line=dict(color=colors["p"], width=2, dash="dash"),
            hovertemplate="T: %{x:.1f} K<br>fit: %{y:.3f} μΩ cm<extra></extra>",
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        text=(
            f"ρ₀ = {fit.rho0:.2f} μΩ cm<br>A₁ = {fit.a1:.4f} μΩ cm/K"
            f"<br>A₂ = {_sf_sci(fit.a2)} μΩ cm/K²"
        ),
    )
    if current is not None:
        fig.add_vline(x=current, line=dict(dash="dash", color=muted, width=1))
    fig.update_layout(
        title=title,
        xaxis_title="T (K)",
        yaxis_title="ρ(0, T) (μΩ cm)",
        legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_sf_hall_plot(
    temperatures: np.ndarray,
    rh_plain: np.ndarray,
    rh_cvc: np.ndarray,
    *,
    drude: float,
    drude_nominal: float,
    current: float | None = None,
    title: str,
    dark: bool = True,
) -> go.Figure:
    """Jones-Zener Hall coefficient R_H(T), in 10⁻¹⁰ m³/C.

    With current-vertex corrections solid, without them dashed; the Drude
    1/(ne) of the fixed x = 0.159 Fermi surface dotted, and of the nominal
    n = 1 − x holes per Cu dash-dot.  ``current`` marks the current temperature.
    """
    colors = _pdw_family_colors(dark)
    _primary, _secondary, muted = _line_colors(dark)
    t_span = [float(temperatures[0]), float(temperatures[-1])]
    hover = "T: %{x:.1f} K<br>R_H: %{y:.3f} × 10⁻¹⁰ m³/C"
    fig = go.Figure()
    for name, xs, ys, color, dash, width in (
        ("with vertex corrections (CVC)", temperatures, rh_cvc, colors["q"], "solid", 2.5),
        ("without vertex corrections", temperatures, rh_plain, colors["p"], "dash", 2.0),
        ("Drude, fixed x = 0.159 FS", t_span, [drude, drude], muted, "dot", 1.5),
        ("Drude, n = 1 − x", t_span, [drude_nominal] * 2, colors["h"], "dashdot", 1.5),
    ):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name=name,
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=f"{hover}<extra>{name}</extra>",
            )
        )
    if current is not None:
        fig.add_vline(x=current, line=dict(dash="dash", color=muted, width=1))
    fig.update_layout(
        title=title,
        xaxis_title="T (K)",
        yaxis=dict(title="R_H (10⁻¹⁰ m³/C)", rangemode="tozero"),
        legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_sf_doping_plot(
    sweep,
    *,
    x_fsr: float,
    current: float | None = None,
    title: str,
    dark: bool = True,
) -> go.Figure:
    """ρ(0, T) and MR(H, T) across the tabulated LCCO dopings, on twin y axes.

    ``sweep`` is a :class:`~waytogocoop.computation.spin_fluctuation.SfDopingSweep`.
    The dotted line marks x_FSR, the end of the Fermi-surface reconstruction;
    ``current`` marks the current doping.
    """
    colors = _pdw_family_colors(dark)
    _primary, _secondary, muted = _line_colors(dark)
    rho_name = f"ρ(0, {sweep.temperature:g} K)"
    mr_name = f"MR({sweep.field:g} T, {sweep.temperature:g} K)"
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=sweep.dopings,
            y=sweep.rho0,
            name=rho_name,
            mode="lines+markers",
            line=dict(color=colors["q"], width=2),
            marker=dict(size=7),
            hovertemplate=f"x: %{{x:.3f}}<br>{rho_name}: %{{y:.2f}} μΩ cm<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=sweep.dopings,
            y=sweep.mr,
            name=mr_name,
            mode="lines+markers",
            line=dict(color=colors["p"], width=2, dash="dash"),
            marker=dict(size=7, symbol="square"),
            hovertemplate=f"x: %{{x:.3f}}<br>{mr_name}: %{{y:.4f}}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_vline(
        x=x_fsr,
        line=dict(dash="dot", color=muted, width=1),
        annotation_text="x_FSR",
        annotation_position="bottom right",
    )
    if current is not None:
        fig.add_vline(
            x=current,
            line=dict(dash="dash", color=_marker_color(dark), width=1),
            annotation_text=f"x = {current:.3f}",
            annotation_position="top right",
        )
    fig.update_layout(
        title=title,
        xaxis_title="Ce doping x",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=60, r=70, t=80, b=50),
        template=_template(dark),
    )
    fig.update_yaxes(title_text=f"{rho_name} (μΩ cm)", secondary_y=False)
    fig.update_yaxes(
        title_text=mr_name, secondary_y=True, showgrid=False, tickmode="auto", tickformat=".3f"
    )
    return fig


def create_sf_coupling_plot(
    lambda_tr: np.ndarray,
    lambda_rho: np.ndarray,
    tc: np.ndarray,
    dopings: np.ndarray,
    curves: list[tuple[str, np.ndarray, np.ndarray, str, bool]],
    *,
    current: tuple[float, float] | None = None,
    operating_point: tuple[float, float] | None = None,
    lambda_c: float | None = None,
    title: str,
    dark: bool = True,
) -> go.Figure:
    """Tc against coupling for the LCCO dopings, with the Tc(λ) pairing maps.

    Filled markers are λ_tr (the FS-averaged T-linear coupling), hollow markers
    λ_ρ (Allen–Drude, from ρ(T)), both at the table Tc(x).  ``curves`` holds
    ``(label, lam, tc, dash, selected)`` per map; the selected map is drawn
    emphasised.  ``current = (λ_tr, Tc)`` rings the current doping,
    ``operating_point = (λ̄, Tc*)`` stars the glue operating point and
    ``lambda_c`` draws the selected map's threshold dotted.
    """
    colors = _pdw_family_colors(dark)
    _primary, _secondary, muted = _line_colors(dark)
    fig = go.Figure()
    for label, lam, tcs, dash, selected in curves:
        fig.add_trace(
            go.Scatter(
                x=lam,
                y=tcs,
                name=label,
                mode="lines",
                line=dict(
                    color=_marker_color(dark) if selected else muted,
                    width=2.5 if selected else 1.2,
                    dash=dash,
                ),
                hovertemplate=f"λ: %{{x:.3f}}<br>Tc: %{{y:.2f}} K<extra>{label}</extra>",
            )
        )
    if lambda_c is not None and lambda_c > 0.0:
        fig.add_vline(
            x=lambda_c,
            line=dict(dash="dot", color=muted, width=1),
            annotation_text=f"λ_c = {lambda_c:.3f}",
            annotation_position="top right",
        )
    custom = np.asarray(dopings, dtype=float)
    for name, xs, symbol in (
        ("λ_tr (FS-averaged T-linear coupling)", lambda_tr, "circle"),
        ("λ_ρ (Allen–Drude, from ρ(T))", lambda_rho, "circle-open"),
    ):
        short = name.split(" ")[0]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=tc,
                name=name,
                mode="markers",
                marker=dict(size=9, symbol=symbol, color=colors["q"], line=dict(width=2)),
                customdata=custom,
                hovertemplate=(
                    f"x = %{{customdata:.3f}}<br>{short}: %{{x:.3f}}<br>Tc: %{{y:.1f}} K"
                    "<extra></extra>"
                ),
            )
        )
    if current is not None:
        fig.add_trace(
            go.Scatter(
                x=[current[0]],
                y=[current[1]],
                name="current x",
                mode="markers",
                marker=dict(size=17, symbol="circle-open", color=colors["p"], line=dict(width=2.5)),
                hovertemplate="current x<br>λ_tr: %{x:.3f}<br>Tc: %{y:.1f} K<extra></extra>",
            )
        )
    if operating_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[operating_point[0]],
                y=[operating_point[1]],
                name="glue operating point (λ̄, Tc*)",
                mode="markers",
                marker=dict(
                    size=17, symbol="star", color=colors["h"], line=dict(width=1, color=muted)
                ),
                hovertemplate="glue operating point<br>λ̄: %{x:.3f}<br>Tc*: %{y:.2f} K"
                "<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis=dict(title="coupling λ", rangemode="tozero"),
        yaxis=dict(title="Tc (K)", rangemode="tozero"),
        legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_sf_imprint_plot(
    tc_targets: np.ndarray,
    dressed: np.ndarray,
    local_exact: np.ndarray,
    local_linear: np.ndarray,
    amplitude_ref: float,
    *,
    coherence_label: str,
    fragment_below: float | None = None,
    current: tuple[float, float] | None = None,
    title: str,
    dark: bool = True,
) -> go.Figure:
    """Glue-route imprint δΔ/Δ̄ against the operating Tc*, beside the amplitude route.

    The coherence-dressed imprint is solid, the local limit (L_m ≫ ξ) dashed and
    its linearisation η·S dotted; in the local limit ``dressed`` equals
    ``local_exact`` and that curve is drawn once, as the dressed one.  The
    dotted horizontal line is the amplitude route's ``amplitude_ref``.
    ``current = (Tc*, dressed)`` stars the operating point.  ``fragment_below``
    shades [tc_targets[0], fragment_below], where the local map has Tc(r) = 0
    regions.  The y axis spans [0, max(0.5, 1.5 max(max(dressed), amplitude_ref))];
    the local curves leave the frame above it.
    """
    colors = _pdw_family_colors(dark)
    _primary, _secondary, muted = _line_colors(dark)
    dressed = np.asarray(dressed, dtype=float)
    local_exact = np.asarray(local_exact, dtype=float)
    x0, x1 = float(tc_targets[0]), float(tc_targets[-1])
    top = max(0.5, 1.5 * max(float(np.max(dressed)), amplitude_ref))
    hover = "Tc*: %{{x:.2f}} K<br>δΔ/Δ̄: %{{y:.4f}}<extra>{name}</extra>"

    fig = go.Figure()
    if fragment_below is not None and fragment_below > x0:
        fig.add_vrect(
            x0=x0,
            x1=fragment_below,
            fillcolor=muted,
            opacity=0.2,
            line_width=0,
            annotation_text="local limit: Tc(r) = 0 in places",
            annotation_position="top right",
        )
    lines = []
    if not np.array_equal(dressed, local_exact):
        lines.append(("local limit (L_m ≫ ξ)", tc_targets, local_exact, colors["p"], "dash", 2.0))
    lines += [
        ("linear η·S (local)", tc_targets, local_linear, muted, "dot", 1.2),
        (_SF_AMPLITUDE_ROUTE_LABEL, [x0, x1], [amplitude_ref] * 2, colors["h"], "dot", 2.0),
        (f"dressed ({coherence_label})", tc_targets, dressed, colors["q"], "solid", 3.0),
    ]
    for name, xs, ys, color, dash, width in lines:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name=name,
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=hover.format(name=name),
            )
        )
    if current is not None:
        fig.add_trace(
            go.Scatter(
                x=[current[0]],
                y=[current[1]],
                name=f"operating point, Tc* = {current[0]:g} K",
                mode="markers",
                marker=dict(
                    size=17,
                    symbol="star",
                    color=_marker_color(dark),
                    line=dict(width=1.5, color=colors["q"]),
                ),
                hovertemplate=hover.format(name="operating point"),
            )
        )
    fig.update_layout(
        title=title,
        xaxis=dict(title="operating Tc* (K)", range=[0.0, x1]),
        yaxis=dict(title="δΔ/Δ̄", range=[0.0, top]),
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig
