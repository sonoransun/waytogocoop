"""Plotly figure constructors for moire visualisation.

Colormap convention (must stay in sync with crates/moire-core/src/colormap.rs).
Colormap data itself lives in :mod:`waytogocoop.components.colormaps`, which
loads a 256-stop LUT generated from the Rust side — the two stacks produce
pixel-identical screenshots so long as that JSON is regenerated after Rust-side
edits.

- ``viridis``  — unsigned scalar fields (moire pattern, surface plots)
- ``coolwarm`` / ``coolwarm_r`` — signed gap modulation and gap-aware overlays (meV)
- ``inferno``  — FFT power spectrum (log10 scaled), Majorana density
- ``plasma``   — susceptibility (arbitrary units)
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


def _resolve_colorscale(name: str) -> list[list[object]]:
    """Accept either a shared-LUT name or a Plotly built-in name.

    Shared-LUT names ('viridis', 'inferno', 'coolwarm', 'plasma', and their
    ``_r`` reversals) are resolved via the cross-stack JSON. Anything else
    passes through unchanged so Plotly's built-in scales ('Hot', 'RdBu_r',
    'Blues', ...) still work for figure types we haven't unified yet.
    """
    key = name.lower()
    base = key[:-2] if key.endswith("_r") else key
    if base in ("viridis", "inferno", "coolwarm", "plasma"):
        return get_plotly_colorscale(key)
    # Special case: "RdBu_r" is a conventional alias we want to map to the
    # shared coolwarm_r LUT so signed-gap plots render identically on both
    # stacks (Rust uses coolwarm for the same role).
    if key in ("rdbu_r", "rdbu"):
        return get_plotly_colorscale("coolwarm_r" if key.endswith("_r") else "coolwarm")
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
) -> go.Figure:
    """3D rotatable surface.

    When ``high_density`` is True the surface is drawn as a triangulated
    :class:`plotly.graph_objects.Mesh3d` with per-vertex intensity and no
    hover-interpolation — that path stays responsive at grid sizes >= 150
    where ``go.Surface`` would stutter. Otherwise we use ``go.Surface`` with
    the shared PBR-ish lighting preset.
    """
    if high_density:
        return create_surface_mesh3d(
            x, y, z, title=title, colorscale=colorscale, z_label=z_label, dark=dark
        )
    surface = _apply_surface_lighting(
        go.Surface(
            z=z,
            x=x,
            y=y,
            colorscale=_resolve_colorscale(colorscale),
            colorbar=dict(title=z_label),
            hovertemplate=(
                "x: %{x:.2f} Å"
                "<br>y: %{y:.2f} Å"
                f"<br>{z_label}: %{{z:.3g}}"
                "<extra></extra>"
            ),
        )
    )
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

    ``surface_count`` is caller-driven so an animation loop in
    ``pages/proximity_3d.py`` can sweep it. ``clip_z`` clips the input mask
    above a user-chosen z-plane — values at ``z > clip_z`` are replaced with
    NaN so Plotly's isosurface algorithm skips them.
    """
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    if iso_min is None:
        iso_min = float(np.percentile(values, 20))
    if iso_max is None:
        iso_max = float(np.percentile(values, 80))

    vals = values.transpose(2, 1, 0).astype(float).copy()
    if clip_z is not None:
        # vals is (nx, ny, nz) after transpose; but we built X/Y/Z with
        # indexing="ij" on (x, y, z) so flatten/mask using the z-meshgrid.
        mask = clip_z < Z
        # Rebuild mask on the transposed orientation for consistency with the
        # flatten-order below.
        vals_flat = vals.flatten()
        vals_flat[mask.flatten()] = np.nan
        vals = vals_flat.reshape(vals.shape)

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

    The opacity ramp is tuned so low-gap regions stay partially transparent
    and strong-gap bulk lights up, while still showing enough mid-range
    contrast that the moire modulation remains visible through the slab.
    The tuple ``[[0, 0], [0.5, 0.25], [1, 0.85]]`` works for both signed
    (coolwarm-like) gap fields and unsigned densities.
    """
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    vals = values.transpose(2, 1, 0).astype(float).copy()
    if clip_z is not None:
        mask = clip_z < Z
        vals_flat = vals.flatten()
        vals_flat[mask.flatten()] = np.nan
        vals = vals_flat.reshape(vals.shape)

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=vals.flatten(),
            isomin=float(np.nanmin(vals)),
            isomax=float(np.nanmax(vals)),
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
    """2D colormap of topological phase index with boundary overlay."""
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
            contours=dict(start=0.5, end=0.5, size=1),
            line=dict(color="white" if dark else "black", width=2),
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
