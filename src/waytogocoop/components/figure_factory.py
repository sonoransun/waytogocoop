"""Plotly figure constructors for moire visualisation.

Colormap convention (must stay in sync with crates/moire-core/src/colormap.rs):
- Viridis   — unsigned scalar fields (moire pattern, surface plots)
- RdBu_r    — signed gap modulation and gap-aware overlays (meV)
- Hot       — FFT power spectrum (log10 scaled), Majorana density
- Plasma    — susceptibility (arbitrary units)
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
            colorscale="Viridis",
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
            colorscale="RdBu_r",
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
            colorscale="Hot",
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
    colorscale: str = "Viridis",
    z_label: str = "Intensity",
    dark: bool = True,
) -> go.Figure:
    """2D contour/topographic map projection of a real-space field."""
    fig = go.Figure(
        data=go.Contour(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
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


def _scene_camera_presets() -> dict:
    """Named camera positions for the preset buttons on 3D figures."""
    return {
        "iso":  dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        "top":  dict(eye=dict(x=0.0, y=0.0, z=2.5), up=dict(x=0, y=1, z=0)),
        "side": dict(eye=dict(x=2.5, y=0.0, z=0.2)),
    }


def _camera_buttons(pos_x: float = 0.02, pos_y: float = 0.98) -> dict:
    presets = _scene_camera_presets()
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
            dict(label="Iso",  method="relayout", args=[{"scene.camera": presets["iso"]}]),
            dict(label="Top",  method="relayout", args=[{"scene.camera": presets["top"]}]),
            dict(label="Side", method="relayout", args=[{"scene.camera": presets["side"]}]),
        ],
    )


def create_3d_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    colorscale: str = "Viridis",
    z_label: str = "Intensity",
    dark: bool = True,
) -> go.Figure:
    """3D rotatable surface plot with camera-preset buttons."""
    fig = go.Figure(
        data=go.Surface(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            colorbar=dict(title=z_label),
            hovertemplate=(
                "x: %{x:.2f} Å"
                "<br>y: %{y:.2f} Å"
                f"<br>{z_label}: %{{z:.3g}}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (Å)",
            yaxis_title="y (Å)",
            zaxis_title=z_label,
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.4),
            camera=_scene_camera_presets()["iso"],
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        template=_template(dark),
        updatemenus=[_camera_buttons()],
    )
    return fig


def create_3d_isosurface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    title: str = "3D Cooper Surface",
    iso_min: float | None = None,
    iso_max: float | None = None,
    colorscale: str = "RdBu_r",
    dark: bool = True,
    annotations: list[dict] | None = None,
) -> go.Figure:
    """3D isosurface of Δ(x,y,z) with camera-preset buttons."""
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    if iso_min is None:
        iso_min = float(np.percentile(values, 20))
    if iso_max is None:
        iso_max = float(np.percentile(values, 80))

    fig = go.Figure(
        data=go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.transpose(2, 1, 0).flatten(),
            isomin=iso_min,
            isomax=iso_max,
            surface_count=5,
            colorscale=colorscale,
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
    scene = dict(
        xaxis_title="x (Å)",
        yaxis_title="y (Å)",
        zaxis_title="z (Å)",
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.5),
        camera=_scene_camera_presets()["iso"],
    )
    if annotations:
        scene["annotations"] = annotations
    fig.update_layout(
        title=title,
        scene=scene,
        margin=dict(l=20, r=20, t=50, b=20),
        template=_template(dark),
        updatemenus=[_camera_buttons()],
    )
    return fig


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
            colorscale="RdBu_r",
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
            colorscale="Hot",
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
            colorscale="RdBu_r",
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
            colorscale="Plasma",
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
