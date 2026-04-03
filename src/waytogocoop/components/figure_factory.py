"""Plotly figure constructors for moire visualisation."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _template(dark: bool = True) -> str:
    """Return the Plotly template name for the current theme."""
    return "plotly_dark" if dark else "plotly_white"


def _line_colors(dark: bool = True) -> tuple[str, str, str]:
    """Return (primary, secondary, muted) line colors for the theme."""
    if dark:
        return "#5dade2", "#ec7063", "#888"
    return "steelblue", "firebrick", "gray"


def _marker_color(dark: bool = True) -> str:
    """Return marker color contrasting with the background."""
    return "white" if dark else "black"


def create_moire_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    pattern: np.ndarray,
    title: str = "Moire Pattern",
    dark: bool = True,
) -> go.Figure:
    """Create a heatmap of the real-space moire pattern.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate axes (Angstrom).
    pattern : np.ndarray
        2D pattern values.
    title : str
        Figure title.
    dark : bool
        Whether to use dark theme styling.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=pattern,
            x=x,
            y=y,
            colorscale="Viridis",
            colorbar=dict(title="Intensity"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x (Angstrom)",
        yaxis_title="y (Angstrom)",
        yaxis_scaleanchor="x",
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
    """Create a heatmap of the spatially varying superconducting gap.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate axes (Angstrom).
    gap_field : np.ndarray
        2D gap values (meV).
    title : str
        Figure title.
    dark : bool
        Whether to use dark theme styling.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=gap_field,
            x=x,
            y=y,
            colorscale="RdBu_r",
            colorbar=dict(title="Delta (meV)"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x (Angstrom)",
        yaxis_title="y (Angstrom)",
        yaxis_scaleanchor="x",
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
    """Create a heatmap of the FFT power spectrum.

    Parameters
    ----------
    kx, ky : np.ndarray
        1D wavevector axes (inverse Angstrom).
    power_spectrum : np.ndarray
        2D log-scaled power spectrum.
    title : str
        Figure title.
    dark : bool
        Whether to use dark theme styling.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=power_spectrum,
            x=kx,
            y=ky,
            colorscale="Hot",
            colorbar=dict(title="log10(|F|^2)"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="kx (1/Angstrom)",
        yaxis_title="ky (1/Angstrom)",
        yaxis_scaleanchor="x",
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
    )
    return fig


def create_sweep_plot(
    param_values: np.ndarray,
    periods: np.ndarray,
    amplitudes: np.ndarray,
    param_name: str = "Parameter",
    dark: bool = True,
) -> go.Figure:
    """Create a dual-axis line plot for parameter sweeps.

    Left y-axis: moire period.  Right y-axis: CPDM amplitude.

    Parameters
    ----------
    param_values : np.ndarray
        Swept parameter values.
    periods : np.ndarray
        Moire periods (Angstrom).
    amplitudes : np.ndarray
        CPDM amplitudes (dimensionless).
    param_name : str
        Label for the x-axis.
    dark : bool
        Whether to use dark theme styling.
    """
    primary, secondary, _muted = _line_colors(dark)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=param_values,
            y=periods,
            name="Moire period (A)",
            mode="lines+markers",
            line=dict(color=primary),
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
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Parameter Sweep: {param_name}",
        xaxis_title=param_name,
        margin=dict(l=60, r=60, t=50, b=50),
        template=_template(dark),
    )
    fig.update_yaxes(title_text="Moire period (Angstrom)", secondary_y=False)
    fig.update_yaxes(title_text="CPDM amplitude", secondary_y=True)
    return fig


def create_3d_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    colorscale: str = "Viridis",
    z_label: str = "Intensity",
    dark: bool = True,
) -> go.Figure:
    """3D rotatable surface plot using go.Surface.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate axes (Angstrom).
    z : np.ndarray
        2D array of surface heights.
    title : str
        Figure title.
    colorscale : str
        Plotly colorscale name.
    z_label : str
        Label for the colour bar and z-axis.
    dark : bool
        Whether to use dark theme styling.
    """
    fig = go.Figure(
        data=go.Surface(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            colorbar=dict(title=z_label),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (Angstrom)",
            yaxis_title="y (Angstrom)",
            zaxis_title=z_label,
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.4),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        margin=dict(l=20, r=20, t=50, b=20),
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
    """2D contour/topographic map projection.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate axes (Angstrom).
    z : np.ndarray
        2D array of field values.
    title : str
        Figure title.
    colorscale : str
        Plotly colorscale name.
    z_label : str
        Label for the colour bar.
    dark : bool
        Whether to use dark theme styling.
    """
    fig = go.Figure(
        data=go.Contour(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            colorbar=dict(title=z_label),
            contours=dict(showlabels=True, labelfont=dict(size=10)),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x (Angstrom)",
        yaxis_title="y (Angstrom)",
        yaxis_scaleanchor="x",
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
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
            colorbar=dict(title="Delta (meV)"),
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
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="x (Angstrom)",
        yaxis_title="y (Angstrom)",
        yaxis_scaleanchor="x",
        margin=dict(l=60, r=20, t=50, b=50),
        template=_template(dark),
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
) -> go.Figure:
    """3D isosurface of Delta(x,y,z) using go.Isosurface.

    Parameters
    ----------
    x, y, z : np.ndarray
        1D coordinate arrays.
    values : np.ndarray
        3D array (nz, ny, nx).
    dark : bool
        Whether to use dark theme styling.
    """
    nz, ny, nx = values.shape
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
            colorbar=dict(title="Delta (meV)"),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (Angstrom)",
            yaxis_title="y (Angstrom)",
            zaxis_title="z (Angstrom)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        template=_template(dark),
    )
    return fig


def create_z_decay_profile(
    z: np.ndarray,
    profile: np.ndarray,
    title: str = "Proximity Decay Profile",
    dark: bool = True,
) -> go.Figure:
    """Line plot of the proximity decay profile f(z)."""
    primary, _secondary, muted = _line_colors(dark)
    fig = go.Figure(
        data=go.Scatter(
            x=z, y=profile,
            mode="lines",
            line=dict(color=primary, width=2),
            name="f(z)",
        )
    )
    fig.add_vline(x=0, line=dict(dash="dash", color=muted), annotation_text="Interface")
    fig.update_layout(
        title=title,
        xaxis_title="z (Angstrom)",
        yaxis_title="f(z)",
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
    """Heatmap of Majorana zero-mode probability density."""
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=density, x=x, y=y,
            colorscale="Hot",
            colorbar=dict(title="|psi|^2"),
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
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="x (Angstrom)",
        yaxis_title="y (Angstrom)",
        yaxis_scaleanchor="x",
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
    """Background heatmap with arrow overlay for vector field.

    Parameters
    ----------
    skip : int
        Subsample every ``skip`` grid points for arrows.
    dark : bool
        Whether to use dark theme styling.
    """
    arrow_color = "white" if dark else "black"
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=base_field, x=x, y=y,
            colorscale="RdBu_r",
            colorbar=dict(title="Delta (meV)"),
            opacity=0.6,
        )
    )

    # Subsample for arrows
    xs, ys = x[::skip], y[::skip]
    jxs, jys = jx[::skip, ::skip], jy[::skip, ::skip]

    # Scale factor for arrow length
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
        xaxis_title="x (Angstrom)",
        yaxis_title="y (Angstrom)",
        yaxis_scaleanchor="x",
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
            colorbar=dict(title="chi (a.u.)"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x (Angstrom)",
        yaxis_title="y (Angstrom)",
        yaxis_scaleanchor="x",
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
    y_label: str = "Delta (meV)",
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
            name="a_v (Angstrom)",
            mode="lines",
            line=dict(color=primary),
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
    )
    fig.update_yaxes(title_text="Vortex period (Angstrom)", secondary_y=False)
    fig.update_yaxes(title_text="a_v / L_m", secondary_y=True)
    return fig
