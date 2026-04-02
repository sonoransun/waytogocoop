"""Plotly figure constructors for moire visualisation."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_moire_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    pattern: np.ndarray,
    title: str = "Moire Pattern",
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
    )
    return fig


def create_gap_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    gap_field: np.ndarray,
    title: str = "Gap Modulation",
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
    )
    return fig


def create_fft_heatmap(
    kx: np.ndarray,
    ky: np.ndarray,
    power_spectrum: np.ndarray,
    title: str = "FFT Power Spectrum",
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
    )
    return fig


def create_sweep_plot(
    param_values: np.ndarray,
    periods: np.ndarray,
    amplitudes: np.ndarray,
    param_name: str = "Parameter",
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
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=param_values,
            y=periods,
            name="Moire period (A)",
            mode="lines+markers",
            line=dict(color="steelblue"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=param_values,
            y=amplitudes,
            name="CPDM amplitude",
            mode="lines+markers",
            line=dict(color="firebrick"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Parameter Sweep: {param_name}",
        xaxis_title=param_name,
        margin=dict(l=60, r=60, t=50, b=50),
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
    )
    return fig


def create_2d_contour(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    colorscale: str = "Viridis",
    z_label: str = "Intensity",
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
    )
    return fig
