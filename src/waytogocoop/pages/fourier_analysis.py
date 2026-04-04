"""Fourier analysis page — FFT power spectrum and peak detection."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html

from waytogocoop.components.figure_factory import create_fft_heatmap
from waytogocoop.components.material_selector import create_material_selector
from waytogocoop.components.parameter_panel import create_parameter_panel
from waytogocoop.computation.fourier import fft_2d, identify_peaks
from waytogocoop.computation.moire import generate_moire_pattern
from waytogocoop.materials.database import get_material

dash.register_page(
    __name__,
    path="/fourier",
    name="Fourier Analysis",
    title="Good Job Coop! - Fourier",
)

_PREFIX = "fourier"

layout = dbc.Container(
    [
        html.Br(),
        html.H2("Fourier Analysis"),
        html.Hr(),
        dbc.Row(
            [
                # Left column — controls
                dbc.Col(
                    [
                        create_material_selector(_PREFIX),
                        create_parameter_panel(_PREFIX),
                    ],
                    md=3,
                ),
                # Right column — FFT figure and peaks table
                dbc.Col(
                    [
                        dcc.Loading(dcc.Graph(id=f"{_PREFIX}-fft-graph")),
                        html.Br(),
                        html.H5("Detected Peaks"),
                        dash_table.DataTable(
                            id=f"{_PREFIX}-peaks-table",
                            columns=[
                                {"name": "kx (1/A)", "id": "kx"},
                                {"name": "ky (1/A)", "id": "ky"},
                                {"name": "Amplitude", "id": "amplitude"},
                            ],
                            data=[],
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "6px"},
                            style_header={
                                "fontWeight": "bold",
                                "backgroundColor": "#2c3e50",
                                "color": "white",
                            },
                        ),
                    ],
                    md=9,
                ),
            ]
        ),
    ],
    fluid=True,
    className="p-4",
)


@callback(
    Output(f"{_PREFIX}-fft-graph", "figure"),
    Output(f"{_PREFIX}-peaks-table", "data"),
    Input(f"{_PREFIX}-substrate-dropdown", "value"),
    Input(f"{_PREFIX}-overlayer-dropdown", "value"),
    Input(f"{_PREFIX}-twist-slider", "value"),
    Input(f"{_PREFIX}-grid-size", "value"),
    Input(f"{_PREFIX}-physical-extent", "value"),
    Input("theme-store", "data"),
)
def _update_fourier(
    substrate_formula: str,
    overlayer_formula: str,
    twist_angle: float,
    grid_size: int,
    physical_extent: float,
    theme: str,
):
    try:
        dark = theme == "dark"
        substrate = get_material(substrate_formula)
        overlayer = get_material(overlayer_formula)

        grid_size = int(grid_size) if grid_size is not None else 200
        physical_extent = float(physical_extent) if physical_extent is not None else 100.0
        twist_angle = float(twist_angle) if twist_angle is not None else 0.0

        result = generate_moire_pattern(
            substrate_a=substrate.a,
            overlayer_a=overlayer.a,
            overlayer_lattice_type=overlayer.lattice_type,
            twist_angle_deg=twist_angle,
            grid_size=grid_size,
            physical_extent=physical_extent,
        )

        dx = (result["x"][-1] - result["x"][0]) / (len(result["x"]) - 1)
        fft_result = fft_2d(result["pattern"], dx)

        fft_fig = create_fft_heatmap(
            fft_result["kx"],
            fft_result["ky"],
            fft_result["power_spectrum"],
            title=f"FFT: {substrate.formula} / {overlayer.formula}",
            dark=dark,
        )

        peaks = identify_peaks(
            fft_result["power_spectrum"],
            fft_result["kx"],
            fft_result["ky"],
        )

        peaks_data = [
            {
                "kx": f"{p['kx']:.4f}",
                "ky": f"{p['ky']:.4f}",
                "amplitude": f"{p['amplitude']:.2f}",
            }
            for p in peaks[:20]  # show at most 20 peaks
        ]

        return fft_fig, peaks_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Computation error: {e}")
        return error_fig, []
