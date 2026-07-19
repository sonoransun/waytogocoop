"""Fourier analysis page — FFT power spectrum and peak detection."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html

from waytogocoop.components.controls import loading_spinner, open_in_viewer_button
from waytogocoop.components.figure_factory import create_fft_heatmap
from waytogocoop.components.material_selector import create_material_selector
from waytogocoop.components.parameter_panel import create_parameter_panel
from waytogocoop.computation.fourier import fft_2d, identify_peaks
from waytogocoop.computation.moire import generate_moire_pattern
from waytogocoop.config import DEFAULT_FFT_THRESHOLD_FRACTION
from waytogocoop.materials.database import get_material
from waytogocoop.state import register_url_sync

dash.register_page(
    __name__,
    path="/fourier",
    name="Fourier Analysis",
    title="Good Job Coop! - Fourier",
)

_PREFIX = "fourier"
_URL_ID = f"{_PREFIX}-url"

_VIEWER_BINDINGS = [
    (f"{_PREFIX}-substrate-dropdown", "value", "substrate"),
    (f"{_PREFIX}-overlayer-dropdown", "value", "overlayer"),
    (f"{_PREFIX}-twist-slider", "value", "twist"),
    (f"{_PREFIX}-grid-size", "value", "grid_size"),
    (f"{_PREFIX}-physical-extent", "value", "extent"),
]

layout = dbc.Container(
    [
        dcc.Location(id=_URL_ID, refresh=False),
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
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Peak Detection", className="card-title"),
                                    dbc.Label("Threshold (fraction of max power)"),
                                    dcc.Slider(
                                        id=f"{_PREFIX}-peak-threshold",
                                        min=0.05,
                                        max=0.95,
                                        step=0.05,
                                        value=DEFAULT_FFT_THRESHOLD_FRACTION,
                                        marks={0.05: "0.05", 0.3: "0.3", 0.6: "0.6", 0.95: "0.95"},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                ]
                            ),
                            className="mb-3",
                        ),
                    ],
                    xs=12, md=4, lg=3,
                ),
                # Right column — FFT figure and peaks table
                dbc.Col(
                    [
                        loading_spinner(
                            dcc.Graph(id=f"{_PREFIX}-fft-graph"),
                            "Computing FFT power spectrum…",
                        ),
                        html.Br(),
                        html.Div(
                            [
                                html.H5("Detected Peaks", style={"display": "inline-block"}),
                                html.Span(
                                    open_in_viewer_button(_PREFIX, _VIEWER_BINDINGS),
                                    style={"marginLeft": "16px"},
                                ),
                            ]
                        ),
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
                    xs=12, md=8, lg=9,
                ),
            ]
        ),
    ],
    fluid=True,
    className="p-4",
)


register_url_sync(
    _URL_ID,
    [
        (f"{_PREFIX}-substrate-dropdown", "value", "sub"),
        (f"{_PREFIX}-overlayer-dropdown", "value", "over"),
        (f"{_PREFIX}-twist-slider", "value", "tw"),
        (f"{_PREFIX}-grid-size", "value", "gs"),
        (f"{_PREFIX}-physical-extent", "value", "ext"),
        (f"{_PREFIX}-peak-threshold", "value", "th"),
    ],
)


@callback(
    Output(f"{_PREFIX}-fft-graph", "figure"),
    Output(f"{_PREFIX}-peaks-table", "data"),
    Input(f"{_PREFIX}-substrate-dropdown", "value"),
    Input(f"{_PREFIX}-overlayer-dropdown", "value"),
    Input(f"{_PREFIX}-twist-slider", "value"),
    Input(f"{_PREFIX}-grid-size", "value"),
    Input(f"{_PREFIX}-physical-extent", "value"),
    Input(f"{_PREFIX}-peak-threshold", "value"),
    Input("theme-store", "data"),
)
def _update_fourier(
    substrate_formula: str,
    overlayer_formula: str,
    twist_angle: float,
    grid_size: int,
    physical_extent: float,
    threshold_fraction: float,
    theme: str,
):
    try:
        dark = theme == "dark"
        substrate = get_material(substrate_formula)
        overlayer = get_material(overlayer_formula)

        grid_size = int(grid_size) if grid_size is not None else 200
        physical_extent = float(physical_extent) if physical_extent is not None else 100.0
        twist_angle = float(twist_angle) if twist_angle is not None else 0.0
        threshold = (
            float(threshold_fraction)
            if threshold_fraction is not None
            else DEFAULT_FFT_THRESHOLD_FRACTION
        )

        result = generate_moire_pattern(
            substrate_a=substrate.a,
            overlayer_a=overlayer.a,
            overlayer_lattice_type=overlayer.lattice_type,
            substrate_lattice_type=substrate.lattice_type,
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
            threshold_fraction=threshold,
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
