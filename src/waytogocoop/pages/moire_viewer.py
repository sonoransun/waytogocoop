"""Moire Viewer page — interactive real-space pattern and gap modulation."""

from __future__ import annotations

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from waytogocoop.components.material_selector import create_material_selector
from waytogocoop.components.parameter_panel import create_parameter_panel
from waytogocoop.components.figure_factory import (
    create_moire_heatmap,
    create_gap_heatmap,
    create_3d_surface,
    create_2d_contour,
)
from waytogocoop.computation.moire import generate_moire_pattern
from waytogocoop.computation.superconducting import gap_modulation, cpdm_amplitude
from waytogocoop.config import DELTA_AVG, DELTA_AMPLITUDE
from waytogocoop.materials.database import get_material

dash.register_page(
    __name__, path="/viewer", name="Moire Viewer", title="Good Job Coop! - Viewer"
)

_PREFIX = "viewer"

layout = dbc.Container(
    [
        html.Br(),
        html.H2("Moire Pattern Viewer"),
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
                                    html.H5("View Mode", className="card-title"),
                                    dcc.RadioItems(
                                        id=f"{_PREFIX}-view-mode",
                                        options=[
                                            {"label": "Heatmap", "value": "heatmap"},
                                            {"label": "Contour Map", "value": "contour"},
                                            {"label": "3D Surface", "value": "3d"},
                                        ],
                                        value="heatmap",
                                        inline=True,
                                        inputStyle={"marginRight": "4px"},
                                        labelStyle={"marginRight": "16px"},
                                    ),
                                ]
                            ),
                            className="mb-3",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Computed Info", className="card-title"),
                                    html.Div(id=f"{_PREFIX}-info"),
                                ]
                            ),
                            className="mb-3",
                        ),
                    ],
                    md=3,
                ),
                # Right column — figures
                dbc.Col(
                    [
                        dcc.Loading(dcc.Graph(id=f"{_PREFIX}-moire-graph")),
                        dcc.Loading(dcc.Graph(id=f"{_PREFIX}-gap-graph")),
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
    Output(f"{_PREFIX}-moire-graph", "figure"),
    Output(f"{_PREFIX}-gap-graph", "figure"),
    Output(f"{_PREFIX}-info", "children"),
    Input(f"{_PREFIX}-substrate-dropdown", "value"),
    Input(f"{_PREFIX}-overlayer-dropdown", "value"),
    Input(f"{_PREFIX}-twist-slider", "value"),
    Input(f"{_PREFIX}-grid-size", "value"),
    Input(f"{_PREFIX}-physical-extent", "value"),
    Input(f"{_PREFIX}-view-mode", "value"),
)
def _update_viewer(
    substrate_formula: str,
    overlayer_formula: str,
    twist_angle: float,
    grid_size: int,
    physical_extent: float,
    view_mode: str,
):
    substrate = get_material(substrate_formula)
    overlayer = get_material(overlayer_formula)

    grid_size = int(grid_size or 200)
    physical_extent = float(physical_extent or 100.0)
    twist_angle = float(twist_angle or 0.0)
    view_mode = view_mode or "heatmap"

    result = generate_moire_pattern(
        substrate_a=substrate.a,
        overlayer_a=overlayer.a,
        overlayer_lattice_type=overlayer.lattice_type,
        twist_angle_deg=twist_angle,
        grid_size=grid_size,
        physical_extent=physical_extent,
    )

    gap = gap_modulation(result["pattern"], DELTA_AVG, DELTA_AMPLITUDE)

    x, y = result["x"], result["y"]
    pattern = result["pattern"]
    moire_title = f"Moire: {substrate.formula} / {overlayer.formula}"
    gap_title = "Superconducting Gap Modulation"

    if view_mode == "contour":
        moire_fig = create_2d_contour(
            x, y, pattern, title=moire_title,
            colorscale="Viridis", z_label="Intensity",
        )
        gap_fig = create_2d_contour(
            x, y, gap, title=gap_title,
            colorscale="RdBu_r", z_label="Delta (meV)",
        )
    elif view_mode == "3d":
        # Subsample for performance when grid is large
        if grid_size > 150:
            x_s, y_s = x[::2], y[::2]
            pattern_s = pattern[::2, ::2]
            gap_s = gap[::2, ::2]
        else:
            x_s, y_s, pattern_s, gap_s = x, y, pattern, gap
        moire_fig = create_3d_surface(
            x_s, y_s, pattern_s, title=moire_title,
            colorscale="Viridis", z_label="Intensity",
        )
        gap_fig = create_3d_surface(
            x_s, y_s, gap_s, title=gap_title,
            colorscale="RdBu_r", z_label="Delta (meV)",
        )
    else:
        moire_fig = create_moire_heatmap(
            x, y, pattern, title=moire_title,
        )
        gap_fig = create_gap_heatmap(
            x, y, gap, title=gap_title,
        )

    mismatch = abs(substrate.a - overlayer.a) / substrate.a * 100.0
    cpdm_amp = cpdm_amplitude(result["moire_period"])

    info = [
        html.P(f"Substrate: {substrate.formula} (a = {substrate.a} A)"),
        html.P(f"Overlayer: {overlayer.formula} (a = {overlayer.a} A)"),
        html.P(f"Lattice mismatch: {mismatch:.2f}%"),
        html.P(f"Twist angle: {twist_angle:.1f} deg"),
        html.P(f"Moire period: {result['moire_period']:.2f} A"),
        html.P(f"CPDM amplitude: {cpdm_amp:.4f}"),
    ]

    return moire_fig, gap_fig, info
