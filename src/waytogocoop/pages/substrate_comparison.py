"""Substrate Comparison page — side-by-side view of all overlayer materials on FeTe."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from waytogocoop.components.figure_factory import (
    create_2d_contour,
    create_3d_surface,
    create_gap_heatmap,
    create_moire_heatmap,
)
from waytogocoop.computation.moire import generate_moire_pattern
from waytogocoop.computation.superconducting import cpdm_amplitude, gap_modulation
from waytogocoop.config import DELTA_AMPLITUDE, DELTA_AVG
from waytogocoop.materials.database import get_material

dash.register_page(
    __name__,
    path="/comparison",
    name="Substrate Comparison",
    title="Good Job Coop! - Comparison",
)

_PREFIX = "comparison"

# Fixed overlayer formulas to display (all three overlayers on FeTe)
_OVERLAYER_FORMULAS = ["Sb2Te3", "Bi2Te3", "Sb2Te"]


def _make_column(idx: int, formula: str) -> dbc.Col:
    """Build a single material column with header and two figure placeholders."""
    mat = get_material(formula)
    return dbc.Col(
        [
            html.H5(
                f"{mat.formula} ({mat.name})",
                className="text-center",
            ),
            html.P(
                f"a = {mat.a} A | {mat.lattice_type}",
                className="text-center text-muted",
            ),
            dcc.Loading(dcc.Graph(id=f"{_PREFIX}-moire-{idx}")),
            dcc.Loading(dcc.Graph(id=f"{_PREFIX}-gap-{idx}")),
            html.Div(id=f"{_PREFIX}-info-{idx}", className="mt-2"),
        ],
        md=4,
    )


layout = dbc.Container(
    [
        html.Br(),
        html.H2("Substrate Comparison"),
        html.P(
            "All three overlayer materials on FeTe, shown side-by-side "
            "with shared parameter controls."
        ),
        html.Hr(),
        # View mode toggle
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
        # Parameter controls
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Parameters", className="card-title"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Twist angle (deg)"),
                                    dcc.Slider(
                                        id=f"{_PREFIX}-twist-slider",
                                        min=0,
                                        max=30,
                                        step=0.1,
                                        value=0.0,
                                        marks={i: str(i) for i in range(0, 31, 5)},
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Grid size"),
                                    dbc.Input(
                                        id=f"{_PREFIX}-grid-size",
                                        type="number",
                                        value=100,
                                        min=50,
                                        max=500,
                                        step=50,
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Physical extent (Angstrom)"),
                                    dbc.Input(
                                        id=f"{_PREFIX}-physical-extent",
                                        type="number",
                                        value=100.0,
                                        min=10,
                                        max=1000,
                                        step=10,
                                    ),
                                ],
                                md=3,
                            ),
                        ]
                    ),
                ]
            ),
            className="mb-3",
        ),
        # Three-column material comparison
        dbc.Row(
            [_make_column(i, formula) for i, formula in enumerate(_OVERLAYER_FORMULAS)]
        ),
    ],
    fluid=True,
    className="p-4",
)


@callback(
    # 3 moire figures + 3 gap figures + 3 info divs = 9 outputs
    Output(f"{_PREFIX}-moire-0", "figure"),
    Output(f"{_PREFIX}-moire-1", "figure"),
    Output(f"{_PREFIX}-moire-2", "figure"),
    Output(f"{_PREFIX}-gap-0", "figure"),
    Output(f"{_PREFIX}-gap-1", "figure"),
    Output(f"{_PREFIX}-gap-2", "figure"),
    Output(f"{_PREFIX}-info-0", "children"),
    Output(f"{_PREFIX}-info-1", "children"),
    Output(f"{_PREFIX}-info-2", "children"),
    Input(f"{_PREFIX}-twist-slider", "value"),
    Input(f"{_PREFIX}-grid-size", "value"),
    Input(f"{_PREFIX}-physical-extent", "value"),
    Input(f"{_PREFIX}-view-mode", "value"),
    Input("theme-store", "data"),
)
def _update_comparison(
    twist_angle: float,
    grid_size: int,
    physical_extent: float,
    view_mode: str,
    theme: str,
):
    try:
        dark = theme == "dark"
        grid_size = int(grid_size) if grid_size is not None else 100
        physical_extent = float(physical_extent) if physical_extent is not None else 100.0
        twist_angle = float(twist_angle) if twist_angle is not None else 0.0
        view_mode = view_mode or "heatmap"

        substrate = get_material("FeTe")

        moire_figs = []
        gap_figs = []
        info_divs = []

        for formula in _OVERLAYER_FORMULAS:
            overlayer = get_material(formula)

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
            moire_title = f"Moire: FeTe / {overlayer.formula}"
            gap_title = f"Gap: FeTe / {overlayer.formula}"

            if view_mode == "contour":
                moire_fig = create_2d_contour(
                    x, y, pattern, title=moire_title,
                    colorscale="Viridis", z_label="Intensity",
                    dark=dark,
                )
                gap_fig = create_2d_contour(
                    x, y, gap, title=gap_title,
                    colorscale="RdBu_r", z_label="Delta (meV)",
                    dark=dark,
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
                    dark=dark,
                )
                gap_fig = create_3d_surface(
                    x_s, y_s, gap_s, title=gap_title,
                    colorscale="RdBu_r", z_label="Delta (meV)",
                    dark=dark,
                )
            else:
                moire_fig = create_moire_heatmap(
                    x, y, pattern, title=moire_title,
                    dark=dark,
                )
                gap_fig = create_gap_heatmap(
                    x, y, gap, title=gap_title,
                    dark=dark,
                )

            moire_figs.append(moire_fig)
            gap_figs.append(gap_fig)

            mismatch = abs(substrate.a - overlayer.a) / substrate.a * 100.0
            cpdm_amp = cpdm_amplitude(result["moire_period"])

            info = [
                html.P(f"Lattice mismatch: {mismatch:.2f}%"),
                html.P(f"Moire period: {result['moire_period']:.2f} A"),
                html.P(f"CPDM amplitude: {cpdm_amp:.4f}"),
            ]
            info_divs.append(info)

        return (
            moire_figs[0], moire_figs[1], moire_figs[2],
            gap_figs[0], gap_figs[1], gap_figs[2],
            info_divs[0], info_divs[1], info_divs[2],
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Computation error: {e}")
        error_info = html.P(str(e), style={"color": "red"})
        return (
            error_fig, error_fig, error_fig,
            error_fig, error_fig, error_fig,
            error_info, error_info, error_info,
        )
