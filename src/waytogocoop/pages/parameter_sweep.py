"""Parameter sweep page — sweep lattice ratio or twist angle."""

from __future__ import annotations

import numpy as np
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

from waytogocoop.components.figure_factory import create_sweep_plot
from waytogocoop.computation.moire import moire_periodicity_1d, moire_periodicity_with_twist
from waytogocoop.computation.superconducting import cpdm_amplitude
from waytogocoop.materials.database import get_material

dash.register_page(
    __name__, path="/sweep", name="Parameter Sweep", title="Good Job Coop! - Sweep"
)

layout = dbc.Container(
    [
        html.Br(),
        html.H2("Parameter Sweep"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Sweep Configuration"),
                                dbc.Label("Sweep parameter"),
                                dcc.Dropdown(
                                    id="sweep-param-dropdown",
                                    options=[
                                        {
                                            "label": "Overlayer lattice constant",
                                            "value": "lattice_constant",
                                        },
                                        {
                                            "label": "Twist angle",
                                            "value": "twist_angle",
                                        },
                                    ],
                                    value="lattice_constant",
                                    clearable=False,
                                ),
                                html.Br(),
                                dbc.Label("Substrate"),
                                dbc.Input(
                                    id="sweep-substrate-a",
                                    type="number",
                                    value=3.82,
                                    step=0.01,
                                ),
                                html.Br(),
                                dbc.Label("Range start"),
                                dbc.Input(
                                    id="sweep-range-start",
                                    type="number",
                                    value=3.9,
                                    step=0.01,
                                ),
                                html.Br(),
                                dbc.Label("Range end"),
                                dbc.Input(
                                    id="sweep-range-end",
                                    type="number",
                                    value=5.0,
                                    step=0.01,
                                ),
                                html.Br(),
                                dbc.Label("Number of points"),
                                dbc.Input(
                                    id="sweep-num-points",
                                    type="number",
                                    value=100,
                                    min=10,
                                    max=500,
                                    step=10,
                                ),
                                html.Br(),
                                dbc.Button(
                                    "Run Sweep",
                                    id="sweep-run-button",
                                    color="primary",
                                    className="w-100",
                                ),
                            ]
                        )
                    ),
                    md=3,
                ),
                dbc.Col(
                    [
                        dcc.Loading(dcc.Graph(id="sweep-graph")),
                        html.Div(id="sweep-material-markers"),
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
    Output("sweep-graph", "figure"),
    Output("sweep-material-markers", "children"),
    Input("sweep-run-button", "n_clicks"),
    State("sweep-param-dropdown", "value"),
    State("sweep-substrate-a", "value"),
    State("sweep-range-start", "value"),
    State("sweep-range-end", "value"),
    State("sweep-num-points", "value"),
    prevent_initial_call=False,
)
def _run_sweep(n_clicks, sweep_param, substrate_a, range_start, range_end, num_points):
    substrate_a = float(substrate_a or 3.82)
    range_start = float(range_start or 3.9)
    range_end = float(range_end or 5.0)
    num_points = int(num_points or 100)

    param_values = np.linspace(range_start, range_end, num_points)
    periods = np.empty(num_points)
    amplitudes = np.empty(num_points)

    if sweep_param == "twist_angle":
        # Sweep twist angle in degrees; range values are degrees
        for i, theta in enumerate(param_values):
            if abs(theta) < 1e-6:
                periods[i] = np.inf
                amplitudes[i] = 0.0
            else:
                periods[i] = moire_periodicity_with_twist(substrate_a, theta)
                amplitudes[i] = cpdm_amplitude(periods[i])
        param_name = "Twist angle (deg)"
    else:
        # Sweep overlayer lattice constant
        for i, a_over in enumerate(param_values):
            if abs(a_over - substrate_a) < 1e-12:
                periods[i] = np.inf
                amplitudes[i] = 0.0
            else:
                periods[i] = moire_periodicity_1d(substrate_a, a_over)
                amplitudes[i] = cpdm_amplitude(periods[i])
        param_name = "Overlayer lattice constant (A)"

    # Cap infinite periods for plotting
    finite_mask = np.isfinite(periods)
    if finite_mask.any():
        max_period = np.max(periods[finite_mask]) * 1.1
    else:
        max_period = 1000.0
    periods = np.where(finite_mask, periods, max_period)

    fig = create_sweep_plot(param_values, periods, amplitudes, param_name)

    # Mark actual material lattice constants
    markers_info = []
    if sweep_param == "lattice_constant":
        for formula in ("Sb2Te3", "Bi2Te3", "Sb2Te"):
            mat = get_material(formula)
            if range_start <= mat.a <= range_end:
                period_val = moire_periodicity_1d(substrate_a, mat.a)
                amp_val = cpdm_amplitude(period_val)
                fig.add_vline(
                    x=mat.a,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=mat.formula,
                )
                markers_info.append(
                    html.Li(
                        f"{mat.formula}: a = {mat.a} A, "
                        f"moire period = {period_val:.2f} A, "
                        f"CPDM = {amp_val:.4f}"
                    )
                )

    marker_div = (
        dbc.Card(
            dbc.CardBody(
                [html.H5("Material Reference Points"), html.Ul(markers_info)]
            ),
            className="mt-3",
        )
        if markers_info
        else html.Div()
    )

    return fig, marker_div
