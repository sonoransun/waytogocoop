"""Parameter sweep page — sweep lattice ratio or twist angle."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from waytogocoop.components.controls import loading_spinner
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
                    xs=12, md=4, lg=3,
                ),
                dbc.Col(
                    [
                        loading_spinner(
                            dcc.Graph(id="sweep-graph"),
                            "Computing parameter sweep…",
                        ),
                        html.Div(id="sweep-material-markers"),
                    ],
                    xs=12, md=8, lg=9,
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
    Input("theme-store", "data"),
    prevent_initial_call=False,
)
def _run_sweep(n_clicks, sweep_param, substrate_a, range_start, range_end, num_points, theme):
    try:
        dark = theme == "dark"
        substrate_a = float(substrate_a) if substrate_a is not None else 3.82
        range_start = float(range_start) if range_start is not None else 3.9
        range_end = float(range_end) if range_end is not None else 5.0
        num_points = int(num_points) if num_points is not None else 100

        param_values = np.linspace(range_start, range_end, num_points)

        if sweep_param == "twist_angle":
            # Sweep twist angle in degrees — vectorized
            periods = moire_periodicity_with_twist(substrate_a, param_values)
            amplitudes = cpdm_amplitude(periods)
            param_name = "Twist angle (deg)"
        else:
            # Sweep overlayer lattice constant — vectorized
            periods = moire_periodicity_1d(substrate_a, param_values)
            amplitudes = cpdm_amplitude(periods)
            param_name = "Overlayer lattice constant (A)"

        # Cap infinite periods for plotting
        finite_mask = np.isfinite(periods)
        if finite_mask.any():
            max_period = np.max(periods[finite_mask]) * 1.1
        else:
            max_period = 1000.0
        periods = np.where(finite_mask, periods, max_period)

        fig = create_sweep_plot(param_values, periods, amplitudes, param_name, dark=dark)

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
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Computation error: {e}")
        return error_fig, html.P(str(e), style={"color": "red"})
