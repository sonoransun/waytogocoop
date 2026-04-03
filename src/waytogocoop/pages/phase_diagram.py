"""Phase Diagram page — topological phase boundaries and commensuration."""

from __future__ import annotations

import numpy as np

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from waytogocoop.components.material_selector import create_material_selector
from waytogocoop.components.figure_factory import (
    create_phase_colormap,
    create_commensuration_sweep,
)
from waytogocoop.computation.topological import phase_diagram_sweep
from waytogocoop.computation.magnetic import vortex_lattice_period
from waytogocoop.computation.moire import moire_periodicity_1d
from waytogocoop.config import DELTA_AVG, G_FACTOR_TSS
from waytogocoop.materials.database import get_material

dash.register_page(
    __name__,
    path="/phase",
    name="Phase Diagram",
    title="Good Job Coop! - Phase Diagram",
)

_PREFIX = "phase"

layout = dbc.Container(
    [
        html.Br(),
        html.H2("Topological Phase Diagram (SPECULATIVE)"),
        dbc.Alert(
            "All results on this page are SPECULATIVE — simplified Fu-Kane "
            "criterion, qualitative only.",
            color="danger",
            className="mb-3",
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        create_material_selector(_PREFIX),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Sweep Parameters"),
                                    dbc.Label("B field range (Tesla)"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Input(
                                                    id=f"{_PREFIX}-b-min",
                                                    type="number",
                                                    value=0,
                                                    min=0,
                                                    step=1,
                                                ),
                                                width=6,
                                            ),
                                            dbc.Col(
                                                dbc.Input(
                                                    id=f"{_PREFIX}-b-max",
                                                    type="number",
                                                    value=100,
                                                    min=1,
                                                    step=1,
                                                ),
                                                width=6,
                                            ),
                                        ],
                                        className="mb-2",
                                    ),
                                    dbc.Label("Delta range (meV)"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Input(
                                                    id=f"{_PREFIX}-d-min",
                                                    type="number",
                                                    value=0.1,
                                                    min=0.01,
                                                    step=0.1,
                                                ),
                                                width=6,
                                            ),
                                            dbc.Col(
                                                dbc.Input(
                                                    id=f"{_PREFIX}-d-max",
                                                    type="number",
                                                    value=10.0,
                                                    min=0.1,
                                                    step=0.1,
                                                ),
                                                width=6,
                                            ),
                                        ],
                                        className="mb-2",
                                    ),
                                    dbc.Label("Chemical potential mu (meV)"),
                                    dcc.Slider(
                                        id=f"{_PREFIX}-mu",
                                        min=-5.0,
                                        max=5.0,
                                        step=0.1,
                                        value=0.0,
                                        marks={-5: "-5", 0: "0", 5: "5"},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Br(),
                                    dbc.Label("g-factor"),
                                    dcc.Slider(
                                        id=f"{_PREFIX}-g-factor",
                                        min=1,
                                        max=50,
                                        step=1,
                                        value=int(G_FACTOR_TSS),
                                        marks={1: "1", 20: "20", 30: "30", 50: "50"},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Br(),
                                    dbc.Label("Grid resolution"),
                                    dcc.Slider(
                                        id=f"{_PREFIX}-resolution",
                                        min=20,
                                        max=200,
                                        step=10,
                                        value=80,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                ]
                            ),
                            className="mb-3",
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        dcc.Graph(id=f"{_PREFIX}-phase-graph"),
                        html.Hr(),
                        dcc.Graph(id=f"{_PREFIX}-comm-graph"),
                    ],
                    md=9,
                ),
            ]
        ),
    ],
    fluid=True,
)


@callback(
    Output(f"{_PREFIX}-phase-graph", "figure"),
    Output(f"{_PREFIX}-comm-graph", "figure"),
    Input(f"{_PREFIX}-substrate", "value"),
    Input(f"{_PREFIX}-overlayer", "value"),
    Input(f"{_PREFIX}-b-min", "value"),
    Input(f"{_PREFIX}-b-max", "value"),
    Input(f"{_PREFIX}-d-min", "value"),
    Input(f"{_PREFIX}-d-max", "value"),
    Input(f"{_PREFIX}-mu", "value"),
    Input(f"{_PREFIX}-g-factor", "value"),
    Input(f"{_PREFIX}-resolution", "value"),
)
def update_phase(
    substrate_key, overlayer_key,
    b_min, b_max, d_min, d_max, mu, g_factor, resolution,
):
    substrate = get_material(substrate_key)
    overlayer = get_material(overlayer_key)

    b_min = float(b_min or 0)
    b_max = float(b_max or 100)
    d_min = float(d_min or 0.1)
    d_max = float(d_max or 10)
    mu = float(mu or 0)
    g_factor = float(g_factor or G_FACTOR_TSS)
    resolution = int(resolution or 80)

    # Phase diagram
    B_values = np.linspace(b_min, b_max, resolution)
    delta_values = np.linspace(d_min, d_max, resolution)
    phase = phase_diagram_sweep(B_values, delta_values, g_factor, mu)

    phase_fig = create_phase_colormap(
        B_values, delta_values, phase,
        title=f"Topological Phase: {overlayer.formula}/{substrate.formula}",
    )

    # Mark the material's actual delta
    phase_fig.add_hline(
        y=DELTA_AVG,
        line=dict(dash="dash", color="lime", width=2),
        annotation_text=f"Delta_avg = {DELTA_AVG:.2f} meV",
    )

    # Commensuration sweep
    moire_period = moire_periodicity_1d(substrate.a, overlayer.a)
    B_comm_range = np.linspace(max(b_min, 0.01), b_max, 200)
    a_v_values = np.array([vortex_lattice_period(b) for b in B_comm_range])
    # Replace inf with NaN for plotting
    a_v_values = np.where(np.isfinite(a_v_values), a_v_values, np.nan)

    comm_fig = create_commensuration_sweep(
        B_comm_range, a_v_values, moire_period,
        title=f"Commensuration: a_v/L_m ({overlayer.formula}/{substrate.formula})",
    )

    return phase_fig, comm_fig
