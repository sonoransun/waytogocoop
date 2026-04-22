"""Dash component for magnetic field and topological controls."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def create_magnetic_panel(id_prefix: str) -> dbc.Card:
    """Return a Card with magnetic field and proximity controls.

    Parameters
    ----------
    id_prefix : str
        Prefix for Dash component IDs.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Magnetic / Topological", className="card-title"),
                dbc.Alert(
                    "Includes SPECULATIVE features — simplified models.",
                    color="danger",
                    className="py-1 px-2 mb-2",
                    style={"fontSize": "0.8em"},
                ),
                # --- Perpendicular field Bz ---
                dbc.Label("Perpendicular field Bz (Tesla)"),
                dcc.Slider(
                    id=f"{id_prefix}-bz",
                    min=0,
                    max=20,
                    step=0.1,
                    value=0.5,
                    marks={0: "0", 5: "5", 10: "10", 15: "15", 20: "20"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Br(),
                # --- In-plane field ---
                dbc.Button(
                    "In-Plane Field",
                    id=f"{id_prefix}-inplane-toggle",
                    color="secondary",
                    size="sm",
                    className="mb-2",
                    n_clicks=0,
                ),
                dbc.Collapse(
                    id=f"{id_prefix}-inplane-collapse",
                    is_open=False,
                    children=[
                        dbc.Label("Bx (Tesla)"),
                        dcc.Slider(
                            id=f"{id_prefix}-bx",
                            min=-10,
                            max=10,
                            step=0.1,
                            value=0.0,
                            marks={-10: "-10", 0: "0", 10: "10"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        dbc.Label("By (Tesla)"),
                        dcc.Slider(
                            id=f"{id_prefix}-by",
                            min=-10,
                            max=10,
                            step=0.1,
                            value=0.0,
                            marks={-10: "-10", 0: "0", 10: "10"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                ),
                html.Hr(),
                # --- Proximity parameters ---
                dbc.Button(
                    "Proximity / Topological",
                    id=f"{id_prefix}-prox-toggle",
                    color="secondary",
                    size="sm",
                    className="mb-2",
                    n_clicks=0,
                ),
                dbc.Collapse(
                    id=f"{id_prefix}-prox-collapse",
                    is_open=False,
                    children=[
                        dbc.Label("Proximity coherence length (Angstrom)"),
                        dcc.Slider(
                            id=f"{id_prefix}-xi-prox",
                            min=10,
                            max=500,
                            step=10,
                            value=100,
                            marks={10: "10", 100: "100", 250: "250", 500: "500"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        dbc.Label("Interface transparency"),
                        dcc.Slider(
                            id=f"{id_prefix}-transparency",
                            min=0.1,
                            max=1.0,
                            step=0.05,
                            value=0.8,
                            marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        dbc.Label("g-factor"),
                        dcc.Slider(
                            id=f"{id_prefix}-g-factor",
                            min=1,
                            max=50,
                            step=1,
                            value=30,
                            marks={1: "1", 20: "20", 30: "30", 50: "50"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                ),
                html.Hr(),
                # --- Visualisation selector ---
                dbc.Label("View"),
                dcc.RadioItems(
                    id=f"{id_prefix}-viz-mode",
                    options=[
                        {"label": "Vortex Lattice", "value": "vortex"},
                        {"label": "Combined Gap", "value": "combined"},
                        {"label": "Screening Currents (3D cones)", "value": "currents"},
                        {"label": "Susceptibility (speculative)", "value": "chi"},
                        {"label": "Majorana 2D (speculative)", "value": "majorana"},
                        {"label": "Majorana 3D (speculative)", "value": "majorana3d"},
                    ],
                    value="vortex",
                    className="mb-2",
                ),
                html.Hr(),
                html.Div(id=f"{id_prefix}-mag-info"),
            ]
        ),
        className="mb-3",
    )
