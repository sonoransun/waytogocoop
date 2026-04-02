"""Dash component for computation parameter controls."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from waytogocoop.config import (
    DEFAULT_GRID_SIZE,
    DEFAULT_PHYSICAL_EXTENT,
    DEFAULT_TWIST_ANGLE,
)


def create_parameter_panel(id_prefix: str) -> dbc.Card:
    """Return a Card with twist-angle slider and grid controls.

    Parameters
    ----------
    id_prefix : str
        Prefix for Dash component IDs.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Parameters", className="card-title"),
                # Twist angle
                dbc.Label("Twist angle (deg)"),
                dcc.Slider(
                    id=f"{id_prefix}-twist-slider",
                    min=0,
                    max=30,
                    step=0.1,
                    value=DEFAULT_TWIST_ANGLE,
                    marks={i: str(i) for i in range(0, 31, 5)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Br(),
                # Grid size
                dbc.Label("Grid size"),
                dbc.Input(
                    id=f"{id_prefix}-grid-size",
                    type="number",
                    value=DEFAULT_GRID_SIZE,
                    min=50,
                    max=1000,
                    step=50,
                ),
                html.Br(),
                # Physical extent
                dbc.Label("Physical extent (Angstrom)"),
                dbc.Input(
                    id=f"{id_prefix}-physical-extent",
                    type="number",
                    value=DEFAULT_PHYSICAL_EXTENT,
                    min=10,
                    max=1000,
                    step=10,
                ),
            ]
        ),
        className="mb-3",
    )
