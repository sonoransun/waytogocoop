"""Dash component for selecting substrate and overlayer materials."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from waytogocoop.materials.database import list_materials


def create_material_selector(id_prefix: str) -> dbc.Card:
    """Return a Card containing substrate and overlayer dropdowns.

    Parameters
    ----------
    id_prefix : str
        Prefix for Dash component IDs (e.g. ``"viewer"`` gives
        ``"viewer-substrate-dropdown"``).
    """
    substrates = list_materials(role="substrate")
    overlayers = list_materials(role="overlayer")

    substrate_options = [
        {"label": f"{m.formula} ({m.name})", "value": m.formula}
        for m in substrates
    ]
    overlayer_options = [
        {"label": f"{m.formula} ({m.name})", "value": m.formula}
        for m in overlayers
    ]

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Materials", className="card-title"),
                dbc.Label("Substrate"),
                dcc.Dropdown(
                    id=f"{id_prefix}-substrate-dropdown",
                    options=substrate_options,
                    value="FeTe",
                    clearable=False,
                ),
                html.Br(),
                dbc.Label("Overlayer"),
                dcc.Dropdown(
                    id=f"{id_prefix}-overlayer-dropdown",
                    options=overlayer_options,
                    value="Sb2Te3",
                    clearable=False,
                ),
            ]
        ),
        className="mb-3",
    )
