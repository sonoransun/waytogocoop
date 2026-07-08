"""Dash components for graphene stack and curvature controls."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from waytogocoop.components.controls import labeled_with_help
from waytogocoop.computation.curvature import CURVATURE_GEOMETRIES
from waytogocoop.computation.graphene import STACKING_PRESETS
from waytogocoop.materials.database import list_materials

STACK_LABELS: dict[str, str] = {
    "AA": "AA bilayer",
    "AB": "AB (Bernal) bilayer",
    "twisted_bilayer": "Twisted bilayer",
    "ABA": "ABA trilayer",
    "ABC": "ABC trilayer",
    "alternating_trilayer": "Alternating-twist trilayer",
}

CURVATURE_LABELS: dict[str, str] = {
    "flat": "Flat (planar)",
    "gaussian_bump": "Gaussian bump",
    "sinusoidal_ripple": "Sinusoidal ripple",
    "cylindrical_bend": "Cylindrical bend",
    "spherical_cap": "Spherical cap",
}


def create_graphene_stack_panel(id_prefix: str) -> dbc.Card:
    """Return a Card with graphene stacking and flat-band filling controls.

    Parameters
    ----------
    id_prefix : str
        Prefix for Dash component IDs.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Graphene Stack", className="card-title"),
                dbc.Alert(
                    "Flat-band gap and Tc models are SPECULATIVE - qualitative only.",
                    color="danger",
                    className="py-1 px-2 mb-2",
                    style={"fontSize": "0.8em"},
                ),
                dbc.Label("Stacking"),
                dcc.Dropdown(
                    id=f"{id_prefix}-stack-dropdown",
                    options=[
                        {"label": STACK_LABELS[s], "value": s}
                        for s in STACKING_PRESETS
                    ],
                    value="twisted_bilayer",
                    clearable=False,
                ),
                html.Br(),
                labeled_with_help(
                    "Filling |nu| (electrons per moire cell)",
                    f"{id_prefix}-filling-help",
                    "Electron filling of the flat band per moire unit cell; "
                    "superconductivity domes around |nu| ~ 2-3 in magic-angle "
                    "graphene.",
                ),
                dcc.Slider(
                    id=f"{id_prefix}-filling-slider",
                    min=0,
                    max=4,
                    step=0.1,
                    value=2.4,
                    marks={0: "0", 2: "2", 4: "4"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                dbc.Switch(
                    id=f"{id_prefix}-honeycomb-switch",
                    label="Honeycomb 2-atom basis",
                    value=False,
                    className="mt-2",
                ),
                dbc.Label("Valley"),
                dcc.RadioItems(
                    id=f"{id_prefix}-valley-radio",
                    options=[
                        {"label": "K", "value": 1},
                        {"label": "K'", "value": -1},
                    ],
                    value=1,
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "16px"},
                ),
                html.Br(),
                dbc.Label("Heterostrain ε (%)"),
                dcc.Slider(
                    id=f"{id_prefix}-strain-percent",
                    min=0,
                    max=2,
                    step=0.05,
                    value=0,
                    marks={0: "0", 1: "1", 2: "2"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                dbc.Label("Strain angle (° from zigzag)"),
                dcc.Slider(
                    id=f"{id_prefix}-strain-angle",
                    min=0,
                    max=90,
                    step=5,
                    value=0,
                    marks={0: "0", 30: "30", 60: "60", 90: "90"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Hr(),
                html.Div(id=f"{id_prefix}-stack-info"),
            ]
        ),
        className="mb-3",
    )


def create_supermoire_panel(id_prefix: str) -> dbc.Card:
    """Return a Card with supermoire overlayer controls.

    The overlayer dropdown offers the non-graphene overlayer materials from
    the database plus a "None" entry that disables the supermoire pathway.

    Parameters
    ----------
    id_prefix : str
        Prefix for Dash component IDs.
    """
    overlayers = [
        m for m in list_materials(role="overlayer")
        if not m.formula.startswith("Graphene")
    ]
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Supermoire Overlayer", className="card-title"),
                dbc.Label("Overlayer material"),
                dcc.Dropdown(
                    id=f"{id_prefix}-overlayer-dropdown",
                    options=[{"label": "None", "value": "none"}]
                    + [
                        {"label": f"{m.name} ({m.formula})", "value": m.formula}
                        for m in overlayers
                    ],
                    value="none",
                    clearable=False,
                ),
                html.Br(),
                dbc.Label("Interface twist (°)"),
                dcc.Slider(
                    id=f"{id_prefix}-interface-twist",
                    min=0,
                    max=30,
                    step=0.1,
                    value=0,
                    marks={0: "0", 10: "10", 20: "20", 30: "30"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        className="mb-3",
    )


def create_curvature_panel(id_prefix: str) -> dbc.Card:
    """Return a Card with sheet-curvature geometry controls.

    Parameters
    ----------
    id_prefix : str
        Prefix for Dash component IDs.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Sheet Curvature", className="card-title"),
                dbc.Alert(
                    "Pseudo-magnetic-field pair-breaking model is SPECULATIVE.",
                    color="danger",
                    className="py-1 px-2 mb-2",
                    style={"fontSize": "0.8em"},
                ),
                dbc.Label("Geometry"),
                dcc.Dropdown(
                    id=f"{id_prefix}-curvature-dropdown",
                    options=[
                        {"label": CURVATURE_LABELS[g], "value": g}
                        for g in CURVATURE_GEOMETRIES
                    ],
                    value="flat",
                    clearable=False,
                ),
                html.Br(),
                dbc.Label("Height h0 (A)"),
                dcc.Slider(
                    id=f"{id_prefix}-curv-amplitude",
                    min=0,
                    max=20,
                    step=0.5,
                    value=5,
                    marks={0: "0", 5: "5", 10: "10", 20: "20"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                dbc.Label("Feature size (A): sigma / wavelength / radius"),
                dcc.Slider(
                    id=f"{id_prefix}-curv-feature",
                    min=20,
                    max=2000,
                    step=10,
                    value=100,
                    marks={50: "50", 100: "100", 500: "500", 1000: "1000", 2000: "2000"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                dbc.Label("Orientation phi (deg from zigzag)"),
                dcc.Slider(
                    id=f"{id_prefix}-curv-orientation",
                    min=0,
                    max=60,
                    step=1,
                    value=30,
                    marks={0: "zigzag", 30: "armchair", 60: "60"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                dbc.Switch(
                    id=f"{id_prefix}-warp-switch",
                    label="Warp moire pattern (back-reaction, SPECULATIVE)",
                    value=False,
                    className="mt-2",
                ),
            ]
        ),
        className="mb-3",
    )
