"""Dash component for speculative isotope-effect controls."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, clientside_callback, dcc, html

from waytogocoop.config import DEFAULT_ISOTOPE_EXPONENT
from waytogocoop.materials.isotopes import ELEMENTS, natural_average_mass

_C_MASS_VISIBILITY_JS = """
function(substrate, overlayer) {
    const isG = (v) => typeof v === 'string' && v.indexOf('Graphene') === 0;
    return (isG(substrate) || isG(overlayer)) ? {} : {display: 'none'};
}
"""

_registered_prefixes: set[str] = set()


def _register_c_mass_visibility(id_prefix: str) -> None:
    """Show the C-mass slider only for graphene-family materials, on first use."""
    if id_prefix in _registered_prefixes:
        return
    _registered_prefixes.add(id_prefix)

    from dash import Output

    clientside_callback(
        _C_MASS_VISIBILITY_JS,
        Output(f"{id_prefix}-c-mass-container", "style"),
        Input(f"{id_prefix}-substrate-dropdown", "value"),
        Input(f"{id_prefix}-overlayer-dropdown", "value"),
    )


def create_isotope_panel(id_prefix: str) -> dbc.Card:
    """Return a collapsible Card with isotope enrichment controls.

    Exposes per-element mass sliders for Fe, Te, Sb (Bi is monoisotopic),
    a BCS isotope exponent slider, and a comparison toggle. A C-mass slider
    is shown only when a graphene-family material is selected.

    Parameters
    ----------
    id_prefix : str
        Prefix for Dash component IDs.
    """
    fe = ELEMENTS["Fe"]
    te = ELEMENTS["Te"]
    sb = ELEMENTS["Sb"]
    c = ELEMENTS["C"]

    fe_nat = natural_average_mass("Fe")
    te_nat = natural_average_mass("Te")
    sb_nat = natural_average_mass("Sb")
    c_nat = natural_average_mass("C")

    _register_c_mass_visibility(id_prefix)

    def _isotope_marks(elem_data):
        return {
            iso.mass_number: {"label": str(iso.mass_number)}
            for iso in elem_data.isotopes
        }

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Isotope Effects (Speculative)", className="card-title"),
                dbc.Checklist(
                    id=f"{id_prefix}-isotope-enabled",
                    options=[{"label": "Enable", "value": "on"}],
                    value=[],
                    inline=True,
                    className="mb-2",
                ),
                html.Div(
                    id=f"{id_prefix}-isotope-controls",
                    children=[
                        dbc.Alert(
                            "SPECULATIVE - simplified models, qualitative results only.",
                            color="danger",
                            className="py-1 px-2 mb-2",
                            style={"fontSize": "0.8em"},
                        ),
                        # --- Fe mass ---
                        dbc.Label(f"Fe mass (amu) — natural: {fe_nat:.2f}"),
                        dcc.Slider(
                            id=f"{id_prefix}-fe-mass",
                            min=fe.isotopes[0].atomic_mass,
                            max=fe.isotopes[-1].atomic_mass,
                            step=0.01,
                            value=fe_nat,
                            marks=_isotope_marks(fe),
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),
                        # --- Te mass ---
                        dbc.Label(f"Te mass (amu) — natural: {te_nat:.2f}"),
                        dcc.Slider(
                            id=f"{id_prefix}-te-mass",
                            min=te.isotopes[0].atomic_mass,
                            max=te.isotopes[-1].atomic_mass,
                            step=0.01,
                            value=te_nat,
                            marks=_isotope_marks(te),
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),
                        # --- Sb mass ---
                        dbc.Label(f"Sb mass (amu) — natural: {sb_nat:.2f}"),
                        dcc.Slider(
                            id=f"{id_prefix}-sb-mass",
                            min=sb.isotopes[0].atomic_mass,
                            max=sb.isotopes[-1].atomic_mass,
                            step=0.01,
                            value=sb_nat,
                            marks=_isotope_marks(sb),
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),
                        # --- C mass (graphene-family materials only) ---
                        html.Div(
                            id=f"{id_prefix}-c-mass-container",
                            style={"display": "none"},
                            children=[
                                dbc.Label(f"C mass (amu) — natural: {c_nat:.2f}"),
                                dcc.Slider(
                                    id=f"{id_prefix}-c-mass",
                                    min=c.isotopes[0].atomic_mass,
                                    max=c.isotopes[-1].atomic_mass,
                                    step=0.001,
                                    value=c_nat,
                                    marks=_isotope_marks(c),
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                                html.Br(),
                            ],
                        ),
                        # --- BCS isotope exponent ---
                        dbc.Label("BCS isotope exponent (α)"),
                        dcc.Slider(
                            id=f"{id_prefix}-isotope-alpha",
                            min=-0.5,
                            max=1.0,
                            step=0.01,
                            value=DEFAULT_ISOTOPE_EXPONENT,
                            marks={
                                -0.18: "-0.18 (inverse)",
                                0: "0",
                                0.4: "0.4 (consensus)",
                                0.5: "0.5 (BCS)",
                                0.81: "0.81 (FeSe)",
                            },
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),
                        # --- Comparison toggle ---
                        dbc.Checklist(
                            id=f"{id_prefix}-isotope-comparison",
                            options=[
                                {
                                    "label": "Show natural vs. enriched comparison",
                                    "value": "on",
                                }
                            ],
                            value=[],
                            inline=True,
                        ),
                        html.Hr(),
                        # --- Computed effects readout ---
                        html.Div(id=f"{id_prefix}-isotope-info"),
                    ],
                ),
            ]
        ),
        className="mb-3",
    )
