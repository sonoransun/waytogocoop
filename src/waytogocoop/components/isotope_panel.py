"""Dash component for speculative isotope-effect controls."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, clientside_callback, dcc, html

from waytogocoop.config import DEFAULT_ISOTOPE_EXPONENT, EXOTIC_MASS_RANGES
from waytogocoop.materials.isotopes import (
    ELEMENTS,
    humanize_half_life,
    natural_average_mass,
    nearest_isotope_info,
)

_C_MASS_VISIBILITY_JS = """
function(substrate, overlayer) {
    const isG = (v) => typeof v === 'string' && v.indexOf('Graphene') === 0;
    return (isG(substrate) || isG(overlayer)) ? {} : {display: 'none'};
}
"""

_EXOTIC_WARNING = (
    "HIGHLY SPECULATIVE — synthetic isotopes are radioactive (several half-lives are "
    "far too short to grow or measure a film), and masses between/beyond known "
    "isotopes are purely hypothetical. Outputs are what-if illustrations only."
)

# Elements with a mass slider in the panel, in layout order
_SLIDER_ELEMENTS = ("Fe", "Te", "Sb", "C")

_registered_prefixes: set[str] = set()
_registered_exotic_prefixes: set[str] = set()


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


def _isotope_marks(elem_data):
    """Slider marks for the stable isotopes of an element."""
    return {
        iso.mass_number: {"label": str(iso.mass_number)}
        for iso in elem_data.isotopes
    }


def _exotic_marks(elem_data):
    """Slider marks for stable plus synthetic isotopes (synthetic starred)."""
    marks = _isotope_marks(elem_data)
    for iso in elem_data.synthetic_isotopes:
        marks[iso.mass_number] = {"label": f"{iso.mass_number}{elem_data.symbol}*"}
    return marks


def _nearest_isotope_text(symbol: str, mass: float | None) -> str:
    """Readout string for the nearest known isotope to a slider value."""
    if mass is None:
        return ""
    info = nearest_isotope_info(symbol, float(mass))
    if info.kind == "hypothetical":
        return "hypothetical mass — no known isotope"
    if info.kind == "synthetic":
        return f"≈ {info.label}* (t½ ≈ {humanize_half_life(info.half_life_s)})"
    return f"≈ {info.label}"


def _register_exotic_mode_callbacks(id_prefix: str) -> None:
    """Wire the exotic-mode switch and nearest-isotope readouts, on first use."""
    if id_prefix in _registered_exotic_prefixes:
        return
    _registered_exotic_prefixes.add(id_prefix)

    from dash import Output, State, callback, no_update

    range_outputs = []
    value_states = []
    for sym in _SLIDER_ELEMENTS:
        slider_id = f"{id_prefix}-{sym.lower()}-mass"
        range_outputs.extend(
            [
                Output(slider_id, "min"),
                Output(slider_id, "max"),
                Output(slider_id, "marks"),
                Output(slider_id, "value"),
            ]
        )
        value_states.append(State(slider_id, "value"))

    @callback(
        Output(f"{id_prefix}-iso-exotic-warning", "is_open"),
        *range_outputs,
        Input(f"{id_prefix}-iso-exotic-mode", "value"),
        *value_states,
    )
    def _toggle_exotic_mode(exotic_on, *slider_values):
        exotic_on = bool(exotic_on)
        outputs = [exotic_on]
        for sym, value in zip(_SLIDER_ELEMENTS, slider_values, strict=True):
            elem = ELEMENTS[sym]
            if exotic_on:
                lo, hi = EXOTIC_MASS_RANGES[sym]
                outputs.extend([lo, hi, _exotic_marks(elem), no_update])
            else:
                # Mirror the Rust app's clamp_overrides_to_stable: a mass left
                # outside the stable range would otherwise keep driving the
                # physics while the exotic warning is hidden.
                lo = elem.isotopes[0].atomic_mass
                hi = elem.isotopes[-1].atomic_mass
                clamped = (
                    min(max(float(value), lo), hi) if value is not None else no_update
                )
                outputs.extend([lo, hi, _isotope_marks(elem), clamped])
        return tuple(outputs)

    @callback(
        *[Output(f"{id_prefix}-iso-{s.lower()}-nearest", "children")
          for s in _SLIDER_ELEMENTS],
        *[Input(f"{id_prefix}-{s.lower()}-mass", "value")
          for s in _SLIDER_ELEMENTS],
    )
    def _update_nearest_readouts(fe_mass, te_mass, sb_mass, c_mass):
        values = (fe_mass, te_mass, sb_mass, c_mass)
        return tuple(
            _nearest_isotope_text(sym, val)
            for sym, val in zip(_SLIDER_ELEMENTS, values, strict=True)
        )


def create_isotope_panel(id_prefix: str) -> dbc.Card:
    """Return a collapsible Card with isotope enrichment controls.

    Exposes per-element mass sliders for Fe, Te, Sb (Bi is monoisotopic),
    a BCS isotope exponent slider, and a comparison toggle. A C-mass slider
    is shown only when a graphene-family material is selected. An exotic-
    isotope switch (HIGHLY SPECULATIVE) widens the slider ranges to include
    synthetic isotopes and hypothetical masses; each slider gets a nearest-
    known-isotope readout.

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
    _register_exotic_mode_callbacks(id_prefix)

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
                        # --- Exotic-isotope mode (HIGHLY SPECULATIVE) ---
                        dbc.Switch(
                            id=f"{id_prefix}-iso-exotic-mode",
                            label="Exotic isotopes (HIGHLY SPECULATIVE)",
                            value=False,
                            className="mb-1",
                        ),
                        dbc.Alert(
                            _EXOTIC_WARNING,
                            id=f"{id_prefix}-iso-exotic-warning",
                            color="danger",
                            is_open=False,
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
                        html.Div(
                            id=f"{id_prefix}-iso-fe-nearest",
                            className="small text-muted",
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
                        html.Div(
                            id=f"{id_prefix}-iso-te-nearest",
                            className="small text-muted",
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
                        html.Div(
                            id=f"{id_prefix}-iso-sb-nearest",
                            className="small text-muted",
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
                                html.Div(
                                    id=f"{id_prefix}-iso-c-nearest",
                                    className="small text-muted",
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
