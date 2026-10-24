"""Dash component for pair-density-wave (PDW) model controls."""

from __future__ import annotations

import math

import dash_bootstrap_components as dbc
from dash import dcc, html

from waytogocoop.components.controls import labeled_with_help
from waytogocoop.config import (
    PDW_KAPPA_H,
    PDW_KAPPA_HARM,
    PDW_KAPPA_P,
    PDW_KAPPA_Q,
    PDW_VISIBILITY_FLOOR,
    T_PDW_UTE2,
    TC_UTE2,
)

GEOMETRY_LABELS: dict[str, str] = {
    "ute2": "UTe2 surface (Zhu et al. 2026)",
    "moire": "Moire-locked heterostructure (what-if)",
}

# Field axes follow the paper's vector magnet: x ~ c*, y = a, z ~ b*.
FIELD_AXES: dict[str, tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}
FIELD_AXIS_LABELS: dict[str, str] = {
    "x": "Bx (≈ c* in UTe2)",
    "y": "By (= a in UTe2)",
    "z": "Bz (≈ b* in UTe2)",
    "along": "Along the applied B",
}

VIEW_OPTIONS: list[dict[str, str]] = [
    {"label": "Pair amplitude |d(r)|", "value": "pair"},
    {"label": "Charge density ρ(r)", "value": "charge"},
    {"label": "Spin density S_z(r) (speculative)", "value": "spin"},
    {"label": "STM-style FFT of ρ(r)", "value": "fft"},
    {"label": "Peak map (k-space)", "value": "peaks"},
]

T_PDW_RATIO_UTE2: float = T_PDW_UTE2 / TC_UTE2


def _field_input(id_prefix: str, axis: str) -> dbc.InputGroup:
    return dbc.InputGroup(
        [
            dbc.InputGroupText(FIELD_AXIS_LABELS[axis], style={"minWidth": "9.5em"}),
            dbc.Input(
                id=f"{id_prefix}-b{axis}",
                type="number",
                value=0.0,
                min=0,
                max=60,
                step=0.05,
            ),
            dbc.InputGroupText("T"),
        ],
        size="sm",
        className="mb-1",
    )


def _coupling_slider(id_prefix: str, name: str, label: str, value: float) -> list:
    return [
        dbc.Label(label, className="small mb-0"),
        dcc.Slider(
            id=f"{id_prefix}-{name}",
            min=0,
            max=2,
            step=0.05,
            value=value,
            marks={0: "0", 1: "1", 2: "2"},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ]


def create_pdw_panel(id_prefix: str) -> dbc.Card:
    """Return a Card with PDW geometry, temperature, field and coupling controls.

    Parameters
    ----------
    id_prefix : str
        Prefix for Dash component IDs.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Pair Density Wave", className="card-title"),
                dbc.Alert(
                    "SPECULATIVE — Landau model of composite orders, not a fit.",
                    color="danger",
                    className="py-1 px-2 mb-2",
                    style={"fontSize": "0.8em"},
                ),
                dbc.Label("Geometry"),
                dcc.RadioItems(
                    id=f"{id_prefix}-geometry",
                    options=[{"label": v, "value": k} for k, v in GEOMETRY_LABELS.items()],
                    value="ute2",
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"display": "block"},
                    className="mb-2",
                ),
                labeled_with_help(
                    "Temperature T (K)",
                    f"{id_prefix}-temperature-help",
                    "Uniform superconductivity sets in at Tc (2.1 K for UTe2, "
                    "13.5 K for the FeTe films); the PDW at T_PDW.",
                ),
                dcc.Slider(
                    id=f"{id_prefix}-temperature",
                    min=0,
                    max=6,
                    step=0.05,
                    value=0.3,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Br(),
                labeled_with_help(
                    "T_PDW / Tc",
                    f"{id_prefix}-tpdw-help",
                    "PDW onset relative to the uniform Tc. Above 1 opens the "
                    "'Cheshire cat' window where only the q = 2p and h charge "
                    "orders survive. UTe2: ~5 K / 2.1 K. A moire-imprinted CPDM "
                    "has no window (ratio 1).",
                ),
                dcc.Slider(
                    id=f"{id_prefix}-tpdw-ratio",
                    min=0.5,
                    max=3.0,
                    step=0.01,
                    value=T_PDW_RATIO_UTE2,
                    marks={1: "1", T_PDW_RATIO_UTE2: "UTe2", 3: "3"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Br(),
                labeled_with_help(
                    "Applied field",
                    f"{id_prefix}-field-help",
                    "Vector field in the paper's magnet frame. UTe2 critical "
                    "fields follow b* > c* > a; the field also rotates the "
                    "triplet d-vectors, switching on the p1 / p2 charge order.",
                ),
                _field_input(id_prefix, "x"),
                _field_input(id_prefix, "y"),
                _field_input(id_prefix, "z"),
                dbc.Label("Field-sweep axis", className="mt-2"),
                dcc.Dropdown(
                    id=f"{id_prefix}-sweep-axis",
                    options=[{"label": v, "value": k} for k, v in FIELD_AXIS_LABELS.items()],
                    value="z",
                    clearable=False,
                ),
                html.Hr(),
                dbc.Button(
                    "Landau couplings",
                    id=f"{id_prefix}-couplings-toggle",
                    color="secondary",
                    size="sm",
                    className="mb-2",
                    n_clicks=0,
                ),
                dbc.Collapse(
                    id=f"{id_prefix}-couplings-collapse",
                    is_open=False,
                    children=[
                        *_coupling_slider(
                            id_prefix, "kappa-p", "κ_p — p CDW (PDW × uniform)", PDW_KAPPA_P
                        ),
                        *_coupling_slider(
                            id_prefix, "kappa-q", "κ_q — q = 2p CDW (PDW alone)", PDW_KAPPA_Q
                        ),
                        *_coupling_slider(
                            id_prefix, "kappa-h", "κ_h — h harmonics", PDW_KAPPA_H
                        ),
                        *_coupling_slider(
                            id_prefix,
                            "kappa-harm",
                            "κ_harm — q as a harmonic of p",
                            PDW_KAPPA_HARM,
                        ),
                        labeled_with_help(
                            "Detection floor log₁₀(|ρ|²)",
                            f"{id_prefix}-floor-help",
                            "Peaks weaker than this are treated as FFT background. "
                            "The default -2 reproduces the paper's onset temperatures.",
                        ),
                        dcc.Slider(
                            id=f"{id_prefix}-floor",
                            min=-4,
                            max=-1,
                            step=0.1,
                            value=round(math.log10(PDW_VISIBILITY_FLOOR), 1),
                            marks={-4: "-4", -3: "-3", -2: "-2", -1: "-1"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
            ]
        ),
        className="mb-3",
    )
