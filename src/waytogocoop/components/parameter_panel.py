"""Dash component for computation parameter controls."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, State, clientside_callback, dcc, html

from waytogocoop.components.controls import labeled_with_help
from waytogocoop.config import (
    DEFAULT_GRID_SIZE,
    DEFAULT_PHYSICAL_EXTENT,
    DEFAULT_TWIST_ANGLE,
)

GRID_MIN, GRID_MAX = 50, 1000
EXTENT_MIN, EXTENT_MAX = 10, 1000

_VALIDATION_JS = """
function(value, minv, maxv) {
    if (value === null || value === undefined || value === '') return false;
    const v = Number(value);
    if (Number.isNaN(v)) return true;
    return v < minv || v > maxv;
}
"""

_registered_prefixes: set[str] = set()


def _register_validation(id_prefix: str) -> None:
    """Wire browser-side validation for the numeric inputs on first use."""
    if id_prefix in _registered_prefixes:
        return
    _registered_prefixes.add(id_prefix)

    from dash import Output

    clientside_callback(
        _VALIDATION_JS,
        Output(f"{id_prefix}-grid-size", "invalid"),
        Input(f"{id_prefix}-grid-size", "value"),
        State(f"{id_prefix}-grid-size", "min"),
        State(f"{id_prefix}-grid-size", "max"),
    )
    clientside_callback(
        _VALIDATION_JS,
        Output(f"{id_prefix}-physical-extent", "invalid"),
        Input(f"{id_prefix}-physical-extent", "value"),
        State(f"{id_prefix}-physical-extent", "min"),
        State(f"{id_prefix}-physical-extent", "max"),
    )


def create_parameter_panel(id_prefix: str) -> dbc.Card:
    """Return a Card with twist-angle, grid and extent controls.

    Hover the ⓘ badge next to any label for a short description of the
    parameter. Numeric inputs surface a red error message when the value is
    out of range; callbacks downstream should also guard against invalid
    values.
    """
    twist_id = f"{id_prefix}-twist-slider"
    grid_id = f"{id_prefix}-grid-size"
    extent_id = f"{id_prefix}-physical-extent"

    _register_validation(id_prefix)

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Parameters", className="card-title"),

                labeled_with_help(
                    "Twist angle (deg)",
                    f"{id_prefix}-twist-help",
                    "Rotation of the overlayer relative to the substrate. "
                    "0° for an aligned heterostructure; small angles (<5°) "
                    "produce long-period moire patterns.",
                ),
                dcc.Slider(
                    id=twist_id,
                    min=0,
                    max=30,
                    step=0.1,
                    value=DEFAULT_TWIST_ANGLE,
                    marks={i: str(i) for i in range(0, 31, 5)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Br(),

                labeled_with_help(
                    "Grid size",
                    f"{id_prefix}-grid-help",
                    "Number of samples per axis (NxN). Higher values produce "
                    f"sharper features but increase compute time. Range: "
                    f"{GRID_MIN}–{GRID_MAX}.",
                ),
                dbc.Input(
                    id=grid_id,
                    type="number",
                    value=DEFAULT_GRID_SIZE,
                    min=GRID_MIN,
                    max=GRID_MAX,
                    step=50,
                    invalid=False,
                ),
                dbc.FormFeedback(
                    f"Grid size must be between {GRID_MIN} and {GRID_MAX}.",
                    type="invalid",
                ),
                html.Br(),

                labeled_with_help(
                    "Physical extent (Å)",
                    f"{id_prefix}-extent-help",
                    "Side length of the real-space window in Ångstrom. "
                    "Choose large enough to contain at least one moire period.",
                ),
                dbc.Input(
                    id=extent_id,
                    type="number",
                    value=DEFAULT_PHYSICAL_EXTENT,
                    min=EXTENT_MIN,
                    max=EXTENT_MAX,
                    step=10,
                    invalid=False,
                ),
                dbc.FormFeedback(
                    f"Physical extent must be between {EXTENT_MIN} and {EXTENT_MAX} Å.",
                    type="invalid",
                ),
            ]
        ),
        className="mb-3",
    )


def is_valid_grid_size(value) -> bool:
    """Return True if `value` is a valid grid size."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return False
    return GRID_MIN <= v <= GRID_MAX


def is_valid_extent(value) -> bool:
    """Return True if `value` is a valid physical extent."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return EXTENT_MIN <= v <= EXTENT_MAX
