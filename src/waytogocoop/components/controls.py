"""Shared Dash controls: help tooltips, preset dropdowns, reset buttons.

Usage from a page module:

    from waytogocoop.components.controls import (
        help_tooltip,
        preset_and_reset_card,
        register_preset_reset_callback,
        loading_spinner,
    )

Pass explicit default values to `register_preset_reset_callback` so the reset
button can restore them.
"""

from __future__ import annotations

import json
from string import Template
from typing import Any, Mapping, Sequence

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, clientside_callback, dcc, html, no_update


# ---------------------------------------------------------------------------
# Help tooltip — an info-style badge paired with a Bootstrap tooltip.
# ---------------------------------------------------------------------------


def help_tooltip(target_id: str, text: str) -> list:
    """Return [badge, dbc.Tooltip] to place next to a label.

    The badge renders a small circled "?" users can hover to see `text`.

    Parameters
    ----------
    target_id : str
        Dash component ID of the badge (must be unique per page).
    text : str
        The help text shown on hover.
    """
    badge = html.Span(
        "ⓘ",
        id=target_id,
        style={
            "marginLeft": "6px",
            "cursor": "help",
            "opacity": "0.7",
            "fontSize": "0.85em",
        },
    )
    tip = dbc.Tooltip(text, target=target_id, placement="right")
    return [badge, tip]


def labeled_with_help(label: str, tooltip_id: str, help_text: str) -> html.Div:
    """A `dbc.Label` followed by a help badge + tooltip."""
    return html.Div(
        [dbc.Label(label), *help_tooltip(tooltip_id, help_text)],
        style={"display": "flex", "alignItems": "center"},
    )


# ---------------------------------------------------------------------------
# Preset dropdown + reset button.
# ---------------------------------------------------------------------------


def preset_and_reset_card(
    page_id: str,
    presets: Mapping[str, Mapping[str, Any]],
    help_text: str = "Load a preset to populate the controls, or reset them to defaults.",
) -> dbc.Card:
    """Return a Card containing a preset dropdown and a reset button."""
    options = [{"label": name, "value": name} for name in presets]
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(
                    [
                        "Presets",
                        *help_tooltip(f"{page_id}-preset-help", help_text),
                    ],
                    className="card-title",
                    style={"display": "flex", "alignItems": "center"},
                ),
                dcc.Dropdown(
                    id=f"{page_id}-preset",
                    options=options,
                    placeholder="Select a preset…",
                    clearable=True,
                ),
                html.Br(),
                dbc.Button(
                    "Reset defaults",
                    id=f"{page_id}-reset",
                    color="secondary",
                    size="sm",
                    outline=True,
                ),
            ]
        ),
        className="mb-3",
    )


def register_preset_reset_callback(
    page_id: str,
    presets: Mapping[str, Mapping[str, Any]],
    defaults: Mapping[str, Any],
    output_bindings: Sequence[tuple[str, str, str]],
) -> None:
    """Wire the preset dropdown and reset button to populate controls.

    `output_bindings` is a sequence of `(component_id, property, preset_key)`
    triples telling the callback where to write each field of a preset dict.
    When "Reset defaults" is clicked, each binding receives `defaults[key]`.
    """
    outputs = [Output(cid, prop) for (cid, prop, _) in output_bindings]
    keys = [key for (_, _, key) in output_bindings]

    @callback(
        *outputs,
        Input(f"{page_id}-preset", "value"),
        Input(f"{page_id}-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def _apply(preset_name: str | None, _reset_clicks: int | None):
        from dash import ctx

        if ctx.triggered_id == f"{page_id}-reset":
            return tuple(defaults.get(k, no_update) for k in keys)

        if preset_name and preset_name in presets:
            preset = presets[preset_name]
            return tuple(preset.get(k, no_update) for k in keys)

        return tuple(no_update for _ in keys)


# ---------------------------------------------------------------------------
# Loading wrapper with a descriptive "Computing…" subtitle visible during load.
# ---------------------------------------------------------------------------


def loading_spinner(component, message: str = "Computing…"):
    """Wrap a component in `dcc.Loading` with a small muted subtitle.

    The subtitle sits beneath the component and fades with it while the
    callback that supplies the component's output is in flight. This is a
    lightweight substitute for Dash's `custom_spinner`.
    """
    return dcc.Loading(
        children=[
            component,
            html.Div(
                message,
                className="text-muted small",
                style={"textAlign": "center", "marginTop": "4px"},
            ),
        ],
        type="circle",
        color="#5dade2",
    )


# ---------------------------------------------------------------------------
# Cross-page navigation — "Open in Moire Viewer" button with URL state.
# ---------------------------------------------------------------------------


_OPEN_VIEWER_JS = Template("""
function(n_clicks, ...values) {
    if (!n_clicks) return window.dash_clientside.no_update;
    const keys = $keys;
    const state = {};
    for (let i = 0; i < keys.length; i++) {
        const v = values[i];
        if (v !== null && v !== undefined) {
            state[keys[i]] = v;
        }
    }
    const raw = JSON.stringify(state);
    const b64 = btoa(unescape(encodeURIComponent(raw)))
        .replace(/\\+/g, '-').replace(/\\//g, '_').replace(/=+$$/, '');
    return '/viewer?q=' + b64;
}
""")


def open_in_viewer_button(
    page_id: str,
    bindings: Sequence[tuple[str, str, str]],
    label: str = "Open in Moire Viewer",
) -> list:
    """Return [button, dcc.Location] and register the click callback.

    ``bindings`` is a sequence of ``(component_id, property, state_key)``
    triples mirroring the target viewer page's preset bindings. On click, a
    base64-encoded JSON payload is written to the Location's ``href`` so the
    viewer initializes with the matching configuration.
    """
    btn_id = f"{page_id}-open-viewer"
    nav_id = f"{page_id}-open-viewer-nav"
    keys_json = json.dumps([k for (_, _, k) in bindings])

    button = dbc.Button(
        label,
        id=btn_id,
        color="primary",
        outline=True,
        size="sm",
        className="mt-2",
    )
    nav = dcc.Location(id=nav_id, refresh=True)

    clientside_callback(
        _OPEN_VIEWER_JS.substitute(keys=keys_json),
        Output(nav_id, "href"),
        Input(btn_id, "n_clicks"),
        *[State(cid, prop) for (cid, prop, _) in bindings],
        prevent_initial_call=True,
    )

    return [button, nav]


# Re-export ALL for callers that want pattern-matching IDs.
__all__ = [
    "ALL",
    "State",
    "help_tooltip",
    "labeled_with_help",
    "preset_and_reset_card",
    "register_preset_reset_callback",
    "loading_spinner",
    "open_in_viewer_button",
]
