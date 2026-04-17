"""URL state sharing for the Dash web UI.

Each page with shareable parameters calls :func:`register_url_sync` once at
module load. This registers clientside callbacks that:

1. Decode the `?q=` query parameter on navigation and populate bound controls.
2. Encode any bound control value into `?q=` whenever a control changes.

The state payload is a JSON dict wrapped in URL-safe base64 so long URLs stay
under typical browser limits.
"""

from __future__ import annotations

import base64
import json
from string import Template
from typing import Any, Mapping, Sequence

from dash import Input, Output, State, clientside_callback


def encode_state(state: Mapping[str, Any]) -> str:
    raw = json.dumps(state, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_state(query: str | None) -> dict[str, Any]:
    if not query:
        return {}
    # Accept either "?q=..." or bare base64.
    if query.startswith("?"):
        query = query[1:]
    for pair in query.split("&"):
        if "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        if key == "q":
            padding = "=" * (-len(value) % 4)
            try:
                raw = base64.urlsafe_b64decode(value + padding)
                decoded = json.loads(raw.decode("utf-8"))
                if isinstance(decoded, dict):
                    return decoded
            except (ValueError, json.JSONDecodeError):
                return {}
    return {}


# ---------------------------------------------------------------------------
# Clientside sync helpers.
# ---------------------------------------------------------------------------


# Binding = (component_id, property, state_key). The `state_key` is the key
# inside the JSON payload; using a short stable key keeps URLs compact.


_DECODE_JS = Template("""
function(pathname, search) {
    const match = (search || '').match(/[?&]q=([^&]+)/);
    if (!match) return Array($n).fill(window.dash_clientside.no_update);
    try {
        const padded = match[1] + '==='.slice((match[1].length + 3) % 4);
        const text = atob(padded.replace(/-/g, '+').replace(/_/g, '/'));
        const state = JSON.parse(text);
        const keys = $keys;
        return keys.map(k => (k in state) ? state[k] : window.dash_clientside.no_update);
    } catch (e) {
        return Array($n).fill(window.dash_clientside.no_update);
    }
}
""")


_ENCODE_JS = Template("""
function(...args) {
    const keys = $keys;
    const state = {};
    for (let i = 0; i < keys.length; i++) {
        const v = args[i];
        if (v !== null && v !== undefined) {
            state[keys[i]] = v;
        }
    }
    const raw = JSON.stringify(state);
    const b64 = btoa(unescape(encodeURIComponent(raw)))
        .replace(/\\+/g, '-').replace(/\\//g, '_').replace(/=+$$/, '');
    return '?q=' + b64;
}
""")


def register_url_sync(
    url_id: str,
    bindings: Sequence[tuple[str, str, str]],
) -> None:
    """Wire bidirectional URL state sync for a page.

    ``url_id`` is the ID of a ``dcc.Location`` on the page (typically
    ``"<page>-url"``).  ``bindings`` is a sequence of
    ``(component_id, property, state_key)`` triples.
    """
    if not bindings:
        return
    keys = [k for (_, _, k) in bindings]
    keys_json = json.dumps(keys)
    # Decode: URL search -> controls.
    decode_js = _DECODE_JS.substitute(n=len(bindings), keys=keys_json)
    outputs_decode = [Output(cid, prop, allow_duplicate=True) for (cid, prop, _) in bindings]
    clientside_callback(
        decode_js,
        *outputs_decode,
        Input(url_id, "pathname"),
        State(url_id, "search"),
        prevent_initial_call="initial_duplicate",
    )
    # Encode: control values -> URL search.
    encode_js = _ENCODE_JS.substitute(keys=keys_json)
    inputs_encode = [Input(cid, prop) for (cid, prop, _) in bindings]
    clientside_callback(
        encode_js,
        Output(url_id, "search", allow_duplicate=True),
        *inputs_encode,
        prevent_initial_call=True,
    )


__all__ = ["decode_state", "encode_state", "register_url_sync"]
