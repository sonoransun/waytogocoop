"""3D Proximity Effect page — volumetric Cooper surface and z-slice browser.

The Isosurface/Volume view supports an animated iso-range sweep (Play/Pause
with a ``dcc.Interval``) and a clipping plane that removes z > clip_z so the
layered decay of the gap into the topological insulator is directly visible.

URL state is wired through :func:`waytogocoop.state.register_url_sync` so any
combination of material, parameter, iso-range, clip plane, and view mode is
shareable via a single ``?q=<base64>`` string.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from waytogocoop.components.controls import loading_spinner
from waytogocoop.components.figure_factory import (
    create_3d_isosurface,
    create_3d_volume,
    create_gap_heatmap,
    create_z_decay_profile,
)
from waytogocoop.components.material_selector import create_material_selector
from waytogocoop.components.parameter_panel import create_parameter_panel
from waytogocoop.computation.moire import generate_moire_pattern
from waytogocoop.computation.superconducting import gap_modulation
from waytogocoop.computation.topological import (
    ProximityConfig,
    gap_3d,
)
from waytogocoop.config import DELTA_AMPLITUDE, DELTA_AVG
from waytogocoop.materials.database import get_material
from waytogocoop.state import register_url_sync

dash.register_page(
    __name__,
    path="/proximity3d",
    name="3D Proximity",
    title="Good Job Coop! - 3D Proximity",
)

_PREFIX = "prox3d"
_URL_ID = f"{_PREFIX}-url"


layout = dbc.Container(
    [
        dcc.Location(id=_URL_ID),
        dcc.Store(id=f"{_PREFIX}-anim-phase", data=0.0),
        dcc.Interval(id=f"{_PREFIX}-anim-tick", interval=350, disabled=True),
        html.Br(),
        html.H2("3D Cooper-Pair Density Surface"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        create_material_selector(_PREFIX),
                        create_parameter_panel(_PREFIX),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Proximity Parameters"),
                                    dbc.Label("Proximity coherence length (Angstrom)"),
                                    dcc.Slider(
                                        id=f"{_PREFIX}-xi-prox",
                                        min=10,
                                        max=500,
                                        step=10,
                                        value=100,
                                        marks={10: "10", 100: "100", 250: "250", 500: "500"},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Br(),
                                    dbc.Label("Interface transparency"),
                                    dcc.Slider(
                                        id=f"{_PREFIX}-transparency",
                                        min=0.1,
                                        max=1.0,
                                        step=0.05,
                                        value=0.8,
                                        marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0"},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Hr(),
                                    dbc.Label("View mode"),
                                    dcc.RadioItems(
                                        id=f"{_PREFIX}-view-mode",
                                        options=[
                                            {"label": "Isosurface", "value": "iso"},
                                            {"label": "Volume", "value": "volume"},
                                            {"label": "Z-Slice (2D)", "value": "slice"},
                                        ],
                                        value="iso",
                                        inline=True,
                                        inputStyle={"marginRight": "4px"},
                                        labelStyle={"marginRight": "16px"},
                                    ),
                                    html.Hr(),
                                    dbc.Label("Iso-level range (fraction of peak Δ)"),
                                    dcc.RangeSlider(
                                        id=f"{_PREFIX}-iso-range",
                                        min=0.0, max=1.0, step=0.02,
                                        value=[0.2, 0.8],
                                        marks={0: "0", 0.5: "0.5", 1: "1"},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Br(),
                                    dbc.Label("Clip plane (z, Å) — ∞ disables"),
                                    dcc.Slider(
                                        id=f"{_PREFIX}-clip-z",
                                        min=-50.0, max=300.0, step=5.0,
                                        value=300.0,
                                        marks={-50: "-50", 0: "0", 150: "150", 300: "300"},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Br(),
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button("▶ Play iso sweep",
                                                       id=f"{_PREFIX}-play-btn",
                                                       color="primary",
                                                       size="sm",
                                                       n_clicks=0),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Hr(),
                                    dbc.Label("Z-slice index (2D view only)"),
                                    dcc.Slider(
                                        id=f"{_PREFIX}-z-slice",
                                        min=0,
                                        max=29,
                                        step=1,
                                        value=15,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                ]
                            ),
                            className="mb-3",
                        ),
                    ],
                    xs=12, md=4, lg=3,
                ),
                dbc.Col(
                    [
                        loading_spinner(
                            dcc.Graph(id=f"{_PREFIX}-main-graph"),
                            "Computing 3D proximity volume…",
                        ),
                        html.Hr(),
                        loading_spinner(
                            dcc.Graph(id=f"{_PREFIX}-decay-graph"),
                            "Computing decay profile…",
                        ),
                        html.Hr(),
                        html.Div(id=f"{_PREFIX}-info"),
                    ],
                    xs=12, md=8, lg=9,
                ),
            ]
        ),
    ],
    fluid=True,
)


# --- Play/Pause toggles the Interval; button label tracks state ---
@callback(
    Output(f"{_PREFIX}-anim-tick", "disabled"),
    Output(f"{_PREFIX}-play-btn", "children"),
    Input(f"{_PREFIX}-play-btn", "n_clicks"),
    State(f"{_PREFIX}-anim-tick", "disabled"),
    prevent_initial_call=True,
)
def _toggle_play(n_clicks, is_disabled):
    if not n_clicks:
        return True, "▶ Play iso sweep"
    new_disabled = not is_disabled
    label = "▶ Play iso sweep" if new_disabled else "⏸ Pause"
    return new_disabled, label


# --- Animation tick: advance a 0..1 phase and sweep the iso-range midpoint ---
@callback(
    Output(f"{_PREFIX}-anim-phase", "data"),
    Output(f"{_PREFIX}-iso-range", "value", allow_duplicate=True),
    Input(f"{_PREFIX}-anim-tick", "n_intervals"),
    State(f"{_PREFIX}-anim-phase", "data"),
    State(f"{_PREFIX}-iso-range", "value"),
    prevent_initial_call=True,
)
def _advance_phase(n, phase, iso_range):
    if phase is None:
        phase = 0.0
    # ping-pong 0..1..0 — swap between percentile-20/80 and 40/60 bands
    phase = (float(phase) + 0.05) % 2.0
    t = phase if phase <= 1.0 else 2.0 - phase  # triangle wave in [0, 1]
    width = float(iso_range[1] - iso_range[0]) if iso_range else 0.6
    mid = 0.25 + 0.5 * t  # sweep midpoint 0.25 -> 0.75
    half = max(min(width / 2.0, mid - 0.0, 1.0 - mid), 0.05)
    return phase, [round(mid - half, 3), round(mid + half, 3)]


@callback(
    Output(f"{_PREFIX}-main-graph", "figure"),
    Output(f"{_PREFIX}-decay-graph", "figure"),
    Output(f"{_PREFIX}-info", "children"),
    Input(f"{_PREFIX}-substrate-dropdown", "value"),
    Input(f"{_PREFIX}-overlayer-dropdown", "value"),
    Input(f"{_PREFIX}-twist-slider", "value"),
    Input(f"{_PREFIX}-grid-size", "value"),
    Input(f"{_PREFIX}-physical-extent", "value"),
    Input(f"{_PREFIX}-xi-prox", "value"),
    Input(f"{_PREFIX}-transparency", "value"),
    Input(f"{_PREFIX}-z-slice", "value"),
    Input(f"{_PREFIX}-view-mode", "value"),
    Input(f"{_PREFIX}-iso-range", "value"),
    Input(f"{_PREFIX}-clip-z", "value"),
    Input("theme-store", "data"),
)
def update_proximity(
    substrate_key, overlayer_key, twist, grid_size, extent,
    xi_prox, transparency, z_slice_idx, view_mode, iso_range, clip_z, theme,
):
    try:
        dark = theme == "dark"
        substrate = get_material(substrate_key)
        overlayer = get_material(overlayer_key)

        xi_prox = float(xi_prox) if xi_prox is not None else 100.0
        transparency = float(transparency) if transparency is not None else 0.8
        z_slice_idx = int(z_slice_idx) if z_slice_idx is not None else 0
        iso_range = iso_range or [0.2, 0.8]
        clip_z = float(clip_z) if clip_z is not None else 300.0

        # Cap grid size for 3D to avoid memory issues
        grid_3d = min(grid_size, 80)
        n_z = 30

        # Generate moire pattern
        result = generate_moire_pattern(
            substrate_a=substrate.a,
            overlayer_a=overlayer.a,
            overlayer_lattice_type=overlayer.lattice_type,
            substrate_lattice_type=substrate.lattice_type,
            twist_angle_deg=twist,
            grid_size=grid_3d,
            physical_extent=extent,
        )
        x, y, pattern = result["x"], result["y"], result["pattern"]

        # Gap modulation
        gap_field = gap_modulation(pattern, DELTA_AVG, DELTA_AMPLITUDE)

        # Proximity z-decay
        config = ProximityConfig(
            xi_prox=xi_prox,
            n_z_layers=n_z,
            z_min=-50.0,
            z_max=300.0,
            interface_transparency=transparency,
        )
        prox_result = gap_3d(gap_field, config)

        # Decay profile figure
        decay_fig = create_z_decay_profile(
            prox_result.z_coords, prox_result.decay_profile, dark=dark
        )

        # Clamp z-slice index
        z_idx = min(max(int(z_slice_idx), 0), n_z - 1)
        z_val = prox_result.z_coords[z_idx]

        # Iso-range maps fractional [0, 1] to real gap magnitudes via peak.
        peak_delta = float(np.nanmax(prox_result.gap_3d)) or 1.0
        iso_min = iso_range[0] * peak_delta
        iso_max = iso_range[1] * peak_delta
        # clip_z at the z-max effectively disables clipping.
        clip_arg = None if clip_z >= float(prox_result.z_coords[-1]) else clip_z

        if view_mode in ("iso", "volume"):
            # Annotate the interface plane and the proximity decay length to
            # orient the viewer inside the isosurface volume.
            x_mid = float((x[0] + x[-1]) / 2.0)
            y_mid = float((y[0] + y[-1]) / 2.0)
            xi_z = min(xi_prox, float(prox_result.z_coords[-1]))
            iso_annotations = [
                dict(
                    x=x_mid, y=y_mid, z=0.0,
                    text="Interface (z=0)",
                    showarrow=True, arrowhead=2, ax=30, ay=-30,
                    font=dict(size=11, color="white" if dark else "black"),
                ),
                dict(
                    x=x_mid, y=y_mid, z=xi_z,
                    text=f"ξ_prox ≈ {xi_prox:.0f} Å",
                    showarrow=True, arrowhead=2, ax=40, ay=-40,
                    font=dict(size=11, color="white" if dark else "black"),
                ),
            ]
            if view_mode == "iso":
                main_fig = create_3d_isosurface(
                    x, y, prox_result.z_coords, prox_result.gap_3d,
                    title=f"3D Cooper Surface — {overlayer.formula}/{substrate.formula}",
                    iso_min=iso_min,
                    iso_max=iso_max,
                    surface_count=5,
                    dark=dark,
                    annotations=iso_annotations,
                    clip_z=clip_arg,
                )
            else:
                main_fig = create_3d_volume(
                    x, y, prox_result.z_coords, prox_result.gap_3d,
                    title=f"3D Cooper Volume — {overlayer.formula}/{substrate.formula}",
                    opacity=0.1,
                    surface_count=17,
                    dark=dark,
                    annotations=iso_annotations,
                    clip_z=clip_arg,
                )
        else:
            # Z-slice heatmap
            slice_data = prox_result.gap_3d[z_idx]
            main_fig = create_gap_heatmap(
                x, y, slice_data,
                title=f"Gap at z = {z_val:.1f} A",
                dark=dark,
            )

        info = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Proximity 3D Info"),
                    html.P(f"xi_prox = {xi_prox:.0f} A"),
                    html.P(f"Interface transparency T = {transparency:.2f}"),
                    html.P(f"z range: {config.z_min:.0f} to {config.z_max:.0f} A ({n_z} layers)"),
                    html.P(f"Selected z-slice: {z_val:.1f} A (index {z_idx})"),
                    html.P(
                        f"Iso-range: [{iso_min:.3f}, {iso_max:.3f}] meV "
                        f"(peak Δ = {peak_delta:.3f} meV)"
                    ),
                    html.P(
                        f"Clip z: {'off' if clip_arg is None else f'{clip_arg:.0f} Å'}"
                    ),
                    html.P(f"Gap at z-slice: {prox_result.gap_3d[z_idx].min():.3f} — "
                            f"{prox_result.gap_3d[z_idx].max():.3f} meV"),
                ]
            ),
            className="mb-3",
        )

        return main_fig, decay_fig, info
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Computation error: {e}")
        return error_fig, error_fig, html.P(str(e), style={"color": "red"})


# --- URL state sync (shareable ?q=<base64>) -------------------------------
register_url_sync(
    _URL_ID,
    [
        (f"{_PREFIX}-substrate-dropdown", "value", "sub"),
        (f"{_PREFIX}-overlayer-dropdown", "value", "over"),
        (f"{_PREFIX}-twist-slider", "value", "tw"),
        (f"{_PREFIX}-grid-size", "value", "gs"),
        (f"{_PREFIX}-physical-extent", "value", "ext"),
        (f"{_PREFIX}-xi-prox", "value", "xi"),
        (f"{_PREFIX}-transparency", "value", "T"),
        (f"{_PREFIX}-view-mode", "value", "vm"),
        (f"{_PREFIX}-iso-range", "value", "ir"),
        (f"{_PREFIX}-clip-z", "value", "cz"),
        (f"{_PREFIX}-z-slice", "value", "zs"),
    ],
)
