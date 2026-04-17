"""3D Proximity Effect page — volumetric Cooper surface and z-slice browser."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from waytogocoop.components.controls import loading_spinner
from waytogocoop.components.figure_factory import (
    create_3d_isosurface,
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

dash.register_page(
    __name__,
    path="/proximity3d",
    name="3D Proximity",
    title="Good Job Coop! - 3D Proximity",
)

_PREFIX = "prox3d"

layout = dbc.Container(
    [
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
                                    html.Br(),
                                    dbc.Label("Z-slice index"),
                                    dcc.Slider(
                                        id=f"{_PREFIX}-z-slice",
                                        min=0,
                                        max=29,
                                        step=1,
                                        value=15,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Br(),
                                    dbc.Label("View mode"),
                                    dcc.RadioItems(
                                        id=f"{_PREFIX}-view-mode",
                                        options=[
                                            {"label": "Isosurface (3D)", "value": "iso"},
                                            {"label": "Z-Slice (2D)", "value": "slice"},
                                        ],
                                        value="slice",
                                        inline=True,
                                        inputStyle={"marginRight": "4px"},
                                        labelStyle={"marginRight": "16px"},
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
    Input("theme-store", "data"),
)
def update_proximity(
    substrate_key, overlayer_key, twist, grid_size, extent,
    xi_prox, transparency, z_slice_idx, view_mode, theme,
):
    try:
        dark = theme == "dark"
        substrate = get_material(substrate_key)
        overlayer = get_material(overlayer_key)

        xi_prox = float(xi_prox) if xi_prox is not None else 100.0
        transparency = float(transparency) if transparency is not None else 0.8
        z_slice_idx = int(z_slice_idx) if z_slice_idx is not None else 0

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

        if view_mode == "iso":
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
            main_fig = create_3d_isosurface(
                x, y, prox_result.z_coords, prox_result.gap_3d,
                title=f"3D Cooper Surface — {overlayer.formula}/{substrate.formula}",
                dark=dark,
                annotations=iso_annotations,
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
