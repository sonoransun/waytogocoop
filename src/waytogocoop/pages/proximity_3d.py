"""3D Proximity Effect page — volumetric Cooper surface and z-slice browser."""

from __future__ import annotations

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from waytogocoop.components.material_selector import create_material_selector
from waytogocoop.components.parameter_panel import create_parameter_panel
from waytogocoop.components.figure_factory import (
    create_3d_isosurface,
    create_z_decay_profile,
    create_gap_heatmap,
)
from waytogocoop.computation.moire import generate_moire_pattern
from waytogocoop.computation.superconducting import gap_modulation
from waytogocoop.computation.topological import (
    ProximityConfig,
    gap_3d,
)
from waytogocoop.config import DELTA_AVG, DELTA_AMPLITUDE
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
                    md=3,
                ),
                dbc.Col(
                    [
                        dcc.Loading(dcc.Graph(id=f"{_PREFIX}-main-graph")),
                        html.Hr(),
                        dcc.Loading(dcc.Graph(id=f"{_PREFIX}-decay-graph")),
                        html.Hr(),
                        html.Div(id=f"{_PREFIX}-info"),
                    ],
                    md=9,
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
    dark = theme == "dark"
    substrate = get_material(substrate_key)
    overlayer = get_material(overlayer_key)

    xi_prox = xi_prox or 100.0
    transparency = transparency or 0.8
    z_slice_idx = z_slice_idx or 0

    # Cap grid size for 3D to avoid memory issues
    grid_3d = min(grid_size, 80)
    n_z = 30

    # Generate moire pattern
    result = generate_moire_pattern(
        substrate_a=substrate.a,
        overlayer_a=overlayer.a,
        overlayer_lattice_type=overlayer.lattice_type,
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
    decay_fig = create_z_decay_profile(prox_result.z_coords, prox_result.decay_profile, dark=dark)

    # Clamp z-slice index
    z_idx = min(max(int(z_slice_idx), 0), n_z - 1)
    z_val = prox_result.z_coords[z_idx]

    if view_mode == "iso":
        # Isosurface view
        main_fig = create_3d_isosurface(
            x, y, prox_result.z_coords, prox_result.gap_3d,
            title=f"3D Cooper Surface — {overlayer.formula}/{substrate.formula}",
            dark=dark,
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
