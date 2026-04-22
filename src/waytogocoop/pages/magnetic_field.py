"""Magnetic Field page — vortex lattice, screening, susceptibility, Majorana."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from waytogocoop.components.figure_factory import (
    create_3d_cone_field,
    create_3d_majorana_isosurface,
    create_gap_heatmap,
    create_majorana_density_map,
    create_susceptibility_heatmap,
    create_vortex_overlay_heatmap,
)
from waytogocoop.components.magnetic_panel import create_magnetic_panel
from waytogocoop.components.material_selector import create_material_selector
from waytogocoop.components.parameter_panel import create_parameter_panel
from waytogocoop.computation.magnetic import (
    MagneticFieldConfig,
    combined_gap_with_vortices,
    commensuration_field,
    compute_zeeman,
    flux_per_moire_cell,
    generate_vortex_positions,
    local_susceptibility,
    screening_currents,
    vortex_lattice_period,
    vortex_suppression_field,
)
from waytogocoop.computation.moire import generate_moire_pattern
from waytogocoop.computation.superconducting import gap_modulation
from waytogocoop.computation.topological import (
    majorana_probability_density,
    majorana_probability_density_3d,
)
from waytogocoop.config import DEFAULT_COHERENCE_LENGTH, DELTA_AMPLITUDE, DELTA_AVG
from waytogocoop.materials.database import get_material

dash.register_page(
    __name__,
    path="/magnetic",
    name="Magnetic Field",
    title="Good Job Coop! - Magnetic",
)

_PREFIX = "mag"

layout = dbc.Container(
    [
        html.Br(),
        html.H2("Magnetic Field Effects"),
        html.Hr(),
        dbc.Row(
            [
                # Left — controls
                dbc.Col(
                    [
                        create_material_selector(_PREFIX),
                        create_parameter_panel(_PREFIX),
                        create_magnetic_panel(_PREFIX),
                    ],
                    xs=12, md=4, lg=3,
                ),
                # Right — visualization
                dbc.Col(
                    [
                        dcc.Loading(dcc.Graph(id=f"{_PREFIX}-main-graph")),
                        html.Hr(),
                        html.Div(id=f"{_PREFIX}-info-panel"),
                    ],
                    xs=12, md=8, lg=9,
                ),
            ]
        ),
    ],
    fluid=True,
)


# --- Collapse toggles ---
@callback(
    Output(f"{_PREFIX}-inplane-collapse", "is_open"),
    Input(f"{_PREFIX}-inplane-toggle", "n_clicks"),
    State(f"{_PREFIX}-inplane-collapse", "is_open"),
)
def _toggle_inplane(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output(f"{_PREFIX}-prox-collapse", "is_open"),
    Input(f"{_PREFIX}-prox-toggle", "n_clicks"),
    State(f"{_PREFIX}-prox-collapse", "is_open"),
)
def _toggle_prox(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


# --- Main computation callback ---
@callback(
    Output(f"{_PREFIX}-main-graph", "figure"),
    Output(f"{_PREFIX}-info-panel", "children"),
    Output(f"{_PREFIX}-mag-info", "children"),
    Input(f"{_PREFIX}-substrate-dropdown", "value"),
    Input(f"{_PREFIX}-overlayer-dropdown", "value"),
    Input(f"{_PREFIX}-twist-slider", "value"),
    Input(f"{_PREFIX}-grid-size", "value"),
    Input(f"{_PREFIX}-physical-extent", "value"),
    Input(f"{_PREFIX}-bz", "value"),
    Input(f"{_PREFIX}-bx", "value"),
    Input(f"{_PREFIX}-by", "value"),
    Input(f"{_PREFIX}-viz-mode", "value"),
    Input(f"{_PREFIX}-xi-prox", "value"),
    Input(f"{_PREFIX}-g-factor", "value"),
    Input("theme-store", "data"),
)
def update_magnetic(
    substrate_key, overlayer_key, twist, grid_size, extent,
    Bz, Bx, By, viz_mode, xi_prox, g_factor, theme,
):
    try:
        dark = theme == "dark"
        substrate = get_material(substrate_key)
        overlayer = get_material(overlayer_key)

        # Default safety for slider values
        Bz = float(Bz) if Bz is not None else 0.0
        Bx = float(Bx) if Bx is not None else 0.0
        By = float(By) if By is not None else 0.0
        xi_prox = float(xi_prox) if xi_prox is not None else 100.0
        g_factor = float(g_factor) if g_factor is not None else 30.0

        # Generate moire pattern
        result = generate_moire_pattern(
            substrate_a=substrate.a,
            overlayer_a=overlayer.a,
            overlayer_lattice_type=overlayer.lattice_type,
            substrate_lattice_type=substrate.lattice_type,
            twist_angle_deg=twist,
            grid_size=grid_size,
            physical_extent=extent,
        )
        x, y, pattern = result["x"], result["y"], result["pattern"]
        moire_period = result["moire_period"]

        # Gap modulation
        gap_field = gap_modulation(pattern, DELTA_AVG, DELTA_AMPLITUDE)

        # Magnetic field computation
        config = MagneticFieldConfig(Bx=Bx, By=By, Bz=Bz)
        vortex_pos = generate_vortex_positions(Bz, extent, grid_size)
        a_v = vortex_lattice_period(Bz)
        suppression = vortex_suppression_field(x, y, vortex_pos, DEFAULT_COHERENCE_LENGTH)
        combined_gap = combined_gap_with_vortices(gap_field, suppression)
        flux = flux_per_moire_cell(Bz, moire_period)
        comm_B = commensuration_field(moire_period)
        zeeman = compute_zeeman(config, DELTA_AVG)

        # Build figure based on viz_mode
        if viz_mode == "vortex":
            fig = create_vortex_overlay_heatmap(x, y, gap_field, vortex_pos, dark=dark)
        elif viz_mode == "combined":
            fig = create_gap_heatmap(
                x, y, combined_gap, title="Combined Gap (Moire + Vortex)", dark=dark
            )
        elif viz_mode == "currents":
            Jx, Jy = screening_currents(x, y, vortex_pos)
            fig = create_3d_cone_field(
                x, y, 0.0, Jx, Jy, jz=None,
                base_surface=combined_gap, skip=8, dark=dark,
            )
        elif viz_mode == "chi":
            chi = local_susceptibility(combined_gap)
            fig = create_susceptibility_heatmap(x, y, chi, dark=dark)
        elif viz_mode == "majorana":
            mzm = majorana_probability_density(x, y, vortex_pos)
            fig = create_majorana_density_map(x, y, mzm.probability_density, vortex_pos, dark=dark)
        elif viz_mode == "majorana3d":
            # Sparse z-grid keeps the isosurface cheap to build; resolution of
            # xy is already capped upstream by the moire grid.
            z3d = np.linspace(-40.0, 200.0, 32)
            density_3d = majorana_probability_density_3d(
                x, y, z3d, vortex_pos, xi_prox=xi_prox,
            )
            fig = create_3d_majorana_isosurface(x, y, z3d, density_3d, vortex_pos, dark=dark)
        else:
            fig = create_vortex_overlay_heatmap(x, y, gap_field, vortex_pos, dark=dark)

        # Info panel
        comm_ratio = a_v / moire_period if np.isfinite(a_v) and moire_period > 0 else float("inf")
        info = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Magnetic Field Results"),
                    html.P(
                        f"Vortex period: {a_v:.1f} A"
                        if np.isfinite(a_v) else "No vortices (Bz=0)"
                    ),
                    html.P(f"Flux per moire cell: {flux:.3f} Phi_0"),
                    html.P(
                        f"Commensuration field: {comm_B:.1f} T"
                        if np.isfinite(comm_B) else "N/A"
                    ),
                    html.P(f"a_v / L_m: {comm_ratio:.3f}" if np.isfinite(comm_ratio) else "N/A"),
                    html.Hr(),
                    html.P(f"Zeeman energy: {zeeman.zeeman_energy:.4f} meV"),
                    html.P(f"Pauli limit: {zeeman.pauli_limit_field:.1f} T"),
                    html.P(f"Depairing ratio: {zeeman.depairing_ratio:.4f}"),
                ]
            ),
            className="mb-3",
        )

        # Short sidebar readout
        sidebar_info = html.Small(
            f"a_v={a_v:.0f} A | Phi/cell={flux:.2f}"
            if np.isfinite(a_v) else "No vortices"
        )

        return fig, info, sidebar_info
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Computation error: {e}")
        err_msg = html.P(str(e), style={"color": "red"})
        return error_fig, err_msg, err_msg
