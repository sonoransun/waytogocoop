"""Moire Viewer page — interactive real-space pattern and gap modulation."""

from __future__ import annotations

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from waytogocoop.components.material_selector import create_material_selector
from waytogocoop.components.parameter_panel import create_parameter_panel
from waytogocoop.components.isotope_panel import create_isotope_panel
from waytogocoop.components.figure_factory import (
    create_moire_heatmap,
    create_gap_heatmap,
    create_3d_surface,
    create_2d_contour,
)
from waytogocoop.computation.moire import generate_moire_pattern
from waytogocoop.computation.superconducting import gap_modulation, cpdm_amplitude
from waytogocoop.computation.isotope_effects import compute_isotope_effects
from waytogocoop.config import DELTA_1, DELTA_2, DELTA_AVG, DELTA_AMPLITUDE
from waytogocoop.materials.database import get_material

dash.register_page(
    __name__, path="/viewer", name="Moire Viewer", title="Good Job Coop! - Viewer"
)

_PREFIX = "viewer"

layout = dbc.Container(
    [
        html.Br(),
        html.H2("Moire Pattern Viewer"),
        html.Hr(),
        dbc.Row(
            [
                # Left column — controls
                dbc.Col(
                    [
                        create_material_selector(_PREFIX),
                        create_parameter_panel(_PREFIX),
                        create_isotope_panel(_PREFIX),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("View Mode", className="card-title"),
                                    dcc.RadioItems(
                                        id=f"{_PREFIX}-view-mode",
                                        options=[
                                            {"label": "Heatmap", "value": "heatmap"},
                                            {"label": "Contour Map", "value": "contour"},
                                            {"label": "3D Surface", "value": "3d"},
                                        ],
                                        value="heatmap",
                                        inline=True,
                                        inputStyle={"marginRight": "4px"},
                                        labelStyle={"marginRight": "16px"},
                                    ),
                                ]
                            ),
                            className="mb-3",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Computed Info", className="card-title"),
                                    html.Div(id=f"{_PREFIX}-info"),
                                ]
                            ),
                            className="mb-3",
                        ),
                    ],
                    md=3,
                ),
                # Right column — figures
                dbc.Col(
                    [
                        dcc.Loading(dcc.Graph(id=f"{_PREFIX}-moire-graph")),
                        dcc.Loading(dcc.Graph(id=f"{_PREFIX}-gap-graph")),
                    ],
                    md=9,
                ),
            ]
        ),
    ],
    fluid=True,
    className="p-4",
)


@callback(
    Output(f"{_PREFIX}-moire-graph", "figure"),
    Output(f"{_PREFIX}-gap-graph", "figure"),
    Output(f"{_PREFIX}-info", "children"),
    Output(f"{_PREFIX}-isotope-info", "children"),
    Input(f"{_PREFIX}-substrate-dropdown", "value"),
    Input(f"{_PREFIX}-overlayer-dropdown", "value"),
    Input(f"{_PREFIX}-twist-slider", "value"),
    Input(f"{_PREFIX}-grid-size", "value"),
    Input(f"{_PREFIX}-physical-extent", "value"),
    Input(f"{_PREFIX}-view-mode", "value"),
    Input(f"{_PREFIX}-isotope-enabled", "value"),
    Input(f"{_PREFIX}-fe-mass", "value"),
    Input(f"{_PREFIX}-te-mass", "value"),
    Input(f"{_PREFIX}-sb-mass", "value"),
    Input(f"{_PREFIX}-isotope-alpha", "value"),
    Input(f"{_PREFIX}-isotope-comparison", "value"),
    Input("theme-store", "data"),
)
def _update_viewer(
    substrate_formula: str,
    overlayer_formula: str,
    twist_angle: float,
    grid_size: int,
    physical_extent: float,
    view_mode: str,
    isotope_enabled: list,
    fe_mass: float,
    te_mass: float,
    sb_mass: float,
    isotope_alpha: float,
    isotope_comparison: list,
    theme: str,
):
    dark = theme == "dark"
    substrate = get_material(substrate_formula)
    overlayer = get_material(overlayer_formula)

    grid_size = int(grid_size or 200)
    physical_extent = float(physical_extent or 100.0)
    twist_angle = float(twist_angle or 0.0)
    view_mode = view_mode or "heatmap"

    isotope_on = bool(isotope_enabled and "on" in isotope_enabled)
    show_comparison = bool(isotope_comparison and "on" in isotope_comparison)

    # Compute isotope effects when enabled
    sub_a = substrate.a
    over_a = overlayer.a
    dw_sub = 1.0
    dw_over = 1.0
    d_avg = DELTA_AVG
    d_amp = DELTA_AMPLITUDE
    coh_len = 20.0
    isotope_info_children = []

    if isotope_on:
        mass_overrides = {
            "Fe": float(fe_mass or 0),
            "Te": float(te_mass or 0),
            "Sb": float(sb_mass or 0),
        }
        alpha = float(isotope_alpha or 0.4)

        effects = compute_isotope_effects(
            substrate_formula=substrate.formula,
            overlayer_formula=overlayer.formula,
            substrate_a=substrate.a,
            overlayer_a=overlayer.a,
            overlayer_lattice_type=overlayer.lattice_type,
            delta_1=DELTA_1,
            delta_2=DELTA_2,
            mass_overrides=mass_overrides,
            alpha=alpha,
        )

        sub_a = effects.substrate_a_modified
        over_a = effects.overlayer_a_modified
        dw_sub = effects.dw_factor_substrate
        dw_over = effects.dw_factor_overlayer
        d_avg = (effects.delta_1_modified + effects.delta_2_modified) / 2.0
        d_amp = (effects.delta_2_modified - effects.delta_1_modified) / 2.0
        coh_len = effects.coherence_length_modified

        isotope_info_children = [
            html.Small(
                [
                    html.Strong("Isotope modifications (speculative):"),
                    html.Br(),
                    f"Substrate da: {effects.substrate_delta_a:+.5f} A",
                    html.Br(),
                    f"Overlayer da: {effects.overlayer_delta_a:+.5f} A",
                    html.Br(),
                    f"Delta1: {effects.delta_1_modified:.3f} meV (was {DELTA_1:.2f})",
                    html.Br(),
                    f"Delta2: {effects.delta_2_modified:.3f} meV (was {DELTA_2:.2f})",
                    html.Br(),
                    f"Coherence: {effects.coherence_length_modified:.2f} A (was 20.00)",
                    html.Br(),
                    f"DW sub: {effects.dw_factor_substrate:.6f}",
                    html.Br(),
                    f"DW over: {effects.dw_factor_overlayer:.6f}",
                    html.Br(),
                    f"125Te spin fraction: {effects.te_125_spin_fraction:.3f}"
                    f" (nat: 0.071)",
                ]
            )
        ]

    result = generate_moire_pattern(
        substrate_a=sub_a,
        overlayer_a=over_a,
        overlayer_lattice_type=overlayer.lattice_type,
        twist_angle_deg=twist_angle,
        grid_size=grid_size,
        physical_extent=physical_extent,
        dw_factor_substrate=dw_sub,
        dw_factor_overlayer=dw_over,
    )

    gap = gap_modulation(result["pattern"], d_avg, d_amp)

    x, y = result["x"], result["y"]
    pattern = result["pattern"]

    suffix = " [isotope-modified]" if isotope_on else ""
    moire_title = f"Moire: {substrate.formula} / {overlayer.formula}{suffix}"
    gap_title = f"Superconducting Gap Modulation{suffix}"

    if view_mode == "contour":
        moire_fig = create_2d_contour(
            x, y, pattern, title=moire_title,
            colorscale="Viridis", z_label="Intensity",
            dark=dark,
        )
        gap_fig = create_2d_contour(
            x, y, gap, title=gap_title,
            colorscale="RdBu_r", z_label="Delta (meV)",
            dark=dark,
        )
    elif view_mode == "3d":
        if grid_size > 150:
            x_s, y_s = x[::2], y[::2]
            pattern_s = pattern[::2, ::2]
            gap_s = gap[::2, ::2]
        else:
            x_s, y_s, pattern_s, gap_s = x, y, pattern, gap
        moire_fig = create_3d_surface(
            x_s, y_s, pattern_s, title=moire_title,
            colorscale="Viridis", z_label="Intensity",
            dark=dark,
        )
        gap_fig = create_3d_surface(
            x_s, y_s, gap_s, title=gap_title,
            colorscale="RdBu_r", z_label="Delta (meV)",
            dark=dark,
        )
    else:
        moire_fig = create_moire_heatmap(
            x, y, pattern, title=moire_title,
            dark=dark,
        )
        gap_fig = create_gap_heatmap(
            x, y, gap, title=gap_title,
            dark=dark,
        )

    # If comparison mode, overlay the natural pattern as contour lines
    if isotope_on and show_comparison:
        natural_result = generate_moire_pattern(
            substrate_a=substrate.a,
            overlayer_a=overlayer.a,
            overlayer_lattice_type=overlayer.lattice_type,
            twist_angle_deg=twist_angle,
            grid_size=grid_size,
            physical_extent=physical_extent,
        )
        import plotly.graph_objects as go

        nat_period = natural_result["moire_period"]
        mod_period = result["moire_period"]
        period_shift = mod_period - nat_period

        isotope_info_children.append(
            html.Small(
                [
                    html.Br(),
                    html.Strong("Comparison:"),
                    html.Br(),
                    f"Natural period: {nat_period:.4f} A",
                    html.Br(),
                    f"Modified period: {mod_period:.4f} A",
                    html.Br(),
                    f"Period shift: {period_shift:+.4f} A",
                ]
            )
        )

        if view_mode == "heatmap":
            diff = result["pattern"] - natural_result["pattern"]
            moire_fig.add_trace(
                go.Contour(
                    x=x,
                    y=y,
                    z=diff,
                    contours=dict(
                        coloring="lines",
                        showlabels=True,
                    ),
                    line=dict(width=1),
                    colorscale="RdBu_r",
                    opacity=0.5,
                    showscale=False,
                    name="Difference",
                )
            )
            moire_fig.update_layout(
                title=f"Moire: {substrate.formula} / {overlayer.formula} "
                "[enriched + difference contours]"
            )

    mismatch = abs(sub_a - over_a) / sub_a * 100.0
    cpdm_amp = cpdm_amplitude(result["moire_period"], coherence_length=coh_len)

    info = [
        html.P(f"Substrate: {substrate.formula} (a = {sub_a:.4f} A)"),
        html.P(f"Overlayer: {overlayer.formula} (a = {over_a:.4f} A)"),
        html.P(f"Lattice mismatch: {mismatch:.2f}%"),
        html.P(f"Twist angle: {twist_angle:.1f} deg"),
        html.P(f"Moire period: {result['moire_period']:.2f} A"),
        html.P(f"CPDM amplitude: {cpdm_amp:.4f}"),
    ]

    return moire_fig, gap_fig, info, isotope_info_children
