"""Graphene page — stacked/twisted graphene moire patterns and curvature effects."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from waytogocoop.components.controls import (
    loading_spinner,
    preset_and_reset_card,
    register_preset_reset_callback,
)
from waytogocoop.components.figure_factory import (
    create_2d_contour,
    create_3d_surface,
    create_band_structure_plot,
    create_dos_plot,
    create_fft_heatmap,
    create_flat_band_plot,
    create_gap_heatmap,
    create_moire_heatmap,
    create_pseudo_field_heatmap,
)
from waytogocoop.components.graphene_panel import (
    STACK_LABELS,
    create_curvature_panel,
    create_graphene_stack_panel,
    create_supermoire_panel,
)
from waytogocoop.components.parameter_panel import create_parameter_panel
from waytogocoop.computation.bm_model import (
    BMConfig,
    compute_band_structure,
    compute_dos,
    flat_band_width_mev,
)
from waytogocoop.computation.curvature import (
    CurvatureConfig,
    compute_curvature_effects,
    displacement_field,
)
from waytogocoop.computation.fourier import fft_2d
from waytogocoop.computation.graphene import (
    compute_flat_band_sc,
    dirac_velocity_ratio,
    generate_stack_pattern_v2,
    generate_supermoire_pattern,
    magic_angle_deg,
)
from waytogocoop.computation.superconducting import cpdm_amplitude, gap_modulation
from waytogocoop.config import XI_TBG
from waytogocoop.materials.database import get_material
from waytogocoop.state import register_url_sync

dash.register_page(
    __name__, path="/graphene", name="Graphene", title="Good Job Coop! - Graphene"
)

_PREFIX = "graphene"
_URL_ID = f"{_PREFIX}-url"

# Neutral values for the v2 controls, shared by every preset that does not
# override them.
_V2_NEUTRAL = {
    "honeycomb": False,
    "valley": 1,
    "strain_pct": 0,
    "strain_ang": 0,
    "overlayer": "none",
    "iface_twist": 0,
    "warp": False,
}

_PRESETS = {
    "Magic-angle TBG": {
        "stack": "twisted_bilayer",
        "twist": 1.08,
        "filling": 2.4,
        "curvature": "flat",
        "amp": 5,
        "feature": 100,
        "phi": 30,
        "grid_size": 200,
        "extent": 200.0,
        "view_mode": "pattern",
        **_V2_NEUTRAL,
    },
    "Alt-twist trilayer": {
        "stack": "alternating_trilayer",
        "twist": 1.52,
        "filling": 2.4,
        "curvature": "flat",
        "amp": 5,
        "feature": 100,
        "phi": 30,
        "grid_size": 200,
        "extent": 200.0,
        "view_mode": "gap",
        **_V2_NEUTRAL,
    },
    "AB (Bernal) bilayer": {
        "stack": "AB",
        "twist": 0.0,
        "filling": 2.4,
        "curvature": "flat",
        "amp": 5,
        "feature": 100,
        "phi": 30,
        "grid_size": 150,
        "extent": 15.0,
        "view_mode": "pattern",
        **_V2_NEUTRAL,
    },
    "Armchair ripple pseudo-field": {
        "stack": "twisted_bilayer",
        "twist": 1.08,
        "filling": 2.4,
        "curvature": "sinusoidal_ripple",
        "amp": 2,
        "feature": 100,
        "phi": 30,
        "grid_size": 200,
        "extent": 200.0,
        "view_mode": "pseudo_field",
        **_V2_NEUTRAL,
    },
    "Gaussian bump (curved 3D)": {
        "stack": "twisted_bilayer",
        "twist": 1.08,
        "filling": 2.4,
        "curvature": "gaussian_bump",
        "amp": 5,
        "feature": 50,
        "phi": 30,
        "grid_size": 200,
        "extent": 200.0,
        "view_mode": "curved3d",
        **_V2_NEUTRAL,
    },
    "Supermoire on Sb2Te3": {
        "stack": "twisted_bilayer",
        "twist": 1.08,
        "filling": 2.4,
        "curvature": "flat",
        "amp": 5,
        "feature": 100,
        "phi": 30,
        "grid_size": 200,
        "extent": 200.0,
        "view_mode": "pattern",
        **_V2_NEUTRAL,
        "overlayer": "Sb2Te3",
        "iface_twist": 0,
    },
    "Magic-angle flat bands": {
        "stack": "twisted_bilayer",
        "twist": 1.08,
        "filling": 2.4,
        "curvature": "flat",
        "amp": 5,
        "feature": 100,
        "phi": 30,
        "grid_size": 200,
        "extent": 200.0,
        "view_mode": "bands",
        **_V2_NEUTRAL,
    },
}

_DEFAULTS = dict(_PRESETS["Magic-angle TBG"])

_PRESET_BINDINGS = [
    (f"{_PREFIX}-stack-dropdown", "value", "stack"),
    (f"{_PREFIX}-twist-slider", "value", "twist"),
    (f"{_PREFIX}-filling-slider", "value", "filling"),
    (f"{_PREFIX}-honeycomb-switch", "value", "honeycomb"),
    (f"{_PREFIX}-valley-radio", "value", "valley"),
    (f"{_PREFIX}-strain-percent", "value", "strain_pct"),
    (f"{_PREFIX}-strain-angle", "value", "strain_ang"),
    (f"{_PREFIX}-overlayer-dropdown", "value", "overlayer"),
    (f"{_PREFIX}-interface-twist", "value", "iface_twist"),
    (f"{_PREFIX}-curvature-dropdown", "value", "curvature"),
    (f"{_PREFIX}-curv-amplitude", "value", "amp"),
    (f"{_PREFIX}-curv-feature", "value", "feature"),
    (f"{_PREFIX}-curv-orientation", "value", "phi"),
    (f"{_PREFIX}-warp-switch", "value", "warp"),
    (f"{_PREFIX}-grid-size", "value", "grid_size"),
    (f"{_PREFIX}-physical-extent", "value", "extent"),
    (f"{_PREFIX}-view-mode", "value", "view_mode"),
]

layout = dbc.Container(
    [
        dcc.Location(id=_URL_ID, refresh=False),
        html.Br(),
        html.H2("Graphene Stacks & Curvature"),
        html.Hr(),
        dbc.Row(
            [
                # Left column — controls
                dbc.Col(
                    [
                        preset_and_reset_card(_PREFIX, _PRESETS),
                        create_graphene_stack_panel(_PREFIX),
                        create_supermoire_panel(_PREFIX),
                        create_parameter_panel(
                            _PREFIX,
                            twist_max=5.0,
                            twist_step=0.01,
                            twist_marks={
                                0: "0",
                                1.08: "1.08",
                                1.52: "1.52",
                                3: "3",
                                5: "5",
                            },
                            twist_default=1.08,
                            extent_min=10,
                            extent_max=400,
                            extent_default=200.0,
                        ),
                        create_curvature_panel(_PREFIX),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("View Mode", className="card-title"),
                                    dcc.RadioItems(
                                        id=f"{_PREFIX}-view-mode",
                                        options=[
                                            {"label": "Pattern", "value": "pattern"},
                                            {
                                                "label": "Gap map (speculative)",
                                                "value": "gap",
                                            },
                                            {
                                                "label": "Pseudo-field (speculative)",
                                                "value": "pseudo_field",
                                            },
                                            {"label": "Strain map", "value": "strain"},
                                            {
                                                "label": "Curved 3D sheet",
                                                "value": "curved3d",
                                            },
                                            {"label": "FFT", "value": "fourier"},
                                            {
                                                "label": "Band structure",
                                                "value": "bands",
                                            },
                                            {"label": "DOS", "value": "dos"},
                                        ],
                                        value="pattern",
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
                    xs=12, md=4, lg=3,
                    className="sidebar-col",
                ),
                # Right column — figures
                dbc.Col(
                    [
                        loading_spinner(
                            dcc.Graph(id=f"{_PREFIX}-main-graph"),
                            "Computing graphene stack…",
                        ),
                        loading_spinner(
                            dcc.Graph(id=f"{_PREFIX}-flatband-graph"),
                            "Computing flat-band curve…",
                        ),
                    ],
                    xs=12, md=8, lg=9,
                ),
            ]
        ),
    ],
    fluid=True,
    className="p-4",
)


register_preset_reset_callback(_PREFIX, _PRESETS, _DEFAULTS, _PRESET_BINDINGS)
register_url_sync(_URL_ID, _PRESET_BINDINGS)


def _fmt_period(period: float) -> str:
    return "infinite" if np.isinf(period) else f"{period:.2f} A"


@callback(
    Output(f"{_PREFIX}-main-graph", "figure"),
    Output(f"{_PREFIX}-flatband-graph", "figure"),
    Output(f"{_PREFIX}-info", "children"),
    Output(f"{_PREFIX}-stack-info", "children"),
    Input(f"{_PREFIX}-stack-dropdown", "value"),
    Input(f"{_PREFIX}-twist-slider", "value"),
    Input(f"{_PREFIX}-filling-slider", "value"),
    Input(f"{_PREFIX}-honeycomb-switch", "value"),
    Input(f"{_PREFIX}-valley-radio", "value"),
    Input(f"{_PREFIX}-strain-percent", "value"),
    Input(f"{_PREFIX}-strain-angle", "value"),
    Input(f"{_PREFIX}-overlayer-dropdown", "value"),
    Input(f"{_PREFIX}-interface-twist", "value"),
    Input(f"{_PREFIX}-curvature-dropdown", "value"),
    Input(f"{_PREFIX}-curv-amplitude", "value"),
    Input(f"{_PREFIX}-curv-feature", "value"),
    Input(f"{_PREFIX}-curv-orientation", "value"),
    Input(f"{_PREFIX}-warp-switch", "value"),
    Input(f"{_PREFIX}-grid-size", "value"),
    Input(f"{_PREFIX}-physical-extent", "value"),
    Input(f"{_PREFIX}-view-mode", "value"),
    Input("theme-store", "data"),
)
def _update_graphene(
    stack: str,
    twist: float,
    filling: float,
    honeycomb: bool,
    valley: int,
    strain_pct: float,
    strain_ang: float,
    overlayer: str,
    iface_twist: float,
    curv_geom: str,
    amp: float,
    feature: float,
    phi: float,
    warp: bool,
    grid_size: int,
    extent: float,
    view_mode: str,
    theme: str,
):
    try:
        dark = theme == "dark"
        stack = stack or "twisted_bilayer"
        twist = float(twist) if twist is not None else 1.08
        filling = float(filling) if filling is not None else 2.4
        honeycomb = bool(honeycomb)
        valley = valley if valley in (1, -1) else 1
        strain_pct = float(strain_pct) if strain_pct is not None else 0.0
        strain_ang = float(strain_ang) if strain_ang is not None else 0.0
        overlayer = overlayer or "none"
        iface_twist = float(iface_twist) if iface_twist is not None else 0.0
        curv_geom = curv_geom or "flat"
        amp = float(amp) if amp is not None else 5.0
        feature = float(feature) if feature is not None else 100.0
        phi = float(phi) if phi is not None else 30.0
        warp = bool(warp)
        grid_size = int(grid_size) if grid_size is not None else 200
        extent = float(extent) if extent is not None else 200.0
        view_mode = view_mode or "pattern"

        # Curvature must run before the pattern: the warp displacement feeds
        # the layer potentials.  This linspace matches the one
        # generate_stack_pattern_v2 rebuilds internally, so the displacement
        # arrays line up with the pattern grid.
        x = np.linspace(-extent, extent, grid_size)
        y = np.linspace(-extent, extent, grid_size)

        # One feature-size slider feeds whichever length scale the geometry uses.
        curv_cfg = CurvatureConfig(
            geometry=curv_geom,
            amplitude=amp,
            sigma=feature,
            wavelength=feature,
            radius=feature,
            orientation_deg=phi,
            valley=valley,
        )
        curv = compute_curvature_effects(curv_cfg, x, y)

        supermoire = overlayer != "none"
        over_mat = get_material(overlayer) if supermoire else None
        if supermoire:
            result = generate_supermoire_pattern(
                stack,
                twist,
                overlayer_a=over_mat.a,
                overlayer_lattice_type=over_mat.lattice_type,
                interface_twist_deg=iface_twist,
                grid_size=grid_size,
                physical_extent=extent,
                honeycomb=honeycomb,
            )
            moire_period = result["stack_period"]
        else:
            disp = (
                displacement_field(curv.height, x, y)
                if warp and curv_geom != "flat"
                else None
            )
            result = generate_stack_pattern_v2(
                stack,
                twist,
                grid_size=grid_size,
                physical_extent=extent,
                honeycomb=honeycomb,
                heterostrain_percent=strain_pct,
                heterostrain_angle_deg=strain_ang,
                displacement=disp,
            )
            moire_period = result["moire_period"]
        x, y, pattern = result["x"], result["y"], result["pattern"]
        # Supermoire results count the overlayer; the flat-band model wants
        # the graphene layers only.
        n_gr_layers = result["n_layers"] - 1 if supermoire else result["n_layers"]

        twisted = (
            stack in ("twisted_bilayer", "alternating_trilayer")
            and bool(twist)
            and twist > 0
        )
        fb = compute_flat_band_sc(twist, n_gr_layers, filling) if twisted else None
        delta_max = fb.delta_mev if fb else 0.0

        gap = gap_modulation(pattern, delta_max, 0.5 * delta_max) * curv.gap_suppression
        strain_mag = np.sqrt(
            curv.strain_xx**2 + curv.strain_yy**2 + 2.0 * curv.strain_xy**2
        )
        cpdm = cpdm_amplitude(moire_period, coherence_length=XI_TBG)
        stack_label = STACK_LABELS.get(stack, stack)

        # BM band structure / DOS are bilayer-only: trilayer stacks show the
        # bilayer result with the approximation flagged in the title.  The BM
        # solve only runs in these views so the other views stay responsive.
        w_mev = None
        if view_mode in ("bands", "dos"):
            bm_cfg = BMConfig(
                twist_angle_deg=twist if twisted else magic_angle_deg(2),
                valley=valley,
            )
            bm_note = ": bilayer approximation" if n_gr_layers == 3 else ""

        if view_mode == "gap":
            fig = create_gap_heatmap(
                x, y, gap, title="Flat-Band Gap Map (SPECULATIVE)", dark=dark
            )
        elif view_mode == "pseudo_field":
            fig = create_pseudo_field_heatmap(x, y, curv.pseudo_field, dark=dark)
        elif view_mode == "strain":
            fig = create_2d_contour(
                x, y, strain_mag, title="Strain Magnitude",
                colorscale="Viridis", z_label="|eps|",
                dark=dark,
            )
        elif view_mode == "curved3d":
            if grid_size > 150:
                x_s, y_s = x[::2], y[::2]
                height_s = curv.height[::2, ::2]
                gap_s = gap[::2, ::2]
            else:
                x_s, y_s, height_s, gap_s = x, y, curv.height, gap
            fig = create_3d_surface(
                x_s, y_s, height_s,
                title="Curved Sheet - Gap Overlay (SPECULATIVE)",
                colorscale="RdBu_r", z_label="h (A)",
                dark=dark,
                surfacecolor=gap_s,
                color_label="Delta (meV)",
            )
        elif view_mode == "fourier":
            dx = (x[-1] - x[0]) / (len(x) - 1)
            fft_result = fft_2d(pattern, dx)
            fig = create_fft_heatmap(
                fft_result["kx"],
                fft_result["ky"],
                fft_result["power_spectrum"],
                title=f"FFT: {stack_label}",
                dark=dark,
            )
        elif view_mode == "bands":
            bands = compute_band_structure(bm_cfg)
            w_mev = bands.flat_bandwidth_mev
            fig = create_band_structure_plot(
                bands.k_distances,
                bands.energies_mev,
                bands.tick_positions,
                bands.tick_labels,
                bands.flat_bandwidth_mev,
                title=f"Moire Band Structure (BM model{bm_note})",
                dark=dark,
            )
        elif view_mode == "dos":
            dos_result = compute_dos(bm_cfg)
            w_mev = flat_band_width_mev(bm_cfg)
            fig = create_dos_plot(
                dos_result["energies_mev"],
                dos_result["dos"],
                title=f"Density of States (BM model{bm_note})",
                dark=dark,
            )
        else:
            fig = create_moire_heatmap(
                x, y, pattern, title=f"Moire Pattern: {stack_label}", dark=dark
            )

        thetas = np.linspace(0.3, 3.0, 200)
        n_layers_curve = 2 if stack in ("AA", "AB", "twisted_bilayer") else 3
        velocity = dirac_velocity_ratio(thetas)
        deltas = np.array(
            [
                compute_flat_band_sc(t, n_layers_curve, filling).delta_mev
                for t in thetas
            ]
        )
        theta_magic = magic_angle_deg(n_gr_layers if n_gr_layers in (2, 3) else 2)
        theta_current = twist if twisted else theta_magic
        flat_fig = create_flat_band_plot(
            thetas, velocity, deltas, theta_current, dark=dark
        )

        valley_label = "K" if valley == 1 else "K'"
        info = [
            html.P(f"Stack: {stack_label} ({n_gr_layers} layers)"),
            html.P(f"Twist angle: {twist:.2f} deg"),
            html.P(f"Valley: {valley_label}"),
        ]
        if twisted:
            info.append(html.P(f"v*/v_F: {fb.velocity_ratio:.4f}"))
        if abs(strain_pct) > 0:
            info.append(
                html.P(f"Heterostrain eps: {strain_pct:.2f}% at {strain_ang:.0f} deg")
            )
        if supermoire:
            info.extend(
                [
                    html.P(f"Overlayer: {over_mat.formula}"),
                    html.P(f"Stack period: {_fmt_period(result['stack_period'])}"),
                    html.P(
                        f"Interface period: {_fmt_period(result['interface_period'])}"
                    ),
                    html.P(
                        f"Supermoire period: {_fmt_period(result['supermoire_period'])}"
                    ),
                ]
            )
        else:
            info.append(html.P(f"Moire period: {_fmt_period(moire_period)}"))
        if w_mev is not None:
            info.append(html.P(f"Flat-band width W: {w_mev:.1f} meV"))
        info.extend(
            [
                html.P(f"theta_magic ({n_gr_layers}L): {theta_magic:.3f} deg"),
                html.P(f"Delta_max: {delta_max:.3f} meV"),
                html.P(f"Tc: {fb.tc_kelvin:.3f} K" if fb else "Tc: 0.000 K"),
                html.P(
                    f"Dome factor: {fb.dome_factor:.3f}" if fb else "Dome factor: 0.000"
                ),
                html.P(f"CPDM amplitude: {cpdm:.4f}"),
                html.P(f"max |B_ps|: {curv.max_abs_field:.2f} T"),
                html.P(f"max strain: {float(strain_mag.max()):.4f}"),
            ]
        )

        notes = [
            html.Strong("Speculative models:"),
            html.Br(),
            "Flat-band Delta/Tc and pseudo-field pair-breaking are "
            "qualitative toy models, not fits to experiment.",
        ]
        if supermoire:
            notes.extend(
                [
                    html.Br(),
                    "Heterostrain applies inside the graphene stack only; the "
                    "supermoire pattern above does not include it.",
                ]
            )
        stack_info = html.Small(notes)

        return fig, flat_fig, info, stack_info
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Computation error: {e}")
        return error_fig, error_fig, html.P(str(e), style={"color": "red"}), []
