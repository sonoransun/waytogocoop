"""Pair Density Waves page — PDW modalities and the charge orders they leave behind."""

from __future__ import annotations

from dataclasses import replace

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from waytogocoop.components.controls import (
    loading_spinner,
    preset_and_reset_card,
    register_preset_reset_callback,
)
from waytogocoop.components.figure_factory import (
    create_gap_heatmap,
    create_pdw_peak_map,
    create_pdw_regime_map,
    create_pdw_sweep_plot,
    create_signed_field_heatmap,
)
from waytogocoop.components.material_selector import create_material_selector
from waytogocoop.components.parameter_panel import (
    create_parameter_panel,
    is_valid_extent,
    is_valid_grid_size,
)
from waytogocoop.components.pdw_panel import (
    FIELD_AXES,
    FIELD_AXIS_LABELS,
    T_PDW_RATIO_UTE2,
    VIEW_OPTIONS,
    create_pdw_panel,
)
from waytogocoop.computation.pdw import (
    PdwConfig,
    charge_spectrum,
    compute_pdw_state,
    critical_field_along,
    field_sweep,
    moire_locked_config,
    moire_locked_pdw_ratio,
    moire_locked_period,
    onset_field_along,
    pdw_peaks,
    pdw_real_space,
    reduced_field,
    regime_map,
    temperature_sweep,
)
from waytogocoop.config import (
    DELTA_AVG,
    PDW_KAPPA_H,
    PDW_KAPPA_HARM,
    PDW_KAPPA_P,
    PDW_KAPPA_Q,
    TC_FETE_FILM,
    TC_UTE2,
)
from waytogocoop.materials.database import get_material
from waytogocoop.state import register_url_sync

dash.register_page(
    __name__,
    path="/pdw",
    name="Pair Density Waves",
    title="Good Job Coop! - Pair Density Waves",
)

_PREFIX = "pdw"
_URL_ID = f"{_PREFIX}-url"

_REGIME_TEXT: dict[str, tuple[str, str]] = {
    "normal": ("Normal", "No pairing order: every composite charge order is gone."),
    "pdw_only": (
        "PDW only (Cheshire cat)",
        "Cooper pairs without uniform superconductivity. Only the q = 2p and h "
        "charge orders survive, the vestige seen above Tc in UTe2.",
    ),
    "uniform_only": (
        "Uniform SC only",
        "The field has destroyed the PDW but not the uniform condensate, so no "
        "composite charge order remains.",
    ),
    "coexistence": (
        "PDW + uniform SC",
        "Both orders present: the p CDW (PDW × uniform SC) joins the q and h peaks.",
    ),
}

# Slider range per geometry: UTe2 lives below ~6 K, the FeTe films near 13.5 K.
_T_SLIDER: dict[str, tuple[float, dict]] = {
    "ute2": (6.0, {0: "0", TC_UTE2: "Tc", 6: "6"}),
    "moire": (30.0, {0: "0", TC_FETE_FILM: "Tc", 30: "30"}),
}

_BASE_PRESET = {
    "geometry": "ute2",
    "temperature": 0.3,
    "tpdw_ratio": T_PDW_RATIO_UTE2,
    "bx": 0.0,
    "by": 0.0,
    "bz": 0.0,
    "sweep_axis": "z",
    "kappa_p": PDW_KAPPA_P,
    "kappa_q": PDW_KAPPA_Q,
    "kappa_h": PDW_KAPPA_H,
    "kappa_harm": PDW_KAPPA_HARM,
    "floor": -2.0,
    "substrate": "FeTe",
    "overlayer": "Sb2Te3",
    "twist": 0.0,
    "grid_size": 200,
    "extent": 40.0,
    "view_mode": "charge",
}

_PRESETS = {
    "UTe2 300 mK, zero field": dict(_BASE_PRESET),
    "UTe2 3 K — Cheshire-cat window": {
        **_BASE_PRESET,
        "temperature": 3.0,
        "view_mode": "fft",
    },
    "UTe2 field-induced p2 (Bz ≈ b* = 6 T)": {
        **_BASE_PRESET,
        "bz": 6.0,
        "view_mode": "peaks",
    },
    "UTe2 4.2 K, By = a = 0.2 T (q fading)": {
        **_BASE_PRESET,
        "temperature": 4.2,
        "by": 0.2,
        "sweep_axis": "y",
        "view_mode": "peaks",
    },
    "Moire-locked CPDM (FeTe / Sb2Te3)": {
        **_BASE_PRESET,
        "geometry": "moire",
        "temperature": 4.0,
        "tpdw_ratio": 1.0,
        "extent": 100.0,
        "view_mode": "pair",
    },
    "Moire what-if: intrinsic PDW above Tc": {
        **_BASE_PRESET,
        "geometry": "moire",
        "temperature": 16.0,
        "tpdw_ratio": 1.5,
        "extent": 100.0,
        "view_mode": "charge",
    },
}

_DEFAULTS = dict(_BASE_PRESET)

# T_PDW / Tc each geometry starts from: UTe2's measured ratio, and 1 for the
# imprinted moire CPDM (no PDW-only window).
_TPDW_RATIO_DEFAULT: dict[str, float] = {"ute2": T_PDW_RATIO_UTE2, "moire": 1.0}

# How the sweep axis reads in an axis title ("|B| along ...").
_AXIS_PHRASE: dict[str, str] = {**FIELD_AXIS_LABELS, "along": "the applied B"}

_PRESET_BINDINGS = [
    (f"{_PREFIX}-geometry", "value", "geometry"),
    (f"{_PREFIX}-temperature", "value", "temperature"),
    (f"{_PREFIX}-tpdw-ratio", "value", "tpdw_ratio"),
    (f"{_PREFIX}-bx", "value", "bx"),
    (f"{_PREFIX}-by", "value", "by"),
    (f"{_PREFIX}-bz", "value", "bz"),
    (f"{_PREFIX}-sweep-axis", "value", "sweep_axis"),
    (f"{_PREFIX}-kappa-p", "value", "kappa_p"),
    (f"{_PREFIX}-kappa-q", "value", "kappa_q"),
    (f"{_PREFIX}-kappa-h", "value", "kappa_h"),
    (f"{_PREFIX}-kappa-harm", "value", "kappa_harm"),
    (f"{_PREFIX}-floor", "value", "floor"),
    (f"{_PREFIX}-substrate-dropdown", "value", "substrate"),
    (f"{_PREFIX}-overlayer-dropdown", "value", "overlayer"),
    (f"{_PREFIX}-twist-slider", "value", "twist"),
    (f"{_PREFIX}-grid-size", "value", "grid_size"),
    (f"{_PREFIX}-physical-extent", "value", "extent"),
    (f"{_PREFIX}-view-mode", "value", "view_mode"),
]

layout = dbc.Container(
    [
        dcc.Location(id=_URL_ID, refresh=False),
        html.Br(),
        html.H2("Pair Density Waves (SPECULATIVE)"),
        dbc.Alert(
            [
                "A phenomenological Landau model of the PDW modalities that "
                "vector-field STM resolves in UTe2 — Zhu et al., ",
                html.A(
                    "PNAS 123, e2602117123 (2026)",
                    href="https://doi.org/10.1073/pnas.2602117123",
                    target="_blank",
                ),
                ". The PDW at ±p and the uniform condensate leave three charge "
                "orders: p (needs both), q = 2p (the PDW alone, surviving above Tc) "
                "and the h harmonics. Qualitative illustration, not a prediction.",
            ],
            color="danger",
            className="mb-3",
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        preset_and_reset_card(_PREFIX, _PRESETS),
                        create_pdw_panel(_PREFIX),
                        html.Div(
                            create_material_selector(_PREFIX),
                            id=f"{_PREFIX}-moire-controls",
                        ),
                        create_parameter_panel(
                            _PREFIX,
                            extent_min=10,
                            extent_max=400,
                            extent_default=_DEFAULTS["extent"],
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("View Mode", className="card-title"),
                                    dcc.RadioItems(
                                        id=f"{_PREFIX}-view-mode",
                                        options=VIEW_OPTIONS,
                                        value=_DEFAULTS["view_mode"],
                                        inputStyle={"marginRight": "4px"},
                                        labelStyle={"display": "block"},
                                    ),
                                ]
                            ),
                            className="mb-3",
                        ),
                    ],
                    xs=12, md=4, lg=3,
                    className="sidebar-col",
                ),
                dbc.Col(
                    [
                        loading_spinner(
                            dcc.Graph(id=f"{_PREFIX}-main-graph"),
                            "Computing PDW state…",
                        ),
                        html.Div(id=f"{_PREFIX}-info", className="mb-3"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Loading(dcc.Graph(id=f"{_PREFIX}-tsweep-graph")),
                                    lg=6,
                                ),
                                dbc.Col(
                                    dcc.Loading(dcc.Graph(id=f"{_PREFIX}-bsweep-graph")),
                                    lg=6,
                                ),
                            ]
                        ),
                        dcc.Loading(dcc.Graph(id=f"{_PREFIX}-regime-graph")),
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


@callback(
    Output(f"{_PREFIX}-couplings-collapse", "is_open"),
    Input(f"{_PREFIX}-couplings-toggle", "n_clicks"),
    State(f"{_PREFIX}-couplings-collapse", "is_open"),
)
def _toggle_couplings(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output(f"{_PREFIX}-moire-controls", "style"),
    Output(f"{_PREFIX}-temperature", "max"),
    Output(f"{_PREFIX}-temperature", "marks"),
    Input(f"{_PREFIX}-geometry", "value"),
)
def _geometry_controls(geometry):
    t_max, marks = _T_SLIDER.get(geometry, _T_SLIDER["ute2"])
    style = {} if geometry == "moire" else {"display": "none"}
    return style, t_max, marks


@callback(
    Output(f"{_PREFIX}-temperature", "value", allow_duplicate=True),
    Input(f"{_PREFIX}-geometry", "value"),
    State(f"{_PREFIX}-temperature", "value"),
    prevent_initial_call=True,
)
def _clamp_temperature(geometry, temperature):
    # Switching to UTe2 from a hot moire state would otherwise leave T above
    # the slider's range; presets set T together with the geometry and stay
    # in range, so they pass through untouched.
    t_max = _T_SLIDER.get(geometry, _T_SLIDER["ute2"])[0]
    if temperature is not None and temperature > t_max:
        return t_max
    return dash.no_update


@callback(
    Output(f"{_PREFIX}-tpdw-ratio", "value", allow_duplicate=True),
    Input(f"{_PREFIX}-geometry", "value"),
    State(f"{_PREFIX}-tpdw-ratio", "value"),
    prevent_initial_call=True,
)
def _default_tpdw_ratio(geometry, ratio):
    # A manual geometry switch carrying the *other* geometry's default ratio
    # takes this geometry's default (UTe2 would otherwise put T_PDW near 32 K
    # on the FeTe films).  Presets set a matching ratio and pass untouched, as
    # does any ratio the user chose.
    for other, default in _TPDW_RATIO_DEFAULT.items():
        if other != geometry and ratio is not None and abs(ratio - default) < 1e-9:
            return _TPDW_RATIO_DEFAULT.get(geometry, T_PDW_RATIO_UTE2)
    return dash.no_update


def _number(value, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _build_config(
    geometry, substrate_key, overlayer_key, twist
) -> tuple[PdwConfig, str]:
    if geometry != "moire":
        return PdwConfig(), "UTe2 (01-1) surface"
    substrate = get_material(substrate_key)
    overlayer = get_material(overlayer_key)
    # The shared two-lattice period (not the viewer's estimate) keeps the web
    # and desktop PDW on the same wavevector at any twist.
    period = moire_locked_period(substrate.a, overlayer.a, twist)
    if not np.isfinite(period):
        raise ValueError(
            f"{overlayer.formula}/{substrate.formula} has no moire at this twist; "
            "pick different layers or a nonzero twist"
        )
    config = moire_locked_config(period, twist, DELTA_AVG, moire_locked_pdw_ratio(period))
    return config, f"{overlayer.formula}/{substrate.formula}, L_m = {period:.1f} Å"


def _peak_table(peaks: list) -> dbc.Table:
    header = html.Thead(
        html.Tr([html.Th(c) for c in ("Peak", "|k| (1/Å)", "λ (Å)", "|ρ|²", "Seen")])
    )
    rows = []
    for p in peaks:
        k = float(np.hypot(p.kx, p.ky))
        wavelength = f"{2.0 * np.pi / k:.1f}" if k > 0 else "∞"
        rows.append(
            html.Tr(
                [
                    html.Td(p.label),
                    html.Td(f"{k:.3f}"),
                    html.Td(wavelength),
                    html.Td(f"{p.intensity:.3g}"),
                    html.Td("✓" if p.visible else "–"),
                ]
            )
        )
    return dbc.Table(
        [header, html.Tbody(rows)], bordered=False, hover=True, size="sm", className="mb-0"
    )


@callback(
    Output(f"{_PREFIX}-main-graph", "figure"),
    Output(f"{_PREFIX}-tsweep-graph", "figure"),
    Output(f"{_PREFIX}-bsweep-graph", "figure"),
    Output(f"{_PREFIX}-regime-graph", "figure"),
    Output(f"{_PREFIX}-info", "children"),
    *[Input(cid, prop) for (cid, prop, _key) in _PRESET_BINDINGS],
    Input("theme-store", "data"),
)
def _update_pdw(
    geometry,
    temperature,
    tpdw_ratio,
    bx,
    by,
    bz,
    sweep_axis,
    kappa_p,
    kappa_q,
    kappa_h,
    kappa_harm,
    floor_log,
    substrate_key,
    overlayer_key,
    twist,
    grid_size,
    extent,
    view_mode,
    theme,
):
    try:
        dark = theme == "dark"
        geometry = geometry or "ute2"
        view_mode = view_mode or "charge"
        grid_size = int(grid_size) if is_valid_grid_size(grid_size) else 200
        extent = float(extent) if is_valid_extent(extent) else _DEFAULTS["extent"]
        field = (
            max(_number(bx, 0.0), 0.0),
            max(_number(by, 0.0), 0.0),
            max(_number(bz, 0.0), 0.0),
        )

        base, system = _build_config(
            geometry, substrate_key or "FeTe", overlayer_key or "Sb2Te3", _number(twist, 0.0)
        )
        config = replace(
            base,
            temperature=max(_number(temperature, 0.3), 0.0),
            t_pdw=_number(tpdw_ratio, T_PDW_RATIO_UTE2) * base.tc,
            field=field,
            kappa_p=_number(kappa_p, PDW_KAPPA_P),
            kappa_q=_number(kappa_q, PDW_KAPPA_Q),
            kappa_h=_number(kappa_h, PDW_KAPPA_H),
            kappa_harm=_number(kappa_harm, PDW_KAPPA_HARM),
            floor=10.0 ** _number(floor_log, -2.0),
        )
        state = compute_pdw_state(config)
        peaks = pdw_peaks(config, state)
        regime_name, regime_blurb = _REGIME_TEXT[state.regime]

        # --- Main view ---
        x = np.linspace(-extent, extent, grid_size)
        fields = pdw_real_space(config, state, x, x)
        suffix = f" — {regime_name} (SPECULATIVE)"
        if view_mode == "pair":
            fig = create_gap_heatmap(
                x, x, fields.pair_amplitude, title=f"Pair Amplitude |d(r)|{suffix}", dark=dark
            )
        elif view_mode == "spin":
            fig = create_signed_field_heatmap(
                x, x, fields.spin_density,
                title=f"Composite Spin Density S_z(r){suffix}",
                z_label="S_z (arb.)",
                dark=dark,
            )
        elif view_mode == "fft":
            spectrum = charge_spectrum(config, state, extent)
            fig = create_pdw_peak_map(
                peaks,
                title=f"STM-style FFT of ρ(r){suffix}",
                dark=dark,
                fft=(spectrum["kx"], spectrum["ky"], spectrum["power_spectrum"]),
            )
        elif view_mode == "peaks":
            fig = create_pdw_peak_map(peaks, title=f"Composite CDW Peaks{suffix}", dark=dark)
        else:
            fig = create_signed_field_heatmap(
                x, x, fields.charge_density,
                title=f"Composite Charge Density ρ(r){suffix}",
                z_label="ρ (arb.)",
                dark=dark,
            )

        # --- Field geometry along the sweep axis ---
        b_norm = float(np.linalg.norm(field))
        if sweep_axis == "along" and b_norm > 0:
            direction = np.asarray(field) / b_norm
            axis_label = _AXIS_PHRASE["along"]
        else:
            if sweep_axis not in FIELD_AXES:
                sweep_axis = "z"
            direction = np.asarray(FIELD_AXES[sweep_axis])
            axis_label = _AXIS_PHRASE[sweep_axis]
        b_along = float(abs(np.dot(field, direction)))
        # The current state lies on the field sweep / regime map only when the
        # applied field points along the sweep axis.
        on_axis = abs(b_norm - b_along) < 1e-9
        t_now = config.temperature
        hc2_now = critical_field_along(config.tc, config.hc2, t_now, direction)
        hpdw_now = critical_field_along(config.t_pdw, config.h_pdw, t_now, direction)
        onset = onset_field_along(config.h_rot, direction)
        field_induced = bool(np.any(np.asarray(config.overlap0) < 1.0))
        b_max = 1.15 * max(
            critical_field_along(config.tc, config.hc2, 0.0, direction),
            critical_field_along(config.t_pdw, config.h_pdw, 0.0, direction),
            b_along,
        )

        # --- Temperature sweep (at the applied field) ---
        tc_b = config.tc * max(0.0, 1.0 - reduced_field(field, config.hc2))
        tpdw_b = config.t_pdw * max(0.0, 1.0 - reduced_field(field, config.h_pdw))
        t_max = 1.25 * max(config.tc, config.t_pdw)
        temps = np.linspace(0.0, t_max, 241)
        t_sweep = temperature_sweep(config, temps)
        t_fig = create_pdw_sweep_plot(
            temps,
            t_sweep.intensity_p,
            t_sweep.intensity_q,
            t_sweep.intensity_h,
            x_label="T (K)",
            title=f"Peaks vs Temperature at |B| = {np.linalg.norm(field):.2f} T (SPECULATIVE)",
            floor=config.floor,
            current=t_now,
            markers=[(tc_b, "Tc(B)"), (tpdw_b, "T_PDW(B)")],
            shade=(tc_b, tpdw_b, "PDW only"),
            dark=dark,
        )

        # --- Field sweep (at the current temperature) ---
        mags = np.linspace(0.0, b_max, 241)
        b_sweep = field_sweep(config, mags, direction)
        markers = [(hc2_now, "Hc2(T)"), (hpdw_now, "H_PDW(T)")]
        if field_induced:
            markers.append((onset, "d-vector onset"))
        b_fig = create_pdw_sweep_plot(
            mags,
            b_sweep.intensity_p,
            b_sweep.intensity_q,
            b_sweep.intensity_h,
            x_label=f"|B| along {axis_label} (T)",
            title=f"Peaks vs Field at T = {t_now:.2f} K (SPECULATIVE)",
            floor=config.floor,
            current=b_along if on_axis else None,
            markers=markers,
            shade=(hc2_now, hpdw_now, "PDW only"),
            dark=dark,
        )

        # --- Regime map ---
        map_temps = np.linspace(0.0, t_max, 121)
        map_mags = np.linspace(0.0, b_max, 121)
        regimes = regime_map(config, map_temps, map_mags, direction)
        r_fig = create_pdw_regime_map(
            map_mags,
            map_temps,
            regimes,
            field_label=f"|B| along {axis_label}",
            current=(b_along, t_now) if on_axis else None,
            dark=dark,
        )

        # --- Info card ---
        delta0_mev = config.delta0_mev * state.delta0
        delta_p_mev = config.delta0_mev * config.pdw_ratio * state.delta_p
        overlaps = ", ".join(f"{c:.2f}" for c in state.overlap)
        info = dbc.Card(
            dbc.CardBody(
                [
                    html.H5(f"Regime: {regime_name}", className="card-title"),
                    html.P(regime_blurb, className="text-muted small"),
                    *(
                        []
                        if on_axis
                        else [
                            html.P(
                                "The applied field is not along the sweep axis, so the "
                                "field sweep and regime map show no current-state marker.",
                                className="text-muted small",
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.P(f"System: {system}"),
                                    html.P(
                                        f"T = {t_now:.2f} K, Tc = {config.tc:.2f} K, "
                                        f"T_PDW = {config.t_pdw:.2f} K"
                                    ),
                                    html.P(
                                        f"δ0 = {state.delta0:.3f} "
                                        f"(Δ0 = {delta0_mev:.3f} meV)"
                                    ),
                                    html.P(
                                        f"δp = {state.delta_p:.3f} "
                                        f"(Δp = {delta_p_mev:.3f} meV)"
                                    ),
                                    html.P(f"d̂0 · d̂p (p1, p2, p3): {overlaps}"),
                                    html.P(
                                        f"Along {axis_label}: Hc2(T) = {hc2_now:.2f} T, "
                                        f"H_PDW(T) = {hpdw_now:.2f} T"
                                        + (
                                            f", d-vector onset = {onset:.2f} T"
                                            if field_induced
                                            else ""
                                        )
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(_peak_table(peaks), md=6),
                        ]
                    ),
                ]
            )
        )

        return fig, t_fig, b_fig, r_fig, info
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Computation error: {e}")
        return (
            error_fig,
            error_fig,
            error_fig,
            error_fig,
            html.P(str(e), style={"color": "red"}),
        )
