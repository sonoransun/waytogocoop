"""Spin Fluctuations page — LCCO magnetotransport and what-if pairing routes (SPECULATIVE)."""

from __future__ import annotations

import functools
import math
import traceback
from dataclasses import dataclass, replace

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update

from waytogocoop.components.controls import (
    loading_spinner,
    preset_and_reset_card,
    register_preset_reset_callback,
)
from waytogocoop.components.figure_factory import (
    create_gap_heatmap,
    create_sf_coupling_plot,
    create_sf_derivative_plot,
    create_sf_doping_plot,
    create_sf_fermi_surface,
    create_sf_hall_plot,
    create_sf_imprint_plot,
    create_sf_mr_plot,
    create_sf_resistivity_plot,
)
from waytogocoop.components.material_selector import create_material_selector
from waytogocoop.components.parameter_panel import (
    create_parameter_panel,
    is_valid_extent,
    is_valid_grid_size,
)
from waytogocoop.components.sf_panel import (
    COHERENCE_PLOT_LABELS,
    VIEW_OPTIONS,
    create_glue_panel,
    create_sf_panel,
    doping_value,
)
from waytogocoop.computation.moire import generate_moire_pattern
from waytogocoop.computation.pdw import moire_locked_period
from waytogocoop.computation.spin_fluctuation import (
    SF_COHERENCE_MODELS,
    SF_INELASTIC_FORMS,
    SF_PAIRING_FORMS,
    SfConfig,
    SfDopingSweep,
    SfGlueConfig,
    SfGlueResult,
    SfMagnetoresistance,
    SfPairingMap,
    SfResistivityFit,
    bcs_route_relative_modulation,
    coherence_factor,
    coherence_length,
    cvc_centres,
    cvc_partner_angle,
    doping_regime,
    doping_sweep,
    drude_hall_coefficient,
    drude_hall_coefficient_nominal,
    equivalent_doping,
    fermi_surface,
    fermi_surface_at,
    film_of_x,
    fragmentation_tc,
    g_of_x,
    glue_gap_scale,
    glue_modulated_gap,
    hall_coefficient,
    hall_coefficient_curve,
    hall_ratio_to_case_a,
    high_field_slope,
    hotspot_angles,
    imprint_curve,
    lambda_from_tc,
    lambda_resistive,
    lambda_threshold,
    lambda_tr,
    local_imprint,
    log_sensitivity,
    magnetoresistance,
    pairing_fit_rms,
    quadrature_field_derivative,
    resistivity_fit,
    resistivity_tensor,
    scattering_rate,
    snap_doping,
    tc_from_lambda,
    tc_of_x,
    zero_field_resistivity,
    zone_boundary_crossing_deg,
)
from waytogocoop.config import (
    A_LCCO,
    ALPHA1_LCCO,
    ALPHA2_LCCO,
    GAMMA_IMP_LCCO,
    HOTSPOT_PEAK_FIG9C,
    HOTSPOT_PEAK_LCCO,
    HOTSPOT_WIDTH_LCCO,
    PHI_HS_DEG_LCCO,
    SF_CVC_EPSILON,
    SF_DOPING_REF,
    SF_FIELD_MAX,
    SF_FIG13_RH,
    SF_FIT_FIELD,
    SF_FIT_TEMPERATURE,
    SF_GLUE_COHERENCE,
    SF_GLUE_ETA,
    SF_MR_TEMPERATURES,
    SF_NU,
    SF_RHO_FIT_RANGE,
    SF_SLOPE_FIELD,
    TC_FETE_FILM,
    X_FSR_LCCO,
)
from waytogocoop.materials.database import MATERIALS, get_material
from waytogocoop.state import register_url_sync

dash.register_page(
    __name__,
    path="/spinfluct",
    name="Spin Fluctuations",
    title="Good Job Coop! - Spin Fluctuations",
)

_PREFIX = "sf"
_URL_ID = f"{_PREFIX}-url"

# Header and blurb per doping regime; the desktop tab shows the same text.
_REGIME_TEXT: dict[str, tuple[str, str]] = {
    "fs_reconstruction": (
        "Fermi-surface reconstruction",
        "Short-range AFM reconstructs the Fermi surface below x = 0.14; the measured MR "
        "gains a negative low-T component this model omits (g(x) was fitted at 40 K, "
        "where it is weak).",
    ),
    "superconducting": (
        "Superconducting",
        "Overdoped and superconducting: only the anisotropic T-linear strength g(x) "
        "varies across the series (Tables 3–4).",
    ),
    "beyond_dome": (
        "Beyond the dome",
        "Beyond the dome: Tc = 0 and g(x) has fallen about tenfold from x = 0.116.",
    ),
}

_BASE_PRESET = {
    "doping": doping_value(SF_DOPING_REF),
    "temperature": SF_FIT_TEMPERATURE,
    "field_max": SF_FIELD_MAX,
    "inelastic": "afm_boundary",
    "hotspots": True,
    "gamma_imp": GAMMA_IMP_LCCO,
    "alpha1": ALPHA1_LCCO,
    "alpha2": ALPHA2_LCCO,
    "hotspot_peak": HOTSPOT_PEAK_LCCO,
    "hotspot_width": HOTSPOT_WIDTH_LCCO,
    "nu": SF_NU,
    "phi_hs": PHI_HS_DEG_LCCO,
    "cvc_epsilon": SF_CVC_EPSILON,
    "pairing_form": "power_law",
    "eta": SF_GLUE_ETA,
    "tc_target": TC_FETE_FILM,
    "substrate": "FeTe",
    "overlayer": "Sb2Te3",
    "twist": 0.0,
    "grid_size": 200,
    "extent": 100.0,
    "view_mode": "mr",
    "coherence": SF_GLUE_COHERENCE,
}

# Names are shared with the desktop tab's preset list.
_PRESETS = {
    "LCCO x = 0.159 — paper fit (Duffy Fig. 4)": dict(_BASE_PRESET),
    "LCCO x = 0.116 — near optimal (Tc ≈ 25 K)": {**_BASE_PRESET, "doping": "0.116"},
    "LCCO x = 0.188 — beyond the dome (Tc = 0)": {**_BASE_PRESET, "doping": "0.188"},
    "Across the dome: ρ and MR vs x": {**_BASE_PRESET, "view_mode": "doping"},
    "No quadrature scaling (vs hole-doped)": {**_BASE_PRESET, "view_mode": "derivative"},
    "Hall coefficient with vertex corrections": {**_BASE_PRESET, "view_mode": "hall"},
    "Kohler check: isotropic Γ, no hot spots": {
        **_BASE_PRESET,
        "inelastic": "isotropic",
        "hotspots": False,
    },
    "Ablation: isotropic + strong hot spots (Fig. 9c)": {
        **_BASE_PRESET,
        "inelastic": "isotropic",
        "hotspot_peak": HOTSPOT_PEAK_FIG9C,
    },
    "Ablation: no hot spots (Fig. 9b)": {**_BASE_PRESET, "hotspots": False},
    "Moire glue what-if: 13.5 K operating point (FeTe-film Tc)": {
        **_BASE_PRESET,
        "view_mode": "glue",
    },
    # 5 K sits near the clean imprint's maximum, where the local map first has
    # Tc(r) = 0 regions; that location moves with the calibration and with L_m,
    # so it is not a robust feature.
    "Moire glue what-if: imprint vs operating Tc": {
        **_BASE_PRESET,
        "tc_target": 5.0,
        "view_mode": "imprint",
    },
}

_DEFAULTS = dict(_BASE_PRESET)

# Also the argument order of _update_sf; "coherence" was appended last.
_PRESET_BINDINGS = [
    (f"{_PREFIX}-doping", "value", "doping"),
    (f"{_PREFIX}-temperature", "value", "temperature"),
    (f"{_PREFIX}-field-max", "value", "field_max"),
    (f"{_PREFIX}-inelastic", "value", "inelastic"),
    (f"{_PREFIX}-hotspots", "value", "hotspots"),
    (f"{_PREFIX}-gamma-imp", "value", "gamma_imp"),
    (f"{_PREFIX}-alpha1", "value", "alpha1"),
    (f"{_PREFIX}-alpha2", "value", "alpha2"),
    (f"{_PREFIX}-hotspot-peak", "value", "hotspot_peak"),
    (f"{_PREFIX}-hotspot-width", "value", "hotspot_width"),
    (f"{_PREFIX}-nu", "value", "nu"),
    (f"{_PREFIX}-phi-hs", "value", "phi_hs"),
    (f"{_PREFIX}-cvc-epsilon", "value", "cvc_epsilon"),
    (f"{_PREFIX}-pairing-form", "value", "pairing_form"),
    (f"{_PREFIX}-eta", "value", "eta"),
    (f"{_PREFIX}-tc-target", "value", "tc_target"),
    (f"{_PREFIX}-substrate-dropdown", "value", "substrate"),
    (f"{_PREFIX}-overlayer-dropdown", "value", "overlayer"),
    (f"{_PREFIX}-twist-slider", "value", "twist"),
    (f"{_PREFIX}-grid-size", "value", "grid_size"),
    (f"{_PREFIX}-physical-extent", "value", "extent"),
    (f"{_PREFIX}-view-mode", "value", "view_mode"),
    (f"{_PREFIX}-coherence", "value", "coherence"),
]
_KEYS = tuple(key for _cid, _prop, key in _PRESET_BINDINGS)

_VIEWS = tuple(option["value"] for option in VIEW_OPTIONS)
_GLUE_VIEWS = ("imprint", "glue")
_RHO_TEMPERATURES = np.arange(2.0, 101.0)  # rho(0, T): 2-100 K at 1 K
_HALL_TEMPERATURES = np.arange(2.0, 101.0, 2.0)  # R_H(T): 2-100 K at 2 K
_KOHLER_TEMPERATURES = (20.0, 50.0)  # the MR(20 T) pair of the Kohler check
_IMPRINT_TARGETS = np.linspace(0.5, 25.0, 80)  # Tc* grid of the imprint view
_COUPLING_LAMBDA = np.linspace(0.0, 3.0, 301)  # lambda grid of the Tc(lambda) curves
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
# Its numbers hold at the paper's own settings only (SfConfig()); elsewhere the card
# derives what the CVCs do from the R_H readouts and curves (_cvc_note).
_CVC_NOTE = (
    "In this form CVCs lower R_H (by 28 % for isotropic scattering, and from 31 % at 2 K "
    "to 7 % at 100 K for the paper's form) but flatten its T-dependence and do not give "
    "(d) ≈ ½(a)."
)
_FIG13_NOTE = (
    "Paper Fig. 13 (T not stated): "
    + " · ".join(f"{v:g}" for v in SF_FIG13_RH)
    + " — semi-quantitative; not reproduced by its printed Eqs. 14–15 (see docs)"
)
# The Fig. 13 panel each form is with its hot spots, compared with (a), the Gamma_0-only
# surface: (b) no inelastic term, (c) antinodal, (d) the paper's form. Isotropic
# scattering, or a form without hot spots, has no panel of its own.
_FIG13_PANELS = {
    "none": ("(b)", SF_FIG13_RH[1]),
    "antinodal": ("(c)", SF_FIG13_RH[2]),
    "afm_boundary": ("(d)", SF_FIG13_RH[3]),
}
_FLAT_SPREAD = 1e-3  # R_H(T) spreads below this are flat, or equally steep (as on the desktop)
_KOHLER_TOLERANCE = 5e-4  # |MR ratio at equal B/rho0 - 1| up to this is a Kohler collapse
# Short names for the card; the radio labels already say "(default)" or "— fails".
_PAIRING_NAMES = {"power_law": "Empirical power law", "mcmillan": "McMillan form"}
# Top margin (px) of the two side graphs, whose "<br>(SPECULATIVE)" titles take two
# lines; the figure builders' 50 px fits one.
_TWO_LINE_TITLE_MARGIN = 72

layout = dbc.Container(
    [
        dcc.Location(id=_URL_ID, refresh=False),
        html.Br(),
        html.H2("Spin Fluctuations (SPECULATIVE)"),
        dbc.Alert(
            [
                "A Boltzmann (Shockley–Chambers) model of the magnetotransport of "
                "electron-doped La₂₋ₓCeₓCuO₄ — Duffy et al., ",
                html.A(
                    "Nat. Phys. (2026)",
                    href="https://doi.org/10.1038/s41567-026-03454-6",
                    target="_blank",
                ),
                ". The transport core reproduces the paper's resistivity and "
                "magnetoresistance simulations (Figs. 4–5) to within about 5–12 %; its Hall "
                "numbers are semi-quantitative. The scattering is an anisotropic T-linear "
                "rate, peaked at the antiferromagnetic hot spots, whose strength tracks Tc. "
                "The pairing map and the moire glue modulation are what-ifs. This is "
                "normal-state transport, not Cooper-pair transport, and LCCO is not one of "
                "this project's heterostructures.",
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
                        create_sf_panel(_PREFIX),
                        html.Div(
                            [
                                create_glue_panel(_PREFIX),
                                create_material_selector(_PREFIX),
                                create_parameter_panel(
                                    _PREFIX,
                                    extent_min=10,
                                    extent_max=400,
                                    extent_default=_DEFAULTS["extent"],
                                ),
                            ],
                            id=f"{_PREFIX}-glue-controls",
                            style={"display": "none"},
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("View", className="card-title"),
                                    dbc.RadioItems(
                                        id=f"{_PREFIX}-view-mode",
                                        options=VIEW_OPTIONS,
                                        value=_DEFAULTS["view_mode"],
                                    ),
                                ]
                            ),
                            className="mb-3",
                        ),
                    ],
                    xs=12,
                    md=4,
                    lg=3,
                    className="sidebar-col",
                ),
                dbc.Col(
                    [
                        loading_spinner(
                            dcc.Graph(id=f"{_PREFIX}-main-graph"),
                            "Computing magnetotransport…",
                        ),
                        html.Div(id=f"{_PREFIX}-info", className="mb-3"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Loading(dcc.Graph(id=f"{_PREFIX}-fs-graph")),
                                    lg=5,
                                ),
                                dbc.Col(
                                    dcc.Loading(dcc.Graph(id=f"{_PREFIX}-coupling-graph")),
                                    lg=7,
                                ),
                            ]
                        ),
                    ],
                    xs=12,
                    md=8,
                    lg=9,
                ),
            ]
        ),
    ],
    fluid=True,
    className="p-4",
)


register_preset_reset_callback(_PREFIX, _PRESETS, _DEFAULTS, _PRESET_BINDINGS)
register_url_sync(_URL_ID, _PRESET_BINDINGS)


def _register_collapse(name: str) -> None:
    @callback(
        Output(f"{_PREFIX}-{name}-collapse", "is_open"),
        Input(f"{_PREFIX}-{name}-toggle", "n_clicks"),
        State(f"{_PREFIX}-{name}-collapse", "is_open"),
    )
    def _toggle(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open


for _collapse_name in ("scatter", "hall", "pairing"):
    _register_collapse(_collapse_name)


@callback(
    Output(f"{_PREFIX}-glue-controls", "style"),
    Input(f"{_PREFIX}-view-mode", "value"),
)
def _glue_controls(view_mode):
    return {} if view_mode in _GLUE_VIEWS else {"display": "none"}


# ---------------------------------------------------------------------------
# Input clamps
# ---------------------------------------------------------------------------


def _number(value, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _clamped(value, default: float, lo: float, hi: float) -> float:
    return min(max(_number(value, default), lo), hi)


def _choice(value, options, default: str) -> str:
    return value if isinstance(value, str) and value in options else default


def _clamp(values: dict) -> dict:
    """The binding values (by URL key) as _update_sf uses them, each clamped to its control.

    Doping snaps to the nearest tabulated x, field_max snaps down to the 0.5 T grid and
    nu to an even integer (ties down: 13 -> 12, 13.1 -> 14); unknown choices fall back
    to the defaults.
    """
    d = _DEFAULTS
    nu = _number(values["nu"], d["nu"])
    field_max = _clamped(values["field_max"], d["field_max"], 5.0, 65.0)
    return {
        "doping": snap_doping(values["doping"]),
        "temperature": _clamped(values["temperature"], d["temperature"], 2.0, 100.0),
        "field_max": math.floor(2.0 * field_max + 1e-9) / 2.0,
        "inelastic": _choice(values["inelastic"], SF_INELASTIC_FORMS, d["inelastic"]),
        "hotspots": d["hotspots"] if values["hotspots"] is None else bool(values["hotspots"]),
        "gamma_imp": _clamped(values["gamma_imp"], d["gamma_imp"], 0.5, 100.0),
        "alpha1": _clamped(values["alpha1"], d["alpha1"], 0.0, 3.0),
        "alpha2": _clamped(values["alpha2"], d["alpha2"], 0.0, 0.05),
        "hotspot_peak": _clamped(values["hotspot_peak"], d["hotspot_peak"], 0.0, 200.0),
        "hotspot_width": _clamped(values["hotspot_width"], d["hotspot_width"], 0.01, 0.3),
        "nu": int(min(max(2 * math.ceil(nu / 2.0 - 0.5), 2), 24)),
        "phi_hs": _clamped(values["phi_hs"], d["phi_hs"], 15.0, 40.0),
        "cvc_epsilon": _clamped(values["cvc_epsilon"], d["cvc_epsilon"], 0.0, 1.0),
        "pairing_form": _choice(values["pairing_form"], SF_PAIRING_FORMS, d["pairing_form"]),
        "eta": _clamped(values["eta"], d["eta"], 0.0, 0.5),
        "tc_target": _clamped(values["tc_target"], d["tc_target"], 0.5, 25.0),
        "substrate": _choice(values["substrate"], MATERIALS, d["substrate"]),
        "overlayer": _choice(values["overlayer"], MATERIALS, d["overlayer"]),
        "twist": _number(values["twist"], d["twist"]),
        "grid_size": (
            int(values["grid_size"]) if is_valid_grid_size(values["grid_size"]) else d["grid_size"]
        ),
        "extent": float(values["extent"]) if is_valid_extent(values["extent"]) else d["extent"],
        "view_mode": _choice(values["view_mode"], _VIEWS, d["view_mode"]),
        "coherence": _choice(values["coherence"], SF_COHERENCE_MODELS, d["coherence"]),
    }


def _sf_config(s: dict) -> SfConfig:
    return SfConfig(
        doping=s["doping"],
        inelastic=s["inelastic"],
        hotspots=s["hotspots"],
        gamma_imp=s["gamma_imp"],
        alpha1=s["alpha1"],
        alpha2=s["alpha2"],
        hotspot_peak=s["hotspot_peak"],
        hotspot_width=s["hotspot_width"],
        nu=s["nu"],
        phi_hs_deg=s["phi_hs"],
        cvc_epsilon=s["cvc_epsilon"],
    )


# A hand-edited URL can hold a value no control shows faithfully: the doping select
# falls back to its first option, a radio group selects nothing, a dropdown goes blank
# and a slider's box keeps 150 K while the page computes 100 K. The clamped value is
# written back. Sliders commit only in-range values themselves (typing sets drag_value
# until blur), so this never fights the user; the grid-size and extent inputs pass
# through invalid values while typing and are left alone.
_WRITE_BACK = tuple(key for key in _KEYS if key not in ("grid_size", "extent"))
_BINDING_IDS = {key: cid for cid, _prop, key in _PRESET_BINDINGS}


def _shows(raw, value) -> bool:
    """Whether a control holding ``raw`` already shows the clamped ``value``."""
    if isinstance(value, (bool, str)):  # the switch, the doping select and the choices
        return type(raw) is type(value) and raw == value
    # Sliders: 40 and 40.0 are the same position.
    return isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw == value


@callback(
    *[Output(_BINDING_IDS[key], "value", allow_duplicate=True) for key in _WRITE_BACK],
    *[Input(_BINDING_IDS[key], "value") for key in _WRITE_BACK],
    prevent_initial_call=True,
)
def _write_back_clamps(*raw):
    s = _clamp({**_DEFAULTS, **dict(zip(_WRITE_BACK, raw, strict=True))})
    s["doping"] = doping_value(s["doping"])
    return tuple(
        no_update if _shows(value, s[key]) else s[key]
        for key, value in zip(_WRITE_BACK, raw, strict=True)
    )


# ---------------------------------------------------------------------------
# Transport (paper-faithful core): cached bundles and direct readouts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TransportBundle:
    """The Chambers-heavy transport of one config at one field_max (see _transport)."""

    mr: SfMagnetoresistance  # SF_MR_TEMPERATURES on the 0 .. field_max grid
    rho: np.ndarray  # rho(0, T) on _RHO_TEMPERATURES, micro-Ohm cm
    fit: SfResistivityFit  # rho0 + A1 T + A2 T^2 on SF_RHO_FIT_RANGE
    lambda_rho: float  # Allen-Drude coupling from that fit
    rh_plain: np.ndarray  # R_H(T) on _HALL_TEMPERATURES without CVC, m^3/C
    rh_cvc: np.ndarray  # with CVC


@functools.lru_cache(maxsize=16)
def _transport(config: SfConfig, field_max: float) -> _TransportBundle:
    """MR family, rho(0, T) and its fit, and R_H(T) both ways, for this config."""
    return _TransportBundle(
        mr=magnetoresistance(config, SF_MR_TEMPERATURES, field_max),
        rho=zero_field_resistivity(config, _RHO_TEMPERATURES),
        fit=resistivity_fit(config),
        lambda_rho=lambda_resistive(config),
        rh_plain=hall_coefficient_curve(config, _HALL_TEMPERATURES, cvc=False),
        rh_cvc=hall_coefficient_curve(config, _HALL_TEMPERATURES, cvc=True),
    )


@functools.lru_cache(maxsize=8)
def _doping_bundle(config: SfConfig) -> SfDopingSweep:
    """The 19-doping sweep; doping_sweep ignores config.doping, so callers key it on
    replace(config, doping=SF_DOPING_REF)."""
    return doping_sweep(config)


def _mr_at(config: SfConfig, temperature: float, field: float) -> float:
    """MR(field) at one temperature, evaluated directly (never from the field grid)."""
    rho_xx = resistivity_tensor(config, temperature, [0.0, field])[:, 0]
    return float(rho_xx[1] / rho_xx[0] - 1.0)


@functools.lru_cache(maxsize=16)
def _kohler(config: SfConfig) -> tuple[float, float]:
    """MR(20 T) at 20 K and at 50 K: the Kohler check, independent of field_max."""
    return tuple(_mr_at(config, t, SF_FIT_FIELD) for t in _KOHLER_TEMPERATURES)


@functools.lru_cache(maxsize=16)
def _kohler_ratio(config: SfConfig) -> float:
    """MR(20 T) at 50 K over the 20 K MR at the same B/rho0; Kohler's rule makes it 1.

    A fall of MR(20 T) from 20 K to 50 K alone is not Kohler behaviour: rho0 grows with
    T, so the rule predicts the fall, and its size. Uniform Gamma collapses exactly.
    """
    rho_low, rho_high = zero_field_resistivity(config, _KOHLER_TEMPERATURES)
    reference = _mr_at(config, _KOHLER_TEMPERATURES[0], SF_FIT_FIELD * float(rho_low / rho_high))
    mr_high = _kohler(config)[1]
    if reference == 0.0:  # no MR at all at 20 K (not reached on the sliders' ranges)
        return 1.0 if mr_high == 0.0 else math.inf
    return mr_high / reference


@dataclass(frozen=True)
class _Readouts:
    """Transport at the current temperature, evaluated directly (independent of field_max)."""

    rho0: float  # rho(0, T), micro-Ohm cm (closed form)
    mr_20t: float  # MR(20 T, T)
    slope_33t: float  # gamma_1 = d rho / d mu0 H at 33 T, micro-Ohm cm / T
    rh_plain: float  # m^3/C
    rh_cvc: float  # m^3/C
    rh_ratio_to_case_a: float  # R_H(CVC) / R_H(case (a), CVC)
    rh_drude: float  # fixed x = 0.159 Fermi surface, m^3/C
    rh_drude_nominal: float  # n = 1 - x holes per Cu, m^3/C


def _readouts(config: SfConfig, temperature: float) -> _Readouts:
    return _Readouts(
        rho0=float(zero_field_resistivity(config, [temperature])[0]),
        mr_20t=_mr_at(config, temperature, SF_FIT_FIELD),
        slope_33t=high_field_slope(config, temperature, SF_SLOPE_FIELD),
        rh_plain=hall_coefficient(config, temperature, cvc=False),
        rh_cvc=hall_coefficient(config, temperature, cvc=True),
        rh_ratio_to_case_a=hall_ratio_to_case_a(config, temperature),
        rh_drude=drude_hall_coefficient(config),
        rh_drude_nominal=drude_hall_coefficient_nominal(config),
    )


def _current_curve(
    config: SfConfig, temperature: float, field_max: float
) -> SfMagnetoresistance | None:
    """The current-T MR curve, or None when T is already one of the family's temperatures."""
    if np.any(np.abs(np.asarray(SF_MR_TEMPERATURES) - temperature) < 1e-9):
        return None
    return magnetoresistance(config, (temperature,), field_max)


# ---------------------------------------------------------------------------
# Glue what-if (SPECULATIVE)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GlueState:
    """The glue what-if at the current settings; ``glue`` is None without a moire."""

    pairing: SfPairingMap
    tc_target: float  # K
    coherence: str
    system: str  # "Sb₂Te₃/FeTe, θ = 0.0°"
    no_moire: str | None  # the message when the layers have no moire at this twist
    lambda_bar: float
    sensitivity: float  # S(lambda_bar)
    delta_bar: float  # meV
    coherence_length: float  # Angstrom (the clean value in the local limit)
    x_equivalent: float | None  # equivalent LCCO doping of lambda_bar
    local_exact: float  # pattern-free local imprint
    local_linear: float  # eta S
    glue: SfGlueConfig | None  # with the layers' shared L_m


def _formula(formula: str) -> str:
    return formula.translate(_SUBSCRIPTS)


def _glue_state(s: dict, pairing: SfPairingMap) -> _GlueState:
    substrate = get_material(s["substrate"])
    overlayer = get_material(s["overlayer"])
    period = moire_locked_period(substrate.a, overlayer.a, s["twist"])
    # coherence_length and glue_gap_scale never read moire_period, so the L_m-free
    # lines come from a config at the default period; only ``glue`` carries the layers'
    # own L_m, and only when it is finite.
    free = SfGlueConfig(
        eta=s["eta"], tc_target=s["tc_target"], coherence=s["coherence"], pairing=pairing
    )
    lambda_bar = lambda_from_tc(s["tc_target"], pairing)
    local_exact, local_linear = local_imprint(s["eta"], s["tc_target"], pairing)
    has_moire = math.isfinite(period)
    return _GlueState(
        pairing=pairing,
        tc_target=s["tc_target"],
        coherence=s["coherence"],
        system=(
            f"{_formula(overlayer.formula)}/{_formula(substrate.formula)}, θ = {s['twist']:.1f}°"
        ),
        no_moire=None
        if has_moire
        else (
            f"{overlayer.formula}/{substrate.formula} has no moire at this twist; "
            "pick different layers or a nonzero twist"
        ),
        lambda_bar=lambda_bar,
        sensitivity=float(log_sensitivity(lambda_bar, pairing)),
        delta_bar=glue_gap_scale(free),
        coherence_length=coherence_length(free),
        x_equivalent=equivalent_doping(lambda_bar),
        local_exact=local_exact,
        local_linear=local_linear,
        glue=replace(free, moire_period=period) if has_moire else None,
    )


# A grid-1000 map holds three 1000² arrays (24 MB), so only the last two are kept: enough
# for T and the other transport inputs, and for a clean/local toggle.
@functools.lru_cache(maxsize=2)
def _glue_map(
    substrate_key: str,
    overlayer_key: str,
    twist: float,
    grid_size: int,
    extent: float,
    glue: SfGlueConfig,
) -> tuple[np.ndarray, SfGlueResult]:
    """The moire pattern's x grid and its glue-modulated gap, keyed on what they depend on."""
    substrate = get_material(substrate_key)
    overlayer = get_material(overlayer_key)
    moire = generate_moire_pattern(
        substrate_a=substrate.a,
        overlayer_a=overlayer.a,
        overlayer_lattice_type=overlayer.lattice_type,
        substrate_lattice_type=substrate.lattice_type,
        twist_angle_deg=twist,
        grid_size=grid_size,
        physical_extent=extent,
    )
    return moire["x"], glue_modulated_gap(moire["pattern"], glue)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _error_figure(message: str, dark: bool) -> go.Figure:
    """An empty figure titled ``message``: computation errors and the no-moire views."""
    fig = go.Figure()
    fig.update_layout(
        title=message,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_dark" if dark else "plotly_white",
    )
    return fig


def _main_figure(
    s: dict,
    config: SfConfig,
    bundle: _TransportBundle,
    readouts: _Readouts,
    sweep: SfDopingSweep,
    glue_state: _GlueState,
    glue_map: tuple[np.ndarray, SfGlueResult] | None,
    dark: bool,
) -> go.Figure:
    view = s["view_mode"]
    t = s["temperature"]
    system = f"LCCO x = {config.doping:.3f}"
    if view in _GLUE_VIEWS and glue_state.glue is None:
        return _error_figure(glue_state.no_moire, dark)

    if view in ("mr", "derivative"):
        mr = bundle.mr
        extra = _current_curve(config, t, s["field_max"])
        if view == "mr":
            return create_sf_mr_plot(
                mr.fields,
                mr.temperatures,
                mr.mr,
                current=None if extra is None else (t, extra.mr[0]),
                title=f"Magnetoresistance, {system} (SPECULATIVE)",
                dark=dark,
            )
        temps = mr.temperatures[:, None]
        return create_sf_derivative_plot(
            mr.h_over_t,
            mr.drho_dh,
            mr.temperatures,
            np.broadcast_to(mr.fields[None, :] / temps, mr.mr.shape),
            quadrature_field_derivative(temps, mr.fields[None, :]),
            current=None if extra is None else (t, extra.h_over_t[0], extra.drho_dh[0]),
            # One line that fits the 726 px graph of a 1024 px viewport; the subplot
            # titles name the two forms.
            title=f"No H/T scaling: {system} vs hole-doped (SPECULATIVE)",
            dark=dark,
        )
    if view == "resistivity":
        return create_sf_resistivity_plot(
            _RHO_TEMPERATURES,
            bundle.rho,
            bundle.fit,
            current=t,
            title=f"Zero-field resistivity, {system} (SPECULATIVE)",
            dark=dark,
        )
    if view == "hall":
        return create_sf_hall_plot(
            _HALL_TEMPERATURES,
            bundle.rh_plain * 1e10,
            bundle.rh_cvc * 1e10,
            drude=readouts.rh_drude * 1e10,
            drude_nominal=readouts.rh_drude_nominal * 1e10,
            current=t,
            title=f"Hall coefficient (Jones–Zener), {system} (SPECULATIVE)",
            dark=dark,
        )
    if view == "doping":
        return create_sf_doping_plot(
            sweep,
            x_fsr=X_FSR_LCCO,
            current=config.doping,
            title="Across the LCCO dome: resistivity and magnetoresistance (SPECULATIVE)",
            dark=dark,
        )

    glue = glue_state.glue
    label = COHERENCE_PLOT_LABELS[glue_state.coherence]
    if view == "imprint":
        dressed, local, linear = imprint_curve(glue, _IMPRINT_TARGETS)
        fragment = fragmentation_tc(glue) if glue_state.coherence == "local" else 0.0
        return create_sf_imprint_plot(
            _IMPRINT_TARGETS,
            dressed,
            local,
            linear,
            bcs_route_relative_modulation(),
            coherence_label=label,
            fragment_below=fragment if fragment > _IMPRINT_TARGETS[0] else None,
            current=(glue_state.tc_target, coherence_factor(glue) * glue_state.local_exact),
            # The dressed curve's height against 0.165 holds only at this shared period.
            title=(
                f"Glue imprint vs Tc*, {glue_state.system}, "
                f"L_m = {glue.moire_period:.1f} Å (SPECULATIVE)"
            ),
            dark=dark,
        )
    x, result = glue_map
    fig = create_gap_heatmap(
        x, x, result.gap, title=f"Glue-modulated gap Δ(r), {label} (SPECULATIVE)", dark=dark
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.0,
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        text="red = larger Δ",
    )
    return fig


def _fermi_surface_figure(config: SfConfig, temperature: float, dark: bool) -> go.Figure:
    fs = fermi_surface(config)
    gamma = scattering_rate(fs.phi, temperature, config)

    def points(phi: np.ndarray) -> np.ndarray:
        kf = fermi_surface_at(phi, config)[0] * 1e-10  # 1/m -> 1/Angstrom
        return np.column_stack([kf * np.cos(phi), kf * np.sin(phi)])

    centres = cvc_centres(config)
    hot = config.hotspots and config.hotspot_peak > 0.0
    fig = create_sf_fermi_surface(
        fs.kf * 1e-10 * np.cos(fs.phi),
        fs.kf * 1e-10 * np.sin(fs.phi),
        gamma,
        points(hotspot_angles(config)) if hot else None,
        points(centres),
        points(cvc_partner_angle(centres, config)),
        math.pi / A_LCCO,
        # Two lines: this graph is 288 px wide in the 1024 px layout.
        title=f"Fermi-surface Γ(φ), {temperature:g} K<br>(SPECULATIVE)",
        dark=dark,
    )
    fig.update_layout(margin_t=_TWO_LINE_TITLE_MARGIN)
    return fig


def _coupling_figure(
    config: SfConfig, sweep: SfDopingSweep, glue_state: _GlueState, dark: bool
) -> go.Figure:
    selected = glue_state.pairing.form
    curves = [
        (
            "Tc(λ): empirical power law",
            _COUPLING_LAMBDA,
            tc_from_lambda(_COUPLING_LAMBDA, SfPairingMap(form="power_law")),
            "solid",
            selected == "power_law",
        ),
        (
            "Tc(λ): McMillan best fit (fails)",
            _COUPLING_LAMBDA,
            tc_from_lambda(_COUPLING_LAMBDA, SfPairingMap(form="mcmillan")),
            "dot",
            selected == "mcmillan",
        ),
    ]
    i = int(np.argmin(np.abs(sweep.dopings - config.doping)))
    fig = create_sf_coupling_plot(
        sweep.lambda_tr,
        sweep.lambda_rho,
        sweep.tc,
        sweep.dopings,
        curves,
        current=(float(sweep.lambda_tr[i]), float(sweep.tc[i])),
        operating_point=(glue_state.lambda_bar, glue_state.tc_target),
        lambda_c=lambda_threshold(glue_state.pairing),
        # Two lines: this graph is 414 px wide in the 1024 px layout.
        title="Coupling vs Tc, LCCO dome<br>(SPECULATIVE)",
        dark=dark,
    )
    fig.update_layout(margin_t=_TWO_LINE_TITLE_MARGIN)
    return fig


# ---------------------------------------------------------------------------
# Info card
# ---------------------------------------------------------------------------


def _line(*children, muted: bool = False, className: str = "") -> html.P:
    base = "text-muted small mb-1" if muted else "mb-1"
    return html.P(list(children), className=f"{base} {className}".strip())


def _kohler_verdict(ratio: float) -> str:
    """Kohler's rule from MR(50 K)/MR(20 K) at equal B/rho0 (see _kohler_ratio)."""
    if abs(ratio - 1.0) <= _KOHLER_TOLERANCE:
        return "collapses on B/ρ₀ (Kohler)"
    return "grows (anti-Kohler)" if ratio > 1.0 else "falls faster than Kohler (inverted)"


def _spread(values: np.ndarray) -> float:
    """(max - min) / max |v| of an R_H(T) curve; unlike max/min it survives a sign change."""
    lo, hi = float(np.min(values)), float(np.max(values))
    scale = max(abs(lo), abs(hi))
    return (hi - lo) / scale if scale > 0.0 else 0.0


def _cvc_note(config: SfConfig, bundle: _TransportBundle, readouts: _Readouts) -> str:
    """What the CVCs do at these settings: to R_H at T, and to its T-dependence over 2-100 K.

    The paper's form lowers R_H and flattens R_H(T); the antinodal form raises and steepens it.
    """
    if config.cvc_epsilon == 0.0:
        return "CVCs are off (ε₀ = 0)."
    plain, cvc = readouts.rh_plain, readouts.rh_cvc
    if plain * cvc < 0.0:
        at_t = "At T, CVCs reverse the sign of R_H"
    elif cvc == plain or abs(cvc - plain) < 5e-4 * abs(plain):  # it would print as 0.0 %
        at_t = "At T, CVCs leave R_H unchanged"
    else:
        change = 100.0 * abs(cvc - plain) / abs(plain) if plain != 0.0 else math.inf
        at_t = f"At T, CVCs {'lower' if cvc < plain else 'raise'} R_H by {change:.1f} %"
    spread_plain, spread_cvc = _spread(bundle.rh_plain), _spread(bundle.rh_cvc)
    if max(spread_plain, spread_cvc) < _FLAT_SPREAD:
        over_t = "R_H(T) is flat over 2–100 K"
    elif abs(spread_cvc - spread_plain) < _FLAT_SPREAD:
        over_t = "over 2–100 K they leave its T-dependence nearly unchanged"
    elif spread_cvc < spread_plain:
        over_t = "over 2–100 K they flatten its T-dependence"
    else:
        over_t = "over 2–100 K they steepen its T-dependence"
    return f"{at_t}; {over_t}."


def _transport_column(
    s: dict,
    config: SfConfig,
    bundle: _TransportBundle,
    readouts: _Readouts,
    kohler: tuple[float, float, float],
) -> list:
    x = config.doping
    t = s["temperature"]
    mr_20k, mr_50k, kohler_ratio = kohler
    hot = config.hotspots and config.hotspot_peak > 0.0
    panel = _FIG13_PANELS.get(config.inelastic) if hot else None
    ratio_line = f"R_H/R_H(a) with CVC at T: {readouts.rh_ratio_to_case_a:.2f}"
    if panel is not None:
        name, rh = panel
        ratio_line = (
            f"{name}/(a) with CVC at T: {readouts.rh_ratio_to_case_a:.2f} "
            f"(paper Fig. 13: {rh:g}/{SF_FIG13_RH[0]:g} = {rh / SF_FIG13_RH[0]:.2f})"
        )
    return [
        html.H6("Paper-faithful transport (Duffy et al. 2026)"),
        _line(
            f"x = {x:.3f} (film {film_of_x(x)}) · Tc(x) = {tc_of_x(x):.2f} K · "
            f"g(x) = {g_of_x(x):.2f}"
        ),
        _line(
            f"λ_tr = {lambda_tr(config):.3f} (FS-averaged T-linear coupling) · "
            f"λ_ρ = {bundle.lambda_rho:.3f} (Allen–Drude, from ρ(T))"
        ),
        _line(
            f"At T = {t:g} K: ρ(0, T) = {readouts.rho0:.2f} μΩ cm · "
            f"MR(20 T) = {100.0 * readouts.mr_20t:.3f} % · "
            f"γ₁(33 T) = {readouts.slope_33t:.4f} μΩ cm/T"
        ),
        _line(
            f"Kohler check: MR(20 T) = {100.0 * mr_20k:.3f} % at 20 K → "
            f"{100.0 * mr_50k:.3f} % at 50 K; at equal B/ρ₀, MR(50 K)/MR(20 K) = "
            f"{kohler_ratio:.3f}: {_kohler_verdict(kohler_ratio)}"
        ),
        _line(
            f"R_H at T (10⁻¹⁰ m³/C): no CVC {1e10 * readouts.rh_plain:.2f} · "
            f"CVC {1e10 * readouts.rh_cvc:.2f} · "
            f"Drude, fixed x = 0.159 FS {1e10 * readouts.rh_drude:.2f} · "
            f"Drude, n = 1 − x {1e10 * readouts.rh_drude_nominal:.2f}"
        ),
        _line(ratio_line),
        _line(_cvc_note(config, bundle, readouts), muted=True),
        *([_line(_CVC_NOTE, muted=True)] if config == SfConfig() else []),
        _line(_FIG13_NOTE, muted=True),
        _line(
            f"The Fermi surface meets the AFM zone boundary at "
            f"{zone_boundary_crossing_deg(config):.2f}° (φ_HS = {config.phi_hs_deg:g}°); "
            "the CVCs are centred there."
        ),
    ]


def _whatif_column(
    config: SfConfig,
    glue_state: _GlueState,
    glue_map: tuple[np.ndarray, SfGlueResult] | None,
) -> list:
    pairing = glue_state.pairing
    if pairing.form == "power_law":
        fit_note = "exact through 3 Fig. 5c labels"
    else:
        fit_note = f"best fit μ* → 0, rms {pairing_fit_rms(pairing):.1f} K — fails"
    lam_c = lambda_threshold(pairing)
    x_c = equivalent_doping(lam_c)
    x = config.doping
    x_eq = glue_state.x_equivalent
    lines = [
        html.H6("What-ifs (SPECULATIVE)"),
        _line(html.Strong("Pairing map: "), f"{_PAIRING_NAMES[pairing.form]} ({fit_note})"),
        _line(
            f"Tc(λ_tr) = {float(tc_from_lambda(lambda_tr(config), pairing)):.2f} K at "
            f"x = {x:.3f}, against Tc(x) = {tc_of_x(x):.2f} K (table)"
        ),
        _line(
            f"threshold λ_c = {lam_c:.3f}"
            + (f" (x ≈ {x_c:.3f})" if x_c is not None else " (outside the LCCO table)")
        ),
        _line(html.Strong("Moire glue route")),
        _line(
            f"Tc* = {glue_state.tc_target:g} K · "
            + (
                f"equivalent LCCO doping x ≈ {x_eq:.3f} (LCCO-shaped Tc(λ))"
                if x_eq is not None
                else "equivalent LCCO doping outside the LCCO table"
            )
        ),
        _line(
            f"λ̄ = {glue_state.lambda_bar:.3f} · S = {glue_state.sensitivity:.3f} · "
            f"Δ̄ = {glue_state.delta_bar:.3f} meV"
        ),
    ]
    ref = f"amplitude route, measured gaps (13.5 K): {bcs_route_relative_modulation():.3f}"
    glue = glue_state.glue
    if glue is None:
        lines += [
            _line(f"ξ(Tc*) = {glue_state.coherence_length:.1f} Å"),
            _line(
                f"{glue_state.system}: no moire at this twist; pick different layers or a "
                "nonzero twist.",
                html.Span(" L_m-free lines only.", className="text-muted"),
                className="text-warning",
            ),
            _line(
                f"δΔ/Δ̄: local {glue_state.local_exact:.4f} · "
                f"linear η·S {glue_state.local_linear:.4f}"
            ),
            _line(ref),
        ]
        return lines

    ratio = glue_state.coherence_length / glue.moire_period
    flag = (
        [" ", html.Span("local map unreliable", className="text-warning fw-bold")]
        if ratio > 0.5
        else []
    )
    factor = coherence_factor(glue)
    if glue_state.coherence == "local":
        f_line = _line("F: local limit: no coherence dressing")
    else:
        model = COHERENCE_PLOT_LABELS[glue_state.coherence]
        f_line = _line(f"F = exp(−ξ/L_m) = {factor:.4f} ({model})")
    lines += [
        _line(
            f"ξ(Tc*) = {glue_state.coherence_length:.1f} Å · "
            f"L_m = {glue.moire_period:.2f} Å ({glue_state.system}) · ξ/L_m = {ratio:.3f}",
            *flag,
        ),
        f_line,
        _line(
            f"δΔ/Δ̄: dressed {factor * glue_state.local_exact:.4f} · "
            f"local {glue_state.local_exact:.4f} · linear η·S {glue_state.local_linear:.4f}"
        ),
        _line(ref),
    ]
    if glue_state.coherence == "local" and glue_map is not None:
        lines.append(
            _line(
                f"local limit: Tc(r) = 0 on {100.0 * glue_map[1].normal_fraction:.1f} % of the map"
            )
        )
    return lines


def _info_card(
    s: dict,
    config: SfConfig,
    bundle: _TransportBundle,
    readouts: _Readouts,
    kohler: tuple[float, float, float],
    glue_state: _GlueState,
    glue_map: tuple[np.ndarray, SfGlueResult] | None,
) -> dbc.Card:
    x = config.doping
    t = s["temperature"]
    regime_name, regime_blurb = _REGIME_TEXT[doping_regime(x)]
    notes = []
    if t < tc_of_x(x):
        notes.append(
            f"T = {t:g} K < Tc(x) = {tc_of_x(x):.2f} K: at zero field the film is "
            "superconducting; curves are the normal-state model."
        )
    if t > SF_RHO_FIT_RANGE[1]:
        notes.append(
            f"T = {t:g} K > {SF_RHO_FIT_RANGE[1]:g} K: outside the fitted window "
            f"(the ρ fit spans {SF_RHO_FIT_RANGE[0]:g}–{SF_RHO_FIT_RANGE[1]:g} K)."
        )
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(f"Doping regime: {regime_name}", className="card-title"),
                html.P(regime_blurb, className="text-muted small mb-1"),
                *[html.P(note, className="text-muted small mb-1") for note in notes],
                dbc.Row(
                    [
                        dbc.Col(_transport_column(s, config, bundle, readouts, kohler), md=6),
                        dbc.Col(_whatif_column(config, glue_state, glue_map), md=6),
                    ],
                    className="mt-2",
                ),
            ]
        )
    )


# ---------------------------------------------------------------------------
# Main callback
# ---------------------------------------------------------------------------


@callback(
    Output(f"{_PREFIX}-main-graph", "figure"),
    Output(f"{_PREFIX}-fs-graph", "figure"),
    Output(f"{_PREFIX}-coupling-graph", "figure"),
    Output(f"{_PREFIX}-info", "children"),
    *[Input(cid, prop) for (cid, prop, _key) in _PRESET_BINDINGS],
    Input("theme-store", "data"),
)
def _update_sf(
    doping,
    temperature,
    field_max,
    inelastic,
    hotspots,
    gamma_imp,
    alpha1,
    alpha2,
    hotspot_peak,
    hotspot_width,
    nu,
    phi_hs,
    cvc_epsilon,
    pairing_form,
    eta,
    tc_target,
    substrate_key,
    overlayer_key,
    twist,
    grid_size,
    extent,
    view_mode,
    coherence,
    theme,
):
    dark = theme == "dark"
    try:
        s = _clamp(
            dict(
                zip(
                    _KEYS,
                    (
                        doping,
                        temperature,
                        field_max,
                        inelastic,
                        hotspots,
                        gamma_imp,
                        alpha1,
                        alpha2,
                        hotspot_peak,
                        hotspot_width,
                        nu,
                        phi_hs,
                        cvc_epsilon,
                        pairing_form,
                        eta,
                        tc_target,
                        substrate_key,
                        overlayer_key,
                        twist,
                        grid_size,
                        extent,
                        view_mode,
                        coherence,
                    ),
                    strict=True,
                )
            )
        )
        config = _sf_config(s)
        bundle = _transport(config, s["field_max"])
        sweep = _doping_bundle(replace(config, doping=SF_DOPING_REF))
        readouts = _readouts(config, s["temperature"])
        kohler = (*_kohler(config), _kohler_ratio(config))
        glue_state = _glue_state(s, SfPairingMap(form=s["pairing_form"]))
        # The map itself is drawn only in the glue view; the card also needs it for the
        # local limit's Tc(r) = 0 fraction.
        need_map = s["view_mode"] == "glue" or s["coherence"] == "local"
        glue_map = (
            _glue_map(
                s["substrate"],
                s["overlayer"],
                s["twist"],
                s["grid_size"],
                s["extent"],
                glue_state.glue,
            )
            if need_map and glue_state.glue is not None
            else None
        )

        fig = _main_figure(s, config, bundle, readouts, sweep, glue_state, glue_map, dark)
        fs_fig = _fermi_surface_figure(config, s["temperature"], dark)
        coupling_fig = _coupling_figure(config, sweep, glue_state, dark)
        info = _info_card(s, config, bundle, readouts, kohler, glue_state, glue_map)
        return fig, fs_fig, coupling_fig, info
    except Exception as e:
        traceback.print_exc()
        error_fig = _error_figure(f"Computation error: {e}", dark)
        return (
            error_fig,
            error_fig,
            error_fig,
            html.P(str(e), style={"color": "red"}),
        )
