"""Dash components for the spin-fluctuation (SF) page controls (SPECULATIVE module)."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from waytogocoop.components.controls import help_tooltip, labeled_with_help
from waytogocoop.computation.spin_fluctuation import DOPINGS_LCCO, g_of_x, tc_of_x
from waytogocoop.config import (
    ALPHA1_LCCO,
    ALPHA2_LCCO,
    GAMMA_IMP_LCCO,
    HOTSPOT_PEAK_LCCO,
    HOTSPOT_WIDTH_LCCO,
    PHI_HS_DEG_LCCO,
    SF_CVC_EPSILON,
    SF_DOPING_REF,
    SF_FIELD_MAX,
    SF_FIT_TEMPERATURE,
    SF_GLUE_COHERENCE,
    SF_GLUE_ETA,
    SF_NU,
    TC_FETE_FILM,
)

INELASTIC_LABELS: dict[str, str] = {
    "afm_boundary": "AFM zone boundary — paper (Eq. 1)",
    "isotropic": "Isotropic (Fig. 9c)",
    "antinodal": "Antinodal (Fig. 9e, 13c)",
    "none": "None (impurity + hot spots only)",
}

PAIRING_LABELS: dict[str, str] = {
    "power_law": "Empirical power law (default)",
    "mcmillan": "McMillan form, best fit — fails",
}

COHERENCE_LABELS: dict[str, str] = {
    "clean": "Clean: ξ ∝ 1/Tc (default)",
    "dirty": "Dirty: ξ ∝ Tc^−½",
    "local": "Local limit (no coherence)",
}

# The same models as they read inside plot legends and titles.
COHERENCE_PLOT_LABELS: dict[str, str] = {
    "clean": "clean coherence",
    "dirty": "dirty coherence",
    "local": "local limit",
}

VIEW_OPTIONS: list[dict[str, str]] = [
    {"label": "Magnetoresistance Δρ/ρ(0) vs μ₀H", "value": "mr"},
    {"label": "dρ/dμ₀H vs μ₀H/T (vs hole-doped quadrature)", "value": "derivative"},
    {"label": "Zero-field ρ(T) and its ρ₀ + A₁T + A₂T² fit", "value": "resistivity"},
    {"label": "Hall coefficient R_H(T), with and without vertex corrections", "value": "hall"},
    {"label": "Across the dome: ρ(0, 40 K) and MR(20 T) vs x", "value": "doping"},
    {"label": "Glue route vs amplitude route (speculative)", "value": "imprint"},
    {"label": "Glue-modulated gap Δ(r) (speculative)", "value": "glue"},
]


def doping_value(doping: float) -> str:
    """The doping select's string value for a tabulated x, e.g. "0.159"."""
    return f"{doping:.3f}"


def _doping_label(doping: float) -> str:
    tc = tc_of_x(doping)
    tc_text = f"Tc ≈ {tc:.0f} K" if tc > 0.0 else "Tc = 0"
    return f"x = {doping:.3f} · g = {g_of_x(doping):.2f} · {tc_text}"


DOPING_OPTIONS: list[dict[str, str]] = [
    {"label": _doping_label(x), "value": doping_value(x)} for x in DOPINGS_LCCO
]


def _slider(
    id_prefix: str,
    name: str,
    lo: float,
    hi: float,
    step: float,
    value: float,
    marks: dict | None = None,
) -> dcc.Slider:
    return dcc.Slider(
        id=f"{id_prefix}-{name}",
        min=lo,
        max=hi,
        step=step,
        value=value,
        marks=marks if marks is not None else {lo: f"{lo:g}", hi: f"{hi:g}"},
        tooltip={"placement": "bottom", "always_visible": False},
    )


def _collapse(id_prefix: str, name: str, label: str, children: list) -> list:
    return [
        dbc.Button(
            label,
            id=f"{id_prefix}-{name}-toggle",
            color="secondary",
            size="sm",
            className="mb-2 me-1",
            n_clicks=0,
        ),
        dbc.Collapse(id=f"{id_prefix}-{name}-collapse", is_open=False, children=children),
    ]


def create_sf_panel(id_prefix: str) -> dbc.Card:
    """Return a Card with the LCCO transport controls and the three collapses.

    Parameters
    ----------
    id_prefix : str
        Prefix for Dash component IDs.
    """
    scattering = [
        dbc.Label("Γ_imp, impurity rate (ps⁻¹)", className="small mb-0"),
        _slider(id_prefix, "gamma-imp", 0.5, 100.0, 0.5, GAMMA_IMP_LCCO, {0.5: "0.5", 100: "100"}),
        dbc.Label("α₁, T-linear strength (ps⁻¹ K⁻¹, times g(x))", className="small mb-0"),
        _slider(id_prefix, "alpha1", 0.0, 3.0, 0.01, ALPHA1_LCCO),
        dbc.Label("α₂, T² term (ps⁻¹ K⁻²)", className="small mb-0"),
        _slider(id_prefix, "alpha2", 0.0, 0.05, 0.0002, ALPHA2_LCCO),
        labeled_with_help(
            "Hot-spot height (ps⁻¹)",
            f"{id_prefix}-hotspot-peak-help",
            "Effective height of the hot-spot peaks, as plotted in Fig. 4b1 and used in "
            "the paper's simulations; Table 2 prints 50 ps⁻¹, which does not reproduce "
            "Fig. 4. The Fig. 9c ablation uses 105.",
        ),
        _slider(id_prefix, "hotspot-peak", 0.0, 200.0, 0.1, HOTSPOT_PEAK_LCCO),
        dbc.Label("Hot-spot width w (rad, 1/e half-width)", className="small mb-0"),
        _slider(id_prefix, "hotspot-width", 0.01, 0.30, 0.005, HOTSPOT_WIDTH_LCCO),
        dbc.Label("ν, sharpness of the AFM-boundary peaks", className="small mb-0"),
        _slider(id_prefix, "nu", 2, 24, 2, SF_NU, {2: "2", 12: "12", 24: "24"}),
        dbc.Label("φ_HS, hot-spot angle (°)", className="small mb-0"),
        _slider(id_prefix, "phi-hs", 15.0, 40.0, 0.1, PHI_HS_DEG_LCCO),
    ]
    hall = [
        labeled_with_help(
            "CVC peak ε₀",
            f"{id_prefix}-cvc-help",
            "Peak of the current-vertex-correction weight ε(φ), centred where the Fermi "
            "surface crosses the AFM zone boundary. It enters the Hall coefficient only, "
            "never ρ or the MR.",
        ),
        _slider(id_prefix, "cvc-epsilon", 0.0, 1.0, 0.01, SF_CVC_EPSILON),
    ]
    pairing = [
        labeled_with_help(
            "Pairing map Tc(λ)",
            f"{id_prefix}-pairing-help",
            "SPECULATIVE. An EMPIRICAL power law through the paper's Fig. 5c Tc labels, "
            "or McMillan's phonon form at its best fit (μ* → 0), shown failing. Neither "
            "is a microscopic theory.",
        ),
        dbc.RadioItems(
            id=f"{id_prefix}-pairing-form",
            options=[{"label": v, "value": k} for k, v in PAIRING_LABELS.items()],
            value="power_law",
            className="mb-2",
        ),
    ]
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Spin Fluctuations", className="card-title"),
                dbc.Alert(
                    "SPECULATIVE — LCCO transport model (Duffy et al. 2026) with what-if "
                    "pairing routes",
                    color="danger",
                    className="py-1 px-2 mb-2",
                    style={"fontSize": "0.8em"},
                ),
                labeled_with_help(
                    "Doping x (La₂₋ₓCeₓCuO₄)",
                    f"{id_prefix}-doping-help",
                    "Each value is a combinatorial-film channel; g(x) is the paper's fitted "
                    "strength of the anisotropic T-linear scattering (1 at x = 0.159).",
                ),
                dbc.Select(
                    id=f"{id_prefix}-doping",
                    options=DOPING_OPTIONS,
                    value=doping_value(SF_DOPING_REF),
                    className="mb-2",
                ),
                labeled_with_help(
                    "Temperature T (K)",
                    f"{id_prefix}-temperature-help",
                    "Normal-state transport. Below Tc(x) the film is superconducting at zero "
                    "field; the paper's magnetoresistance runs at 20–50 K.",
                ),
                dcc.Slider(
                    id=f"{id_prefix}-temperature",
                    min=2,
                    max=100,
                    step=0.5,
                    value=SF_FIT_TEMPERATURE,
                    marks={20: "20", 40: "40", 60: "60", 80: "80"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Br(),
                labeled_with_help(
                    "Max field μ₀H (T)",
                    f"{id_prefix}-field-max-help",
                    "Top of the field sweep; the paper's data reach 35 T. The 20 T and 33 T "
                    "readouts do not depend on it.",
                ),
                dcc.Slider(
                    id=f"{id_prefix}-field-max",
                    min=5,
                    max=65,
                    step=1,
                    value=SF_FIELD_MAX,
                    marks={20: "20", 35: "35", 65: "65"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Br(),
                labeled_with_help(
                    "Inelastic anisotropy W(φ)",
                    f"{id_prefix}-inelastic-help",
                    "Angular shape of the T-linear scattering: the paper's Eq. 1 peaks at the "
                    "AFM-zone-boundary hot spots; the others are the paper's Fig. 9 and 13 "
                    "ablations.",
                ),
                dbc.RadioItems(
                    id=f"{id_prefix}-inelastic",
                    options=[{"label": v, "value": k} for k, v in INELASTIC_LABELS.items()],
                    value="afm_boundary",
                    className="mb-2",
                ),
                html.Div(
                    [
                        dbc.Switch(
                            id=f"{id_prefix}-hotspots",
                            label="Hot spots",
                            value=True,
                            className="mb-0",
                        ),
                        *help_tooltip(
                            f"{id_prefix}-hotspots-help",
                            "Sharp scattering where the Fermi surface crosses the AFM zone "
                            "boundary; they impede orbital motion and keep the MR H-linear.",
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center"},
                    className="mb-2",
                ),
                html.Hr(),
                *_collapse(id_prefix, "scatter", "Scattering parameters (Table 2)", scattering),
                *_collapse(id_prefix, "hall", "Hall: vertex corrections", hall),
                *_collapse(id_prefix, "pairing", "Pairing map (SPECULATIVE)", pairing),
            ]
        ),
        className="mb-3",
    )


def create_glue_panel(id_prefix: str) -> dbc.Card:
    """Return a Card with the moire glue what-if controls (SPECULATIVE).

    Parameters
    ----------
    id_prefix : str
        Prefix for Dash component IDs.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Moire glue what-if", className="card-title"),
                dbc.Alert(
                    "SPECULATIVE — only the shape of Tc(λ) is used; not a model of FeTe",
                    color="danger",
                    className="py-1 px-2 mb-2",
                    style={"fontSize": "0.8em"},
                ),
                labeled_with_help(
                    "Glue depth η",
                    f"{id_prefix}-eta-help",
                    "Depth of the what-if moire modulation of the pairing glue, "
                    "λ(r) = λ̄ (1 + η m̃(r)).",
                ),
                _slider(id_prefix, "eta", 0.0, 0.5, 0.01, SF_GLUE_ETA),
                labeled_with_help(
                    "Film Tc target Tc* (K)",
                    f"{id_prefix}-tc-target-help",
                    "Operating Tc* of the film; λ̄ = λ(Tc*) on the LCCO-shaped Tc(λ) map. "
                    "13.5 K is the FeTe film.",
                ),
                _slider(
                    id_prefix,
                    "tc-target",
                    0.5,
                    25.0,
                    0.1,
                    TC_FETE_FILM,
                    {2: "2 K", TC_FETE_FILM: "FeTe film"},
                ),
                labeled_with_help(
                    "Coherence length",
                    f"{id_prefix}-coherence-help",
                    "How ξ(Tc*) dresses the local imprint: F = exp(−ξ/L_m), with ξ = 20 Å at "
                    "13.5 K. The local limit (L_m ≫ ξ) applies no dressing.",
                ),
                dbc.RadioItems(
                    id=f"{id_prefix}-coherence",
                    options=[{"label": v, "value": k} for k, v in COHERENCE_LABELS.items()],
                    value=SF_GLUE_COHERENCE,
                    className="mb-2",
                ),
                html.P(
                    "L_m: the shared two-lattice period of the selected layers and twist.",
                    className="text-muted small mb-0",
                ),
            ]
        ),
        className="mb-3",
    )
