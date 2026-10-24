"""Tests for the Spin Fluctuations page (/spinfluct) and its figure builders (SPECULATIVE).

``dash.register_page`` raises until an app exists, so the page module is reached the
way Dash imports it: through ``waytogocoop.app.create_app()`` (as in
tests/test_capture_parity.py).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import pytest
from dash import dcc, no_update

from waytogocoop.components import figure_factory as ff
from waytogocoop.components.colormaps import get_plotly_colorscale
from waytogocoop.components.sf_panel import (
    COHERENCE_LABELS,
    DOPING_OPTIONS,
    INELASTIC_LABELS,
    PAIRING_LABELS,
    VIEW_OPTIONS,
    create_glue_panel,
    create_sf_panel,
)
from waytogocoop.computation.pdw import moire_locked_period
from waytogocoop.computation.spin_fluctuation import (
    DOPINGS_LCCO,
    SF_COHERENCE_MODELS,
    SfConfig,
    SfGlueConfig,
    SfPairingMap,
    bcs_route_relative_modulation,
    cvc_centres,
    cvc_partner_angle,
    fermi_surface_at,
    fragmentation_tc,
    high_field_slope,
    imprint_curve,
    lambda_from_tc,
    local_imprint,
    log_sensitivity,
    resistivity_tensor,
    zero_field_resistivity,
)
from waytogocoop.materials.database import get_material
from waytogocoop.state import decode_state, encode_state

pytestmark = [
    pytest.mark.speculative,
    # Importing the pages builds the home page's DataTable, which makes Dash 4
    # announce the component's future removal. Unrelated to these tests.
    pytest.mark.filterwarnings(
        r"ignore:(?s).*dash_table\.DataTable will be removed:DeprecationWarning"
    ),
]

_KEYS = (
    "doping",
    "temperature",
    "field_max",
    "inelastic",
    "hotspots",
    "gamma_imp",
    "alpha1",
    "alpha2",
    "hotspot_peak",
    "hotspot_width",
    "nu",
    "phi_hs",
    "cvc_epsilon",
    "pairing_form",
    "eta",
    "tc_target",
    "substrate",
    "overlayer",
    "twist",
    "grid_size",
    "extent",
    "view_mode",
    "coherence",
)
_VIEWS = ("mr", "derivative", "resistivity", "hall", "doping", "imprint", "glue")
_GLUE_VIEWS = ("imprint", "glue")
_PRESET_OVERRIDES = {
    "LCCO x = 0.159 — paper fit (Duffy Fig. 4)": {},
    "LCCO x = 0.116 — near optimal (Tc ≈ 25 K)": {"doping": "0.116"},
    "LCCO x = 0.188 — beyond the dome (Tc = 0)": {"doping": "0.188"},
    "Across the dome: ρ and MR vs x": {"view_mode": "doping"},
    "No quadrature scaling (vs hole-doped)": {"view_mode": "derivative"},
    "Hall coefficient with vertex corrections": {"view_mode": "hall"},
    "Kohler check: isotropic Γ, no hot spots": {"inelastic": "isotropic", "hotspots": False},
    "Ablation: isotropic + strong hot spots (Fig. 9c)": {
        "inelastic": "isotropic",
        "hotspot_peak": 105.0,
    },
    "Ablation: no hot spots (Fig. 9b)": {"hotspots": False},
    "Moire glue what-if: 13.5 K operating point (FeTe-film Tc)": {"view_mode": "glue"},
    "Moire glue what-if: imprint vs operating Tc": {"tc_target": 5.0, "view_mode": "imprint"},
}
# Identical lattices at zero twist: the shared two-lattice period is infinite.
_NO_MOIRE = {"substrate": "Graphene-AB", "overlayer": "Graphene", "twist": 0.0}
_NO_MOIRE_MESSAGE = (
    "Graphene/Graphene-AB has no moire at this twist; pick different layers or a nonzero twist"
)


def _page_module(stem: str):
    """A page module that create_app() imported (Dash names it pages.<stem>)."""
    suffix = f"waytogocoop/pages/{stem}.py"
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).as_posix().endswith(suffix):
            return module
    raise LookupError(f"create_app() did not import the {stem!r} page")


@pytest.fixture(scope="module")
def page():
    from waytogocoop.app import create_app

    create_app()
    return _page_module("spin_fluctuation")


def _args(page, values: dict) -> list:
    return [values[key] for _cid, _prop, key in page._PRESET_BINDINGS]


def _render(page, theme: str = "dark", **override):
    return page._update_sf(*_args(page, {**page._DEFAULTS, **override}), theme)


def _text(node) -> str:
    """All text inside a Dash component tree, space-joined."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, (int, float)):
        return str(node)
    if isinstance(node, (list, tuple)):
        return " ".join(_text(child) for child in node)
    return _text(getattr(node, "children", None))


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    elif children is not None and not isinstance(children, (str, int, float)):
        yield from _walk(children)


def _title(fig: go.Figure) -> str:
    return str(fig.layout.title.text)


def _assert_rendered(out, label: str = "") -> None:
    assert len(out) == 4, label
    for fig in out[:3]:
        assert isinstance(fig, go.Figure), label
        assert not _title(fig).startswith("Computation error"), f"{label}: {_title(fig)}"
    assert isinstance(out[3], dbc.Card), f"{label}: {_text(out[3])}"


def _trace(fig: go.Figure, name: str):
    return next(t for t in fig.data if t.name == name)


def _default_config(**override) -> SfConfig:
    return SfConfig(**override)


def _mr20(config: SfConfig, temperature: float, field: float = 20.0) -> float:
    """MR(field) at one temperature, evaluated directly."""
    rho = resistivity_tensor(config, temperature, [0.0, field])[:, 0]
    return float(rho[1] / rho[0] - 1.0)


# ---------------------------------------------------------------------------
# Presets, bindings and layout
# ---------------------------------------------------------------------------


class TestPresets:
    def test_23_bindings_in_url_key_order(self, page):
        assert tuple(key for _cid, _prop, key in page._PRESET_BINDINGS) == _KEYS
        ids = [cid for cid, _prop, _key in page._PRESET_BINDINGS]
        assert len(ids) == 23 == len(set(ids))
        assert page._PRESET_BINDINGS[-1] == ("sf-coherence", "value", "coherence")

    def test_every_binding_component_is_in_the_layout(self, page):
        ids = {getattr(node, "id", None) for node in _walk(page.layout)}
        for cid, _prop, _key in page._PRESET_BINDINGS:
            assert cid in ids, cid
        for name in ("scatter", "hall", "pairing"):
            assert f"sf-{name}-toggle" in ids and f"sf-{name}-collapse" in ids

    def test_defaults_equal_base_preset(self, page):
        assert page._DEFAULTS == page._BASE_PRESET
        assert set(page._BASE_PRESET) == set(_KEYS)

    def test_eleven_presets_with_their_names_and_overrides(self, page):
        assert list(page._PRESETS) == list(_PRESET_OVERRIDES)
        for name, preset in page._PRESETS.items():
            assert set(preset) == set(_KEYS), name
            changed = {k: v for k, v in preset.items() if page._BASE_PRESET[k] != v}
            assert changed == _PRESET_OVERRIDES[name], name

    def test_base_preset_is_the_paper_fit(self, page):
        base = page._BASE_PRESET
        assert base["doping"] == "0.159" and base["view_mode"] == "mr"
        assert base["coherence"] == "clean" and base["pairing_form"] == "power_law"
        assert (base["temperature"], base["field_max"]) == (40.0, 35.0)
        assert (base["gamma_imp"], base["hotspot_peak"], base["hotspot_width"]) == (
            27.0,
            31.4,
            0.05,
        )

    def test_control_options(self):
        assert [o["value"] for o in VIEW_OPTIONS] == list(_VIEWS)
        assert [o["value"] for o in DOPING_OPTIONS] == [f"{x:.3f}" for x in DOPINGS_LCCO]
        assert DOPING_OPTIONS[12]["label"] == "x = 0.159 · g = 1.00 · Tc ≈ 8 K"
        assert DOPING_OPTIONS[-1]["label"] == "x = 0.188 · g = 0.55 · Tc = 0"
        assert INELASTIC_LABELS["isotropic"] == "Isotropic (Fig. 9c)"
        assert INELASTIC_LABELS["antinodal"] == "Antinodal (Fig. 9e, 13c)"
        assert tuple(COHERENCE_LABELS) == SF_COHERENCE_MODELS
        assert list(COHERENCE_LABELS.values()) == [
            "Clean: ξ ∝ 1/Tc (default)",
            "Dirty: ξ ∝ Tc^−½",
            "Local limit (no coherence)",
        ]
        assert list(PAIRING_LABELS.values()) == [
            "Empirical power law (default)",
            "McMillan form, best fit — fails",
        ]

    def test_new_controls_add_no_dcc_dropdown_or_radioitems(self, page):
        view_card = next(
            node
            for node in _walk(page.layout)
            if isinstance(node, dbc.Card)
            and any(getattr(n, "id", None) == "sf-view-mode" for n in _walk(node))
        )
        for tree in (create_sf_panel("sf"), create_glue_panel("sf"), view_card):
            for node in _walk(tree):
                assert not isinstance(node, (dcc.Dropdown, dcc.RadioItems)), node

    def test_glue_controls_show_only_for_the_glue_views(self, page):
        for view in _VIEWS:
            style = page._glue_controls(view)
            assert style == ({} if view in _GLUE_VIEWS else {"display": "none"}), view

    @pytest.mark.parametrize("view", _VIEWS)
    def test_every_preset_renders_this_view(self, page, view):
        for name, preset in page._PRESETS.items():
            _assert_rendered(
                page._update_sf(*_args(page, {**preset, "view_mode": view}), "dark"),
                f"{name} / {view}",
            )

    @pytest.mark.parametrize("view", _VIEWS)
    def test_light_theme_renders(self, page, view):
        out = _render(page, "light", view_mode=view)
        _assert_rendered(out, view)
        assert all(fig.layout.template.layout.paper_bgcolor == "white" for fig in out[:3])

    def test_navbar_lists_the_page_in_the_speculative_menu_before_pdw(self):
        from waytogocoop.app import create_app, nav_menu_id

        app = create_app()
        menu_id = nav_menu_id("Speculative")
        menu = next(n for n in _walk(app.layout) if getattr(n, "id", None) == menu_id)
        hrefs = [n.href for n in _walk(menu) if isinstance(n, dbc.DropdownMenuItem)]
        assert hrefs.index("/pdw") == hrefs.index("/spinfluct") + 1
        links = [
            n.href
            for n in _walk(app.layout)
            if isinstance(n, (dbc.NavLink, dbc.DropdownMenuItem)) and getattr(n, "href", None)
        ]
        assert links.count("/spinfluct") == 1 and len(links) == 11


# ---------------------------------------------------------------------------
# URL state and clamps
# ---------------------------------------------------------------------------

# (key, raw value, clamped value): one row per clamp of the page's input table.
_CLAMPS = [
    ("doping", "0.15", 0.149),
    ("doping", 0.1435, 0.141),
    ("doping", 0.25, 0.188),
    ("doping", "garbage", 0.159),
    ("doping", float("nan"), 0.159),
    ("temperature", 0.5, 2.0),
    ("temperature", 500.0, 100.0),
    ("temperature", None, 40.0),
    ("field_max", 1.0, 5.0),
    ("field_max", 99.0, 65.0),
    ("field_max", 35.7, 35.5),
    ("inelastic", "bogus", "afm_boundary"),
    ("pairing_form", "bogus", "power_law"),
    ("hotspots", 0, False),
    ("hotspots", 1, True),
    ("hotspots", None, True),
    ("gamma_imp", 0.0, 0.5),
    ("gamma_imp", 1e3, 100.0),
    ("alpha1", -1.0, 0.0),
    ("alpha1", 9.0, 3.0),
    ("alpha2", -1.0, 0.0),
    ("alpha2", 1.0, 0.05),
    ("hotspot_peak", -1.0, 0.0),
    ("hotspot_peak", 1e3, 200.0),
    ("hotspot_width", 0.0, 0.01),
    ("hotspot_width", 1.0, 0.3),
    ("nu", 13, 12),
    ("nu", 13.1, 14),
    ("nu", 11, 10),
    ("nu", 0, 2),
    ("nu", 99, 24),
    ("nu", float("nan"), 12),
    ("phi_hs", 0.0, 15.0),
    ("phi_hs", 90.0, 40.0),
    ("cvc_epsilon", -1.0, 0.0),
    ("cvc_epsilon", 2.0, 1.0),
    ("eta", -0.1, 0.0),
    ("eta", 0.9, 0.5),
    ("tc_target", 0.0, 0.5),
    ("tc_target", 40.0, 25.0),
    ("grid_size", 10, 200),
    ("grid_size", "abc", 200),
    ("extent", 1.0, 100.0),
    ("extent", 5000.0, 100.0),
    ("coherence", "bogus", "clean"),
    ("view_mode", "bogus", "mr"),
    ("substrate", "Unobtainium", "FeTe"),
    ("overlayer", None, "Sb2Te3"),
    ("twist", float("nan"), 0.0),
]


class TestUrlState:
    def test_every_preset_round_trips(self, page):
        for name, preset in page._PRESETS.items():
            assert decode_state("?q=" + encode_state(preset)) == preset, name

    def test_presets_are_fixed_points_of_the_clamps(self, page):
        for name, preset in page._PRESETS.items():
            expected = {**preset, "doping": float(preset["doping"])}
            assert page._clamp(preset) == expected, name

    @pytest.mark.parametrize("coherence", SF_COHERENCE_MODELS)
    def test_coherence_round_trip(self, page, coherence):
        state = {**page._DEFAULTS, "coherence": coherence}
        decoded = decode_state("?q=" + encode_state(state))
        assert decoded["coherence"] == coherence
        assert page._clamp(decoded)["coherence"] == coherence

    def test_coherence_clamp(self, page):
        decoded = decode_state("?q=" + encode_state({**page._DEFAULTS, "coherence": "bogus"}))
        assert page._clamp(decoded)["coherence"] == "clean"

    @pytest.mark.parametrize(
        ("key", "raw", "expected"), _CLAMPS, ids=[f"{k}={r!r}" for k, r, _ in _CLAMPS]
    )
    def test_clamp(self, page, key, raw, expected):
        clamped = page._clamp({**page._DEFAULTS, key: raw})[key]
        assert clamped == expected
        assert type(clamped) is type(expected)

    def test_clamps_reach_the_figures(self, page):
        # nu = 13 must draw exactly what nu = 12 draws.
        raw = _render(page, view_mode="hall", nu=13)
        snapped = _render(page, view_mode="hall", nu=12)
        for a, b in zip(raw[0].data, snapped[0].data, strict=True):
            np.testing.assert_array_equal(np.asarray(a.y, float), np.asarray(b.y, float))

    def test_everything_out_of_range_still_renders(self, page):
        wild = {key: raw for key, raw, _expected in _CLAMPS}
        for view in _VIEWS:
            _assert_rendered(_render(page, **{**wild, "view_mode": view}), view)

    def test_field_max_snaps_down(self, page):
        fig = _render(page, field_max=35.7)[0]
        assert max(fig.data[0].x) == 35.5


# The gate's hand-built URL: an off-table doping, unknown choices, out-of-range sliders.
_BAD_URL = {
    "doping": "0.15",
    "inelastic": "weird",
    "hotspots": None,
    "pairing_form": "x",
    "overlayer": "Unobtainium",
    "view_mode": "nonsense",
    "coherence": "bogus",
    "temperature": 150,
    "field_max": 70,
    "nu": 13,
    "eta": 0.9,
    "tc_target": 40,
}


def _write_back(page, values: dict) -> dict:
    out = page._write_back_clamps(*[values[key] for key in page._WRITE_BACK])
    return dict(zip(page._WRITE_BACK, out, strict=True))


class TestWriteBack:
    """A URL value no control can show is written back clamped, so controls match the figures."""

    def test_wired_to_the_controls_it_reads(self, page):
        from dash import _callback  # Dash's registry of every @callback

        key, entry = next(
            (k, e)
            for k, e in _callback.GLOBAL_CALLBACK_MAP.items()
            if getattr(e.get("callback"), "__name__", "") == "_write_back_clamps"
        )
        inputs = [f"{i['id']}.{i['property']}" for i in entry["inputs"]]
        outputs = [part.split("@")[0] for part in key.strip(".").split("...")]
        assert outputs == inputs
        assert inputs == [
            f"{cid}.{prop}"
            for cid, prop, k in page._PRESET_BINDINGS
            if k not in ("grid_size", "extent")  # typed text: left alone
        ]
        listed = next(c for c in _callback.GLOBAL_CALLBACK_LIST if c["output"] == key)
        assert listed["prevent_initial_call"] is True

    def test_bad_values_come_back_clamped(self, page):
        out = _write_back(page, {**page._DEFAULTS, **_BAD_URL})
        written = {k: v for k, v in out.items() if v is not no_update}
        assert written == {
            "doping": "0.149",  # the select's option string, not the float
            "inelastic": "afm_boundary",
            "hotspots": True,
            "pairing_form": "power_law",
            "overlayer": "Sb2Te3",
            "view_mode": "mr",
            "coherence": "clean",
            "temperature": 100.0,
            "field_max": 65.0,
            "nu": 12,
            "eta": 0.5,
            "tc_target": 25.0,
        }
        assert written["doping"] in {o["value"] for o in DOPING_OPTIONS}

    def test_numeric_doping_and_truthy_switch_take_the_controls_types(self, page):
        out = _write_back(page, {**page._DEFAULTS, "doping": 0.159, "hotspots": 1})
        assert out["doping"] == "0.159" and out["hotspots"] is True

    def test_presets_are_left_alone(self, page):
        for name, preset in page._PRESETS.items():
            assert all(v is no_update for v in _write_back(page, preset).values()), name

    def test_sliders_compare_by_value(self, page):
        # A slider at 40 sends the integer 40; that is the clamped 40.0, not a change.
        values = {**page._DEFAULTS, "temperature": 40, "field_max": 35, "nu": 12.0, "twist": 0}
        assert all(v is no_update for v in _write_back(page, values).values())

    def test_one_write_settles(self, page):
        raw = {**page._DEFAULTS, **_BAD_URL}
        first = _write_back(page, raw)
        shown = {k: raw[k] if v is no_update else v for k, v in first.items()}
        # What the controls show after one write triggers no second write (no loop).
        assert all(v is no_update for v in _write_back(page, shown).values())
        assert page._clamp({**raw, **shown}) == page._clamp(raw)


# ---------------------------------------------------------------------------
# Readouts (independent of field_max)
# ---------------------------------------------------------------------------


class TestReadouts:
    def test_field_max_5_gives_three_figures_and_a_card(self, page):
        for view in _VIEWS:
            out = _render(page, field_max=5, view_mode=view)
            _assert_rendered(out, view)
        fig = _render(page, field_max=5)[0]
        assert max(fig.data[0].x) == 5.0

    def test_card_readouts_equal_direct_evaluations(self, page):
        config = _default_config()
        t = 35.0  # off the family, so nothing can come from the MR grid
        card = _text(_render(page, field_max=5, temperature=t)[3])

        def mr20(temperature: float) -> float:
            return _mr20(config, temperature)

        slope = high_field_slope(config, t, 33.0)
        assert f"MR(20 T) = {100.0 * mr20(t):.3f} %" in card
        assert f"γ₁(33 T) = {slope:.4f} μΩ cm/T" in card
        assert (
            f"MR(20 T) = {100.0 * mr20(20.0):.3f} % at 20 K → {100.0 * mr20(50.0):.3f} % at 50 K"
            in card
        )
        readouts = page._readouts(config, t)
        assert readouts.mr_20t == mr20(t)
        assert readouts.slope_33t == slope
        assert page._kohler(config) == (mr20(20.0), mr20(50.0))
        # Kohler's rule: the 50 K MR at 20 T against the 20 K MR at the same B/rho0.
        rho_20k, rho_50k = (float(zero_field_resistivity(config, [x])[0]) for x in (20.0, 50.0))
        ratio = mr20(50.0) / _mr20(config, 20.0, 20.0 * rho_20k / rho_50k)
        assert page._kohler_ratio(config) == ratio
        assert f"at equal B/ρ₀, MR(50 K)/MR(20 K) = {ratio:.3f}: grows (anti-Kohler)" in card

    def test_card_does_not_depend_on_field_max(self, page):
        for t in (20.0, 35.0, 40.0):
            low = _text(_render(page, field_max=5, temperature=t)[3])
            high = _text(_render(page, field_max=65, temperature=t)[3])
            assert low == high, t

    def test_default_card_lines(self, page):
        card = _text(_render(page)[3])
        assert "Doping regime: Superconducting" in card
        assert "λ_tr = 0.418 (FS-averaged T-linear coupling)" in card
        assert "λ_ρ = 0.435 (Allen–Drude, from ρ(T))" in card
        assert "MR(50 K)/MR(20 K) = 2.055: grows (anti-Kohler)" in card
        assert "Drude, fixed x = 0.159 FS 7.40" in card
        assert "Drude, n = 1 − x 7.39" in card
        assert "(d)/(a) with CVC at T: 0.96 (paper Fig. 13: 4.3/8.7 = 0.49)" in card
        assert "At T, CVCs lower R_H by 16.9 %; over 2–100 K they flatten its T-dependence." in card
        assert page._CVC_NOTE in card  # the paper's own settings
        assert "meets the AFM zone boundary at 27.20° (φ_HS = 27.5°)" in card
        assert "threshold λ_c = 0.305 (x ≈ 0.171)" in card
        assert "equivalent LCCO doping x ≈ 0.144 (LCCO-shaped Tc(λ))" in card
        assert "ξ/L_m = 0.545" in card and "local map unreliable" in card
        assert "amplitude route, measured gaps (13.5 K): 0.165" in card
        # The P10c imprints at 13.5 K: dressed clean 0.0408, local 0.0704, linear 0.0700.
        assert "dressed 0.0408 · local 0.0704 · linear η·S 0.0700" in card
        assert "Tc(r) = 0" not in card  # local-limit line only

    def test_kohler_preset_collapses_on_b_over_rho0(self, page):
        preset = page._PRESETS["Kohler check: isotropic Γ, no hot spots"]
        card = _text(page._update_sf(*_args(page, preset), "dark")[3])
        assert "MR(50 K)/MR(20 K) = 1.000: collapses on B/ρ₀ (Kohler)" in card
        assert "R_H/R_H(a) with CVC" in card

    def test_local_coherence_reports_the_normal_fraction(self, page):
        card = _text(_render(page, coherence="local", tc_target=2.0)[3])
        assert "local limit: Tc(r) = 0 on 30.6 % of the map" in card
        assert "F: local limit: no coherence dressing" in card

    def test_validity_notes(self, page):
        assert "superconducting; curves are the normal-state model" in _text(
            _render(page, temperature=5.0)[3]
        )
        assert "outside the fitted window" in _text(_render(page, temperature=80.0)[3])

    def test_mr_view_adds_the_current_temperature_off_the_family(self, page):
        on = _render(page, temperature=40.0)[0]
        off = _render(page, temperature=35.0)[0]
        assert [t.name for t in on.data] == ["20 K", "30 K", "40 K", "50 K"]
        assert off.data[-1].name == "35 K (current T)"
        assert off.data[-1].line.dash == "dash"


class TestKohlerCheck:
    """The verdict comes from MR at equal B/rho0, not from the sign of the fixed-field change."""

    @pytest.mark.parametrize(
        ("override", "ratio", "verdict"),
        [
            ({}, "2.055", "grows (anti-Kohler)"),  # preset 1, the paper fit
            ({"inelastic": "isotropic", "hotspots": False}, "1.000", "collapses on B/ρ₀ (Kohler)"),
            # Preset 8 (Fig. 9c): MR(20 T) falls, but faster than Kohler's rule allows.
            (
                {"inelastic": "isotropic", "hotspot_peak": 105.0},
                "0.596",
                "falls faster than Kohler (inverted)",
            ),
            # Gamma independent of T: the 20 K and 50 K MR are equal, a trivial collapse.
            ({"inelastic": "none"}, "1.000", "collapses on B/ρ₀ (Kohler)"),
            ({"alpha1": 0.0, "alpha2": 0.0}, "1.000", "collapses on B/ρ₀ (Kohler)"),
        ],
    )
    def test_verdicts(self, page, override, ratio, verdict):
        card = _text(_render(page, **override)[3])
        assert f"at equal B/ρ₀, MR(50 K)/MR(20 K) = {ratio}: {verdict}" in card
        verdicts = (
            "grows (anti-Kohler)",
            "collapses on B/ρ₀ (Kohler)",
            "falls faster than Kohler (inverted)",
        )
        assert sum(v in card for v in verdicts) == 1
        assert "Kohler-like" not in card

    def test_uniform_gamma_collapses_exactly(self, page):
        for config in (
            _default_config(inelastic="isotropic", hotspots=False),
            _default_config(inelastic="none"),
        ):
            assert page._kohler_ratio(config) == pytest.approx(1.0, rel=1e-10)

    def test_equal_pair_is_a_collapse_not_a_fall(self, page):
        config = _default_config(inelastic="none")
        mr_20k, mr_50k = page._kohler(config)
        assert mr_20k == mr_50k
        assert page._kohler_verdict(page._kohler_ratio(config)) == "collapses on B/ρ₀ (Kohler)"

    def test_verdict_thresholds(self, page):
        assert page._kohler_verdict(1.0 + 4e-4) == "collapses on B/ρ₀ (Kohler)"
        assert page._kohler_verdict(1.0 + 1e-3) == "grows (anti-Kohler)"
        assert page._kohler_verdict(1.0 - 1e-3) == "falls faster than Kohler (inverted)"


class TestHallLines:
    """The CVC note and the Fig. 13 comparison follow the form, epsilon_0 and the data."""

    def test_antinodal_is_panel_c_and_the_cvcs_raise_and_steepen(self, page):
        card = _text(_render(page, inelastic="antinodal")[3])
        assert "(c)/(a) with CVC at T: 3.18 (paper Fig. 13: 10.3/8.7 = 1.18)" in card
        assert "At T, CVCs raise R_H by 46.2 %; over 2–100 K they steepen its T-dependence." in card
        assert page._CVC_NOTE not in card and "4.3/8.7" not in card

    def test_no_inelastic_term_is_panel_b_with_a_flat_curve(self, page):
        card = _text(_render(page, inelastic="none")[3])
        assert "(b)/(a) with CVC at T: 0.93 (paper Fig. 13: 5.6/8.7 = 0.64)" in card
        assert "At T, CVCs lower R_H by 32.3 %; R_H(T) is flat over 2–100 K." in card

    @pytest.mark.parametrize(
        "override",
        [
            {"inelastic": "isotropic"},
            {"hotspots": False},
            {"hotspot_peak": 0.0},
            {"inelastic": "none", "hotspots": False},  # case (a) itself
        ],
    )
    def test_settings_without_a_fig13_panel_quote_no_paper_ratio(self, page, override):
        card = _text(_render(page, **override)[3])
        assert "R_H/R_H(a) with CVC at T:" in card
        assert "(paper Fig. 13:" not in card

    def test_cvcs_off(self, page):
        card = _text(_render(page, cvc_epsilon=0.0)[3])
        assert "CVCs are off (ε₀ = 0)." in card
        assert page._CVC_NOTE not in card

    def test_fixed_note_only_at_the_paper_settings(self, page):
        for override in ({"doping": "0.116"}, {"cvc_epsilon": 0.5}, {"gamma_imp": 30.0}):
            assert page._CVC_NOTE not in _text(_render(page, **override)[3]), override
        assert page._CVC_NOTE in _text(_render(page, temperature=2.0)[3])  # T is not a setting

    @pytest.mark.parametrize("t", [2.0, 10.0, 40.0, 100.0])
    def test_change_at_t_is_the_readouts(self, page, t):
        config = _default_config(inelastic="antinodal")
        readouts = page._readouts(config, t)
        change = 100.0 * (readouts.rh_cvc - readouts.rh_plain) / abs(readouts.rh_plain)
        word = "raise" if change > 0 else "lower"
        card = _text(_render(page, inelastic="antinodal", temperature=t)[3])
        assert f"At T, CVCs {word} R_H by {abs(change):.1f} %" in card

    def test_spread(self, page):
        assert page._spread(np.array([1.0, 2.0, 4.0])) == 0.75
        assert page._spread(np.array([-1.0, 3.0])) == pytest.approx(4.0 / 3.0)  # sign change
        assert page._spread(np.zeros(3)) == 0.0


class TestPairingLine:
    @pytest.mark.parametrize(
        ("form", "text"),
        [
            ("power_law", "Empirical power law (exact through 3 Fig. 5c labels)"),
            ("mcmillan", "McMillan form (best fit μ* → 0, rms 2.9 K — fails)"),
        ],
    )
    def test_names_the_form_once(self, page, form, text):
        card = _text(_render(page, pairing_form=form)[3])
        assert text in card
        assert card.count("fails") == (1 if form == "mcmillan" else 0)
        assert card.count("best fit") == (1 if form == "mcmillan" else 0)


# ---------------------------------------------------------------------------
# No moire at this twist
# ---------------------------------------------------------------------------


class TestNoMoire:
    def test_identical_layers_have_no_shared_period(self):
        assert math.isinf(moire_locked_period(2.46, 2.46, 0.0))

    @pytest.mark.parametrize("view", _GLUE_VIEWS)
    def test_glue_views_show_the_no_moire_figure(self, page, view):
        main, fs, coupling, card = _render(page, view_mode=view, **_NO_MOIRE)
        assert _title(main) == _NO_MOIRE_MESSAGE
        assert not main.data
        for fig in (fs, coupling):
            assert _title(fig).endswith("(SPECULATIVE)")
        assert isinstance(card, dbc.Card)

    @pytest.mark.parametrize("view", [v for v in _VIEWS if v not in _GLUE_VIEWS])
    def test_other_views_render_normally(self, page, view):
        out = _render(page, view_mode=view, **_NO_MOIRE)
        _assert_rendered(out, view)
        assert _title(out[0]) != _NO_MOIRE_MESSAGE
        assert out[0].data

    def test_card_keeps_the_lm_free_lines(self, page):
        card = _text(_render(page, **_NO_MOIRE)[3])
        assert "no moire at this twist" in card
        lam = lambda_from_tc(13.5)
        local, linear = local_imprint(0.1, 13.5)
        assert f"λ̄ = {lam:.3f} · S = {float(log_sensitivity(lam)):.3f} · Δ̄ = 3.090 meV" in card
        assert f"δΔ/Δ̄: local {local:.4f} · linear η·S {linear:.4f}" in card
        assert "ξ(Tc*) = 20.0 Å" in card
        assert "equivalent LCCO doping x ≈ 0.144" in card
        assert "amplitude route, measured gaps (13.5 K): 0.165" in card
        for transport in ("λ_tr = 0.418", "λ_ρ = 0.435", "R_H at T", "MR(20 T) ="):
            assert transport in card
        for gone in ("L_m =", "ξ/L_m", "local map unreliable", "F = exp", "dressed"):
            assert gone not in card

    def test_card_lines_match_the_moire_case(self, page):
        # The L_m-free lines read the same with and without a moire.
        with_moire = _text(_render(page)[3])
        without = _text(_render(page, **_NO_MOIRE)[3])
        lam = lambda_from_tc(13.5)
        line = f"λ̄ = {lam:.3f} · S = {float(log_sensitivity(lam)):.3f} · Δ̄ = 3.090 meV"
        assert line in with_moire and line in without

    def test_no_exception_reaches_the_error_path(self, page, monkeypatch):
        real = page._error_figure
        calls: list[tuple[str, str]] = []
        current = {"view": None}

        def guard(message, dark):
            calls.append((current["view"], message))
            return real(message, dark)

        monkeypatch.setattr(page, "_error_figure", guard)
        for view in _VIEWS:
            current["view"] = view
            _render(page, view_mode=view, **_NO_MOIRE)
        assert calls == [(view, _NO_MOIRE_MESSAGE) for view in _VIEWS if view in _GLUE_VIEWS]


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def _imprint(coherence: str = "clean", form: str = "power_law"):
    glue = SfGlueConfig(coherence=coherence, pairing=SfPairingMap(form=form))
    tc = np.linspace(0.5, 25.0, 80)
    return glue, tc, imprint_curve(glue, tc)


class TestBuilders:
    def test_imprint_plot_traces_and_reference_line(self):
        _glue, tc, (dressed, local, linear) = _imprint()
        fig = ff.create_sf_imprint_plot(
            tc,
            dressed,
            local,
            linear,
            bcs_route_relative_modulation(),
            coherence_label="clean coherence",
            current=(13.5, 0.0408),
            title="Imprint (SPECULATIVE)",
        )
        names = [t.name for t in fig.data]
        assert names[:4] == [
            "local limit (L_m ≫ ξ)",
            "linear η·S (local)",
            "amplitude route, measured gaps (13.5 K)",
            "dressed (clean coherence)",
        ]
        ref = _trace(fig, "amplitude route, measured gaps (13.5 K)")
        assert list(ref.y) == [0.1650485436893204] * 2 and ref.line.dash == "dot"
        assert _trace(fig, "dressed (clean coherence)").line.dash == "solid"
        assert _trace(fig, "local limit (L_m ≫ ξ)").line.dash == "dash"
        assert fig.data[-1].marker.symbol == "star"
        assert all(t.type == "scatter" for t in fig.data)
        assert fig.layout.xaxis.title.text == "operating Tc* (K)"
        assert fig.layout.yaxis.title.text == "δΔ/Δ̄"

    @pytest.mark.parametrize("coherence", SF_COHERENCE_MODELS)
    def test_imprint_y_range_rule(self, coherence):
        _glue, tc, (dressed, local, linear) = _imprint(coherence)
        ref = bcs_route_relative_modulation()
        fig = ff.create_sf_imprint_plot(
            tc, dressed, local, linear, ref, coherence_label=coherence, title="x (SPECULATIVE)"
        )
        top = max(0.5, 1.5 * max(float(np.max(dressed)), ref))
        assert tuple(fig.layout.yaxis.range) == (0.0, top)

    def test_local_limit_draws_the_local_curve_once(self):
        _glue, tc, (dressed, local, linear) = _imprint("local")
        np.testing.assert_array_equal(dressed, local)
        fig = ff.create_sf_imprint_plot(
            tc, dressed, local, linear, 0.165, coherence_label="local limit", title="x"
        )
        assert [t.name for t in fig.data] == [
            "linear η·S (local)",
            "amplitude route, measured gaps (13.5 K)",
            "dressed (local limit)",
        ]

    @pytest.mark.parametrize(
        ("coherence", "form", "shaded"),
        [
            ("local", "power_law", True),
            ("clean", "power_law", False),
            ("dirty", "power_law", False),
            ("local", "mcmillan", False),
        ],
    )
    def test_shading_only_for_the_local_power_law(self, page, coherence, form, shaded):
        fig = _render(page, view_mode="imprint", coherence=coherence, pairing_form=form)[0]
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) == (1 if shaded else 0)
        if shaded:
            frag = fragmentation_tc(SfGlueConfig(coherence="local"))
            assert (rects[0].x0, rects[0].x1) == (0.5, frag)
            assert any(a.text == "local limit: Tc(r) = 0 in places" for a in fig.layout.annotations)

    def test_mcmillan_never_fragments(self):
        assert (
            fragmentation_tc(SfGlueConfig(coherence="local", pairing=SfPairingMap(form="mcmillan")))
            == 0.0
        )
        _glue, tc, (dressed, local, linear) = _imprint("local", "mcmillan")
        fig = ff.create_sf_imprint_plot(
            tc,
            dressed,
            local,
            linear,
            0.165,
            coherence_label="local limit",
            fragment_below=None,
            title="x",
        )
        assert not fig.layout.shapes

    def test_imprint_star_at_the_operating_point(self, page):
        fig = _render(page, view_mode="imprint")[0]
        star = fig.data[-1]
        assert star.x[0] == 13.5
        # P10c: the dressed clean imprint at 13.5 K.
        assert star.y[0] == pytest.approx(0.04081885749393931, rel=1e-12)

    def test_imprint_title_states_the_shared_period(self, page):
        # The dressed curve's height against 0.165 holds only at this L_m (amendments §4.3).
        fig = _render(page, view_mode="imprint", tc_target=5.0)[0]
        assert _title(fig) == (
            "Glue imprint vs Tc*, Sb₂Te₃/FeTe, θ = 0.0°, L_m = 36.7 Å (SPECULATIVE)"
        )
        lm = moire_locked_period(get_material("FeTe").a, get_material("Sb2Te3").a, 3.0)
        twisted = _title(_render(page, view_mode="imprint", twist=3.0)[0])
        assert twisted.endswith(f"θ = 3.0°, L_m = {lm:.1f} Å (SPECULATIVE)")

    def test_titles_keep_speculative_at_1024_px(self, page):
        # Re-gate [4]: in the 1024 px layout the side graphs are 288 px (Fermi surface) and
        # 414 px (coupling) wide, so their titles break onto two lines and the page widens
        # the builders' 50 px top margin; the 726 px derivative graph keeps one line because
        # its 80 px margin also holds the two subplot titles.
        main, fs, coupling = _render(page, view_mode="derivative")[:3]
        assert _title(main) == "No H/T scaling: LCCO x = 0.159 vs hole-doped (SPECULATIVE)"
        assert _title(fs) == "Fermi-surface Γ(φ), 40 K<br>(SPECULATIVE)"
        assert _title(coupling) == "Coupling vs Tc, LCCO dome<br>(SPECULATIVE)"
        assert fs.layout.margin.t == coupling.layout.margin.t == page._TWO_LINE_TITLE_MARGIN == 72
        assert main.layout.margin.t == 80

    def test_imprint_view_data_is_the_imprint_curve(self, page):
        fig = _render(page, view_mode="imprint", tc_target=5.0)[0]
        _glue, tc, (dressed, local, linear) = _imprint()
        np.testing.assert_array_equal(_trace(fig, "dressed (clean coherence)").y, dressed)
        np.testing.assert_array_equal(_trace(fig, "local limit (L_m ≫ ξ)").y, local)
        np.testing.assert_array_equal(_trace(fig, "linear η·S (local)").y, linear)

    def test_coupling_plot_has_both_curves_and_hollow_rho_markers(self, page):
        fig = _render(page)[2]
        names = [t.name for t in fig.data]
        assert "Tc(λ): empirical power law" in names
        assert "Tc(λ): McMillan best fit (fails)" in names
        tr = _trace(fig, "λ_tr (FS-averaged T-linear coupling)")
        rho = _trace(fig, "λ_ρ (Allen–Drude, from ρ(T))")
        assert tr.marker.symbol == "circle" and rho.marker.symbol == "circle-open"
        assert len(tr.x) == len(rho.x) == 19
        assert _trace(fig, "glue operating point (λ̄, Tc*)").marker.symbol == "star"
        selected = _trace(fig, "Tc(λ): empirical power law").line.width
        other = _trace(fig, "Tc(λ): McMillan best fit (fails)").line.width
        assert selected > other
        swapped = _render(page, pairing_form="mcmillan")[2]
        assert (
            _trace(swapped, "Tc(λ): McMillan best fit (fails)").line.width
            > _trace(swapped, "Tc(λ): empirical power law").line.width
        )

    def test_fs_partner_segments_start_at_the_cvc_centres(self, page):
        fig = _render(page)[1]
        segments = _trace(fig, "CVC partners (k + Q)")
        config = _default_config()
        centres = cvc_centres(config)
        partners = cvc_partner_angle(centres, config)

        def xy(phi):
            kf = fermi_surface_at(phi, config)[0] * 1e-10
            return kf * np.cos(phi), kf * np.sin(phi)

        cx, cy = xy(centres)
        px, py = xy(partners)
        np.testing.assert_allclose(segments.x[0::3], cx, rtol=0, atol=1e-15)
        np.testing.assert_allclose(segments.y[0::3], cy, rtol=0, atol=1e-15)
        np.testing.assert_allclose(segments.x[1::3], px, rtol=0, atol=1e-15)
        np.testing.assert_allclose(segments.y[1::3], py, rtol=0, atol=1e-15)

    def test_fs_axes_have_equal_aspect(self, page):
        fig = _render(page)[1]
        assert fig.layout.yaxis.scaleanchor == "x"
        assert fig.layout.yaxis.scaleratio == 1
        assert _trace(fig, "Fermi surface").marker.colorbar.title.text == "Γ (ps⁻¹)"

    def test_fs_rings_hot_spots_only_when_they_are_on(self, page):
        assert "hot spots (Γ_HS)" in [t.name for t in _render(page)[1].data]
        assert "hot spots (Γ_HS)" not in [t.name for t in _render(page, hotspots=False)[1].data]

    @pytest.mark.parametrize("dark", [True, False])
    def test_temperature_colours(self, page, dark):
        colors = ff._sf_temperature_colors(4, dark)
        theme = "dark" if dark else "light"
        mr = _render(page, theme)[0]
        assert [t.line.color for t in mr.data] == colors
        deriv = _render(page, theme, view_mode="derivative")[0]
        assert [t.line.color for t in deriv.data[0::2]] == colors
        assert [t.line.color for t in deriv.data[1::2]] == colors

    def test_temperature_colour_stops(self):
        lut = get_plotly_colorscale("plasma")
        assert ff._sf_temperature_colors(4, True) == [lut[i][1] for i in (89, 138, 186, 235)]
        assert ff._sf_temperature_colors(4, False) == [lut[i][1] for i in (13, 75, 137, 199)]
        assert ff._sf_temperature_colors(1, True) == [lut[89][1]]

    @pytest.mark.parametrize("view", _VIEWS)
    def test_titles_end_speculative(self, page, view):
        for fig in _render(page, view_mode=view)[:3]:
            assert _title(fig).endswith("(SPECULATIVE)"), _title(fig)

    def test_trace_types(self, page):
        expected = {
            "mr": {"scatter"},
            "derivative": {"scatter"},
            "resistivity": {"scatter"},
            "hall": {"scatter"},
            "doping": {"scatter"},
            "imprint": {"scatter"},
            "glue": {"heatmap"},
        }
        for view, types in expected.items():
            assert {t.type for t in _render(page, view_mode=view)[0].data} == types, view

    def test_glue_view_draws_the_dressed_gap(self, page):
        fig = _render(page, view_mode="glue", grid_size=50)[0]
        z = np.asarray(fig.data[0].z, dtype=float)
        assert z.shape == (50, 50)
        assert _title(fig) == "Glue-modulated gap Δ(r), clean coherence (SPECULATIVE)"
        # Clean coherence at 13.5 K: P10's dressed range 2.957-3.209 meV bounds the map.
        assert z.min() >= 2.9569962300271597 - 1e-12
        assert z.max() <= 3.2092567693397047 + 1e-12
        assert any(a.text == "red = larger Δ" for a in fig.layout.annotations)

    def test_hall_plot_lines(self, page):
        fig = _render(page, view_mode="hall")[0]
        assert _trace(fig, "with vertex corrections (CVC)").line.dash == "solid"
        assert _trace(fig, "without vertex corrections").line.dash == "dash"
        assert _trace(fig, "Drude, fixed x = 0.159 FS").line.dash == "dot"
        assert _trace(fig, "Drude, n = 1 − x").line.dash == "dashdot"
        assert fig.layout.yaxis.title.text == "R_H (10⁻¹⁰ m³/C)"

    def test_derivative_plot_has_two_panels(self, page):
        fig = _render(page, view_mode="derivative")[0]
        assert {t.xaxis for t in fig.data} == {"x", "x2"}
        titles = [a.text for a in fig.layout.annotations]
        assert "LCCO model: no H/T collapse" in titles


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestGlueMapCache:
    """The glue map is keyed on its own inputs, so transport-only changes reuse it."""

    def test_transport_changes_reuse_the_map(self, page):
        page._glue_map.cache_clear()
        _render(page, coherence="local", grid_size=60)  # the card's Tc(r) = 0 line needs it
        assert page._glue_map.cache_info().misses == 1
        for override in ({"temperature": 35.0}, {"field_max": 20}, {"doping": "0.116"}):
            _render(page, coherence="local", grid_size=60, **override)
        _render(page, coherence="local", grid_size=60, view_mode="glue", temperature=45.5)
        info = page._glue_map.cache_info()
        assert (info.misses, info.hits) == (1, 4)

    def test_glue_inputs_rebuild_it(self, page):
        page._glue_map.cache_clear()
        _render(page, view_mode="glue", grid_size=60)
        for override in ({"eta": 0.2}, {"twist": 1.0}, {"extent": 80.0}, {"coherence": "dirty"}):
            misses = page._glue_map.cache_info().misses
            _render(page, view_mode="glue", grid_size=60, **override)
            assert page._glue_map.cache_info().misses == misses + 1, override

    def test_a_cached_map_draws_what_a_fresh_one_does(self, page):
        page._glue_map.cache_clear()
        _render(page, view_mode="glue", grid_size=60)
        cached = _render(page, view_mode="glue", grid_size=60, temperature=30.0)[0]
        assert page._glue_map.cache_info().hits == 1
        page._glue_map.cache_clear()
        fresh = _render(page, view_mode="glue", grid_size=60, temperature=30.0)[0]
        np.testing.assert_array_equal(
            np.asarray(cached.data[0].z, float), np.asarray(fresh.data[0].z, float)
        )


class TestPerformance:
    def test_defaults_render_fast(self, page):
        for cache in (
            page._transport,
            page._doping_bundle,
            page._kohler,
            page._kohler_ratio,
            page._glue_map,
        ):
            cache.cache_clear()
        t0 = time.perf_counter()
        _assert_rendered(_render(page))
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        _render(page)
        warm = time.perf_counter() - t0
        print(f"\n_update_sf at defaults: cold {1e3 * cold:.0f} ms, warm {1e3 * warm:.0f} ms")
        assert cold < 2.0  # budget 300 ms; a non-flaky guard against a 10x regression
