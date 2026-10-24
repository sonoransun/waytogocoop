use egui::Ui;
use moire_core::spin_fluctuation::{
    self as sf, SfCoherence, SfConfig, SfInelasticForm, SfPairingForm,
};

use crate::app::{CooperView, MoireApp, SfReadouts, SfTransport};
use crate::ui::sf_panel::sf_stale;

/// Below this relative spread over 2–100 K an R_H(T) curve counts as flat,
/// and two curves count as equally steep.
const SF_FLAT_SPREAD: f64 = 1e-3;
/// |κ − 1| up to this counts as a Kohler collapse, κ being MR(50 K)/MR(20 K)
/// at equal B/ρ₀ (the web `_KOHLER_TOLERANCE`).
const SF_KOHLER_TOLERANCE: f64 = 5e-4;

/// Render information about the computed moire pattern.
pub fn show_info_panel(ui: &mut Ui, app: &MoireApp) {
    ui.heading("Results");
    ui.add_space(4.0);

    let substrate = app.substrate_material();
    let overlayer = app.overlayer_material();

    ui.label(format!(
        "Substrate: {} (a = {:.3} A)",
        substrate.formula, substrate.a
    ));
    ui.label(format!(
        "Overlayer: {} (a = {:.3} A)",
        overlayer.formula, overlayer.a
    ));
    ui.label(format!("Twist angle: {:.1} deg", app.twist_angle));

    ui.add_space(8.0);

    if let Some(ref result) = app.moire_result {
        ui.label(format!("Lattice mismatch: {:.2}%", result.mismatch_percent));

        if result.moire_period.is_finite() {
            ui.label(format!("Moire period: {:.2} A", result.moire_period));
        } else {
            ui.label("Moire period: infinite (no mismatch)");
        }

        ui.label(format!(
            "Viewport: {:.0} x {:.0} A",
            result.physical_extent, result.physical_extent
        ));
        ui.label(format!(
            "Resolution: {} x {}",
            result.resolution, result.resolution
        ));
    } else {
        ui.label("No results computed yet.");
    }

    if let Some(ref density) = app.density_result {
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);
        ui.label("Density modulation:");
        let min_gap = density.gap_field.iter().cloned().fold(f64::MAX, f64::min);
        let max_gap = density.gap_field.iter().cloned().fold(f64::MIN, f64::max);
        ui.label(format!("  Gap range: {:.3} - {:.3} meV", min_gap, max_gap));
    }

    // --- Magnetic field results ---
    if let Some(ref vr) = app.vortex_result {
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);
        ui.colored_label(
            egui::Color32::from_rgb(220, 50, 50),
            "Magnetic effects (speculative):",
        );
        if vr.vortex_period.is_finite() {
            ui.label(format!("  Vortex period: {:.1} A", vr.vortex_period));
            ui.label(format!("  Flux/cell: {:.3} Phi_0", vr.flux_per_moire_cell));
            if vr.commensuration_field.is_finite() {
                ui.label(format!("  Comm. field: {:.1} T", vr.commensuration_field));
            }
            if vr.is_commensurate {
                ui.colored_label(egui::Color32::from_rgb(50, 220, 50), "  COMMENSURATE");
            }
            ui.label(format!("  Vortex cores: {}", vr.vortex_positions.len()));
        } else {
            ui.label("  No vortices (Bz ≈ 0)");
        }
    }

    if let Some(ref zr) = app.zeeman_result {
        ui.label(format!("  Zeeman: {:.4} meV", zr.zeeman_energy));
        ui.label(format!("  Pauli limit: {:.1} T", zr.pauli_limit_field));
        ui.label(format!("  Depairing: {:.4}", zr.depairing_ratio));
    }

    // --- Cooper 3D / proximity results ---
    if let Some(ref z_coords) = app.cooper_z_coords {
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);
        ui.label("Cooper 3D / proximity:");
        ui.label(format!(
            "  xi: {:.0} A  T: {:.2}",
            app.proximity_config.xi_prox, app.proximity_config.interface_transparency
        ));
        if let (Some(&z0), Some(&z1)) = (z_coords.first(), z_coords.last()) {
            ui.label(format!(
                "  z range: {:.0} .. {:.0} A ({} layers)",
                z0,
                z1,
                z_coords.len()
            ));
        }
        let iz = app.z_slice_index.min(z_coords.len().saturating_sub(1));
        if let Some(&z_sel) = z_coords.get(iz) {
            ui.label(format!(
                "  z-slice: {:.0} A (#{} / {})",
                z_sel,
                iz,
                z_coords.len()
            ));
        }
        if let Some((lo, hi)) = app.cooper_value_range() {
            ui.label(format!("  Δ(z) range: {:.3} - {:.3} meV", lo, hi));
        }
        if app.cooper_view == CooperView::Majorana {
            if let Some(ref vr) = app.vortex_result {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 50, 50),
                    format!("  MZM vortices: {} (speculative)", vr.vortex_positions.len()),
                );
            }
        }
    }

    // --- Spin fluctuations (speculative) ---
    show_sf_block(ui, app);

    // --- Pair density wave (speculative) ---
    if let Some(ref rt) = app.pdw_result {
        let cfg = &rt.config;
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);
        ui.colored_label(
            egui::Color32::from_rgb(220, 50, 50),
            "Pair density wave (speculative):",
        );
        ui.label(format!("  {}", rt.system));
        ui.label(format!("  Regime: {}", rt.state.regime.label()));
        ui.label(format!(
            "  T = {:.2} K  Tc = {:.2} K  T_PDW = {:.2} K",
            cfg.temperature, cfg.tc, cfg.t_pdw
        ));
        ui.label(format!(
            "  Δ0 = {:.3} meV  Δp = {:.3} meV",
            cfg.delta0_mev * rt.state.delta0,
            cfg.delta0_mev * cfg.pdw_ratio * rt.state.delta_p
        ));
        let [c1, c2, c3] = rt.state.overlap;
        ui.label(format!("  d0·dp (p1 p2 p3): {c1:.2} {c2:.2} {c3:.2}"));
        let visible = rt.peaks.iter().filter(|p| p.visible).count();
        ui.label(format!(
            "  Visible CDW peaks: {visible} / {}",
            rt.peaks.len()
        ));
    }

    if app.isotope_enabled {
        if let Some(ref effects) = app.isotope_effects {
            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);
            ui.colored_label(
                egui::Color32::from_rgb(220, 50, 50),
                "Isotope effects (speculative):",
            );
            ui.label(format!(
                "  Substrate da: {:+.5} A",
                effects.substrate_delta_a
            ));
            ui.label(format!(
                "  Overlayer da: {:+.5} A",
                effects.overlayer_delta_a
            ));
            ui.label(format!(
                "  Delta1: {:.3} meV (was {:.2})",
                effects.delta_1_modified, app.density_config.delta_1
            ));
            ui.label(format!(
                "  Delta2: {:.3} meV (was {:.2})",
                effects.delta_2_modified, app.density_config.delta_2
            ));
            ui.label(format!(
                "  Coherence: {:.2} A (was 20.00)",
                effects.coherence_length_modified
            ));
            ui.label(format!("  DW sub: {:.6}", effects.dw_factor_substrate));
            ui.label(format!("  DW over: {:.6}", effects.dw_factor_overlayer));
            ui.label(format!(
                "  Θ_D (sub/over): {:.2} / {:.2} K",
                effects.theta_d_substrate, effects.theta_d_overlayer
            ));
            ui.label(format!(
                "  125Te spin frac: {:.3} (nat: 0.071)",
                effects.te_125_spin_fraction
            ));
        }
    }
}

/// The Spin Fluctuations block, mirroring the web info card. The stage only
/// recomputes on its own tab, so elsewhere the last values carry a muted note,
/// and before the first run the block points to the tab. Without a moire the
/// glue part keeps only its L_m-free lines.
fn show_sf_block(ui: &mut Ui, app: &MoireApp) {
    ui.add_space(8.0);
    ui.separator();
    ui.add_space(4.0);
    ui.colored_label(
        egui::Color32::from_rgb(220, 50, 50),
        "Spin fluctuations (speculative):",
    );
    let Some(rt) = app.sf_result.as_ref() else {
        match app.sf_error.as_ref() {
            Some(err) => ui.label(format!("  unavailable: {err}")),
            None => ui.weak("  — (open the Spin Fluctuations tab)"),
        };
        return;
    };
    if sf_stale(app) {
        ui.weak("  (updates on the Spin Fluctuations tab)");
    }
    let r = &rt.readouts;
    let cfg = &rt.transport.config;
    let t = rt.temperature;

    // Paper-faithful transport (Duffy et al. 2026).
    ui.label(format!("  Doping regime: {}", r.regime.label()));
    ui.weak(format!("  {}", r.regime.blurb()));
    ui.label(format!(
        "  x = {:.3} (film {}), Tc(x) = {:.1} K, g = {:.2}",
        cfg.doping,
        sf::film_of_x(cfg.doping),
        r.tc,
        r.g
    ));
    if t < r.tc {
        ui.weak("  T < Tc(x): at zero field the film is superconducting; curves are the normal-state model");
    }
    if t > 60.0 {
        ui.weak("  T > 60 K: outside the fitted window");
    }
    ui.label(format!(
        "  λ_tr = {:.3} (FS-averaged T-linear coupling)",
        r.lambda_tr
    ));
    ui.label(format!(
        "  λ_ρ = {:.3} (Allen–Drude, from ρ(T))",
        r.lambda_rho
    ));
    ui.label(format!("  At T = {t:.1} K: ρ(0) = {:.2} μΩ cm", r.rho0));
    ui.label(format!(
        "  MR(20 T) = {:.2} %, γ₁(33 T) = {:.4} μΩ cm/T",
        100.0 * r.mr_20t,
        r.slope_33t
    ));
    // Kohler's rule makes MR a function of B/ρ₀ alone, so the verdict comes
    // from MR(50 K) over MR(20 K) at equal B/ρ₀ (`SfReadouts::kohler_ratio`),
    // not from whether MR(20 T) falls between the two temperatures: ρ₀ grows
    // with T, so the rule itself predicts a fall. The web line, with "->" for
    // its arrow (no glyph in the default fonts).
    let kohler = if (r.kohler_ratio - 1.0).abs() <= SF_KOHLER_TOLERANCE {
        "collapses on B/ρ₀ (Kohler)"
    } else if r.kohler_ratio > 1.0 {
        "grows (anti-Kohler)"
    } else {
        "falls faster than Kohler (inverted)"
    };
    ui.label(format!(
        "  Kohler check: MR(20 T) = {:.3} % at 20 K -> {:.3} % at 50 K; at equal B/ρ₀, \
         MR(50 K)/MR(20 K) = {:.3}: {kohler}",
        100.0 * r.mr_20k_20t,
        100.0 * r.mr_50k_20t,
        r.kohler_ratio
    ));
    ui.label("  R_H at T (1e-10 m³/C):");
    ui.label(format!(
        "    no CVC {:.2} · CVC {:.2}",
        1e10 * r.rh_plain,
        1e10 * r.rh_cvc
    ));
    ui.label(format!(
        "    Drude, fixed x = 0.159 FS: {:.2}",
        1e10 * r.rh_drude
    ));
    ui.label(format!(
        "    Drude, n = 1 − x: {:.2}",
        1e10 * r.rh_drude_nominal
    ));
    // Case (a) is the Γ0-only Fermi surface. With its hot spots each other
    // form is a Fig. 13 panel: (b) no inelastic term, (c) antinodal, (d) the
    // paper's form. Isotropic scattering, or a form without hot spots, has no
    // panel of its own to compare with.
    let hot = cfg.hotspots && cfg.hotspot_peak > 0.0;
    let panel = match cfg.inelastic {
        SfInelasticForm::None if hot => Some(("(b)", sf::SF_FIG13_RH[1])),
        SfInelasticForm::Antinodal if hot => Some(("(c)", sf::SF_FIG13_RH[2])),
        SfInelasticForm::AfmBoundary if hot => Some(("(d)", sf::SF_FIG13_RH[3])),
        _ => None,
    };
    let case_a = sf::SF_FIG13_RH[0];
    match panel {
        Some((name, rh)) => ui.label(format!(
            "  {name}/(a) with CVC at T: {:.2} (paper Fig. 13: {rh}/{case_a} = {:.2})",
            r.rh_ratio_to_case_a,
            rh / case_a
        )),
        None => ui.label(format!(
            "  R_H/R_H(a) with CVC at T: {:.2}",
            r.rh_ratio_to_case_a
        )),
    };
    ui.weak(format!("  {}", cvc_note(r, &rt.transport)));
    // The fixed note's numbers are the paper's own settings (x = 0.159,
    // ε₀ = 1); elsewhere the line above says what the CVCs do.
    if *cfg == SfConfig::default() {
        ui.weak(
            "  In this form CVCs lower R_H (by 28 % for isotropic scattering, and from 31 % \
             at 2 K to 7 % at 100 K for the paper's form) but flatten its T-dependence and do \
             not give (d) ≈ ½(a).",
        );
    }
    ui.weak(
        "  Paper Fig. 13 (T not stated): 8.7 · 5.6 · 10.3 · 4.3, semi-quantitative; not \
         reproduced by its printed Eqs. 14–15 (see docs)",
    );
    ui.label(format!(
        "  The Fermi surface meets the AFM zone boundary at {:.2}° (φ_HS = {:.1}°); the \
         CVCs are centred there.",
        r.crossing_deg, cfg.phi_hs_deg
    ));

    // What-ifs (speculative): the pairing map, then the glue route.
    ui.add_space(4.0);
    ui.label("  What-ifs (speculative):");
    // The short form name: the full labels already end in "(default)" or
    // "best fit — fails", which the fit note says.
    let (form, fit_note) = match rt.pairing.form {
        SfPairingForm::PowerLaw => ("Empirical power law", "exact through 3 Fig. 5c labels"),
        SfPairingForm::McMillan => ("McMillan form", "best fit at μ* = 0, rms 2.9 K — fails"),
    };
    ui.label(format!("  Pairing map: {form} ({fit_note})"));
    ui.label(format!(
        "  Tc(λ_tr) = {:.2} K vs table Tc(x) = {:.2} K",
        r.tc_map_at_x, r.tc
    ));
    let threshold_x = r
        .x_threshold
        .map_or(String::new(), |x| format!(" (x ≈ {x:.3})"));
    ui.label(format!(
        "  threshold λ_c = {:.3}{threshold_x}",
        r.lambda_threshold
    ));
    let glue = &rt.glue_free;
    let x_equiv = r.x_equiv.map_or("outside the LCCO table".to_string(), |x| {
        format!("equivalent LCCO doping x ≈ {x:.3} (LCCO-shaped Tc(λ))")
    });
    ui.label(format!("  Glue: Tc* = {:.1} K, {x_equiv}", glue.tc_target));
    // <λ> and <Δ> stand for λ̄ and Δ̄: the default fonts lack the combining
    // macron.
    ui.label(format!(
        "  <λ> = {:.3}, S = {:.2}, <Δ> = {:.3} meV",
        r.lambda_bar, r.sensitivity, r.delta_bar
    ));
    match (
        r.moire_period,
        r.xi_over_lm,
        r.coherence_factor,
        r.glue_rel_dressed,
    ) {
        (Some(lm), Some(ratio), Some(factor), Some(dressed)) => {
            ui.label(format!(
                "  ξ(Tc*) = {:.1} Å, L_m = {lm:.2} Å ({}), ξ/L_m = {ratio:.3}",
                r.coherence_length, rt.system
            ));
            if ratio > 0.5 {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "  local map unreliable (ξ > L_m/2)",
                );
            }
            if glue.coherence == SfCoherence::Local {
                ui.label("  F: local limit, no coherence dressing");
            } else {
                ui.label(format!("  F = exp(−ξ/L_m) = {factor:.3}"));
            }
            ui.label(format!(
                "  δΔ/<Δ>: dressed {dressed:.3} · local {:.3} · linear η·S {:.3}",
                r.glue_rel_local, r.glue_rel_linear
            ));
        }
        _ => {
            ui.label("  no moire at this twist");
            ui.label(format!("  ξ(Tc*) = {:.1} Å", r.coherence_length));
            ui.label(format!(
                "  δΔ/<Δ>: local {:.3} · linear η·S {:.3}",
                r.glue_rel_local, r.glue_rel_linear
            ));
        }
    }
    ui.label(format!(
        "  amplitude route, measured gaps (13.5 K): {:.3}",
        sf::bcs_route_relative_modulation()
    ));
    if glue.coherence == SfCoherence::Local {
        if let Some(fraction) = r.local_normal_fraction {
            ui.label(format!(
                "  local limit: Tc(r) = 0 on {:.1} % of the map",
                100.0 * fraction
            ));
        }
    }
}

/// What the CVCs do at these settings: to R_H at the current T, and to its
/// T-dependence over the plotted 2–100 K. Depending on the form they lower R_H
/// and flatten R_H(T) (the paper's form) or raise and steepen it (antinodal).
fn cvc_note(r: &SfReadouts, tr: &SfTransport) -> String {
    if tr.config.cvc_epsilon == 0.0 {
        return "CVCs are off (ε₀ = 0).".to_string();
    }
    let change = 100.0 * (r.rh_cvc - r.rh_plain) / r.rh_plain.abs();
    let at_t = if r.rh_plain * r.rh_cvc < 0.0 {
        "At T, CVCs reverse the sign of R_H".to_string()
    } else if change.abs() < 0.05 {
        // It would print as 0.0 %.
        "At T, CVCs leave R_H unchanged".to_string()
    } else if change < 0.0 {
        format!("At T, CVCs lower R_H by {:.1} %", -change)
    } else {
        format!("At T, CVCs raise R_H by {change:.1} %")
    };
    let (plain, cvc) = (spread(&tr.hall_plain), spread(&tr.hall_cvc));
    let over_t = if plain.max(cvc) < SF_FLAT_SPREAD {
        "R_H(T) is flat over 2–100 K"
    } else if (cvc - plain).abs() < SF_FLAT_SPREAD {
        "over 2–100 K they leave its T-dependence nearly unchanged"
    } else if cvc < plain {
        "over 2–100 K they flatten its T-dependence"
    } else {
        "over 2–100 K they steepen its T-dependence"
    };
    format!("{at_t}; {over_t}.")
}

/// (max − min) / max |v|: the relative T-dependence of an R_H(T) curve. Unlike
/// max/min it stays meaningful when R_H changes sign, which some settings do.
fn spread(values: &[f64]) -> f64 {
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scale = lo.abs().max(hi.abs());
    if scale > 0.0 {
        (hi - lo) / scale
    } else {
        0.0
    }
}
