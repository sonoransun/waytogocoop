use egui::{Color32, Ui};
use moire_core::isotopes::{
    exotic_mass_range, humanize_half_life, natural_average_mass, nearest_isotope_info,
    ElementData, IsotopeKind, C, FE, SB, TE,
};

use crate::app::MoireApp;

const WARN_RED: Color32 = Color32::from_rgb(220, 50, 50);

/// Stable slider range: lightest..heaviest stable isotope (amu).
fn stable_mass_range(elem: &'static ElementData) -> (f64, f64) {
    let first = elem
        .isotopes
        .first()
        .expect("isotope table must not be empty");
    let last = elem
        .isotopes
        .last()
        .expect("isotope table must not be empty");
    (first.atomic_mass, last.atomic_mass)
}

/// Slider range for the current mode: stable isotopes, or the HIGHLY
/// SPECULATIVE hypothetical exploration range when exotic mode is on.
fn mass_range(elem: &'static ElementData, exotic: bool) -> (f64, f64) {
    if exotic {
        exotic_mass_range(elem.symbol).unwrap_or_else(|| stable_mass_range(elem))
    } else {
        stable_mass_range(elem)
    }
}

/// Nearest-isotope readout under a mass slider. Synthetic isotopes are
/// starred and show their humanized half-life.
fn nearest_isotope_readout(ui: &mut Ui, symbol: &str, mass: f64) {
    let info = nearest_isotope_info(symbol, mass);
    match info.kind {
        IsotopeKind::Stable => {
            ui.small(format!("≈ {} (stable)", info.label));
        }
        IsotopeKind::Synthetic => {
            let half_life = info
                .half_life_s
                .map(humanize_half_life)
                .unwrap_or_default();
            ui.small(format!("≈ {}* (t½ ≈ {})", info.label, half_life));
        }
        IsotopeKind::Hypothetical => {
            ui.small("hypothetical mass — no known isotope");
        }
    }
}

/// One element mass slider plus nearest-isotope readout.
/// Returns `true` if the override changed.
fn mass_slider(
    ui: &mut Ui,
    elem: &'static ElementData,
    override_mass: &mut Option<f64>,
    exotic: bool,
    step: f64,
    decimals: usize,
) -> bool {
    let nat = natural_average_mass(elem);
    let (min, max) = mass_range(elem, exotic);
    let mut mass = override_mass.unwrap_or(nat).clamp(min, max);

    ui.label(format!(
        "{} mass (amu) — nat: {:.prec$}",
        elem.symbol,
        nat,
        prec = decimals
    ));
    let changed = ui
        .add(egui::Slider::new(&mut mass, min..=max).step_by(step))
        .changed();
    if changed {
        *override_mass = Some(mass);
    }
    nearest_isotope_readout(ui, elem.symbol, mass);
    changed
}

/// Pull any out-of-range overrides back into the stable ranges (used when
/// exotic mode is switched off).
fn clamp_overrides_to_stable(app: &mut MoireApp) {
    let overrides = [
        (&FE, &mut app.fe_mass_override),
        (&TE, &mut app.te_mass_override),
        (&SB, &mut app.sb_mass_override),
        (&C, &mut app.c_mass_override),
    ];
    for (elem, override_mass) in overrides {
        if let Some(mass) = override_mass {
            let (min, max) = stable_mass_range(elem);
            *mass = mass.clamp(min, max);
        }
    }
}

/// Render isotope enrichment controls. Returns `true` if any value changed.
pub fn show_isotope_panel(ui: &mut Ui, app: &mut MoireApp) -> bool {
    let mut changed = false;

    ui.heading("Isotope Effects");
    ui.add_space(4.0);

    if ui
        .checkbox(&mut app.isotope_enabled, "Enable (Speculative)")
        .changed()
    {
        changed = true;
    }

    if !app.isotope_enabled {
        return changed;
    }

    ui.colored_label(WARN_RED, "SPECULATIVE - qualitative only");
    ui.add_space(4.0);

    // --- Exotic-isotope tier (HIGHLY SPECULATIVE) ---
    if ui
        .checkbox(&mut app.exotic_mode, "Exotic isotopes (HIGHLY SPECULATIVE)")
        .changed()
    {
        if !app.exotic_mode {
            clamp_overrides_to_stable(app);
        }
        changed = true;
    }
    if app.exotic_mode {
        ui.colored_label(
            WARN_RED,
            "Synthetic isotopes are radioactive — several half-lives are far \
             too short to grow or measure a film — and masses between/beyond \
             known isotopes are purely hypothetical. Outputs are what-if \
             illustrations only.",
        );
    }
    ui.add_space(4.0);

    let exotic = app.exotic_mode;

    // --- Fe / Te mass sliders ---
    changed |= mass_slider(ui, &FE, &mut app.fe_mass_override, exotic, 0.01, 2);
    changed |= mass_slider(ui, &TE, &mut app.te_mass_override, exotic, 0.01, 2);

    // --- Sb mass slider (only relevant for Sb-containing overlayers) ---
    let overlayer = app.overlayer_material();
    let substrate = app.substrate_material();
    let overlayer_has_sb = overlayer.formula == "Sb2Te3" || overlayer.formula == "Sb2Te";

    if overlayer_has_sb {
        changed |= mass_slider(ui, &SB, &mut app.sb_mass_override, exotic, 0.01, 2);
    }

    // --- C mass slider (only for graphene substrate or overlayer) ---
    let has_graphene =
        substrate.formula.starts_with("Graphene") || overlayer.formula.starts_with("Graphene");
    if has_graphene {
        changed |= mass_slider(ui, &C, &mut app.c_mass_override, exotic, 0.0001, 3);
    }

    ui.add_space(8.0);

    // --- BCS isotope exponent ---
    // Literature: α = 0.35–0.8 (iron chalcogenides); α = −0.18 (inverse, pnictides)
    ui.label("BCS isotope exponent (α):");
    if ui
        .add(egui::Slider::new(&mut app.isotope_alpha, -0.5..=1.0).step_by(0.01))
        .changed()
    {
        changed = true;
    }

    ui.add_space(4.0);

    // --- Reset button ---
    if ui.button("Reset to natural").clicked() {
        app.fe_mass_override = None;
        app.te_mass_override = None;
        app.sb_mass_override = None;
        app.c_mass_override = None;
        app.isotope_alpha = 0.4;
        app.exotic_mode = false;
        changed = true;
    }

    changed
}
