use egui::{Color32, Ui};
use moire_core::isotopes::{natural_average_mass, C, FE, SB, TE};

use crate::app::MoireApp;

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

    ui.colored_label(
        Color32::from_rgb(220, 50, 50),
        "SPECULATIVE - qualitative only",
    );
    ui.add_space(4.0);

    // --- Fe mass slider ---
    let fe_nat = natural_average_mass(&FE);
    let fe_min = FE
        .isotopes
        .first()
        .expect("FE isotopes must not be empty")
        .atomic_mass;
    let fe_max = FE
        .isotopes
        .last()
        .expect("FE isotopes must not be empty")
        .atomic_mass;
    let mut fe_mass = app.fe_mass_override.unwrap_or(fe_nat);

    ui.label(format!("Fe mass (amu) — nat: {:.2}", fe_nat));
    if ui
        .add(egui::Slider::new(&mut fe_mass, fe_min..=fe_max).step_by(0.01))
        .changed()
    {
        app.fe_mass_override = Some(fe_mass);
        changed = true;
    }

    // --- Te mass slider ---
    let te_nat = natural_average_mass(&TE);
    let te_min = TE
        .isotopes
        .first()
        .expect("TE isotopes must not be empty")
        .atomic_mass;
    let te_max = TE
        .isotopes
        .last()
        .expect("TE isotopes must not be empty")
        .atomic_mass;
    let mut te_mass = app.te_mass_override.unwrap_or(te_nat);

    ui.label(format!("Te mass (amu) — nat: {:.2}", te_nat));
    if ui
        .add(egui::Slider::new(&mut te_mass, te_min..=te_max).step_by(0.01))
        .changed()
    {
        app.te_mass_override = Some(te_mass);
        changed = true;
    }

    // --- Sb mass slider (only relevant for Sb-containing overlayers) ---
    let overlayer = app.overlayer_material();
    let substrate = app.substrate_material();
    let overlayer_has_sb = overlayer.formula == "Sb2Te3" || overlayer.formula == "Sb2Te";

    if overlayer_has_sb {
        let sb_nat = natural_average_mass(&SB);
        let sb_min = SB
            .isotopes
            .first()
            .expect("SB isotopes must not be empty")
            .atomic_mass;
        let sb_max = SB
            .isotopes
            .last()
            .expect("SB isotopes must not be empty")
            .atomic_mass;
        let mut sb_mass = app.sb_mass_override.unwrap_or(sb_nat);

        ui.label(format!("Sb mass (amu) — nat: {:.2}", sb_nat));
        if ui
            .add(egui::Slider::new(&mut sb_mass, sb_min..=sb_max).step_by(0.01))
            .changed()
        {
            app.sb_mass_override = Some(sb_mass);
            changed = true;
        }
    }

    // --- C mass slider (only for graphene substrate or overlayer) ---
    let has_graphene = substrate.name == "Graphene" || overlayer.name == "Graphene";
    if has_graphene {
        let c_nat = natural_average_mass(&C);
        let c_min = C
            .isotopes
            .first()
            .expect("C isotopes must not be empty")
            .atomic_mass;
        let c_max = C
            .isotopes
            .last()
            .expect("C isotopes must not be empty")
            .atomic_mass;
        let mut c_mass = app.c_mass_override.unwrap_or(c_nat);

        ui.label(format!("C mass (amu) — nat: {:.3}", c_nat));
        if ui
            .add(egui::Slider::new(&mut c_mass, c_min..=c_max).step_by(0.0001))
            .changed()
        {
            app.c_mass_override = Some(c_mass);
            changed = true;
        }
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
        changed = true;
    }

    changed
}
