use egui::Ui;
use moire_core::materials;

use crate::app::MoireApp;

/// Render the sidebar controls for material selection and simulation parameters.
pub fn show_sidebar(ui: &mut Ui, app: &mut MoireApp) {
    ui.heading("Parameters");
    ui.add_space(8.0);

    // --- Substrate selection ---
    ui.label("Substrate:");
    let substrate = materials::substrate();
    ui.label(format!("  {} (a = {:.3} A)", substrate.formula, substrate.a));
    ui.add_space(4.0);

    // --- Overlayer selection ---
    ui.label("Overlayer:");
    let overlayers = materials::overlayers();
    let current_name = overlayers[app.overlayer_idx.min(overlayers.len() - 1)]
        .name
        .to_string();

    let mut changed = false;

    egui::ComboBox::from_id_salt("overlayer_combo")
        .selected_text(&current_name)
        .show_ui(ui, |ui| {
            for (idx, mat) in overlayers.iter().enumerate() {
                let label = format!("{} (a = {:.3} A)", mat.formula, mat.a);
                if ui
                    .selectable_value(&mut app.overlayer_idx, idx, label)
                    .changed()
                {
                    changed = true;
                }
            }
        });

    ui.add_space(12.0);

    // --- Twist angle ---
    ui.label("Twist angle (deg):");
    if ui
        .add(egui::Slider::new(&mut app.twist_angle, 0.0..=30.0).step_by(0.1))
        .changed()
    {
        changed = true;
    }

    ui.add_space(12.0);

    // --- Resolution ---
    ui.label("Resolution:");
    let resolutions = [128, 256, 512];
    let current_res = format!("{}x{}", app.resolution, app.resolution);
    egui::ComboBox::from_id_salt("resolution_combo")
        .selected_text(&current_res)
        .show_ui(ui, |ui| {
            for &res in &resolutions {
                let label = format!("{}x{}", res, res);
                if ui
                    .selectable_value(&mut app.resolution, res, label)
                    .changed()
                {
                    changed = true;
                }
            }
        });

    ui.add_space(12.0);

    // --- Physical extent ---
    ui.label("Viewport size (A):");
    if ui
        .add(egui::Slider::new(&mut app.physical_extent, 50.0..=500.0))
        .changed()
    {
        changed = true;
    }

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);

    // --- Density parameters ---
    ui.heading("Density Modulation");
    ui.add_space(4.0);

    ui.label("Delta 1 (meV):");
    if ui
        .add(egui::Slider::new(&mut app.density_config.delta_1, 0.5..=10.0).step_by(0.01))
        .changed()
    {
        changed = true;
    }

    ui.label("Delta 2 (meV):");
    if ui
        .add(egui::Slider::new(&mut app.density_config.delta_2, 0.5..=10.0).step_by(0.01))
        .changed()
    {
        changed = true;
    }

    ui.label("Modulation amplitude:");
    if ui
        .add(
            egui::Slider::new(&mut app.density_config.modulation_amplitude, 0.0..=1.0)
                .step_by(0.01),
        )
        .changed()
    {
        changed = true;
    }

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);

    // --- Isotope Effects ---
    if super::isotope_panel::show_isotope_panel(ui, app) {
        changed = true;
    }

    if changed {
        app.needs_recompute = true;
    }

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);

    // --- Substrate Comparison ---
    ui.heading("Comparison");
    ui.add_space(4.0);
    if ui.button("Compare All Substrates").clicked() {
        app.show_comparison = true;
        app.comparison_needs_refresh = true;
    }
}
