use egui::Ui;

use crate::app::MoireApp;

/// Render information about the computed moire pattern.
pub fn show_info_panel(ui: &mut Ui, app: &MoireApp) {
    ui.heading("Results");
    ui.add_space(4.0);

    let substrate = app.substrate_material();
    let overlayer = app.overlayer_material();

    ui.label(format!("Substrate: {} (a = {:.3} A)", substrate.formula, substrate.a));
    ui.label(format!("Overlayer: {} (a = {:.3} A)", overlayer.formula, overlayer.a));
    ui.label(format!("Twist angle: {:.1} deg", app.twist_angle));

    ui.add_space(8.0);

    if let Some(ref result) = app.moire_result {
        ui.label(format!(
            "Lattice mismatch: {:.2}%",
            result.mismatch_percent
        ));

        if result.moire_period.is_finite() {
            ui.label(format!(
                "Moire period: {:.2} A",
                result.moire_period
            ));
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
        }
    }
}
