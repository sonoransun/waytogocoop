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
                ui.colored_label(
                    egui::Color32::from_rgb(50, 220, 50),
                    "  COMMENSURATE",
                );
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
                "  125Te spin frac: {:.3} (nat: 0.071)",
                effects.te_125_spin_fraction
            ));
        }
    }
}
