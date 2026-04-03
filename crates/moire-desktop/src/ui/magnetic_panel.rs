use egui::Ui;

use crate::app::MoireApp;

/// Render the magnetic field control panel. Returns true if any parameter changed.
pub fn show_magnetic_panel(ui: &mut Ui, app: &mut MoireApp) -> bool {
    let mut changed = false;

    ui.heading("Magnetic / Topological");
    ui.add_space(4.0);
    ui.colored_label(
        egui::Color32::from_rgb(220, 50, 50),
        "Includes SPECULATIVE features",
    );
    ui.add_space(8.0);

    // --- Perpendicular field Bz ---
    ui.label("Bz (Tesla) — perpendicular:");
    if ui
        .add(egui::Slider::new(&mut app.magnetic_config.bz, 0.0..=20.0).step_by(0.1))
        .changed()
    {
        changed = true;
    }

    ui.add_space(8.0);

    // --- In-plane fields (collapsible) ---
    egui::CollapsingHeader::new("In-Plane Field")
        .default_open(false)
        .show(ui, |ui| {
            ui.label("Bx (Tesla):");
            if ui
                .add(egui::Slider::new(&mut app.magnetic_config.bx, -10.0..=10.0).step_by(0.1))
                .changed()
            {
                changed = true;
            }
            ui.label("By (Tesla):");
            if ui
                .add(egui::Slider::new(&mut app.magnetic_config.by, -10.0..=10.0).step_by(0.1))
                .changed()
            {
                changed = true;
            }
        });

    ui.add_space(8.0);

    // --- Proximity / topological parameters ---
    egui::CollapsingHeader::new("Proximity / Topological")
        .default_open(false)
        .show(ui, |ui| {
            ui.label("Proximity xi (A):");
            if ui
                .add(egui::Slider::new(&mut app.proximity_config.xi_prox, 10.0..=500.0))
                .changed()
            {
                changed = true;
            }
            ui.label("Interface transparency:");
            if ui
                .add(
                    egui::Slider::new(
                        &mut app.proximity_config.interface_transparency,
                        0.1..=1.0,
                    )
                    .step_by(0.05),
                )
                .changed()
            {
                changed = true;
            }
            ui.label("g-factor:");
            if ui
                .add(egui::Slider::new(&mut app.g_factor, 1.0..=50.0))
                .changed()
            {
                changed = true;
            }
        });

    ui.add_space(8.0);

    // --- Display options ---
    ui.checkbox(&mut app.show_vortices, "Show vortex cores");
    ui.checkbox(&mut app.show_majorana, "Show Majorana density (speculative)");

    // --- Summary readout ---
    if let Some(ref vr) = app.vortex_result {
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);
        if vr.vortex_period.is_finite() {
            ui.label(format!("Vortex period: {:.1} A", vr.vortex_period));
            ui.label(format!("Flux/cell: {:.3} Phi_0", vr.flux_per_moire_cell));
            if vr.is_commensurate {
                ui.colored_label(
                    egui::Color32::from_rgb(50, 220, 50),
                    "COMMENSURATE",
                );
            }
        } else {
            ui.label("No vortices (Bz ≈ 0)");
        }
    }

    if let Some(ref zr) = app.zeeman_result {
        ui.label(format!("Zeeman: {:.4} meV", zr.zeeman_energy));
        ui.label(format!("Pauli limit: {:.1} T", zr.pauli_limit_field));
    }

    changed
}
