use egui::Ui;
use moire_core::colormap::ColormapName;
use moire_core::materials;

use crate::app::MoireApp;

/// Render the sidebar controls for material selection and simulation parameters.
pub fn show_sidebar(ui: &mut Ui, app: &mut MoireApp) {
    // --- Theme toggle ---
    ui.horizontal(|ui| {
        ui.label("Theme:");
        if ui.selectable_label(app.dark_mode, "Dark").clicked() && !app.dark_mode {
            app.dark_mode = true;
            ui.ctx().set_visuals(egui::Visuals::dark());
        }
        if ui.selectable_label(!app.dark_mode, "Light").clicked() && app.dark_mode {
            app.dark_mode = false;
            ui.ctx().set_visuals(egui::Visuals::light());
        }
    });
    ui.add_space(8.0);
    ui.separator();
    ui.add_space(8.0);

    ui.heading("Parameters");
    ui.add_space(8.0);

    let mut changed = false;

    // --- Substrate selection ---
    ui.label("Substrate:");
    let substrates = materials::substrates();
    if substrates.is_empty() {
        ui.label("  (no substrates available)");
        return;
    }
    let current_sub_name = substrates[app.substrate_idx.min(substrates.len() - 1)]
        .name
        .to_string();

    egui::ComboBox::from_id_salt("substrate_combo")
        .selected_text(&current_sub_name)
        .show_ui(ui, |ui| {
            for (idx, mat) in substrates.iter().enumerate() {
                let label = format!("{} (a = {:.3} A)", mat.formula, mat.a);
                if ui
                    .selectable_value(&mut app.substrate_idx, idx, label)
                    .changed()
                {
                    changed = true;
                }
            }
        });

    ui.add_space(8.0);

    // --- Overlayer selection ---
    ui.label("Overlayer:");
    let overlayers = materials::overlayers();
    if overlayers.is_empty() {
        ui.label("  (no overlayers available)");
        return;
    }
    let current_name = overlayers[app.overlayer_idx.min(overlayers.len() - 1)]
        .name
        .to_string();

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

    ui.add_space(12.0);

    // --- 3D view options ---
    ui.label("3D overlays:");
    if ui
        .checkbox(&mut app.show_world_axes, "World axes + scale bar")
        .changed()
    {
        app.needs_surface_rerender = true;
    }
    if ui.checkbox(&mut app.show_wireframe, "Wireframe").changed() {
        app.needs_surface_rerender = true;
    }

    // --- Colormap override ---
    // Auto keeps each view's semantic palette; a pick forces all views (and
    // their colorbars) to one map. Cascades: needs_recompute rebuilds the base
    // + magnetic + cooper textures; the other two flags cover the comparison
    // window and the live 3D surface.
    ui.horizontal(|ui| {
        ui.label("Colormap:");
        let current = app.colormap_override.map(ColormapName::as_str).unwrap_or("Auto");
        let mut changed_cmap = false;
        egui::ComboBox::from_id_salt("colormap_combo")
            .selected_text(current)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(app.colormap_override.is_none(), "Auto")
                    .clicked()
                    && app.colormap_override.is_some()
                {
                    app.colormap_override = None;
                    changed_cmap = true;
                }
                for name in ColormapName::ALL {
                    let selected = app.colormap_override == Some(name);
                    if ui.selectable_label(selected, name.as_str()).clicked() && !selected {
                        app.colormap_override = Some(name);
                        changed_cmap = true;
                    }
                }
            });
        if changed_cmap {
            app.needs_recompute = true;
            app.comparison_needs_refresh = true;
            app.needs_surface_rerender = true;
        }
    });

    // --- Clipping plane ---
    // In normalised surface coords: the mesh spans z in [0, 0.3] (software
    // rasterizer convention) so clip_z = 1.0 means "disabled" for most
    // values. Exposed here so both backends receive it via the Renderer3D
    // FrameInputs.clip_z field.
    ui.horizontal(|ui| {
        if ui.checkbox(&mut app.clip_z_enabled, "Clip plane").changed() {
            app.needs_surface_rerender = true;
        }
        ui.add_enabled_ui(app.clip_z_enabled, |ui| {
            if ui
                .add(egui::Slider::new(&mut app.clip_z, -0.1_f32..=0.3_f32).step_by(0.005))
                .changed()
            {
                app.needs_surface_rerender = true;
            }
        });
    });

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

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);

    // --- Magnetic / Topological ---
    let mag_changed = super::magnetic_panel::show_magnetic_panel(ui, app);
    if mag_changed {
        app.needs_magnetic_recompute = true;
    }

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);

    // --- Cooper 3D / Proximity ---
    if super::cooper_panel::show_cooper_panel(ui, app) {
        app.needs_cooper_recompute = true;
    }

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);

    // --- Graphene Stack ---
    if super::graphene_panel::show_graphene_panel(ui, app) {
        app.needs_graphene_recompute = true;
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
    if ui.button("Phase Diagram (speculative)").clicked() {
        app.show_phase_diagram = true;
        app.phase_needs_refresh = true;
    }
}
