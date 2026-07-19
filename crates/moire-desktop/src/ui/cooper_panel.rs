use egui::Ui;

use crate::app::{CooperView, MoireApp};

/// Render the Cooper 3D / proximity control panel. Returns true when any
/// control changed, signalling `needs_cooper_recompute`.
///
/// The proximity z-decay is established physics; only the Majorana view is
/// speculative, so the red label is scoped to it. The `xi` and interface
/// transparency sliders live here (not the magnetic panel) because they
/// configure this tab's z-decay rather than the vortex compute.
pub fn show_cooper_panel(ui: &mut Ui, app: &mut MoireApp) -> bool {
    let mut changed = false;

    ui.heading("Cooper 3D / Proximity");
    ui.add_space(4.0);

    // --- View selector ---
    ui.label("View:");
    let prev_view = app.cooper_view;
    egui::ComboBox::from_id_salt("cooper_view_combo")
        .selected_text(cooper_view_label(app.cooper_view))
        .show_ui(ui, |ui| {
            for view in [
                CooperView::InterfaceGap,
                CooperView::ZSlice,
                CooperView::DecayProfile,
                CooperView::Majorana,
            ] {
                if ui
                    .selectable_value(&mut app.cooper_view, view, cooper_view_label(view))
                    .changed()
                {
                    changed = true;
                }
            }
        });
    // Selecting the Majorana view enables its computation so the map is never
    // silently empty.
    if app.cooper_view != prev_view && app.cooper_view == CooperView::Majorana {
        app.show_majorana = true;
    }

    ui.add_space(8.0);

    // --- Proximity parameters (established physics) ---
    ui.label("Proximity xi (A):");
    if ui
        .add(egui::Slider::new(
            &mut app.proximity_config.xi_prox,
            10.0..=500.0,
        ))
        .changed()
    {
        // The Majorana z-envelope depends on xi_prox, so drop its cache.
        app.majorana_density = None;
        changed = true;
    }
    ui.label("Interface transparency:");
    if ui
        .add(
            egui::Slider::new(&mut app.proximity_config.interface_transparency, 0.1..=1.0)
                .step_by(0.05),
        )
        .changed()
    {
        changed = true;
    }

    ui.add_space(8.0);

    // --- Z-slice browser (only meaningful for the Z-slice view) ---
    let n_z = app
        .cooper_z_coords
        .as_ref()
        .map(|z| z.len())
        .unwrap_or(app.proximity_config.n_z_layers)
        .max(1);
    let z_coords = app.cooper_z_coords.clone();
    let mut idx = app.z_slice_index.min(n_z - 1);
    let resp = ui.add_enabled_ui(app.cooper_view == CooperView::ZSlice, |ui| {
        ui.label("Z-slice:");
        let slider = egui::Slider::new(&mut idx, 0..=n_z - 1).custom_formatter(move |v, _| {
            z_coords
                .as_ref()
                .and_then(|z| z.get(v as usize))
                .map(|&z| format!("z = {z:.0} Å"))
                .unwrap_or_else(|| format!("{v}"))
        });
        ui.add(slider).changed()
    });
    if resp.inner {
        app.z_slice_index = idx;
        changed = true;
    }

    ui.add_space(8.0);

    // --- Majorana toggle (speculative) ---
    ui.colored_label(
        egui::Color32::from_rgb(220, 50, 50),
        "Majorana view is SPECULATIVE",
    );
    if ui
        .checkbox(&mut app.show_majorana, "Compute Majorana density")
        .changed()
    {
        changed = true;
    }

    ui.add_space(8.0);

    // --- Z-sweep animation ---
    let play_label = if app.cooper_playing {
        "⏸ Pause z sweep"
    } else {
        "▶ Play z sweep"
    };
    if ui.button(play_label).clicked() {
        app.cooper_playing = !app.cooper_playing;
    }

    changed
}

fn cooper_view_label(view: CooperView) -> &'static str {
    match view {
        CooperView::InterfaceGap => "Interface gap",
        CooperView::ZSlice => "Z-slice",
        CooperView::DecayProfile => "Decay profile",
        CooperView::Majorana => "Majorana density",
    }
}
