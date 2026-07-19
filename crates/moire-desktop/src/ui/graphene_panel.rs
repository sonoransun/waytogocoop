use egui::Ui;
use moire_core::curvature::CurvatureGeometry;
use moire_core::graphene::{FlatBandConfig, StackingKind};
use moire_core::materials;

use crate::app::{GrapheneView, MoireApp, ViewMode};

/// One sidebar preset: a complete graphene + curvature parameter set,
/// mirroring the presets on the web /graphene page.
#[derive(Clone, Copy)]
struct GraphenePreset {
    name: &'static str,
    stacking: StackingKind,
    twist: f64,
    filling: f64,
    geometry: CurvatureGeometry,
    amplitude: f64,
    sigma: f64,
    wavelength: f64,
    orientation_deg: f64,
    view: GrapheneView,
    honeycomb: bool,
    valley: i32,
    strain_percent: f64,
    strain_angle_deg: f64,
    warp: bool,
    /// Supermoire overlayer formula (must match the materials database).
    overlayer: Option<&'static str>,
    interface_twist: f64,
    /// Viewport size in Angstrom (shared with the main tabs).
    extent: f64,
    /// Whether the preset opens in the 3D surface view.
    surface_3d: bool,
}

const BASE_PRESET: GraphenePreset = GraphenePreset {
    name: "Magic-angle TBG",
    stacking: StackingKind::TwistedBilayer,
    twist: 1.08,
    filling: 2.4,
    geometry: CurvatureGeometry::Flat,
    amplitude: 5.0,
    sigma: 50.0,
    wavelength: 100.0,
    orientation_deg: 30.0,
    view: GrapheneView::Pattern,
    honeycomb: false,
    valley: 1,
    strain_percent: 0.0,
    strain_angle_deg: 0.0,
    warp: false,
    overlayer: None,
    interface_twist: 0.0,
    extent: 200.0,
    surface_3d: false,
};

const PRESETS: [GraphenePreset; 7] = [
    BASE_PRESET,
    GraphenePreset {
        name: "Alt-twist trilayer",
        stacking: StackingKind::AlternatingTrilayer,
        twist: 1.52,
        view: GrapheneView::GapMap,
        ..BASE_PRESET
    },
    GraphenePreset {
        name: "AB (Bernal) bilayer",
        stacking: StackingKind::AB,
        twist: 0.0,
        extent: 15.0,
        ..BASE_PRESET
    },
    GraphenePreset {
        name: "Armchair ripple",
        geometry: CurvatureGeometry::SinusoidalRipple,
        // Python twin: config.RIPPLE_HEIGHT_DEFAULT (Meyer et al. intrinsic ripples).
        amplitude: 2.0,
        view: GrapheneView::PseudoField,
        ..BASE_PRESET
    },
    GraphenePreset {
        name: "Gaussian bump 3D",
        geometry: CurvatureGeometry::GaussianBump,
        surface_3d: true,
        ..BASE_PRESET
    },
    GraphenePreset {
        name: "Supermoire on Sb2Te3",
        overlayer: Some("Sb₂Te₃"),
        ..BASE_PRESET
    },
    GraphenePreset {
        name: "Magic-angle flat bands",
        view: GrapheneView::Bands,
        ..BASE_PRESET
    },
];

/// Write every graphene / curvature parameter of a preset into the app.
/// `physical_extent` is shared with the main tabs, so the full pipeline is
/// flagged for recompute.
fn apply_preset(app: &mut MoireApp, preset: &GraphenePreset) {
    app.graphene_stack = preset.stacking;
    app.graphene_twist = preset.twist;
    app.graphene_filling = preset.filling;
    app.curvature_config.geometry = preset.geometry;
    app.curvature_config.amplitude = preset.amplitude;
    app.curvature_config.sigma = preset.sigma;
    app.curvature_config.wavelength = preset.wavelength;
    app.curvature_config.orientation_deg = preset.orientation_deg;
    app.graphene_view = preset.view;
    app.graphene_honeycomb = preset.honeycomb;
    app.graphene_valley = preset.valley;
    app.graphene_strain_percent = preset.strain_percent;
    app.graphene_strain_angle = preset.strain_angle_deg;
    app.graphene_warp = preset.warp;
    app.supermoire_overlayer = preset.overlayer.map(str::to_owned);
    app.supermoire_interface_twist = preset.interface_twist;
    app.physical_extent = preset.extent;
    app.view_mode = if preset.surface_3d {
        ViewMode::Surface3D
    } else {
        ViewMode::Flat2D
    };
    app.needs_recompute = true;
    app.needs_surface_rerender = true;
}

/// Render the graphene stack control panel. Returns true if any parameter changed.
pub fn show_graphene_panel(ui: &mut Ui, app: &mut MoireApp) -> bool {
    let mut changed = false;

    ui.heading("Graphene Stack");
    ui.add_space(4.0);
    ui.colored_label(
        egui::Color32::from_rgb(220, 50, 50),
        "SPECULATIVE: flat-band gap model",
    );
    ui.add_space(8.0);

    // --- Presets ---
    ui.label("Preset:");
    egui::ComboBox::from_id_salt("graphene_preset_combo")
        .selected_text("Select preset")
        .show_ui(ui, |ui| {
            for preset in &PRESETS {
                if ui.selectable_label(false, preset.name).clicked() {
                    apply_preset(app, preset);
                    changed = true;
                }
            }
        });

    ui.add_space(8.0);

    // --- Stacking ---
    ui.label("Stacking:");
    egui::ComboBox::from_id_salt("graphene_stacking_combo")
        .selected_text(app.graphene_stack.label())
        .show_ui(ui, |ui| {
            for kind in StackingKind::ALL {
                if ui
                    .selectable_value(&mut app.graphene_stack, kind, kind.label())
                    .changed()
                {
                    changed = true;
                }
            }
        });

    if ui
        .checkbox(&mut app.graphene_honeycomb, "Honeycomb basis (A + B sublattices)")
        .changed()
    {
        changed = true;
    }

    ui.horizontal(|ui| {
        ui.label("Valley:");
        for (valley, label) in [(1, "K"), (-1, "K'")] {
            if ui
                .selectable_value(&mut app.graphene_valley, valley, label)
                .changed()
            {
                changed = true;
            }
        }
    });

    ui.add_space(8.0);

    // --- Twist angle (twisted stackings only) ---
    if app.graphene_stack.is_twisted() {
        ui.label("Twist angle (deg):");
        if ui
            .add(
                egui::Slider::new(&mut app.graphene_twist, 0.0..=5.0)
                    .step_by(0.01)
                    .suffix(" deg"),
            )
            .changed()
        {
            changed = true;
        }
        if ui.small_button("Magic angle").clicked() {
            if let Ok(theta) =
                moire_core::graphene::magic_angle_deg(app.graphene_stack.n_layers())
            {
                app.graphene_twist = theta;
                changed = true;
            }
        }
        ui.add_space(8.0);
    }

    // --- Heterostrain ---
    ui.horizontal(|ui| {
        ui.label("Heterostrain:");
        if ui
            .add(
                egui::DragValue::new(&mut app.graphene_strain_percent)
                    .range(0.0..=2.0)
                    .speed(0.01)
                    .suffix(" %"),
            )
            .changed()
        {
            changed = true;
        }
        if ui
            .add(
                egui::DragValue::new(&mut app.graphene_strain_angle)
                    .range(0.0..=90.0)
                    .speed(0.5)
                    .suffix(" deg"),
            )
            .changed()
        {
            changed = true;
        }
    });
    if app.graphene_strain_percent != 0.0 {
        ui.weak("Uniaxial strain on layer 2 distorts the moire lattice anisotropically");
    }

    ui.add_space(8.0);

    // --- Filling ---
    ui.label("Filling |nu|:");
    if ui
        .add(egui::Slider::new(&mut app.graphene_filling, 0.0..=4.0).step_by(0.1))
        .changed()
    {
        changed = true;
    }

    ui.add_space(8.0);

    // --- Supermoire substrate ---
    egui::CollapsingHeader::new("Supermoire Substrate")
        .default_open(false)
        .show(ui, |ui| {
            ui.label("Overlayer:");
            let selected_text = app
                .supermoire_overlayer
                .clone()
                .unwrap_or_else(|| "None".to_owned());
            egui::ComboBox::from_id_salt("supermoire_overlayer_combo")
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(app.supermoire_overlayer.is_none(), "None")
                        .clicked()
                        && app.supermoire_overlayer.take().is_some()
                    {
                        changed = true;
                    }
                    for mat in materials::overlayers() {
                        // A graphene stack on graphene is just a larger
                        // stack, not a supermoire.
                        if mat.formula.starts_with("Graphene") {
                            continue;
                        }
                        let selected =
                            app.supermoire_overlayer.as_deref() == Some(mat.formula);
                        let label = format!("{} (a = {:.3} A)", mat.formula, mat.a);
                        if ui.selectable_label(selected, label).clicked() && !selected {
                            app.supermoire_overlayer = Some(mat.formula.to_owned());
                            changed = true;
                        }
                    }
                });

            if app.supermoire_overlayer.is_some() {
                ui.label("Interface twist (deg):");
                if ui
                    .add(
                        egui::Slider::new(&mut app.supermoire_interface_twist, 0.0..=30.0)
                            .step_by(0.1),
                    )
                    .changed()
                {
                    changed = true;
                }
                if let Some(ref sm) = app.supermoire_result {
                    let fmt_period = |p: f64| {
                        if p.is_finite() {
                            format!("{:.1} A", p)
                        } else {
                            "inf".to_owned()
                        }
                    };
                    ui.label(format!("Stack period: {}", fmt_period(sm.stack_period)));
                    ui.label(format!(
                        "Interface period: {}",
                        fmt_period(sm.interface_period)
                    ));
                    ui.label(format!(
                        "Supermoire period: {}",
                        fmt_period(sm.supermoire_period)
                    ));
                }
            }
        });

    ui.add_space(8.0);

    // --- Sheet curvature ---
    egui::CollapsingHeader::new("Sheet Curvature (SPECULATIVE)")
        .default_open(false)
        .show(ui, |ui| {
            ui.label("Geometry:");
            egui::ComboBox::from_id_salt("curvature_geometry_combo")
                .selected_text(app.curvature_config.geometry.label())
                .show_ui(ui, |ui| {
                    for geom in CurvatureGeometry::ALL {
                        if ui
                            .selectable_value(&mut app.curvature_config.geometry, geom, geom.label())
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });

            let geometry = app.curvature_config.geometry;
            match geometry {
                CurvatureGeometry::Flat => {}
                CurvatureGeometry::GaussianBump => {
                    ui.label("Amplitude (A):");
                    if ui
                        .add(egui::Slider::new(&mut app.curvature_config.amplitude, 0.0..=20.0))
                        .changed()
                    {
                        changed = true;
                    }
                    ui.label("Sigma (A):");
                    if ui
                        .add(egui::Slider::new(&mut app.curvature_config.sigma, 20.0..=500.0))
                        .changed()
                    {
                        changed = true;
                    }
                }
                CurvatureGeometry::SinusoidalRipple => {
                    ui.label("Amplitude (A):");
                    if ui
                        .add(egui::Slider::new(&mut app.curvature_config.amplitude, 0.0..=20.0))
                        .changed()
                    {
                        changed = true;
                    }
                    ui.label("Wavelength (A):");
                    if ui
                        .add(egui::Slider::new(&mut app.curvature_config.wavelength, 20.0..=500.0))
                        .changed()
                    {
                        changed = true;
                    }
                }
                CurvatureGeometry::CylindricalBend | CurvatureGeometry::SphericalCap => {
                    ui.label("Radius (A):");
                    if ui
                        .add(egui::Slider::new(&mut app.curvature_config.radius, 100.0..=5000.0))
                        .changed()
                    {
                        changed = true;
                    }
                }
            }

            if matches!(
                geometry,
                CurvatureGeometry::SinusoidalRipple | CurvatureGeometry::CylindricalBend
            ) {
                ui.label("Orientation (deg):");
                if ui
                    .add(egui::Slider::new(
                        &mut app.curvature_config.orientation_deg,
                        0.0..=60.0,
                    ))
                    .changed()
                {
                    changed = true;
                }
                ui.weak("0 = zigzag, 30 = armchair");
            }

            if ui
                .checkbox(&mut app.graphene_warp, "Warp stack by displacement field")
                .changed()
            {
                changed = true;
            }
            ui.weak("SPECULATIVE: u = -h grad(h) / 2 applied to all layers");
        });

    ui.add_space(8.0);

    // --- View selector ---
    ui.label("View:");
    ui.horizontal(|ui| {
        for (view, label) in [
            (GrapheneView::Pattern, "Pattern"),
            (GrapheneView::GapMap, "Gap"),
            (GrapheneView::PseudoField, "B_ps"),
            (GrapheneView::Strain, "Strain"),
        ] {
            if ui
                .selectable_value(&mut app.graphene_view, view, label)
                .changed()
            {
                changed = true;
            }
        }
    });
    ui.horizontal(|ui| {
        for (view, label) in [
            (GrapheneView::Fourier, "FFT"),
            (GrapheneView::Bands, "Bands"),
            (GrapheneView::Dos, "DOS"),
        ] {
            if ui
                .selectable_value(&mut app.graphene_view, view, label)
                .changed()
            {
                changed = true;
            }
        }
    });

    if matches!(app.graphene_view, GrapheneView::Bands | GrapheneView::Dos) {
        if let Some(ref bs) = app.band_structure {
            ui.label(format!("Flat bandwidth: {:.2} meV", bs.flat_bandwidth_mev));
            ui.label(format!("Flat-remote gap: {:.2} meV", bs.flat_gap_mev));
        }
    }

    // --- Summary readout ---
    if let Some(ref g) = app.graphene_result {
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);
        if g.moire_period.is_finite() {
            ui.label(format!("Moire period: {:.1} A", g.moire_period));
        }
        if app.graphene_stack.is_twisted() && app.graphene_twist > 0.0 {
            let ratio = moire_core::graphene::dirac_velocity_ratio(app.graphene_twist);
            ui.label(format!("v*/v: {:.3}", ratio));
            let fb = moire_core::graphene::compute_flat_band_sc(&FlatBandConfig {
                twist_angle_deg: app.graphene_twist,
                n_layers: app.graphene_stack.n_layers(),
                filling: app.graphene_filling,
            });
            if let Ok(fb) = fb {
                ui.label(format!("Delta: {:.3} meV", fb.delta_mev));
                ui.label(format!("Tc: {:.2} K", fb.tc_kelvin));
            }
        } else {
            ui.weak("Untwisted stack: set viewport size <= 30 A to resolve the atomic-scale pattern");
        }
    }
    if let Some(ref c) = app.curvature_result {
        ui.label(format!("Max |B_ps|: {:.2} T", c.max_abs_field));
    }

    changed
}
