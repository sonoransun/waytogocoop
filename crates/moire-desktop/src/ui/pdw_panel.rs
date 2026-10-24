//! Pair Density Wave sidebar panel (speculative) plus the PDW palette shared
//! by the viewport plots and the regime-map texture. Mirrors the controls and
//! presets of the web `/pdw` page.

use egui::{Color32, Ui};
use moire_core::pdw::{PdwFamily, PdwGeometry, PdwRegime, TC_FETE_FILM, TC_UTE2, T_PDW_UTE2};

use crate::app::{MoireApp, PdwSettings, PdwSweepAxis, PdwView};

/// Composite-CDW family colour, matching Python `_pdw_family_colors` (the
/// first three slots of the validated categorical palette, per theme).
pub fn family_color(family: PdwFamily, dark: bool) -> Color32 {
    match (family, dark) {
        (PdwFamily::Q, true) => Color32::from_rgb(0x39, 0x87, 0xe5),
        (PdwFamily::P, true) => Color32::from_rgb(0xd9, 0x59, 0x26),
        (PdwFamily::H, true) => Color32::from_rgb(0x19, 0x9e, 0x70),
        (PdwFamily::Q, false) => Color32::from_rgb(0x2a, 0x78, 0xd6),
        (PdwFamily::P, false) => Color32::from_rgb(0xeb, 0x68, 0x34),
        (PdwFamily::H, false) => Color32::from_rgb(0x1b, 0xaf, 0x7a),
    }
}

/// Regime colour, matching Python `_pdw_regime_colors`: neutral grays where
/// no CDW shows, the q / p hues where they do.
pub fn regime_color(regime: PdwRegime, dark: bool) -> Color32 {
    match (regime, dark) {
        (PdwRegime::Normal, true) => Color32::from_rgb(0x2b, 0x2b, 0x29),
        (PdwRegime::Normal, false) => Color32::from_rgb(0xe8, 0xe7, 0xe3),
        (PdwRegime::UniformOnly, true) => Color32::from_rgb(0x6b, 0x6a, 0x65),
        (PdwRegime::UniformOnly, false) => Color32::from_rgb(0xa3, 0xa2, 0x9d),
        (PdwRegime::PdwOnly, _) => family_color(PdwFamily::Q, dark),
        (PdwRegime::Coexistence, _) => family_color(PdwFamily::P, dark),
    }
}

/// One sidebar preset, mirroring `_PRESETS` on the web page. The moire-locked
/// presets use whatever substrate / overlayer the main sidebar has selected.
struct PdwPreset {
    name: &'static str,
    settings: PdwSettings,
    view: PdwView,
}

fn presets() -> [PdwPreset; 6] {
    let base = PdwSettings::default();
    let moire = PdwSettings {
        geometry: PdwGeometry::MoireLocked,
        temperature: 4.0,
        tpdw_ratio: 1.0,
        ..base
    };
    [
        PdwPreset {
            name: "UTe2 300 mK, zero field",
            settings: base,
            view: PdwView::ChargeDensity,
        },
        PdwPreset {
            name: "UTe2 3 K — Cheshire-cat window",
            settings: PdwSettings {
                temperature: 3.0,
                ..base
            },
            view: PdwView::Fourier,
        },
        PdwPreset {
            name: "UTe2 field-induced p2 (Bz ≈ b* = 6 T)",
            settings: PdwSettings {
                field: [0.0, 0.0, 6.0],
                ..base
            },
            view: PdwView::FieldSweep,
        },
        PdwPreset {
            name: "UTe2 4.2 K, By = a = 0.2 T (q fading)",
            settings: PdwSettings {
                temperature: 4.2,
                field: [0.0, 0.2, 0.0],
                sweep_axis: PdwSweepAxis::Y,
                ..base
            },
            view: PdwView::FieldSweep,
        },
        PdwPreset {
            name: "Moire-locked CPDM",
            settings: moire,
            view: PdwView::PairAmplitude,
        },
        PdwPreset {
            name: "Moire what-if: intrinsic PDW above Tc",
            settings: PdwSettings {
                temperature: 16.0,
                tpdw_ratio: 1.5,
                ..moire
            },
            view: PdwView::ChargeDensity,
        },
    ]
}

const FIELD_LABELS: [&str; 3] = ["Bx (≈ c*)", "By (= a)", "Bz (≈ b*)"];

/// T_PDW / Tc each geometry starts from: UTe2's measured ratio, and 1 for the
/// imprinted moire CPDM (no PDW-only window). Mirrors the web page.
fn default_tpdw_ratio(geometry: PdwGeometry) -> f64 {
    match geometry {
        PdwGeometry::Ute2Surface => T_PDW_UTE2 / TC_UTE2,
        PdwGeometry::MoireLocked => 1.0,
    }
}

/// Render the Pair Density Wave control panel. Returns true when any control
/// changed, signalling `needs_pdw_recompute`.
pub fn show_pdw_panel(ui: &mut Ui, app: &mut MoireApp) -> bool {
    let mut changed = false;

    ui.heading("Pair Density Wave");
    ui.add_space(4.0);
    ui.colored_label(
        Color32::from_rgb(220, 50, 50),
        "SPECULATIVE: Landau model (Zhu et al. 2026)",
    );
    ui.add_space(8.0);

    // --- Presets ---
    ui.label("Preset:");
    egui::ComboBox::from_id_salt("pdw_preset_combo")
        .selected_text("Select preset")
        .show_ui(ui, |ui| {
            for preset in presets() {
                if ui.selectable_label(false, preset.name).clicked() {
                    app.pdw_settings = preset.settings;
                    app.pdw_view = preset.view;
                    changed = true;
                }
            }
        });

    ui.add_space(8.0);

    // --- Geometry ---
    ui.label("Geometry:");
    let prev_geometry = app.pdw_settings.geometry;
    egui::ComboBox::from_id_salt("pdw_geometry_combo")
        .selected_text(app.pdw_settings.geometry.label())
        .show_ui(ui, |ui| {
            for geometry in PdwGeometry::ALL {
                changed |= ui
                    .selectable_value(&mut app.pdw_settings.geometry, geometry, geometry.label())
                    .changed();
            }
        });
    // A switch carrying the previous geometry's default T_PDW / Tc takes the
    // new geometry's default (UTe2's 2.38 would put T_PDW near 32 K on the
    // FeTe films); a ratio the user chose is kept.
    if app.pdw_settings.geometry != prev_geometry
        && (app.pdw_settings.tpdw_ratio - default_tpdw_ratio(prev_geometry)).abs() < 1e-9
    {
        app.pdw_settings.tpdw_ratio = default_tpdw_ratio(app.pdw_settings.geometry);
    }

    ui.add_space(8.0);

    // --- Temperatures ---
    let (tc, t_max) = match app.pdw_settings.geometry {
        PdwGeometry::Ute2Surface => (TC_UTE2, 6.0),
        PdwGeometry::MoireLocked => (TC_FETE_FILM, 30.0),
    };
    // Leaving a hot moire state for UTe2 must not strand T above the slider.
    if app.pdw_settings.temperature > t_max {
        app.pdw_settings.temperature = t_max;
        changed = true;
    }
    ui.label(format!("Temperature (Tc = {tc:.1} K):"));
    changed |= ui
        .add(
            egui::Slider::new(&mut app.pdw_settings.temperature, 0.0..=t_max)
                .step_by(0.05)
                .suffix(" K"),
        )
        .changed();
    ui.label("T_PDW / Tc:");
    changed |= ui
        .add(egui::Slider::new(&mut app.pdw_settings.tpdw_ratio, 0.5..=3.0).step_by(0.01))
        .on_hover_text(format!(
            "Above 1 opens the 'Cheshire cat' window where only the q = 2p and h \
             charge orders survive. UTe2: {T_PDW_UTE2} K / {TC_UTE2} K."
        ))
        .changed();

    ui.add_space(8.0);

    // --- Field (paper's vector-magnet frame) ---
    ui.label("Applied field (T):");
    for (component, label) in app.pdw_settings.field.iter_mut().zip(FIELD_LABELS) {
        ui.horizontal(|ui| {
            ui.label(label);
            changed |= ui
                .add(
                    egui::DragValue::new(component)
                        .range(0.0..=60.0)
                        .speed(0.05)
                        .suffix(" T"),
                )
                .changed();
        });
    }
    ui.label("Sweep axis:");
    egui::ComboBox::from_id_salt("pdw_sweep_axis_combo")
        .selected_text(app.pdw_settings.sweep_axis.label())
        .show_ui(ui, |ui| {
            for axis in PdwSweepAxis::ALL {
                changed |= ui
                    .selectable_value(&mut app.pdw_settings.sweep_axis, axis, axis.label())
                    .changed();
            }
        });

    ui.add_space(8.0);

    // --- View ---
    ui.label("View:");
    egui::ComboBox::from_id_salt("pdw_view_combo")
        .selected_text(app.pdw_view.label())
        .show_ui(ui, |ui| {
            for view in PdwView::ALL {
                changed |= ui
                    .selectable_value(&mut app.pdw_view, view, view.label())
                    .changed();
            }
        });

    ui.add_space(8.0);

    // --- Landau couplings ---
    egui::CollapsingHeader::new("Landau couplings")
        .default_open(false)
        .show(ui, |ui| {
            let s = &mut app.pdw_settings;
            for (value, label) in [
                (&mut s.kappa_p, "κ_p (PDW × uniform)"),
                (&mut s.kappa_q, "κ_q (PDW alone)"),
                (&mut s.kappa_h, "κ_h (harmonics)"),
                (&mut s.kappa_harm, "κ_harm (q from p)"),
            ] {
                changed |= ui
                    .add(
                        egui::Slider::new(value, 0.0..=2.0)
                            .step_by(0.05)
                            .text(label),
                    )
                    .changed();
            }
            changed |= ui
                .add(
                    egui::Slider::new(&mut s.floor_log10, -4.0..=-1.0)
                        .step_by(0.1)
                        .text("floor log₁₀|ρ|²"),
                )
                .changed();
        });

    // --- Readout ---
    if let Some(rt) = app.pdw_result.as_ref() {
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
            ui.painter()
                .rect_filled(rect, 2.0, regime_color(rt.state.regime, app.dark_mode));
            ui.label(rt.state.regime.label());
        });
        ui.label(format!(
            "δ0 = {:.3}   δp = {:.3}",
            rt.state.delta0, rt.state.delta_p
        ));
        let visible: Vec<&str> = rt
            .peaks
            .iter()
            .filter(|p| p.visible)
            .map(|p| p.label)
            .collect();
        if visible.is_empty() {
            ui.label("Visible CDW peaks: none");
        } else {
            ui.label(format!("Visible CDW peaks: {}", visible.join(" ")));
        }
    }

    changed
}
