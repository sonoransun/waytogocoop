//! Spin Fluctuations sidebar panel (speculative) plus the SF palette shared by
//! the plots and the readouts. Mirrors the controls and presets of the web
//! `/spinfluct` page; the glue what-if takes its layers, twist and grid from
//! the main sidebar.

use egui::{Color32, Slider, Ui};
use moire_core::colormap::plasma;
use moire_core::spin_fluctuation::{
    self as sf, SfCoherence, SfDopingRegime, SfInelasticForm, SfPairingForm,
};

use crate::app::{MoireApp, SfSettings, SfView, Tab};

/// Colour of the "speculative" headings, as in the other panels.
const SPECULATIVE_RED: Color32 = Color32::from_rgb(220, 50, 50);

/// Categorical series colour: slots 0–2 of the validated palette (blue,
/// orange, green), the hues `pdw_panel::family_color` uses, per theme.
pub fn series_color(slot: usize, dark: bool) -> Color32 {
    match (slot % 3, dark) {
        (0, true) => Color32::from_rgb(0x39, 0x87, 0xe5),
        (1, true) => Color32::from_rgb(0xd9, 0x59, 0x26),
        (_, true) => Color32::from_rgb(0x19, 0x9e, 0x70),
        (0, false) => Color32::from_rgb(0x2a, 0x78, 0xd6),
        (1, false) => Color32::from_rgb(0xeb, 0x68, 0x34),
        (_, false) => Color32::from_rgb(0x1b, 0xaf, 0x7a),
    }
}

/// Colour of temperature curve `i` of `n`, matching Python
/// `_sf_temperature_colors`: the plasma LUT at evenly spaced stops over
/// 0.35–0.92 on dark backgrounds and 0.05–0.78 on light ones (no yellow on
/// white).
pub fn temperature_color(i: usize, n: usize, dark: bool) -> Color32 {
    let (lo, hi) = if dark { (0.35, 0.92) } else { (0.05, 0.78) };
    let t = if n > 1 {
        lo + (hi - lo) * i as f64 / (n - 1) as f64
    } else {
        lo
    };
    let [r, g, b, _] = plasma(t);
    Color32::from_rgb(r, g, b)
}

/// Doping-regime swatch colour: the reconstruction side orange, the dome
/// blue, beyond the dome neutral gray.
pub fn regime_color(regime: SfDopingRegime, dark: bool) -> Color32 {
    match (regime, dark) {
        (SfDopingRegime::FsReconstruction, _) => series_color(1, dark),
        (SfDopingRegime::Superconducting, _) => series_color(0, dark),
        (SfDopingRegime::BeyondDome, true) => Color32::from_rgb(0x6b, 0x6a, 0x65),
        (SfDopingRegime::BeyondDome, false) => Color32::from_rgb(0xa3, 0xa2, 0x9d),
    }
}

/// One sidebar preset, mirroring `_PRESETS` on the web page (same names). The
/// glue presets use whatever layers the main sidebar has selected.
pub struct SfPreset {
    pub name: &'static str,
    pub settings: SfSettings,
    pub view: SfView,
}

/// The 11 web presets, in the web order.
pub fn presets() -> [SfPreset; 11] {
    let base = SfSettings::default();
    let isotropic = SfSettings {
        inelastic: SfInelasticForm::Isotropic,
        ..base
    };
    [
        SfPreset {
            name: "LCCO x = 0.159 — paper fit (Duffy Fig. 4)",
            settings: base,
            view: SfView::Magnetoresistance,
        },
        SfPreset {
            name: "LCCO x = 0.116 — near optimal (Tc ≈ 25 K)",
            settings: SfSettings {
                doping: 0.116,
                ..base
            },
            view: SfView::Magnetoresistance,
        },
        SfPreset {
            name: "LCCO x = 0.188 — beyond the dome (Tc = 0)",
            settings: SfSettings {
                doping: 0.188,
                ..base
            },
            view: SfView::Magnetoresistance,
        },
        SfPreset {
            name: "Across the dome: ρ and MR vs x",
            settings: base,
            view: SfView::Doping,
        },
        SfPreset {
            name: "No quadrature scaling (vs hole-doped)",
            settings: base,
            view: SfView::FieldDerivative,
        },
        SfPreset {
            name: "Hall coefficient with vertex corrections",
            settings: base,
            view: SfView::Hall,
        },
        SfPreset {
            name: "Kohler check: isotropic Γ, no hot spots",
            settings: SfSettings {
                hotspots: false,
                ..isotropic
            },
            view: SfView::Magnetoresistance,
        },
        SfPreset {
            name: "Ablation: isotropic + strong hot spots (Fig. 9c)",
            settings: SfSettings {
                hotspot_peak: sf::HOTSPOT_PEAK_FIG9C,
                ..isotropic
            },
            view: SfView::Magnetoresistance,
        },
        SfPreset {
            name: "Ablation: no hot spots (Fig. 9b)",
            settings: SfSettings {
                hotspots: false,
                ..base
            },
            view: SfView::Magnetoresistance,
        },
        SfPreset {
            name: "Moire glue what-if: 13.5 K operating point (FeTe-film Tc)",
            settings: base,
            view: SfView::GlueGap,
        },
        SfPreset {
            name: "Moire glue what-if: imprint vs operating Tc",
            settings: SfSettings {
                tc_target: 5.0,
                ..base
            },
            view: SfView::Imprint,
        },
    ]
}

/// The web `DOPING_OPTIONS` label, "x = 0.159 · g = 1.00 · Tc ≈ 8 K".
fn doping_label(x: f64) -> String {
    let g = sf::g_of_x(x).unwrap_or(f64::NAN);
    let tc = sf::tc_of_x(x);
    if tc > 0.0 {
        format!("x = {x:.3} · g = {g:.2} · Tc ≈ {tc:.0} K")
    } else {
        format!("x = {x:.3} · g = {g:.2} · Tc = 0")
    }
}

/// Whether the shown SF values are stale: the stage is dirty and recomputes
/// only on its own tab.
pub fn sf_stale(app: &MoireApp) -> bool {
    app.needs_sf_recompute && app.active_tab != Tab::SpinFluctuation
}

/// `SfCoherence::label` (the web `COHERENCE_LABELS`) with "∝" written "~":
/// egui's default fonts have no glyph for it. They also lack "⁻", "≫", "→"
/// and the combining macron, so the SF strings spell those out.
pub fn coherence_label(coherence: SfCoherence) -> String {
    coherence.label().replace('∝', "~")
}

/// The coherence model as plot legends and titles name it (the web
/// `COHERENCE_PLOT_LABELS`).
pub fn coherence_plot_label(coherence: SfCoherence) -> &'static str {
    match coherence {
        SfCoherence::Clean => "clean coherence",
        SfCoherence::Dirty => "dirty coherence",
        SfCoherence::Local => "local limit",
    }
}

/// A slider under its own label line, as in the main sidebar, so that long
/// units never clip at the panel edge.
fn labelled_slider(ui: &mut Ui, label: &str, slider: Slider<'_>) -> egui::Response {
    ui.label(label);
    ui.add(slider)
}

/// Render the Spin Fluctuations control panel. Returns true when any control
/// changed, signalling `needs_sf_recompute`.
pub fn show_sf_panel(ui: &mut Ui, app: &mut MoireApp) -> bool {
    let mut changed = false;

    ui.heading("Spin Fluctuations");
    ui.add_space(4.0);
    ui.colored_label(
        SPECULATIVE_RED,
        "SPECULATIVE: LCCO transport (Duffy et al. 2026) + what-ifs",
    );
    ui.add_space(8.0);

    // --- Presets ---
    ui.label("Preset:");
    egui::ComboBox::from_id_salt("sf_preset_combo")
        .selected_text("Select preset")
        .show_ui(ui, |ui| {
            for preset in presets() {
                if ui.selectable_label(false, preset.name).clicked() {
                    app.sf_settings = preset.settings;
                    app.sf_view = preset.view;
                    changed = true;
                }
            }
        });

    ui.add_space(8.0);

    let s = &mut app.sf_settings;

    // --- Doping, temperature, field ---
    ui.label("Doping x (LCCO film channel):").on_hover_text(
        "Each value is a combinatorial-film channel; g(x) is the paper's fitted strength of \
         the anisotropic T-linear scattering (1 at x = 0.159).",
    );
    // The SF labels are long: wrap them inside the sidebar (a combo lays out
    // its text in a horizontal row, which otherwise widens the panel).
    egui::ComboBox::from_id_salt("sf_doping_combo")
        .selected_text(doping_label(sf::snap_doping(s.doping)))
        .wrap()
        .show_ui(ui, |ui| {
            for x in sf::DOPINGS_LCCO {
                changed |= ui
                    .selectable_value(&mut s.doping, x, doping_label(x))
                    .changed();
            }
        });
    // The steps live on SfSettings: `sf_config` rounds each value to its
    // step's decimals, since egui re-snaps a drawn slider with binary noise.
    ui.label("Temperature:");
    changed |= ui
        .add(
            Slider::new(&mut s.temperature, SfSettings::TEMPERATURE)
                .step_by(SfSettings::TEMPERATURE_STEP)
                .suffix(" K"),
        )
        .changed();
    ui.label("Max field μ₀H:");
    changed |= ui
        .add(
            Slider::new(&mut s.field_max, SfSettings::FIELD_MAX)
                .step_by(SfSettings::FIELD_MAX_STEP)
                .suffix(" T"),
        )
        .changed();

    ui.add_space(8.0);

    // --- Scattering anisotropy ---
    ui.label("Inelastic anisotropy W(φ):");
    egui::ComboBox::from_id_salt("sf_inelastic_combo")
        .selected_text(s.inelastic.label())
        .wrap()
        .show_ui(ui, |ui| {
            for form in SfInelasticForm::ALL {
                changed |= ui
                    .selectable_value(&mut s.inelastic, form, form.label())
                    .changed();
            }
        });
    changed |= ui
        .checkbox(&mut s.hotspots, "Hot spots")
        .on_hover_text(
            "Sharp scattering where the Fermi surface crosses the AFM zone boundary; they \
             impede orbital motion and keep the MR H-linear.",
        )
        .changed();

    ui.add_space(8.0);

    // --- View ---
    ui.label("View:");
    egui::ComboBox::from_id_salt("sf_view_combo")
        .selected_text(app.sf_view.label())
        .wrap()
        .show_ui(ui, |ui| {
            for view in SfView::ALL {
                changed |= ui
                    .selectable_value(&mut app.sf_view, view, view.label())
                    .changed();
            }
        });

    ui.add_space(8.0);

    // --- Collapsible parameter groups (the web collapses and glue card) ---
    egui::CollapsingHeader::new("Scattering parameters (Table 2)")
        .default_open(false)
        .show(ui, |ui| {
            for (label, value, range, step) in [
                (
                    "Γ_imp (1/ps):",
                    &mut s.gamma_imp,
                    SfSettings::GAMMA_IMP,
                    SfSettings::GAMMA_IMP_STEP,
                ),
                (
                    "α₁ (1/(ps K)):",
                    &mut s.alpha1,
                    SfSettings::ALPHA1,
                    SfSettings::ALPHA1_STEP,
                ),
                (
                    "α₂ (1/(ps K²)):",
                    &mut s.alpha2,
                    SfSettings::ALPHA2,
                    SfSettings::ALPHA2_STEP,
                ),
            ] {
                changed |=
                    labelled_slider(ui, label, Slider::new(value, range).step_by(step)).changed();
            }
            changed |= labelled_slider(
                ui,
                "Hot-spot height (1/ps):",
                Slider::new(&mut s.hotspot_peak, SfSettings::HOTSPOT_PEAK)
                    .step_by(SfSettings::HOTSPOT_PEAK_STEP),
            )
            .on_hover_text("The effective height plotted in Fig. 4b1 (31.4); Table 2 prints 50.")
            .changed();
            changed |= labelled_slider(
                ui,
                "Hot-spot width w (rad):",
                Slider::new(&mut s.hotspot_width, SfSettings::HOTSPOT_WIDTH)
                    .step_by(SfSettings::HOTSPOT_WIDTH_STEP),
            )
            .on_hover_text("1/e half-width of the hot-spot Gaussians.")
            .changed();
            changed |= labelled_slider(
                ui,
                "ν (sin^ν exponent):",
                Slider::new(&mut s.nu, SfSettings::NU).step_by(2.0),
            )
            .changed();
            changed |= labelled_slider(
                ui,
                "Hot-spot angle φ_HS:",
                Slider::new(&mut s.phi_hs_deg, SfSettings::PHI_HS)
                    .step_by(SfSettings::PHI_HS_STEP)
                    .suffix("°"),
            )
            .changed();
        });
    egui::CollapsingHeader::new("Hall: vertex corrections")
        .default_open(false)
        .show(ui, |ui| {
            changed |= labelled_slider(
                ui,
                "CVC peak ε₀:",
                Slider::new(&mut s.cvc_epsilon, SfSettings::CVC_EPSILON)
                    .step_by(SfSettings::CVC_EPSILON_STEP),
            )
            .on_hover_text(
                "Peak of ε(φ) at the Fermi-surface × zone-boundary crossings (Eq. 13); \
                 Fig. 12a needs about 1.",
            )
            .changed();
        });
    egui::CollapsingHeader::new("Pairing map (speculative)")
        .default_open(false)
        .show(ui, |ui| {
            for form in SfPairingForm::ALL {
                changed |= ui
                    .radio_value(&mut s.pairing_form, form, form.label())
                    .changed();
            }
        });
    egui::CollapsingHeader::new("Moire glue what-if (speculative)")
        .default_open(false)
        .show(ui, |ui| {
            ui.colored_label(
                SPECULATIVE_RED,
                "SPECULATIVE: only the shape of Tc(λ) is used; not a model of FeTe",
            );
            changed |= labelled_slider(
                ui,
                "Glue depth η:",
                Slider::new(&mut s.eta, SfSettings::ETA).step_by(SfSettings::ETA_STEP),
            )
            .changed();
            changed |= labelled_slider(
                ui,
                "Tc target (13.5 K = FeTe film):",
                Slider::new(&mut s.tc_target, SfSettings::TC_TARGET)
                    .step_by(SfSettings::TC_TARGET_STEP)
                    .suffix(" K"),
            )
            .on_hover_text("The operating Tc* of the what-if.")
            .changed();
            ui.label("Coherence length:");
            egui::ComboBox::from_id_salt("sf_coherence_combo")
                .selected_text(coherence_label(s.coherence))
                .wrap()
                .show_ui(ui, |ui| {
                    for coherence in SfCoherence::ALL {
                        changed |= ui
                            .selectable_value(
                                &mut s.coherence,
                                coherence,
                                coherence_label(coherence),
                            )
                            .changed();
                    }
                });
            ui.weak("L_m: the shared two-lattice period of the selected layers and twist.");
        });

    // --- Readout ---
    ui.add_space(8.0);
    ui.separator();
    ui.add_space(4.0);
    match app.sf_result.as_ref() {
        Some(rt) => {
            let r = &rt.readouts;
            ui.horizontal(|ui| {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                ui.painter()
                    .rect_filled(rect, 2.0, regime_color(r.regime, app.dark_mode));
                ui.label(r.regime.label()).on_hover_text(r.regime.blurb());
            });
            ui.label(format!(
                "λ_tr = {:.3}   MR(20 T) = {:.2} %",
                r.lambda_tr,
                100.0 * r.mr_20t
            ));
            ui.label(format!(
                "R_H = {:.2}, with CVC {:.2} (1e-10 m³/C)",
                1e10 * r.rh_plain,
                1e10 * r.rh_cvc
            ));
            if sf_stale(app) {
                ui.weak("(updates on the Spin Fluctuations tab)");
            }
        }
        None if app.sf_error.is_some() => {
            ui.weak("Spin-fluctuation model unavailable (see the tab)");
        }
        None => {
            ui.weak("— (open the Spin Fluctuations tab)");
        }
    }

    changed
}
