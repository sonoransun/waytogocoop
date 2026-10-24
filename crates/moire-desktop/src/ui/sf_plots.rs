//! Spin Fluctuations plot views (speculative). Every SF view but the glue map
//! lives on its own axes, so the tab draws them with egui_plot instead of as
//! moire-grid textures. They mirror the web `/spinfluct` figure builders
//! (`create_sf_*` in `figure_factory.py`). The eight plots live here so that
//! `viewport.rs` only routes to them.

use std::f64::consts::PI;

use egui::{Align2, Color32, FontId, Rect, RichText, Stroke, Ui};
use egui_plot::{
    Corner, HLine, Legend, Line, LineStyle, MarkerShape, Plot, PlotPoint, Points, Polygon, Text,
    VLine,
};
use moire_core::spin_fluctuation::{self as sf, SfCoherence, SfPairingForm};

use crate::app::{MoireApp, SfRuntime, SfView};
use crate::ui::sf_panel::{coherence_plot_label, series_color, temperature_color};
use crate::ui::viewport::{muted_line_color, show_computing_message};

/// R_H axis unit: 10⁻¹⁰ m³/C.
const RH_SCALE: f64 = 1e10;
/// k in 1/m to 1/Å.
const PER_ANGSTROM: f64 = 1e-10;
/// Colour levels of the Γ(φ) ramp along the Fermi surface.
const GAMMA_LEVELS: usize = 48;

/// Draw the active Spin Fluctuations plot view (see `SfView::is_plot`).
pub fn show(ui: &mut Ui, app: &MoireApp) {
    let Some(rt) = app.sf_result.as_ref() else {
        show_computing_message(ui, "Computing spin-fluctuation transport...");
        return;
    };
    match app.sf_view {
        SfView::Magnetoresistance => show_sf_mr_plot(ui, app, rt),
        SfView::FieldDerivative => show_sf_derivative_plot(ui, app, rt),
        SfView::Resistivity => show_sf_resistivity_plot(ui, app, rt),
        SfView::Hall => show_sf_hall_plot(ui, app, rt),
        SfView::Doping => show_sf_doping_plot(ui, app, rt),
        SfView::FermiSurface => show_sf_fermi_surface_plot(ui, app, rt),
        SfView::Coupling => show_sf_coupling_plot(ui, app, rt),
        SfView::Imprint => show_sf_imprint_plot(ui, app, rt),
        // A texture view: the viewport draws it.
        SfView::GlueGap => {}
    }
}

/// (x, y) points of a line, with y scaled by `scale`.
fn xy(xs: &[f64], ys: &[f64], scale: f64) -> Vec<[f64; 2]> {
    xs.iter().zip(ys).map(|(&x, &y)| [x, scale * y]).collect()
}

/// One-line title above a plot. As on the web page, each ends with
/// "(SPECULATIVE)": the whole module is speculative.
fn plot_title(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).strong());
}

/// A note painted inside the top-left corner of a plot frame.
fn frame_note(ui: &Ui, frame: &Rect, text: &str) {
    ui.painter().text(
        frame.left_top() + egui::vec2(8.0, 6.0),
        Align2::LEFT_TOP,
        text,
        FontId::proportional(12.0),
        ui.visuals().strong_text_color(),
    );
}

/// Split a polyline into the runs at or below `y_max`, each ending on the
/// crossing, so a curve leaves the frame at its top instead of stretching the
/// y axis (the imprint plot's fixed cap).
fn clip_below(xs: &[f64], ys: &[f64], y_max: f64) -> Vec<Vec<[f64; 2]>> {
    let mut runs = Vec::new();
    let mut run: Vec<[f64; 2]> = Vec::new();
    for (i, (&x, &y)) in xs.iter().zip(ys).enumerate() {
        let inside = y.is_finite() && y <= y_max;
        if i > 0 {
            let (xp, yp) = (xs[i - 1], ys[i - 1]);
            let was_inside = yp.is_finite() && yp <= y_max;
            if inside != was_inside && y.is_finite() && yp.is_finite() {
                let f = (y_max - yp) / (y - yp);
                run.push([xp + f * (x - xp), y_max]);
            }
            if was_inside && !inside {
                runs.push(std::mem::take(&mut run));
            }
        }
        if inside {
            run.push([x, y]);
        }
    }
    runs.push(run);
    runs.retain(|r| r.len() > 1);
    runs
}

/// Δρ/ρ(0) against μ₀H: one line per family temperature (the plasma ramp),
/// plus the current T dashed when it is not one of them.
fn show_sf_mr_plot(ui: &mut Ui, app: &MoireApp, rt: &SfRuntime) {
    let dark = app.dark_mode;
    let mr = &rt.transport.mr;
    let n = mr.temperatures.len();
    let in_family = mr
        .temperatures
        .iter()
        .any(|&t| (t - rt.temperature).abs() < 1e-9);
    let current_color = ui.visuals().strong_text_color();
    plot_title(
        ui,
        &format!(
            "Magnetoresistance, LCCO x = {:.3} (SPECULATIVE)",
            rt.transport.config.doping
        ),
    );
    Plot::new("sf_mr")
        .legend(Legend::default().position(Corner::LeftTop))
        .x_axis_label("μ₀H (T)")
        .y_axis_label("Δρ/ρ(0)")
        .include_x(0.0)
        .include_y(0.0)
        .show(ui, |plot_ui| {
            for (i, (&t, row)) in mr.temperatures.iter().zip(&mr.mr).enumerate() {
                plot_ui.line(
                    Line::new(xy(&mr.fields, row, 1.0))
                        .name(format!("{t:.0} K"))
                        .color(temperature_color(i, n, dark))
                        .width(2.0_f32),
                );
            }
            if !in_family {
                plot_ui.line(
                    Line::new(xy(&rt.current.fields, &rt.current.mr[0], 1.0))
                        .name(format!("{:.1} K (current T)", rt.temperature))
                        .color(current_color)
                        .style(LineStyle::dashed_dense())
                        .width(1.5_f32),
                );
            }
        });
}

/// dρ/dμ₀H against μ₀H/T: the LCCO family does not collapse, while the
/// hole-doped quadrature form (muted, dashed; illustrative Tl2201 parameters
/// at the same temperatures) collapses onto one curve. The current T is
/// dashed when it is not a family temperature, as on the MR view.
fn show_sf_derivative_plot(ui: &mut Ui, app: &MoireApp, rt: &SfRuntime) {
    let dark = app.dark_mode;
    let mr = &rt.transport.mr;
    let n = mr.temperatures.len();
    let quad_color = muted_line_color(dark);
    let in_family = mr
        .temperatures
        .iter()
        .any(|&t| (t - rt.temperature).abs() < 1e-9);
    let current_color = ui.visuals().strong_text_color();
    // Headroom above the quadrature plateau keeps the legend off the curves.
    let top = mr
        .drho_dh
        .iter()
        .chain(&rt.transport.quadrature)
        .chain(&rt.current.drho_dh)
        .flatten()
        .fold(0.0_f64, |m, &v| m.max(v));
    plot_title(
        ui,
        &format!(
            "No H/T scaling: LCCO x = {:.3} vs the hole-doped quadrature form (SPECULATIVE)",
            rt.transport.config.doping
        ),
    );
    Plot::new("sf_derivative")
        .legend(Legend::default().position(Corner::RightTop))
        .x_axis_label("μ₀H/T (T/K)")
        .y_axis_label("dρ/dμ₀H (μΩ cm/T)")
        .include_x(0.0)
        .include_y(0.0)
        .include_y(1.3 * top)
        .show(ui, |plot_ui| {
            for (x, q) in mr.h_over_t.iter().zip(&rt.transport.quadrature) {
                plot_ui.line(
                    Line::new(xy(x, q, 1.0))
                        .name("hole-doped quadrature (Ayres 2021, illustrative)")
                        .color(quad_color)
                        .style(LineStyle::dashed_dense())
                        .width(1.5_f32),
                );
            }
            for (i, ((&t, x), d)) in mr
                .temperatures
                .iter()
                .zip(&mr.h_over_t)
                .zip(&mr.drho_dh)
                .enumerate()
            {
                plot_ui.line(
                    Line::new(xy(x, d, 1.0))
                        .name(format!("LCCO {t:.0} K"))
                        .color(temperature_color(i, n, dark))
                        .width(2.0_f32),
                );
            }
            if !in_family {
                plot_ui.line(
                    Line::new(xy(&rt.current.h_over_t[0], &rt.current.drho_dh[0], 1.0))
                        .name(format!("{:.1} K (current T)", rt.temperature))
                        .color(current_color)
                        .style(LineStyle::dashed_dense())
                        .width(1.5_f32),
                );
            }
        });
}

/// Zero-field ρ(T) over 2–100 K with its ρ₀ + A₁T + A₂T² fit dashed over the
/// fit window (dotted verticals in the fit colour) and the current T ringed.
fn show_sf_resistivity_plot(ui: &mut Ui, app: &MoireApp, rt: &SfRuntime) {
    let dark = app.dark_mode;
    let tr = &rt.transport;
    let fit = tr.rho_fit;
    let fit_color = series_color(1, dark);
    let current_color = ui.visuals().strong_text_color();
    let fit_t: Vec<f64> = tr
        .rho_t
        .iter()
        .copied()
        .filter(|&t| t >= fit.t_min && t <= fit.t_max)
        .collect();
    let fit_rho: Vec<f64> = fit_t
        .iter()
        .map(|&t| fit.rho0 + fit.a1 * t + fit.a2 * t * t)
        .collect();
    plot_title(
        ui,
        &format!(
            "Zero-field resistivity, LCCO x = {:.3} (SPECULATIVE)",
            tr.config.doping
        ),
    );
    let response = Plot::new("sf_resistivity")
        .legend(Legend::default().position(Corner::RightBottom))
        .x_axis_label("T (K)")
        .y_axis_label("ρ(0, T) (μΩ cm)")
        .show(ui, |plot_ui| {
            for edge in [fit.t_min, fit.t_max] {
                plot_ui.vline(
                    VLine::new(edge)
                        .name("fit window")
                        .color(fit_color)
                        .style(LineStyle::dotted_loose()),
                );
            }
            plot_ui.line(
                Line::new(xy(&tr.rho_t, &tr.rho, 1.0))
                    .name("ρ(0, T), model")
                    .color(series_color(0, dark))
                    .width(2.0_f32),
            );
            plot_ui.line(
                Line::new(xy(&fit_t, &fit_rho, 1.0))
                    .name("ρ₀ + A₁T + A₂T² fit")
                    .color(fit_color)
                    .style(LineStyle::dashed_dense())
                    .width(2.0_f32),
            );
            plot_ui.points(
                Points::new(vec![[rt.temperature, rt.readouts.rho0]])
                    .name(format!("current T = {:.1} K", rt.temperature))
                    .shape(MarkerShape::Circle)
                    .filled(false)
                    .radius(7.0_f32)
                    .color(current_color),
            );
        });
    frame_note(
        ui,
        response.transform.frame(),
        &format!(
            "ρ₀ = {:.2} μΩ cm   A₁ = {:.4} μΩ cm/K   A₂ = {:.2e} μΩ cm/K²",
            fit.rho0, fit.a1, fit.a2
        ),
    );
}

/// Jones–Zener R_H(T): with CVCs solid, without dashed, both Drude references
/// (dotted on the fixed x = 0.159 Fermi surface, dashed for n = 1 − x), the
/// zero line and the current T ringed on both curves.
fn show_sf_hall_plot(ui: &mut Ui, app: &MoireApp, rt: &SfRuntime) {
    let dark = app.dark_mode;
    let tr = &rt.transport;
    let r = &rt.readouts;
    let ref_color = muted_line_color(dark);
    let current_color = ui.visuals().strong_text_color();
    plot_title(
        ui,
        &format!(
            "Hall coefficient (Jones–Zener), LCCO x = {:.3} (SPECULATIVE)",
            tr.config.doping
        ),
    );
    Plot::new("sf_hall")
        .legend(Legend::default().position(Corner::RightBottom))
        .x_axis_label("T (K)")
        .y_axis_label("R_H (1e-10 m³/C)")
        .include_x(0.0)
        .show(ui, |plot_ui| {
            plot_ui.hline(HLine::new(0.0).color(ref_color).width(1.0_f32));
            plot_ui.hline(
                HLine::new(RH_SCALE * r.rh_drude)
                    .name("Drude, fixed x = 0.159 FS")
                    .color(ref_color)
                    .style(LineStyle::dotted_dense())
                    .width(1.5_f32),
            );
            plot_ui.hline(
                HLine::new(RH_SCALE * r.rh_drude_nominal)
                    .name("Drude, n = 1 − x")
                    .color(ref_color)
                    .style(LineStyle::dashed_loose())
                    .width(1.5_f32),
            );
            plot_ui.line(
                Line::new(xy(&tr.hall_t, &tr.hall_plain, RH_SCALE))
                    .name("without CVC")
                    .color(series_color(1, dark))
                    .style(LineStyle::dashed_dense())
                    .width(2.0_f32),
            );
            plot_ui.line(
                Line::new(xy(&tr.hall_t, &tr.hall_cvc, RH_SCALE))
                    .name("with CVC (Eqs. 14–15)")
                    .color(series_color(0, dark))
                    .width(2.0_f32),
            );
            plot_ui.points(
                Points::new(vec![
                    [rt.temperature, RH_SCALE * r.rh_plain],
                    [rt.temperature, RH_SCALE * r.rh_cvc],
                ])
                .name(format!("current T = {:.1} K", rt.temperature))
                .shape(MarkerShape::Circle)
                .filled(false)
                .radius(7.0_f32)
                .color(current_color),
            );
        });
}

/// ρ(0, 40 K) and MR(20 T, 40 K) against x as two stacked half-height plots
/// with linked x axes (egui_plot has no twin y axis): the end of the
/// Fermi-surface reconstruction dotted, the current x dashed and ringed.
fn show_sf_doping_plot(ui: &mut Ui, app: &MoireApp, rt: &SfRuntime) {
    let dark = app.dark_mode;
    let rows = &rt.transport.doping;
    let x_now = rt.transport.config.doping;
    let ref_color = muted_line_color(dark);
    let current_color = ui.visuals().strong_text_color();
    plot_title(
        ui,
        "Across the LCCO dome: resistivity and magnetoresistance (SPECULATIVE)",
    );
    let dopings: Vec<f64> = rows.iter().map(|p| p.doping).collect();
    let rho: Vec<f64> = rows.iter().map(|p| p.rho0).collect();
    let mr: Vec<f64> = rows.iter().map(|p| p.mr).collect();
    let height = ((ui.available_height() - ui.spacing().item_spacing.y) / 2.0).max(80.0);
    let link = ui.id().with("sf_doping_x");
    for (id, values, y_label, slot) in [
        ("sf_doping_rho", &rho, "ρ(0, 40 K) (μΩ cm)", 0),
        ("sf_doping_mr", &mr, "MR(20 T, 40 K)", 1),
    ] {
        let color = series_color(slot, dark);
        let at_x = rows
            .iter()
            .zip(values.iter())
            .find(|(p, _)| (p.doping - x_now).abs() < 1e-9)
            .map(|(_, &v)| v);
        let top = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Plot::new(id)
            .height(height)
            .link_axis(link, [true, false])
            .link_cursor(link, [true, false])
            .legend(Legend::default().position(Corner::RightTop))
            .x_axis_label("Ce doping x")
            .y_axis_label(y_label)
            .show(ui, |plot_ui| {
                plot_ui.vline(
                    VLine::new(sf::X_FSR_LCCO)
                        .color(ref_color)
                        .style(LineStyle::dotted_loose()),
                );
                plot_ui.text(
                    Text::new(
                        PlotPoint::new(sf::X_FSR_LCCO, top),
                        RichText::new(format!(" x_FSR = {}", sf::X_FSR_LCCO)).size(12.0),
                    )
                    .color(ref_color)
                    .anchor(Align2::LEFT_TOP),
                );
                plot_ui.vline(
                    VLine::new(x_now)
                        .color(ref_color)
                        .style(LineStyle::dashed_loose()),
                );
                plot_ui.line(
                    Line::new(xy(&dopings, values, 1.0))
                        .color(color)
                        .width(2.0_f32),
                );
                plot_ui.points(
                    Points::new(xy(&dopings, values, 1.0))
                        .shape(MarkerShape::Circle)
                        .radius(3.0_f32)
                        .color(color),
                );
                if let Some(v) = at_x {
                    plot_ui.points(
                        Points::new(vec![[x_now, v]])
                            .name(format!("current x = {x_now:.3}"))
                            .shape(MarkerShape::Circle)
                            .filled(false)
                            .radius(7.0_f32)
                            .color(current_color),
                    );
                }
            });
    }
}

/// Γ(φ) on the Fermi surface about M in 1/Å, equal aspect: the contour
/// coloured by Γ (viridis, or the colormap override), the dashed AFM zone
/// boundary |k_x| + |k_y| = π/a, ringed hot spots, thin chords from the CVC
/// centres (the FS × zone-boundary crossings) to their k + Q partners, and
/// "N" / "AN" labels.
fn show_sf_fermi_surface_plot(ui: &mut Ui, app: &MoireApp, rt: &SfRuntime) {
    let dark = app.dark_mode;
    let cfg = &rt.transport.config;
    let fs = &rt.fermi;
    let cmap = app.cmap(moire_core::colormap::viridis);
    let ref_color = muted_line_color(dark);
    let strong = ui.visuals().strong_text_color();
    let to_color = |t: f64| {
        let [r, g, b, _] = cmap(t);
        Color32::from_rgb(r, g, b)
    };
    let point = |phi: f64| -> [f64; 2] {
        let kf = sf::fermi_surface_at(phi, cfg).0 * PER_ANGSTROM;
        [kf * phi.cos(), kf * phi.sin()]
    };
    let lo = rt.gamma.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = rt.gamma.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;

    // Runs of equal colour level along the closed contour.
    let n = fs.phi.len();
    let level = |i: usize| -> usize {
        if span > 1e-12 {
            (((rt.gamma[i % n] - lo) / span * GAMMA_LEVELS as f64) as usize).min(GAMMA_LEVELS - 1)
        } else {
            GAMMA_LEVELS / 2
        }
    };
    let contour: Vec<[f64; 2]> = (0..=n)
        .map(|i| {
            let k = fs.kf[i % n] * PER_ANGSTROM;
            let phi = fs.phi[i % n];
            [k * phi.cos(), k * phi.sin()]
        })
        .collect();
    let mut runs: Vec<(usize, Vec<[f64; 2]>)> = Vec::new();
    for i in 0..n {
        let lvl = level(i);
        match runs.last_mut() {
            Some((l, pts)) if *l == lvl => pts.push(contour[i + 1]),
            _ => runs.push((lvl, vec![contour[i], contour[i + 1]])),
        }
    }

    let q = PI / sf::A_LCCO;
    plot_title(
        ui,
        &format!("Fermi-surface Γ(φ) at {} K (SPECULATIVE)", rt.temperature),
    );
    let response = Plot::new("sf_fermi_surface")
        .data_aspect(1.0)
        .legend(Legend::default().position(Corner::LeftTop))
        .x_axis_label("k_x − M (1/Å)")
        .y_axis_label("k_y − M (1/Å)")
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(vec![[q, 0.0], [0.0, q], [-q, 0.0], [0.0, -q], [q, 0.0]])
                    .name("AFM zone boundary")
                    .color(ref_color)
                    .style(LineStyle::dashed_dense())
                    .width(1.0_f32),
            );
            for (&c, &p) in rt.cvc_centres.iter().zip(&rt.cvc_partners) {
                plot_ui.line(
                    Line::new(vec![point(c), point(p)])
                        .name("CVC centres to their k + Q partners")
                        .color(ref_color)
                        .width(1.0_f32),
                );
            }
            for (lvl, pts) in runs {
                let t = (lvl as f64 + 0.5) / GAMMA_LEVELS as f64;
                plot_ui.line(Line::new(pts).color(to_color(t)).width(3.0_f32));
            }
            if cfg.hotspots && cfg.hotspot_peak > 0.0 {
                let rings: Vec<[f64; 2]> = rt.hotspots.iter().map(|&p| point(p)).collect();
                plot_ui.points(
                    Points::new(rings)
                        .name(format!("hot spots (φ_HS = {:.1}°)", cfg.phi_hs_deg))
                        .shape(MarkerShape::Circle)
                        .filled(false)
                        .radius(7.0_f32)
                        .color(strong),
                );
            }
            for (phi, label) in [(PI / 4.0, "N"), (0.0, "AN")] {
                let [x, y] = point(phi);
                plot_ui.text(
                    Text::new(
                        PlotPoint::new(1.15 * x, 1.15 * y),
                        RichText::new(label).size(15.0).strong(),
                    )
                    .color(strong)
                    .anchor(Align2::CENTER_CENTER),
                );
            }
        });
    paint_gamma_bar(ui, response.transform.frame(), cmap, lo, hi);
}

/// A small horizontal colorbar for Γ(φ) in the bottom-right of the frame.
fn paint_gamma_bar(ui: &Ui, frame: &Rect, cmap: fn(f64) -> [u8; 4], lo: f64, hi: f64) {
    const STEPS: usize = 32;
    let size = egui::vec2(140.0, 10.0);
    let bar = Rect::from_min_size(frame.right_bottom() - size - egui::vec2(12.0, 22.0), size);
    let painter = ui.painter();
    let text_color = ui.visuals().strong_text_color();
    painter.rect_filled(bar.expand(4.0), 3.0, ui.visuals().extreme_bg_color);
    let step = bar.width() / STEPS as f32;
    for k in 0..STEPS {
        let [r, g, b, _] = cmap((k as f64 + 0.5) / STEPS as f64);
        let cell = Rect::from_min_size(
            bar.left_top() + egui::vec2(k as f32 * step, 0.0),
            egui::vec2(step + 0.5, size.y),
        );
        painter.rect_filled(cell, 0.0, Color32::from_rgb(r, g, b));
    }
    painter.text(
        bar.center_top() - egui::vec2(0.0, 4.0),
        Align2::CENTER_BOTTOM,
        "Γ (1/ps)",
        FontId::proportional(12.0),
        text_color,
    );
    for (pos, align, value) in [
        (bar.left_bottom(), Align2::LEFT_TOP, lo),
        (bar.right_bottom(), Align2::RIGHT_TOP, hi),
    ] {
        painter.text(
            pos + egui::vec2(0.0, 3.0),
            align,
            format!("{value:.0}"),
            FontId::proportional(11.0),
            text_color,
        );
    }
}

/// SPECULATIVE coupling against Tc: λ_tr per doping (filled), λ_ρ (hollow),
/// both pairing maps (power law solid, McMillan dotted, the selected one
/// emphasised), the threshold of the selected map dotted, the current x
/// ringed and the glue operating point as a star.
fn show_sf_coupling_plot(ui: &mut Ui, app: &MoireApp, rt: &SfRuntime) {
    let dark = app.dark_mode;
    let r = &rt.readouts;
    let curves = &rt.pairing_curves;
    let ref_color = muted_line_color(dark);
    let strong = ui.visuals().strong_text_color();
    let rows = &rt.transport.doping;
    let lambda_tr: Vec<[f64; 2]> = rows.iter().map(|p| [p.lambda_tr, p.tc]).collect();
    let lambda_rho: Vec<[f64; 2]> = rows.iter().map(|p| [p.lambda_rho, p.tc]).collect();
    plot_title(ui, "Coupling vs Tc across the LCCO dome (SPECULATIVE)");
    Plot::new("sf_coupling")
        .legend(Legend::default().position(Corner::LeftTop))
        .x_axis_label("coupling λ")
        .y_axis_label("Tc (K)")
        .include_x(0.0)
        .include_y(0.0)
        .show(ui, |plot_ui| {
            for (form, values, name, color) in [
                (
                    SfPairingForm::PowerLaw,
                    &curves.power_law,
                    "empirical power law",
                    series_color(2, dark),
                ),
                (
                    SfPairingForm::McMillan,
                    &curves.mcmillan,
                    "McMillan form, best fit (fails)",
                    ref_color,
                ),
            ] {
                let width = if form == rt.pairing.form {
                    2.5_f32
                } else {
                    1.2_f32
                };
                let mut line = Line::new(xy(&curves.lambda, values, 1.0))
                    .name(name)
                    .color(color)
                    .width(width);
                if form == SfPairingForm::McMillan {
                    line = line.style(LineStyle::dotted_dense());
                }
                plot_ui.line(line);
            }
            plot_ui.vline(
                VLine::new(r.lambda_threshold)
                    .name("threshold λ_c (selected map)")
                    .color(ref_color)
                    .style(LineStyle::dotted_loose()),
            );
            plot_ui.points(
                Points::new(lambda_tr)
                    .name("λ_tr (FS-averaged T-linear coupling)")
                    .shape(MarkerShape::Circle)
                    .radius(4.0_f32)
                    .color(series_color(0, dark)),
            );
            plot_ui.points(
                Points::new(lambda_rho)
                    .name("λ_ρ (Allen–Drude, from ρ(T))")
                    .shape(MarkerShape::Circle)
                    .filled(false)
                    .radius(4.5_f32)
                    .color(series_color(1, dark)),
            );
            plot_ui.points(
                Points::new(vec![[r.lambda_tr, r.tc]])
                    .name(format!("current x = {:.3}", rt.transport.config.doping))
                    .shape(MarkerShape::Circle)
                    .filled(false)
                    .radius(8.0_f32)
                    .color(strong),
            );
            plot_ui.points(
                Points::new(vec![[r.lambda_bar, rt.glue_free.tc_target]])
                    .name(format!(
                        "glue operating point, Tc* = {:.1} K",
                        rt.glue_free.tc_target
                    ))
                    .shape(MarkerShape::Asterisk)
                    .radius(8.0_f32)
                    .color(strong),
            );
        });
}

/// SPECULATIVE glue imprint against the operating Tc*: the dressed curve
/// (solid), the local limit (dashed; drawn once, as the dressed curve, in the
/// local model), the linear η·S (dotted), the amplitude-route reference, the
/// operating point as a star and, in the local limit only, the "Tc(r) = 0"
/// shading. The y axis is capped at max(0.5, 1.5 × max(dressed, 0.165)); the
/// local curves leave the frame above it.
fn show_sf_imprint_plot(ui: &mut Ui, app: &MoireApp, rt: &SfRuntime) {
    let Some(imprint) = rt.imprint.as_ref() else {
        let message = rt
            .glue_message
            .as_deref()
            .unwrap_or("Computing the glue imprint...");
        show_computing_message(ui, message);
        return;
    };
    let dark = app.dark_mode;
    let coherence = rt.glue_free.coherence;
    let reference = sf::bcs_route_relative_modulation();
    let ref_color = muted_line_color(dark);
    let strong = ui.visuals().strong_text_color();
    let max_dressed = imprint.dressed.iter().copied().fold(0.0_f64, f64::max);
    let y_top = (1.5 * max_dressed.max(reference)).max(0.5);
    let x_min = imprint.tc_targets.first().copied().unwrap_or(0.0);
    let [sr, sg, sb, _] = series_color(1, dark).to_array();
    let shade = Color32::from_rgba_unmultiplied(sr, sg, sb, 28);
    // The comparison with the amplitude route holds at this L_m only.
    let period = rt
        .readouts
        .moire_period
        .map_or(String::new(), |lm| format!(", L_m = {lm:.1} Å"));
    plot_title(
        ui,
        &format!(
            "Glue imprint vs operating Tc*, {}{period} (SPECULATIVE)",
            rt.system
        ),
    );
    Plot::new("sf_imprint")
        .legend(Legend::default().position(Corner::RightTop))
        .x_axis_label("operating Tc* (K)")
        .y_axis_label("δΔ/<Δ>")
        .set_margin_fraction(egui::vec2(0.05, 0.0))
        .include_x(0.0)
        .include_y(0.0)
        .include_y(y_top)
        .show(ui, |plot_ui| {
            if let Some(tc_frag) = imprint.fragment_below {
                plot_ui.polygon(
                    Polygon::new(vec![
                        [x_min, 0.0],
                        [tc_frag, 0.0],
                        [tc_frag, y_top],
                        [x_min, y_top],
                    ])
                    .name("local limit: Tc(r) = 0 in places")
                    .fill_color(shade)
                    .stroke(Stroke::NONE),
                );
            }
            plot_ui.hline(
                HLine::new(reference)
                    .name("amplitude route, measured gaps (13.5 K)")
                    .color(ref_color)
                    .style(LineStyle::dotted_dense())
                    .width(1.5_f32),
            );
            for run in clip_below(&imprint.tc_targets, &imprint.linear, y_top) {
                plot_ui.line(
                    Line::new(run)
                        .name("linear η·S (local)")
                        .color(series_color(2, dark))
                        .style(LineStyle::dotted_dense())
                        .width(1.2_f32),
                );
            }
            if coherence != SfCoherence::Local {
                for run in clip_below(&imprint.tc_targets, &imprint.local, y_top) {
                    plot_ui.line(
                        Line::new(run)
                            .name("local limit (L_m >> ξ)")
                            .color(series_color(1, dark))
                            .style(LineStyle::dashed_dense())
                            .width(1.5_f32),
                    );
                }
            }
            for run in clip_below(&imprint.tc_targets, &imprint.dressed, y_top) {
                plot_ui.line(
                    Line::new(run)
                        .name(format!("dressed ({})", coherence_plot_label(coherence)))
                        .color(series_color(0, dark))
                        .width(2.5_f32),
                );
            }
            if let Some(dressed) = rt.readouts.glue_rel_dressed {
                plot_ui.points(
                    Points::new(vec![[rt.glue_free.tc_target, dressed]])
                        .name(format!(
                            "operating point, Tc* = {:.1} K",
                            rt.glue_free.tc_target
                        ))
                        .shape(MarkerShape::Asterisk)
                        .radius(8.0_f32)
                        .color(strong),
                );
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_below_splits_at_the_cap() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0, 2.0, 0.0, 0.5, 0.25];
        let runs = clip_below(&xs, &ys, 1.0);
        assert_eq!(runs.len(), 2);
        // Up to the crossing at x = 0.5, then back from x = 1.5.
        assert_eq!(runs[0], vec![[0.0, 0.0], [0.5, 1.0]]);
        assert_eq!(
            runs[1],
            vec![[1.5, 1.0], [2.0, 0.0], [3.0, 0.5], [4.0, 0.25]]
        );
        // A curve wholly inside is one run; wholly above, none.
        assert_eq!(clip_below(&xs, &ys, 5.0).len(), 1);
        assert!(clip_below(&xs, &[9.0; 5], 1.0).is_empty());
    }
}
