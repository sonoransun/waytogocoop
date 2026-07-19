//! Topological phase-diagram popup window (speculative).
//!
//! Mirrors the web `/phase` page: a binary Fu-Kane phase map over (B, Δ) with a
//! dashed reference at the average gap, plus a vortex-period commensuration
//! plot. Kept in its own module (not a Tab) because the phase diagram lives on
//! (B, Δ) axes, not the shared moire grid — like the Compare-All-Substrates
//! window, it does not participate in the 2D/3D toggle or the texture pipeline.

use egui::{Color32, Pos2, Rect, Sense, Stroke};
use egui_plot::{HLine, Line, Plot};

use crate::app::MoireApp;
use crate::render::axes::{self, AxesSpec};

/// Phase-map grid resolution per axis.
const PHASE_N: usize = 160;
/// Trivial-phase swatch (muted gray-blue).
const TRIVIAL_COLOR: Color32 = Color32::from_rgb(70, 90, 120);
/// Topological-phase swatch (viridis-yellow endpoint).
const TOPO_COLOR: Color32 = Color32::from_rgb(253, 231, 37);

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![a];
    }
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}

/// Rebuild the binary phase-map texture from the current (B, Δ, μ, g) settings.
pub(crate) fn refresh_phase(ctx: &egui::Context, app: &mut MoireApp) {
    let b_values = linspace(0.0, app.phase_b_max, PHASE_N);
    // Δ starts just above 0 so the criterion has a nonzero gap on the first row.
    let delta_values = linspace(0.1, app.phase_delta_max, PHASE_N);
    let phase = moire_core::topological::phase_diagram_sweep(
        &b_values,
        &delta_values,
        app.g_factor,
        app.phase_mu,
    );

    // Sweep is row-major delta_idx * n_b + b_idx. Flip rows so Δ increases
    // upward in the image (row 0 = top = max Δ).
    let mut pixels = Vec::with_capacity(PHASE_N * PHASE_N);
    for row in 0..PHASE_N {
        let delta_idx = PHASE_N - 1 - row;
        for b_idx in 0..PHASE_N {
            let topo = phase[delta_idx * PHASE_N + b_idx] == 1;
            pixels.push(if topo { TOPO_COLOR } else { TRIVIAL_COLOR });
        }
    }
    let img = egui::ColorImage {
        size: [PHASE_N, PHASE_N],
        pixels,
    };
    app.phase_texture = Some(ctx.load_texture("phase_diagram", img, egui::TextureOptions::LINEAR));
}

/// Render the phase-diagram window, refreshing the texture on demand.
pub fn show(ctx: &egui::Context, app: &mut MoireApp) {
    if app.phase_needs_refresh || app.phase_texture.is_none() {
        refresh_phase(ctx, app);
        app.phase_needs_refresh = false;
    }

    let mut open = app.show_phase_diagram;
    egui::Window::new("Topological Phase Diagram")
        .open(&mut open)
        .default_width(520.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.colored_label(
                Color32::from_rgb(220, 50, 50),
                "SPECULATIVE — illustrative Fu-Kane boundary, not a prediction",
            );
            ui.add_space(6.0);

            phase_image(ui, app);

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                swatch(ui, TRIVIAL_COLOR, "trivial");
                ui.add_space(12.0);
                swatch(ui, TOPO_COLOR, "topological");
            });

            ui.add_space(8.0);
            ui.separator();

            let mut changed = false;
            changed |= ui
                .add(egui::Slider::new(&mut app.phase_b_max, 10.0..=200.0).text("B max (T)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut app.phase_delta_max, 1.0..=20.0).text("Δ max (meV)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut app.phase_mu, -5.0..=5.0).text("μ (meV)"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut app.g_factor, 1.0..=50.0).text("g-factor"))
                .changed();
            if changed {
                app.phase_needs_refresh = true;
            }

            ui.add_space(8.0);
            ui.separator();
            commensuration_plot(ui, app);
        });
    app.show_phase_diagram = open;
}

/// Paint the phase image with (B, Δ) axes and a dashed line at the average gap.
fn phase_image(ui: &mut egui::Ui, app: &MoireApp) {
    let Some(tex) = app.phase_texture.as_ref() else {
        ui.label("Computing...");
        return;
    };
    let img_side = 380.0_f32;
    let (area, _resp) = ui.allocate_exact_size(
        egui::vec2(img_side + 52.0, img_side + 34.0),
        Sense::hover(),
    );
    let image_rect = Rect::from_min_size(
        area.min + egui::vec2(44.0, 8.0),
        egui::vec2(img_side, img_side),
    );
    let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
    ui.painter().image(tex.id(), image_rect, uv, Color32::WHITE);

    let spec = AxesSpec {
        x_range: (0.0, app.phase_b_max),
        y_range: (0.1, app.phase_delta_max),
        x_label: "B (T)",
        y_label: "Δ (meV)",
        ticks: 5,
    };
    axes::draw_axes(ui.painter(), image_rect, &spec, app.dark_mode);

    // Dashed lime line at the average gap Δ_avg (mirrors the web add_hline).
    let delta_avg = (app.density_config.delta_1 + app.density_config.delta_2) / 2.0;
    if delta_avg >= 0.1 && delta_avg <= app.phase_delta_max {
        let frac = (delta_avg - 0.1) / (app.phase_delta_max - 0.1);
        let py = image_rect.max.y - frac as f32 * image_rect.height();
        let stroke = Stroke::new(1.5, Color32::from_rgb(160, 240, 60));
        let dashes = egui::Shape::dashed_line(
            &[
                Pos2::new(image_rect.min.x, py),
                Pos2::new(image_rect.max.x, py),
            ],
            stroke,
            6.0,
            4.0,
        );
        ui.painter().extend(dashes);
    }
}

fn swatch(ui: &mut egui::Ui, color: Color32, label: &str) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), Sense::hover());
    ui.painter().rect_filled(rect, 2.0, color);
    ui.label(label);
}

/// Vortex-lattice period vs B, with a reference line at the moire period. The
/// commensuration field (period match) is read off `vortex_result`.
fn commensuration_plot(ui: &mut egui::Ui, app: &MoireApp) {
    let moire_period = app
        .moire_result
        .as_ref()
        .map(|m| m.moire_period)
        .filter(|p| p.is_finite())
        .unwrap_or(0.0);

    ui.label("Vortex-period commensuration");
    if let Some(vr) = app.vortex_result.as_ref() {
        if vr.commensuration_field.is_finite() {
            ui.label(format!(
                "Commensuration field: {:.2} T",
                vr.commensuration_field
            ));
        }
    }

    // period(B) = a_v(B) ∝ 1/√B spikes at low B; cap the plotted range so the
    // crossing with the moire period stays visible.
    let cap = (moire_period * 4.0).max(200.0);
    let points: Vec<[f64; 2]> = linspace(0.01, app.phase_b_max, 200)
        .into_iter()
        .map(|b| [b, moire_core::magnetic::vortex_lattice_period(b)])
        .filter(|[_, p]| p.is_finite() && *p <= cap)
        .collect();

    Plot::new("commensuration_plot")
        .height(160.0)
        .x_axis_label("B (T)")
        .y_axis_label("period (Å)")
        .show(ui, |plot_ui| {
            plot_ui.line(Line::new(points).name("vortex period"));
            if moire_period > 0.0 {
                plot_ui.hline(
                    HLine::new(moire_period)
                        .color(Color32::from_rgb(255, 140, 0))
                        .name("moire period"),
                );
            }
        });
}
