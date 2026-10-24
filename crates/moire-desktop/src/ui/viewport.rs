use std::ops::RangeInclusive;

use egui::{Color32, Pos2, Rect, Ui};
use egui_plot::{
    GridInput, GridMark, HLine, Legend, Line, LineStyle, Plot, PlotPoint, Text, VLine,
};
use moire_core::pdw::{self, PdwFamily, PdwRegime};

use crate::app::{CooperView, GrapheneView, MagneticView, MoireApp, PdwView, Tab, ViewMode};
use crate::render::axes::{self, AxesSpec, ColorbarSpec};
use crate::ui::pdw_panel::{family_color, regime_color};

/// Camera rotation sensitivity (radians per pixel of drag).
const CAMERA_ROTATION_SENSITIVITY: f32 = 0.01;
/// Width of the tab bar's 2D/3D toggle group before its first frame has
/// measured it (see `show_tab_bar`).
const VIEW_TOGGLE_WIDTH_ESTIMATE: f32 = 72.0;
/// Camera zoom sensitivity (distance per pixel of scroll).
const CAMERA_ZOOM_SENSITIVITY: f32 = 0.005;
/// Bands whose energies never enter this +/- window (meV) are not drawn, so
/// the default plot bounds stay focused on the flat bands.
const BAND_PLOT_WINDOW_MEV: f64 = 250.0;
/// Highlight color for the two flat bands and the DOS trace; readable on
/// both the dark and light themes.
const FLAT_BAND_COLOR: Color32 = Color32::from_rgb(255, 140, 0);

/// Render the main visualization viewport with tab bar and 2D/3D toggle.
pub fn show_viewport(ui: &mut Ui, app: &mut MoireApp) {
    let prev_tab = app.active_tab;

    show_tab_bar(ui, app);

    if app.active_tab != prev_tab && app.view_mode == ViewMode::Surface3D {
        app.needs_surface_rerender = true;
    }

    ui.separator();

    // Bands and DOS are egui_plot line charts, not textures: route them past
    // the texture paths in either view mode, so the 2D/3D toggle is inert.
    if app.active_tab == Tab::Graphene {
        match app.graphene_view {
            GrapheneView::Bands => {
                show_band_plot(ui, app);
                return;
            }
            GrapheneView::Dos => {
                show_dos_plot(ui, app);
                return;
            }
            _ => {}
        }
    }

    // The Cooper decay profile is a line chart too; route it past both texture
    // paths so the 2D/3D toggle is inert.
    if app.active_tab == Tab::CooperSurface3D && app.cooper_view == CooperView::DecayProfile {
        show_decay_plot(ui, app);
        return;
    }

    // A failed PDW computation (e.g. identical lattices at zero twist in the
    // moire-locked geometry) explains itself instead of "Computing...".
    if app.active_tab == Tab::PairDensityWave {
        if let Some(err) = app.pdw_error.as_ref() {
            show_computing_message(ui, &format!("PDW model unavailable: {err}"));
            return;
        }
    }

    // The PDW sweeps and regime map live on (T, |B|) axes, not the moire grid.
    if app.active_tab == Tab::PairDensityWave && app.pdw_view.is_plot() {
        match app.pdw_view {
            PdwView::TemperatureSweep => show_pdw_sweep_plot(ui, app, false),
            PdwView::FieldSweep => show_pdw_sweep_plot(ui, app, true),
            _ => show_pdw_regime_map(ui, app),
        }
        return;
    }

    // Every SF view but the glue map is a plot on its own axes. A failed SF
    // computation, or a glue map with no moire to build on, explains itself
    // instead of "Computing...".
    if app.active_tab == Tab::SpinFluctuation {
        if let Some(err) = app.sf_error.as_ref() {
            show_computing_message(ui, &format!("Spin-fluctuation model unavailable: {err}"));
            return;
        }
        if app.sf_view.is_plot() {
            crate::ui::sf_plots::show(ui, app);
            return;
        }
        if let Some(message) = app
            .sf_result
            .as_ref()
            .and_then(|rt| rt.glue_message.as_ref())
        {
            show_computing_message(ui, message);
            return;
        }
    }

    match app.view_mode {
        ViewMode::Flat2D => show_flat_2d(ui, app),
        ViewMode::Surface3D => show_surface_3d(ui, app),
    }
}

/// The tab bar and the 2D/3D toggle. Returns the bar's response. It fits one
/// row of the default 1280 px window (a test checks it) and wraps in narrower
/// windows (the minimum is 800 px) or a zoomed UI, so that neither a tab nor
/// the toggle is clipped.
pub fn show_tab_bar(ui: &mut Ui, app: &mut MoireApp) -> egui::Response {
    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(&mut app.active_tab, Tab::Pattern, "Moire Pattern");
        ui.selectable_value(&mut app.active_tab, Tab::Density, "Density Modulation");
        ui.selectable_value(&mut app.active_tab, Tab::Fourier, "Fourier Spectrum");
        ui.selectable_value(&mut app.active_tab, Tab::MagneticField, "Magnetic");
        ui.selectable_value(&mut app.active_tab, Tab::CooperSurface3D, "Cooper 3D");
        ui.selectable_value(&mut app.active_tab, Tab::Graphene, "Graphene");
        ui.selectable_value(
            &mut app.active_tab,
            Tab::SpinFluctuation,
            "Spin Fluctuations",
        );
        ui.selectable_value(
            &mut app.active_tab,
            Tab::PairDensityWave,
            "Pair Density Wave",
        );
        // The separator and the two toggle buttons move to the next row as one
        // unit (never "2D" ending a row and "3D" starting the next). A nested
        // `ui.horizontal` would not do: egui hands it the rest of the current
        // row, so it overflows instead of wrapping. The group therefore asks
        // the wrapping layout for its own width, measured on the last frame.
        let toggle_id = egui::Id::new("tab_bar_view_toggle");
        let toggle_width = ui
            .data(|d| d.get_temp::<f32>(toggle_id))
            .unwrap_or(VIEW_TOGGLE_WIDTH_ESTIMATE);
        let toggle = ui.allocate_ui_with_layout(
            egui::vec2(toggle_width, ui.spacing().interact_size.y),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.separator();
                let _ = ui.selectable_value(&mut app.view_mode, ViewMode::Flat2D, "2D");
                if ui
                    .selectable_value(&mut app.view_mode, ViewMode::Surface3D, "3D")
                    .changed()
                {
                    app.needs_surface_rerender = true;
                }
            },
        );
        ui.data_mut(|d| d.insert_temp(toggle_id, toggle.response.rect.width()));
    })
    .response
}

/// Per-tab axis + colorbar metadata for the 2D view.
struct ViewMeta {
    x_range: (f64, f64),
    y_range: (f64, f64),
    x_label: &'static str,
    y_label: &'static str,
    colormap: fn(f64) -> [u8; 4],
    value_range: (f64, f64),
    value_label: &'static str,
}

fn view_meta(app: &MoireApp) -> Option<ViewMeta> {
    let extent = app
        .moire_result
        .as_ref()
        .map(|m| m.physical_extent)
        .unwrap_or(app.physical_extent);
    let half = extent / 2.0;
    let resolution = app
        .moire_result
        .as_ref()
        .map(|m| m.resolution)
        .unwrap_or(app.resolution);

    let density_range = || -> (f64, f64) {
        if let Some(ref d) = app.density_result {
            let min = d.gap_field.iter().copied().fold(f64::INFINITY, f64::min);
            let max = d
                .gap_field
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if min.is_finite() && max.is_finite() && (max - min).abs() > 1e-12 {
                return (min, max);
            }
        }
        let avg = (app.density_config.delta_1 + app.density_config.delta_2) / 2.0;
        (avg - 0.5, avg + 0.5)
    };

    Some(match app.active_tab {
        Tab::Pattern => ViewMeta {
            x_range: (-half, half),
            y_range: (-half, half),
            x_label: "x (Å)",
            y_label: "y (Å)",
            colormap: app.cmap(moire_core::colormap::viridis),
            value_range: (0.0, 1.0),
            value_label: "intensity",
        },
        Tab::Density => ViewMeta {
            x_range: (-half, half),
            y_range: (-half, half),
            x_label: "x (Å)",
            y_label: "y (Å)",
            colormap: app.cmap(moire_core::colormap::coolwarm),
            value_range: density_range(),
            value_label: "Δ (meV)",
        },
        Tab::CooperSurface3D => match app.cooper_view {
            CooperView::Majorana => ViewMeta {
                x_range: (-half, half),
                y_range: (-half, half),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap: app.cmap(moire_core::colormap::viridis),
                value_range: (0.0, 1.0),
                value_label: "|ψ|² (norm)",
            },
            // Gap views (InterfaceGap / ZSlice); DecayProfile is routed to a
            // plot before reaching here.
            _ => ViewMeta {
                x_range: (-half, half),
                y_range: (-half, half),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap: app.cmap(moire_core::colormap::coolwarm),
                value_range: app.cooper_value_range().unwrap_or((0.0, 1.0)),
                value_label: "Δ(z) (meV)",
            },
        },
        Tab::Fourier => {
            let k_nyq = if extent > 0.0 {
                std::f64::consts::PI * resolution as f64 / extent
            } else {
                1.0
            };
            ViewMeta {
                x_range: (-k_nyq, k_nyq),
                y_range: (-k_nyq, k_nyq),
                x_label: "kx (1/Å)",
                y_label: "ky (1/Å)",
                colormap: app.cmap(moire_core::colormap::inferno),
                value_range: (0.0, 1.0),
                value_label: "log₁₀(|F|²) norm",
            }
        }
        Tab::MagneticField => match app.magnetic_view {
            MagneticView::Susceptibility => ViewMeta {
                x_range: (-half, half),
                y_range: (-half, half),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap: app.cmap(moire_core::colormap::plasma),
                value_range: (-1.0, 0.0),
                value_label: "χ (norm)",
            },
            MagneticView::ScreeningCurrent => ViewMeta {
                x_range: (-half, half),
                y_range: (-half, half),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap: app.cmap(moire_core::colormap::plasma),
                value_range: (0.0, 1.0),
                value_label: "|j| / j_max",
            },
            MagneticView::CombinedGap => ViewMeta {
                x_range: (-half, half),
                y_range: (-half, half),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap: app.cmap(moire_core::colormap::coolwarm),
                value_range: density_range(),
                value_label: "Δ (meV)",
            },
        },
        Tab::Graphene => {
            // Fourier lives in k-space; Bands / Dos bypass the texture
            // display entirely but still get sensible plot-shaped meta.
            match app.graphene_view {
                GrapheneView::Fourier => {
                    let (g_res, g_extent) = app
                        .graphene_result
                        .as_ref()
                        .map(|g| (g.resolution, g.physical_extent))
                        .unwrap_or((resolution, extent));
                    let k_nyq = if g_extent > 0.0 {
                        std::f64::consts::PI * g_res as f64 / g_extent
                    } else {
                        1.0
                    };
                    return Some(ViewMeta {
                        x_range: (-k_nyq, k_nyq),
                        y_range: (-k_nyq, k_nyq),
                        x_label: "kx (1/Å)",
                        y_label: "ky (1/Å)",
                        colormap: app.cmap(moire_core::colormap::inferno),
                        value_range: (0.0, 1.0),
                        value_label: "log₁₀(|F|²) norm",
                    });
                }
                GrapheneView::Bands => {
                    let k_max = app
                        .band_structure
                        .as_ref()
                        .and_then(|b| b.k_distances.last().copied())
                        .filter(|&k| k > 0.0)
                        .unwrap_or(1.0);
                    return Some(ViewMeta {
                        x_range: (0.0, k_max),
                        y_range: (-BAND_PLOT_WINDOW_MEV, BAND_PLOT_WINDOW_MEV),
                        x_label: "k-path (1/Å)",
                        y_label: "E (meV)",
                        colormap: app.cmap(moire_core::colormap::viridis),
                        value_range: (-BAND_PLOT_WINDOW_MEV, BAND_PLOT_WINDOW_MEV),
                        value_label: "E (meV)",
                    });
                }
                GrapheneView::Dos => {
                    let (e_min, e_max, d_max) = app
                        .dos_result
                        .as_ref()
                        .map(|d| {
                            (
                                d.energies_mev.first().copied().unwrap_or(-1.0),
                                d.energies_mev.last().copied().unwrap_or(1.0),
                                d.dos.iter().copied().fold(0.0_f64, f64::max).max(1e-12),
                            )
                        })
                        .unwrap_or((-1.0, 1.0, 1.0));
                    return Some(ViewMeta {
                        x_range: (e_min, e_max),
                        y_range: (0.0, d_max),
                        x_label: "E (meV)",
                        y_label: "DOS",
                        colormap: app.cmap(moire_core::colormap::viridis),
                        value_range: (0.0, d_max),
                        value_label: "DOS (meV⁻¹ k⁻¹)",
                    });
                }
                _ => {}
            }
            let ghalf = app
                .graphene_result
                .as_ref()
                .map(|g| g.physical_extent)
                .unwrap_or(extent)
                / 2.0;
            #[allow(clippy::type_complexity)]
            let (colormap, value_range, value_label): (fn(f64) -> [u8; 4], (f64, f64), &'static str) =
                match app.graphene_view {
                GrapheneView::Pattern => {
                    (app.cmap(moire_core::colormap::viridis), (0.0, 1.0), "intensity")
                }
                GrapheneView::GapMap => {
                    let range = app
                        .graphene_gap
                        .as_ref()
                        .and_then(|g| {
                            let min = g.iter().copied().fold(f64::INFINITY, f64::min);
                            let max = g.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            if min.is_finite() && max.is_finite() && (max - min).abs() > 1e-12 {
                                Some((min, max))
                            } else {
                                None
                            }
                        })
                        .unwrap_or((0.0, 1.0));
                    (app.cmap(moire_core::colormap::coolwarm), range, "Δ (meV)")
                }
                GrapheneView::PseudoField => {
                    let b_max = app
                        .curvature_result
                        .as_ref()
                        .map(|c| c.max_abs_field)
                        .filter(|&m| m > 1e-15)
                        .unwrap_or(1.0);
                    (app.cmap(moire_core::colormap::coolwarm), (-b_max, b_max), "B_ps (T)")
                }
                GrapheneView::Strain => {
                    let s_max = app
                        .curvature_result
                        .as_ref()
                        .map(|c| {
                            c.strain_xx
                                .iter()
                                .zip(&c.strain_yy)
                                .zip(&c.strain_xy)
                                .map(|((&xx, &yy), &xy)| {
                                    (xx * xx + yy * yy + 2.0 * xy * xy).sqrt()
                                })
                                .fold(0.0_f64, f64::max)
                        })
                        .filter(|&m| m > 0.0)
                        .unwrap_or(1.0);
                    (app.cmap(moire_core::colormap::viridis), (0.0, s_max), "|ε|")
                }
                // Handled by the early returns above.
                GrapheneView::Fourier | GrapheneView::Bands | GrapheneView::Dos => {
                    unreachable!()
                }
            };
            ViewMeta {
                x_range: (-ghalf, ghalf),
                y_range: (-ghalf, ghalf),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap,
                value_range,
                value_label,
            }
        }
        Tab::SpinFluctuation => {
            // Only the glue map reaches here; the plot views route earlier.
            let glue = app
                .sf_result
                .as_ref()
                .and_then(|rt| rt.glue.as_ref().map(|g| (&g.gap, rt.glue_extent)));
            let shalf = glue.map_or(half, |(_, e)| e / 2.0);
            let value_range = glue.map_or((0.0, 1.0), |(gap, _)| {
                let lo = gap.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = gap.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                // A uniform map (η = 0) sits mid-colormap: centre a small span on it.
                if hi - lo > 1e-12 {
                    (lo, hi)
                } else {
                    (lo - 5e-4, hi + 5e-4)
                }
            });
            ViewMeta {
                x_range: (-shalf, shalf),
                y_range: (-shalf, shalf),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap: app.cmap(moire_core::colormap::coolwarm),
                value_range,
                value_label: "Δ (meV)",
            }
        }
        Tab::PairDensityWave => {
            let (p_res, p_extent) = app
                .pdw_result
                .as_ref()
                .map(|r| (r.fields.resolution, r.fields.physical_extent))
                .unwrap_or((resolution, extent));
            if app.pdw_view == PdwView::Fourier {
                let k_nyq = if p_extent > 0.0 {
                    std::f64::consts::PI * p_res as f64 / p_extent
                } else {
                    1.0
                };
                return Some(ViewMeta {
                    x_range: (-k_nyq, k_nyq),
                    y_range: (-k_nyq, k_nyq),
                    x_label: "kx (1/Å)",
                    y_label: "ky (1/Å)",
                    colormap: app.cmap(moire_core::colormap::inferno),
                    value_range: (0.0, 1.0),
                    value_label: "log₁₀(|F|²) norm",
                });
            }
            // Signed densities are mapped symmetrically, so the colorbar is too.
            let symmetric = |field: &[f64]| {
                let m = field.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1e-12);
                (-m, m)
            };
            let (value_range, value_label) = match (app.pdw_view, app.pdw_result.as_ref()) {
                (PdwView::PairAmplitude, Some(r)) => {
                    let pair = &r.fields.pair_amplitude;
                    let lo = pair.iter().copied().fold(f64::INFINITY, f64::min);
                    let hi = pair.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    ((lo, hi.max(lo + 1e-3)), "|d| (meV)")
                }
                (PdwView::SpinDensity, Some(r)) => {
                    (symmetric(&r.fields.spin_density), "S_z (arb.)")
                }
                (_, Some(r)) => (symmetric(&r.fields.charge_density), "ρ (arb.)"),
                (_, None) => ((-1.0, 1.0), "ρ (arb.)"),
            };
            let phalf = p_extent / 2.0;
            ViewMeta {
                x_range: (-phalf, phalf),
                y_range: (-phalf, phalf),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap: app.cmap(moire_core::colormap::coolwarm),
                value_range,
                value_label,
            }
        }
    })
}

fn show_flat_2d(ui: &mut Ui, app: &MoireApp) {
    let texture = match app.active_tab {
        Tab::Pattern => app.pattern_texture.as_ref(),
        Tab::Density => app.density_texture.as_ref(),
        Tab::Fourier => app.fft_texture.as_ref(),
        Tab::MagneticField => app.magnetic_texture.as_ref(),
        Tab::CooperSurface3D => app.cooper_texture.as_ref(),
        Tab::Graphene => app.graphene_texture.as_ref(),
        Tab::SpinFluctuation => app.sf_texture.as_ref(),
        Tab::PairDensityWave => app.pdw_texture.as_ref(),
    };

    // The Fourier tab tucks a collapsible peak table below the spectrum. Only
    // shrink the image when the table is open, so the spectrum stays near
    // full-height by default.
    if app.active_tab == Tab::Fourier {
        let full = ui.available_rect_before_wrap();
        let header_id = ui.make_persistent_id("fft_peaks_header");
        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            header_id,
            false,
        );
        let table_reserve = if state.is_open() {
            (full.height() * 0.4).clamp(160.0, 300.0)
        } else {
            30.0
        };
        let image_h = (full.height() - table_reserve).max(120.0);
        let image_area = Rect::from_min_size(full.min, egui::vec2(full.width(), image_h));
        ui.allocate_rect(image_area, egui::Sense::hover());
        paint_texture_view(ui, app, texture, image_area);
        show_fft_peak_table(ui, app, state, table_reserve);
        return;
    }

    let available = ui.available_rect_before_wrap();
    ui.allocate_rect(available, egui::Sense::hover());
    paint_texture_view(ui, app, texture, available);
}

/// Paint a texture view (image + axes + colorbar) into `area`, or a centered
/// "Computing..." placeholder when the texture is missing.
fn paint_texture_view(ui: &Ui, app: &MoireApp, texture: Option<&egui::TextureHandle>, area: Rect) {
    let (image_rect, colorbar_rect) = axes::layout(area);
    let Some(meta) = view_meta(app) else {
        return;
    };

    if let Some(tex) = texture {
        // Data row 0 holds y_min (moire-core samples y = (iy / (n - 1) - 0.5) *
        // extent; the shifted FFT's row 0 is -k_nyq), so flip v to put it at
        // the bottom, where `draw_axes` labels `y_range.0`, as the web
        // heatmaps and the 3D surface's +y axis do. The 3D image
        // (`show_surface_3d`) and the regime map (built with T rising upward)
        // keep uv (0,0)-(1,1).
        let uv = Rect::from_min_max(Pos2::new(0.0, 1.0), Pos2::new(1.0, 0.0));
        ui.painter().image(tex.id(), image_rect, uv, Color32::WHITE);

        let axes_spec = AxesSpec {
            x_range: meta.x_range,
            y_range: meta.y_range,
            x_label: meta.x_label,
            y_label: meta.y_label,
            ticks: 5,
        };
        axes::draw_axes(ui.painter(), image_rect, &axes_spec, app.dark_mode);

        let cb_spec = ColorbarSpec {
            colormap: meta.colormap,
            value_range: meta.value_range,
            label: meta.value_label,
            ticks: 5,
        };
        axes::draw_colorbar(ui.painter(), colorbar_rect, &cb_spec, app.dark_mode);
    } else {
        ui.painter().text(
            image_rect.center(),
            egui::Align2::CENTER_CENTER,
            "Computing...",
            egui::FontId::default(),
            ui.visuals().text_color(),
        );
    }
}

/// Collapsible table of the strongest FFT peaks (kx, ky, |k|, wavelength,
/// amplitude). Scrolls internally so it never overflows its reserved band.
fn show_fft_peak_table(
    ui: &mut Ui,
    app: &MoireApp,
    state: egui::collapsing_header::CollapsingState,
    reserve: f32,
) {
    state
        .show_header(ui, |ui| {
            ui.strong("FFT peaks (top 20)");
        })
        .body(|ui| {
            let peaks = app.fft_peaks.as_deref().unwrap_or(&[]);
            if peaks.is_empty() {
                ui.weak("No peaks above threshold.");
                return;
            }
            egui::ScrollArea::vertical()
                .max_height((reserve - 28.0).max(60.0))
                .show(ui, |ui| {
                    render_fft_table(ui, peaks);
                });
        });
}

fn render_fft_table(ui: &mut Ui, peaks: &[moire_core::fft::FftPeak]) {
    use egui_extras::{Column, TableBuilder};

    TableBuilder::new(ui)
        .striped(true)
        .columns(Column::auto(), 4)
        .column(Column::remainder())
        .header(18.0, |mut header| {
            for label in [
                "kx (1/Å)",
                "ky (1/Å)",
                "|k| (1/Å)",
                "λ = 2π/|k| (Å)",
                "amplitude",
            ] {
                header.col(|ui| {
                    ui.strong(label);
                });
            }
        })
        .body(|mut body| {
            for p in peaks {
                let k_mag = (p.kx * p.kx + p.ky * p.ky).sqrt();
                let lambda = if k_mag > 1e-12 {
                    2.0 * std::f64::consts::PI / k_mag
                } else {
                    f64::INFINITY
                };
                body.row(16.0, |mut row| {
                    row.col(|ui| {
                        ui.monospace(format!("{:.4}", p.kx));
                    });
                    row.col(|ui| {
                        ui.monospace(format!("{:.4}", p.ky));
                    });
                    row.col(|ui| {
                        ui.monospace(format!("{k_mag:.4}"));
                    });
                    row.col(|ui| {
                        ui.monospace(format!("{lambda:.1}"));
                    });
                    row.col(|ui| {
                        ui.monospace(format!("{:.4}", p.amplitude));
                    });
                });
            }
        });
}

fn show_surface_3d(ui: &mut Ui, app: &mut MoireApp) {
    let available = ui.available_size();
    let side = available.x.min(available.y);
    let size = egui::vec2(side, side);

    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());

    if response.dragged() {
        let delta = response.drag_delta();
        // Drag right: the world turns counter-clockwise seen from above, so
        // the near edge follows the pointer.
        app.camera.yaw += delta.x * CAMERA_ROTATION_SENSITIVITY;
        // Drag down (egui's delta.y > 0) to look more from above (pitch toward
        // 0, the plan view), up to look more edge-on: the orbit convention of
        // three.js, Blender and Plotly 3D.
        app.camera.pitch =
            (app.camera.pitch - delta.y * CAMERA_ROTATION_SENSITIVITY).clamp(-1.4, 1.4);
        app.needs_surface_rerender = true;
    }

    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            app.camera.distance =
                (app.camera.distance - scroll * CAMERA_ZOOM_SENSITIVITY).clamp(1.0, 5.0);
            app.needs_surface_rerender = true;
        }
    }

    if let Some(ref tex) = app.surface_texture {
        let uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        ui.painter().image(tex.id(), rect, uv, Color32::WHITE);
    } else {
        ui.painter()
            .rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Computing 3D view...",
            egui::FontId::default(),
            ui.visuals().text_color(),
        );
    }

    if response.hovered() {
        ui.painter().text(
            rect.left_bottom() + egui::vec2(8.0, -20.0),
            egui::Align2::LEFT_BOTTOM,
            "Drag to rotate \u{2022} Scroll to zoom",
            egui::FontId::proportional(12.0),
            ui.visuals().weak_text_color(),
        );
    }
}

/// Centered placeholder text while a plot's backing data is missing.
pub(crate) fn show_computing_message(ui: &mut Ui, message: &str) {
    let rect = ui.available_rect_before_wrap();
    ui.allocate_rect(rect, egui::Sense::hover());
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        message,
        egui::FontId::default(),
        ui.visuals().text_color(),
    );
}

/// Line color for remote (non-flat) bands and reference lines.
pub(crate) fn muted_line_color(dark_mode: bool) -> Color32 {
    if dark_mode {
        Color32::from_gray(110)
    } else {
        Color32::from_gray(160)
    }
}

/// Draw the BM band structure: one line per band entering the display
/// window, the two central (flat) bands highlighted, x-axis ticks at the
/// K / Γ / M / K' path corners, and a flat-bandwidth overlay label.
fn show_band_plot(ui: &mut Ui, app: &MoireApp) {
    let Some(bs) = app.band_structure.as_ref() else {
        show_computing_message(ui, "Computing band structure...");
        return;
    };
    let n_bands = bs.energies_mev.first().map_or(0, Vec::len);
    if bs.k_distances.is_empty() || n_bands < 2 {
        show_computing_message(ui, "Computing band structure...");
        return;
    }

    let flat_lo = n_bands / 2 - 1;
    let remote_color = muted_line_color(app.dark_mode);
    let k_span = bs
        .k_distances
        .last()
        .copied()
        .unwrap_or(1.0)
        .max(f64::MIN_POSITIVE);
    let tick_tol = k_span * 1e-3;
    let spacer_ticks = bs.tick_positions.clone();
    let format_ticks: Vec<(f64, &'static str)> = bs
        .tick_positions
        .iter()
        .copied()
        .zip(bs.tick_labels.iter().copied())
        .collect();

    let response = Plot::new("graphene_bands")
        .x_axis_label("k-path")
        .y_axis_label("E (meV)")
        .x_grid_spacer(move |_input: GridInput| {
            spacer_ticks
                .iter()
                .map(|&value| GridMark {
                    value,
                    step_size: k_span,
                })
                .collect()
        })
        .x_axis_formatter(move |mark: GridMark, _range: &RangeInclusive<f64>| {
            format_ticks
                .iter()
                .find(|(pos, _)| (mark.value - pos).abs() < tick_tol)
                .map(|(_, label)| (*label).to_string())
                .unwrap_or_default()
        })
        .show(ui, |plot_ui| {
            for band in 0..n_bands {
                let enters_window = bs
                    .energies_mev
                    .iter()
                    .any(|energies| energies[band].abs() <= BAND_PLOT_WINDOW_MEV);
                if !enters_window {
                    continue;
                }
                let points: Vec<[f64; 2]> = bs
                    .k_distances
                    .iter()
                    .zip(bs.energies_mev.iter())
                    .map(|(&k, energies)| [k, energies[band]])
                    .collect();
                let line = if band == flat_lo || band == flat_lo + 1 {
                    Line::new(points).color(FLAT_BAND_COLOR).width(2.5_f32)
                } else {
                    Line::new(points).color(remote_color).width(1.0_f32)
                };
                plot_ui.line(line);
            }
        });

    ui.painter().text(
        response.response.rect.left_top() + egui::vec2(56.0, 8.0),
        egui::Align2::LEFT_TOP,
        format!(
            "flat bandwidth {:.2} meV \u{2022} gap to remote {:.2} meV",
            bs.flat_bandwidth_mev, bs.flat_gap_mev
        ),
        egui::FontId::proportional(12.0),
        ui.visuals().strong_text_color(),
    );
}

/// Draw the BM density of states as a single line with a dashed vertical
/// reference at the charge-neutrality point E = 0.
fn show_dos_plot(ui: &mut Ui, app: &MoireApp) {
    let Some(dos) = app.dos_result.as_ref() else {
        show_computing_message(ui, "Computing density of states...");
        return;
    };
    let points: Vec<[f64; 2]> = dos
        .energies_mev
        .iter()
        .zip(dos.dos.iter())
        .map(|(&e, &d)| [e, d])
        .collect();
    let zero_color = muted_line_color(app.dark_mode);

    Plot::new("graphene_dos")
        .x_axis_label("E (meV)")
        .y_axis_label("DOS (meV⁻¹ k⁻¹)")
        .include_y(0.0)
        .show(ui, |plot_ui| {
            plot_ui.vline(
                VLine::new(0.0)
                    .color(zero_color)
                    .style(LineStyle::dashed_loose()),
            );
            plot_ui.line(Line::new(points).color(FLAT_BAND_COLOR).width(2.0_f32));
        });
}

/// Draw the proximity decay profile f(z): unity inside the superconductor
/// (z < 0), exponential decay into the TI beyond the interface. Dashed
/// verticals mark the interface (z = 0) and the coherence length xi_prox.
fn show_decay_plot(ui: &mut Ui, app: &MoireApp) {
    let (Some(z_coords), Some(decay)) =
        (app.cooper_z_coords.as_ref(), app.cooper_decay.as_ref())
    else {
        show_computing_message(ui, "Computing proximity decay...");
        return;
    };
    let points: Vec<[f64; 2]> = z_coords
        .iter()
        .zip(decay.iter())
        .map(|(&z, &f)| [z, f])
        .collect();
    let ref_color = muted_line_color(app.dark_mode);
    let xi = app.proximity_config.xi_prox;

    Plot::new("cooper_decay")
        .x_axis_label("z (Å)")
        .y_axis_label("f(z)")
        .include_y(0.0)
        .show(ui, |plot_ui| {
            plot_ui.vline(
                VLine::new(0.0)
                    .color(ref_color)
                    .style(LineStyle::dashed_loose()),
            );
            plot_ui.vline(
                VLine::new(xi)
                    .color(ref_color)
                    .style(LineStyle::dashed_loose()),
            );
            plot_ui.line(Line::new(points).color(FLAT_BAND_COLOR).width(2.0_f32));
        });
}

/// Draw the PDW composite-peak intensities |ρ_k|² on a log₁₀ axis against
/// temperature (`field_axis = false`) or |B| along the sweep axis. Members 1
/// and 2 of a family are mirror partners and share one dashed line; member 3
/// is solid. Dotted verticals mark the transitions, dashed the current value.
fn show_pdw_sweep_plot(ui: &mut Ui, app: &MoireApp, field_axis: bool) {
    let Some(rt) = app.pdw_result.as_ref() else {
        show_computing_message(ui, "Computing PDW sweep...");
        return;
    };
    let cfg = &rt.config;
    let dark = app.dark_mode;
    let sweep = if field_axis { &rt.b_sweep } else { &rt.t_sweep };
    let floor = cfg.floor.max(1e-12).log10();
    let y_min = floor - 2.0;

    let pair_max =
        |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x.max(*y)).collect() };
    let curves = [
        (
            "p₁,₂ (field-induced)",
            PdwFamily::P,
            pair_max(&sweep.intensity_p[0], &sweep.intensity_p[1]),
            true,
        ),
        ("p₃", PdwFamily::P, sweep.intensity_p[2].clone(), false),
        (
            "q₁,₂",
            PdwFamily::Q,
            pair_max(&sweep.intensity_q[0], &sweep.intensity_q[1]),
            true,
        ),
        ("q₃", PdwFamily::Q, sweep.intensity_q[2].clone(), false),
        (
            "h₁,₂",
            PdwFamily::H,
            pair_max(&sweep.intensity_h[0], &sweep.intensity_h[1]),
            false,
        ),
    ];
    let y_top = curves
        .iter()
        .flat_map(|(_, _, values, _)| values.iter())
        .fold(floor, |m, &v| if v > 0.0 { m.max(v.log10()) } else { m })
        + 0.3;

    let (x_label, current, marks) = if field_axis {
        let t = cfg.temperature;
        let mut marks = vec![
            (
                pdw::critical_field_along(cfg.tc, &cfg.hc2, t, &rt.direction).unwrap_or(0.0),
                "Hc2(T)",
            ),
            (
                pdw::critical_field_along(cfg.t_pdw, &cfg.h_pdw, t, &rt.direction).unwrap_or(0.0),
                "H_PDW(T)",
            ),
        ];
        if cfg.overlap0.iter().any(|&c| c < 1.0) {
            let onset = pdw::onset_field_along(&cfg.h_rot, &rt.direction).unwrap_or(0.0);
            marks.push((onset, "d-vector onset"));
        }
        // Off-axis fields put the current state off this sweep: no marker.
        let (along, on_axis) = rt.field_along_axis();
        (
            format!(
                "|B| along {} (T)",
                app.pdw_settings.sweep_axis.phrase(cfg.field)
            ),
            on_axis.then_some(along),
            marks,
        )
    } else {
        let shifted = |t_star: f64, critical: &[f64; 3]| {
            t_star * (1.0 - pdw::reduced_field(&cfg.field, critical).unwrap_or(1.0)).max(0.0)
        };
        (
            "T (K)".to_string(),
            Some(cfg.temperature),
            vec![
                (shifted(cfg.tc, &cfg.hc2), "Tc(B)"),
                (shifted(cfg.t_pdw, &cfg.h_pdw), "T_PDW(B)"),
            ],
        )
    };
    let ref_color = muted_line_color(dark);
    let x_end = sweep.values.last().copied().unwrap_or(1.0);

    Plot::new(if field_axis {
        "pdw_field_sweep"
    } else {
        "pdw_temperature_sweep"
    })
    .legend(Legend::default())
    .x_axis_label(x_label)
    .y_axis_label("|ρ_k|² (log scale)")
    .y_axis_formatter(|mark: GridMark, _range: &RangeInclusive<f64>| {
        format!("{:.0e}", 10f64.powf(mark.value))
    })
    .include_y(y_min)
    .include_y(y_top)
    .show(ui, |plot_ui| {
        plot_ui.hline(
            HLine::new(floor)
                .color(ref_color)
                .style(LineStyle::dotted_dense()),
        );
        plot_ui.text(
            Text::new(PlotPoint::new(x_end, floor), "detection floor")
                .color(ref_color)
                .anchor(egui::Align2::RIGHT_BOTTOM),
        );
        for (x, label) in marks {
            if x.is_finite() && x > 0.0 {
                plot_ui.vline(
                    VLine::new(x)
                        .color(ref_color)
                        .style(LineStyle::dotted_loose()),
                );
                plot_ui.text(
                    Text::new(PlotPoint::new(x, y_top), label)
                        .color(ref_color)
                        .anchor(egui::Align2::LEFT_TOP),
                );
            }
        }
        if let Some(current) = current {
            plot_ui.vline(
                VLine::new(current)
                    .color(ref_color)
                    .style(LineStyle::dashed_loose()),
            );
        }
        for (name, family, values, dashed) in curves {
            let points: Vec<[f64; 2]> = sweep
                .values
                .iter()
                .zip(&values)
                .filter(|(_, &v)| v > 0.0 && v.log10() >= y_min)
                .map(|(&x, &v)| [x, v.log10()])
                .collect();
            let mut line = Line::new(points)
                .name(name)
                .color(family_color(family, dark))
                .width(2.0_f32);
            if dashed {
                line = line.style(LineStyle::dashed_dense());
            }
            plot_ui.line(line);
        }
    });
}

/// Paint the PDW regime texture on (|B| along the sweep axis, T) axes with a
/// swatch legend underneath and a cross at the current (|B|, T).
fn show_pdw_regime_map(ui: &mut Ui, app: &MoireApp) {
    let (Some(rt), Some(tex)) = (app.pdw_result.as_ref(), app.pdw_texture.as_ref()) else {
        show_computing_message(ui, "Computing regime map...");
        return;
    };
    // Room below the x-axis label for the swatch row.
    const LEGEND_HEIGHT: f32 = 30.0;
    let area = ui.available_rect_before_wrap();
    ui.allocate_rect(area, egui::Sense::hover());
    let map_area = Rect::from_min_max(area.min, Pos2::new(area.max.x, area.max.y - LEGEND_HEIGHT));
    let (image_rect, _colorbar) = axes::layout(map_area);

    let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
    ui.painter().image(tex.id(), image_rect, uv, Color32::WHITE);
    let x_label = format!(
        "|B| along {} (T)",
        app.pdw_settings.sweep_axis.phrase(rt.config.field)
    );
    let spec = AxesSpec {
        x_range: (0.0, rt.b_max),
        y_range: (0.0, rt.t_max),
        x_label: &x_label,
        y_label: "T (K)",
        ticks: 5,
    };
    axes::draw_axes(ui.painter(), image_rect, &spec, app.dark_mode);

    // Cross at the current state, which lies on this map only when the
    // applied field points along the sweep axis.
    let (along, on_axis) = rt.field_along_axis();
    let fx = (along / rt.b_max) as f32;
    let fy = (rt.config.temperature / rt.t_max) as f32;
    if on_axis && (0.0..=1.0).contains(&fx) && (0.0..=1.0).contains(&fy) {
        let c = Pos2::new(
            image_rect.min.x + fx * image_rect.width(),
            image_rect.max.y - fy * image_rect.height(),
        );
        let stroke = egui::Stroke::new(2.0_f32, ui.visuals().strong_text_color());
        let r = 6.0;
        ui.painter()
            .line_segment([c + egui::vec2(-r, -r), c + egui::vec2(r, r)], stroke);
        ui.painter()
            .line_segment([c + egui::vec2(-r, r), c + egui::vec2(r, -r)], stroke);
    }

    // Swatch legend row.
    let mut cursor = Pos2::new(image_rect.min.x, image_rect.max.y + 38.0);
    for regime in PdwRegime::ALL {
        let swatch = Rect::from_min_size(cursor, egui::vec2(12.0, 12.0));
        ui.painter()
            .rect_filled(swatch, 2.0, regime_color(regime, app.dark_mode));
        let label_rect = ui.painter().text(
            swatch.right_center() + egui::vec2(4.0, 0.0),
            egui::Align2::LEFT_CENTER,
            regime.label(),
            egui::FontId::proportional(12.0),
            ui.visuals().text_color(),
        );
        cursor.x = label_rect.max.x + 14.0;
    }
}
